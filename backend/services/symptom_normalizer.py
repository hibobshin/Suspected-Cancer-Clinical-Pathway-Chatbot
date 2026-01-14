"""
Symptom Normalizer Service

Uses pre-computed embeddings to normalize user-provided symptoms to the NG12 vocabulary.
This provides typo tolerance and semantic matching without per-call LLM costs.

Design:
1. Extract symptom vocabulary from parsed rules + terms_index (one-time)
2. Pre-compute embeddings for all symptoms (cached to disk)
3. At runtime: embed user input, find nearest neighbors via cosine similarity
4. Return normalized symptom if confidence exceeds threshold
"""

import json
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Lazy imports to avoid loading model on module import
_model = None
_normalizer_instance = None


def get_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model", model="all-MiniLM-L6-v2")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded")
    return _model


class SymptomNormalizer:
    """
    Normalizes user-provided symptoms to NG12 vocabulary using embeddings.
    
    Features:
    - Typo tolerance via embedding similarity
    - Semantic matching (e.g., "coughing blood" → "haemoptysis")
    - Cached embeddings for fast runtime inference
    - Auditable match scores
    """
    
    # Similarity threshold for accepting a match
    MATCH_THRESHOLD = 0.65
    
    # Higher threshold for high-confidence matches
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    
    # Cache directory
    CACHE_DIR = Path(__file__).parent.parent.parent / "data" / ".cache" / "embeddings"
    
    def __init__(self):
        self.vocabulary: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._initialized = False
    
    def initialize(self, symptoms: list[str], synonyms: dict[str, list[str]] = None):
        """
        Initialize with symptom vocabulary.
        
        Args:
            symptoms: List of canonical symptom names from NG12
            synonyms: Optional dict mapping canonical symptoms to their synonyms
        """
        # Build expanded vocabulary (canonical + synonyms)
        self.vocabulary = []
        self.canonical_map = {}  # Maps expanded term → canonical term
        
        for symptom in symptoms:
            canonical = symptom.lower().strip()
            self.vocabulary.append(canonical)
            self.canonical_map[canonical] = canonical
            
            # Add synonyms if provided
            if synonyms and canonical in synonyms:
                for syn in synonyms[canonical]:
                    syn_lower = syn.lower().strip()
                    if syn_lower not in self.vocabulary:
                        self.vocabulary.append(syn_lower)
                        self.canonical_map[syn_lower] = canonical
        
        # Try to load cached embeddings
        cache_key = self._compute_cache_key()
        if self._load_cached_embeddings(cache_key):
            logger.info("Loaded cached symptom embeddings", vocab_size=len(self.vocabulary))
        else:
            # Compute embeddings
            self._compute_embeddings()
            self._save_cached_embeddings(cache_key)
            logger.info("Computed and cached symptom embeddings", vocab_size=len(self.vocabulary))
        
        self._initialized = True
    
    def _compute_cache_key(self) -> str:
        """Compute cache key based on vocabulary."""
        vocab_str = "|".join(sorted(self.vocabulary))
        return hashlib.md5(vocab_str.encode()).hexdigest()[:16]
    
    def _load_cached_embeddings(self, cache_key: str) -> bool:
        """Load embeddings from cache if available."""
        cache_file = self.CACHE_DIR / f"symptoms_{cache_key}.npz"
        vocab_file = self.CACHE_DIR / f"symptoms_{cache_key}_vocab.json"
        
        if cache_file.exists() and vocab_file.exists():
            try:
                data = np.load(cache_file)
                self.embeddings = data['embeddings']
                
                with open(vocab_file, 'r') as f:
                    cached_vocab = json.load(f)
                
                # Verify vocabulary matches
                if cached_vocab == self.vocabulary:
                    return True
            except Exception as e:
                logger.warning("Failed to load cached embeddings", error=str(e))
        
        return False
    
    def _compute_embeddings(self):
        """Compute embeddings for all vocabulary terms."""
        model = get_model()
        
        # Embed all terms
        logger.debug("Computing embeddings", term_count=len(self.vocabulary))
        self.embeddings = model.encode(
            self.vocabulary,
            normalize_embeddings=True,  # Pre-normalize for cosine similarity
            show_progress_bar=False
        )
    
    def _save_cached_embeddings(self, cache_key: str):
        """Save embeddings to cache."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        cache_file = self.CACHE_DIR / f"symptoms_{cache_key}.npz"
        vocab_file = self.CACHE_DIR / f"symptoms_{cache_key}_vocab.json"
        
        try:
            np.savez_compressed(cache_file, embeddings=self.embeddings)
            with open(vocab_file, 'w') as f:
                json.dump(self.vocabulary, f)
        except Exception as e:
            logger.warning("Failed to cache embeddings", error=str(e))
    
    def normalize(self, user_symptom: str) -> Optional[tuple[str, float]]:
        """
        Normalize a user-provided symptom to the NG12 vocabulary.
        
        Args:
            user_symptom: The symptom text from user input
            
        Returns:
            Tuple of (canonical_symptom, confidence_score) or None if no match
        """
        if not self._initialized:
            raise RuntimeError("SymptomNormalizer not initialized. Call initialize() first.")
        
        if not user_symptom or not user_symptom.strip():
            return None
        
        user_symptom = user_symptom.lower().strip()
        
        # Check for exact match first (fast path)
        if user_symptom in self.canonical_map:
            canonical = self.canonical_map[user_symptom]
            logger.debug("Exact symptom match", input=user_symptom, canonical=canonical)
            return (canonical, 1.0)
        
        # Compute embedding for user input
        model = get_model()
        user_embedding = model.encode(
            [user_symptom],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        
        # Compute cosine similarities (dot product since embeddings are normalized)
        similarities = np.dot(self.embeddings, user_embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        best_match = self.vocabulary[best_idx]
        canonical = self.canonical_map[best_match]
        
        logger.debug(
            "Symptom similarity match",
            input=user_symptom,
            best_match=best_match,
            canonical=canonical,
            score=round(best_score, 3)
        )
        
        if best_score >= self.MATCH_THRESHOLD:
            return (canonical, best_score)
        
        return None
    
    def normalize_multiple(self, user_symptoms: list[str]) -> list[tuple[str, float]]:
        """
        Normalize multiple symptoms.
        
        Returns list of (canonical_symptom, confidence_score) for matched symptoms.
        """
        results = []
        seen = set()
        
        for symptom in user_symptoms:
            result = self.normalize(symptom)
            if result and result[0] not in seen:
                results.append(result)
                seen.add(result[0])
        
        return results
    
    def get_top_matches(self, user_symptom: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Get top-k matches for a symptom (for debugging/UI).
        
        Returns list of (canonical_symptom, score) sorted by score descending.
        """
        if not self._initialized:
            raise RuntimeError("SymptomNormalizer not initialized.")
        
        user_symptom = user_symptom.lower().strip()
        
        model = get_model()
        user_embedding = model.encode(
            [user_symptom],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        
        similarities = np.dot(self.embeddings, user_embedding)
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            match = self.vocabulary[idx]
            canonical = self.canonical_map[match]
            score = float(similarities[idx])
            if canonical not in [r[0] for r in results]:
                results.append((canonical, score))
        
        return results[:k]


def get_symptom_normalizer() -> SymptomNormalizer:
    """Get or create the singleton SymptomNormalizer instance."""
    global _normalizer_instance
    
    if _normalizer_instance is None:
        _normalizer_instance = SymptomNormalizer()
        
        # Initialize with vocabulary from rules and terms_index
        symptoms, synonyms = _build_vocabulary()
        _normalizer_instance.initialize(symptoms, synonyms)
    
    return _normalizer_instance


def _build_vocabulary() -> tuple[list[str], dict[str, list[str]]]:
    """
    Build symptom vocabulary from parsed rules and terms_index.
    
    Returns:
        Tuple of (canonical_symptoms, synonyms_dict)
    """
    from services.rule_parser import get_rule_parser
    from services.terms_index import get_terms_index
    
    symptoms = set()
    synonyms = {}
    
    # Extract symptoms from parsed rules
    parser = get_rule_parser()
    rules = parser.get_rules()
    
    for rule in rules:
        for condition in rule.conditions:
            _extract_symptoms_from_condition(condition, symptoms)
    
    # Add symptoms from terms_index
    terms_index = get_terms_index()
    for canonical, syns in terms_index.SYMPTOM_SYNONYMS.items():
        symptoms.add(canonical.lower())
        synonyms[canonical.lower()] = [s.lower() for s in syns]
    
    # Add common symptom phrases from terms
    common_symptoms = [
        "unexplained weight loss",
        "fatigue",
        "night sweats",
        "fever",
        "loss of appetite",
        "abdominal pain",
        "back pain",
        "bone pain",
        "chest pain",
        "dysphagia",
        "dyspepsia",
        "nausea",
        "vomiting",
        "diarrhoea",
        "constipation",
        "rectal bleeding",
        "change in bowel habit",
        "haematuria",
        "urinary frequency",
        "urinary retention",
        "breast lump",
        "nipple discharge",
        "skin changes",
        "lymphadenopathy",
        "hepatomegaly",
        "splenomegaly",
        "ascites",
        "jaundice",
        "hoarseness",
        "persistent cough",
        "shortness of breath",
        "haemoptysis",
        "pleural effusion",
        "headache",
        "seizure",
        "neurological symptoms",
        "visual disturbance",
        "vulval bleeding",
        "vaginal bleeding",
        "post-menopausal bleeding",
        "intermenstrual bleeding",
        "pelvic mass",
        "testicular mass",
        "scrotal swelling",
        "prostate symptoms",
        "haematospermia",
        "erectile dysfunction",
        "unexplained bruising",
        "petechiae",
        "pallor",
        "thrombocytosis",
        "anaemia",
    ]
    
    for symptom in common_symptoms:
        symptoms.add(symptom.lower())
    
    logger.info("Built symptom vocabulary", 
                symptom_count=len(symptoms), 
                synonym_count=len(synonyms))
    
    return list(symptoms), synonyms


def _extract_symptoms_from_condition(condition, symptoms: set):
    """Recursively extract symptom values from a condition."""
    from models.rule_models import AtomicCondition, CountCondition, CompositeCondition
    
    if isinstance(condition, AtomicCondition):
        if condition.type == "symptom":
            # Split on common separators to get individual symptoms
            value = condition.value.lower()
            # Handle lists like "vulval lump, ulceration or bleeding"
            parts = value.replace(" or ", ", ").replace(" and ", ", ").split(", ")
            for part in parts:
                part = part.strip()
                if part and len(part) > 2:
                    symptoms.add(part)
            symptoms.add(value)  # Also add the full phrase
    
    elif isinstance(condition, CountCondition):
        for sub in condition.conditions:
            _extract_symptoms_from_condition(sub, symptoms)
    
    elif isinstance(condition, CompositeCondition):
        for sub in condition.conditions:
            _extract_symptoms_from_condition(sub, symptoms)
