"""
Section Retriever for NG12 Document.

Loads the pre-parsed sections_index.json and provides hybrid BM25 + semantic search.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with section data and score."""
    section_id: str
    header: str
    header_path: list[str]
    content: str
    start_line: int
    end_line: int
    section_type: str
    has_criteria: bool
    criteria_spec: Optional[dict]
    cancer_site: Optional[str]
    score: float
    level: int = 4  # Header level (1-4)
    
    @classmethod
    def from_section(cls, section: dict, score: float) -> "RetrievalResult":
        """Create from parsed section dict."""
        return cls(
            section_id=section["id"],
            header=section["header"],
            header_path=section["header_path"],
            content=section["content"],
            start_line=section["start_line"],
            end_line=section["end_line"],
            section_type=section["section_type"],
            has_criteria=section["has_criteria"],
            criteria_spec=section.get("criteria_spec"),
            cancer_site=section.get("cancer_site"),
            score=score,
            level=section.get("level", 4)
        )
    
    def to_artifact_dict(self) -> dict:
        """Convert to artifact format for API response."""
        return {
            "section": " > ".join(self.header_path) if self.header_path else self.header,
            "text": self.content,
            "rule_id": self.section_id if self.section_id[0].isdigit() else None,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "source": "NICE NG12",
            "source_url": "https://www.nice.org.uk/guidance/ng12",
            "relevance_score": self.score
        }


class SectionRetriever:
    """
    Retriever for NG12 sections using hybrid BM25 + semantic search.
    
    Loads pre-parsed sections from JSON and builds search indices on init.
    """
    
    # Embedding model for semantic search
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Weight for header vs content in BM25
    HEADER_WEIGHT = 2.0
    
    # Hybrid search weights
    BM25_WEIGHT = 0.5
    SEMANTIC_WEIGHT = 0.5
    
    def __init__(self, index_path: str = "data/sections_index.json"):
        """
        Initialize the retriever.
        
        Args:
            index_path: Path to the sections index JSON file
        """
        self.index_path = Path(index_path)
        self.sections: list[dict] = []
        self.metadata: dict = {}
        
        # Search indices
        self._bm25: Optional[BM25Okapi] = None
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        
        # Load and build indices
        self._load_index()
        self._build_indices()
        
        logger.info(
            "SectionRetriever initialized",
            total_sections=len(self.sections),
            sections_with_criteria=sum(1 for s in self.sections if s["has_criteria"])
        )
    
    def _load_index(self):
        """Load the sections index from JSON."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Sections index not found: {self.index_path}\n"
                f"Run: python -m backend.scripts.parse_sections"
            )
        
        with open(self.index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metadata = data.get("metadata", {})
        self.sections = data.get("sections", [])
        
        logger.info(
            "Loaded sections index",
            source=self.metadata.get("source"),
            parsed_at=self.metadata.get("parsed_at"),
            total_sections=self.metadata.get("total_sections")
        )
    
    def _build_indices(self):
        """Build BM25 and semantic search indices."""
        if not self.sections:
            logger.warning("No sections to index")
            return
        
        # Build BM25 index
        # Combine header (weighted) + content for each section
        corpus = []
        for section in self.sections:
            # Weight header by repeating it
            header_tokens = self._tokenize(section["header"])
            content_tokens = self._tokenize(section["content"])
            
            # Repeat header tokens for weighting
            weighted_tokens = header_tokens * int(self.HEADER_WEIGHT) + content_tokens
            corpus.append(weighted_tokens)
        
        self._bm25 = BM25Okapi(corpus)
        logger.info("Built BM25 index", corpus_size=len(corpus))
        
        # Build semantic embeddings
        try:
            self._embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
            
            # Create combined text for embedding
            texts = [
                f"{section['header']}\n{section['content'][:500]}"  # Limit content length
                for section in self.sections
            ]
            
            self._embeddings = self._embedding_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            logger.info("Built semantic embeddings", shape=self._embeddings.shape)
            
        except Exception as e:
            logger.warning(f"Failed to build semantic embeddings: {e}")
            self._embeddings = None
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        # Lowercase, split on non-alphanumeric, filter short tokens
        import re
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return [t for t in tokens if len(t) > 1]
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        section_types: Optional[list[str]] = None,
        require_criteria: bool = False,
        header_levels: Optional[list[int]] = None,
        include_overlap: bool = False
    ) -> list[RetrievalResult]:
        """
        Search for relevant sections using hybrid BM25 + semantic search.
        
        Args:
            query: The search query
            top_k: Number of results to return
            section_types: Optional filter for section types
            require_criteria: Only return sections with has_criteria=True
            header_levels: Optional filter for header levels (e.g., [2, 3] for H2/H3 only)
            include_overlap: If True, include adjacent sections for context
            
        Returns:
            List of RetrievalResult sorted by relevance score
        """
        if not self.sections:
            return []
        
        # Get BM25 scores
        query_tokens = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores
        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_normalized = bm25_scores / bm25_max
        
        # Get semantic scores if available
        if self._embeddings is not None and self._embedding_model is not None:
            query_embedding = self._embedding_model.encode(
                [query], 
                show_progress_bar=False,
                convert_to_numpy=True
            )[0]
            
            # Cosine similarity
            semantic_scores = np.dot(self._embeddings, query_embedding) / (
                np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Combine scores
            combined_scores = (
                self.BM25_WEIGHT * bm25_normalized + 
                self.SEMANTIC_WEIGHT * semantic_scores
            )
        else:
            combined_scores = bm25_normalized
        
        # Boost sections that have criteria with symptoms matching query terms
        # This ensures actionable recommendations are prioritized
        query_lower = query.lower()
        query_words = set(query_lower.split())
        criteria_boost = np.zeros(len(self.sections))
        
        # Extract key medical terms from query for matching (handle typos)
        # Common symptoms to look for
        key_symptoms = ['thrombocytosis', 'haematuria', 'haemoptysis', 'lymphadenopathy', 
                        'clubbing', 'fatigue', 'weight loss', 'cough', 'hoarseness']
        
        query_symptoms = []
        for symptom in key_symptoms:
            # Check if symptom or partial match is in query (handles typos like "thrombocytos")
            symptom_root = symptom[:8] if len(symptom) > 8 else symptom[:5]
            if symptom_root.lower() in query_lower:
                query_symptoms.append(symptom.lower())
        
        for idx, section in enumerate(self.sections):
            content_lower = section.get("content", "").lower()
            
            # Boost 1: Direct content match for key symptoms
            for symptom in query_symptoms:
                if symptom in content_lower:
                    # Extra boost for sections with criteria (actionable)
                    if section.get("has_criteria"):
                        criteria_boost[idx] = max(criteria_boost[idx], 0.4)
                    else:
                        criteria_boost[idx] = max(criteria_boost[idx], 0.15)
            
            # Boost 2: Criteria spec symptom match
            if section.get("has_criteria") and section.get("criteria_spec"):
                spec = section["criteria_spec"]
                for group in spec.get("criteria_groups", []):
                    for criterion in group.get("criteria", []):
                        if criterion.get("field") == "symptoms":
                            symptoms = criterion.get("value", [])
                            if isinstance(symptoms, list):
                                for symptom in symptoms:
                                    symptom_lower = symptom.lower()
                                    # Check for any query symptom matching
                                    for qs in query_symptoms:
                                        if qs in symptom_lower or symptom_lower in qs:
                                            criteria_boost[idx] = max(criteria_boost[idx], 0.5)
                                            break
        
        # Apply criteria boost
        combined_scores = combined_scores + criteria_boost
        
        # Get top indices
        top_indices = np.argsort(combined_scores)[::-1]
        
        # Build results with filtering
        results = []
        selected_indices = set()
        
        for idx in top_indices:
            section = self.sections[idx]
            score = float(combined_scores[idx])
            
            # Apply filters
            if section_types and section["section_type"] not in section_types:
                continue
            if require_criteria and not section["has_criteria"]:
                continue
            if header_levels and section.get("level", 4) not in header_levels:
                continue
            
            # Skip very low scores
            if score < 0.1:
                continue
            
            results.append(RetrievalResult.from_section(section, score))
            selected_indices.add(idx)
            
            if len(results) >= top_k:
                break
        
        # Add overlap (adjacent sections) for context if requested
        if include_overlap and results:
            overlap_results = []
            for idx in selected_indices:
                current_section = self.sections[idx]
                current_level = current_section.get("level", 4)
                
                # Add previous section if exists and not already included
                if idx > 0 and (idx - 1) not in selected_indices:
                    prev_section = self.sections[idx - 1]
                    prev_level = prev_section.get("level", 4)
                    # Only add if passes header_levels filter (if specified)
                    if header_levels is None or prev_level in header_levels:
                        overlap_results.append(
                            RetrievalResult.from_section(prev_section, score=0.5)
                        )
                # Add next section if exists and not already included
                if idx < len(self.sections) - 1 and (idx + 1) not in selected_indices:
                    next_section = self.sections[idx + 1]
                    next_level = next_section.get("level", 4)
                    # Only add if passes header_levels filter (if specified)
                    if header_levels is None or next_level in header_levels:
                        overlap_results.append(
                            RetrievalResult.from_section(next_section, score=0.5)
                        )
            
            # Add unique overlap sections (limit to 1 to keep context focused)
            seen_ids = {r.section_id for r in results}
            for overlap in overlap_results[:1]:  # Limit overlap to 1
                if overlap.section_id not in seen_ids:
                    results.append(overlap)
                    seen_ids.add(overlap.section_id)
        
        return results
    
    def get_by_id(self, section_id: str) -> Optional[RetrievalResult]:
        """
        Get a section by its ID.
        
        Args:
            section_id: The section ID (e.g., "1.1.2" or "terms-unexplained")
            
        Returns:
            RetrievalResult or None if not found
        """
        for section in self.sections:
            if section["id"] == section_id:
                return RetrievalResult.from_section(section, score=1.0)
        return None
    
    def get_sections_with_criteria(self) -> list[RetrievalResult]:
        """Get all sections that have criteria for pathway UI."""
        return [
            RetrievalResult.from_section(s, score=1.0)
            for s in self.sections
            if s["has_criteria"]
        ]
    
    def get_definition(self, term: str) -> Optional[RetrievalResult]:
        """
        Search for a term definition in the Terms section.
        
        Args:
            term: The term to look up
            
        Returns:
            RetrievalResult for the definition or None
        """
        term_lower = term.lower()
        
        for section in self.sections:
            if section["section_type"] == "definition":
                if term_lower in section["header"].lower():
                    return RetrievalResult.from_section(section, score=1.0)
        
        # Fall back to search
        results = self.search(
            f"definition of {term}",
            top_k=1,
            section_types=["definition"]
        )
        return results[0] if results else None


# Singleton instance
_retriever_instance: Optional[SectionRetriever] = None


def get_section_retriever(index_path: Optional[str] = None) -> SectionRetriever:
    """
    Get or create the singleton SectionRetriever instance.
    
    Args:
        index_path: Path to the sections index JSON. If None, uses default location.
        
    Returns:
        SectionRetriever instance
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        if index_path is None:
            # Default path relative to this file's location
            backend_dir = Path(__file__).parent.parent
            index_path = str(backend_dir.parent / "data" / "sections_index.json")
        
        _retriever_instance = SectionRetriever(index_path)
    
    return _retriever_instance
