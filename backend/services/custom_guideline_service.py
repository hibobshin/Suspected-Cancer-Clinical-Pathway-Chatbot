"""
Custom guideline service for NG12 with two-level chunking and hybrid retrieval.

This service is completely independent from guideline_service.py:
- Separate cache files (custom_* vs guideline_*)
- Different chunking strategy (two-level: section containers + rule chunks)
- Different retrieval method (hybrid BM25 + embeddings)
- Fresh parsing of final.md

IMPORTANT: This is a one-time preprocessing step, not runtime.
Chunks are created once and cached, re-run only on document change.
"""

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.custom_config import CustomPipelineSettings, get_custom_settings
from config.logging_config import get_logger
from models.custom_models import (
    ActionType,
    InheritedMetadata,
    LocalMetadata,
    MetadataQuality,
    NG12Chunk,
    NG12ChunkMetadata,
    RetrievalResult,
    SectionContainer,
)
from services.metadata_extractor import MetadataExtractor, get_metadata_extractor

logger = get_logger(__name__)

# Paths
GUIDELINE_PATH = Path(__file__).parent.parent.parent / "data" / "final.md"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_CHUNKS = CACHE_DIR / "custom_chunks.json"
CACHE_CONTAINERS = CACHE_DIR / "custom_section_containers.json"
CACHE_EMBEDDINGS = CACHE_DIR / "custom_embeddings.pkl"
CACHE_BM25 = CACHE_DIR / "custom_bm25_index.pkl"
CACHE_METADATA_INDEX = CACHE_DIR / "custom_metadata_index.json"
CACHE_DOCUMENT_HASH = CACHE_DIR / "custom_document_hash.txt"


class CustomGuidelineService:
    """
    Service for parsing and retrieving NG12 guidelines with two-level chunking.
    
    Level 1: Section containers (navigation only, not indexed)
    Level 2: Rule chunks (retrieval + evidence, indexed and embedded)
    """
    
    def __init__(
        self,
        settings: CustomPipelineSettings | None = None,
        metadata_extractor: MetadataExtractor | None = None,
    ):
        """
        Initialize the custom guideline service.
        
        Args:
            settings: Custom pipeline settings. Uses default if not provided.
            metadata_extractor: Metadata extractor. Uses default if not provided.
        """
        self.settings = settings or get_custom_settings()
        self.metadata_extractor = metadata_extractor or get_metadata_extractor()
        
        # State
        self._section_containers: list[SectionContainer] = []
        self._rule_chunks: list[NG12Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._bm25_index: BM25Okapi | None = None
        self._embedding_model: SentenceTransformer | None = None
        self._loaded = False
        
        # Load or create chunks
        self._load_guideline()
    
    def _load_guideline(self) -> None:
        """Load guideline file and create chunks (one-time preprocessing)."""
        if self._loaded:
            return
        
        try:
            logger.info(
                "Loading guideline",
                path=str(GUIDELINE_PATH),
                exists=GUIDELINE_PATH.exists(),
            )
            
            if not GUIDELINE_PATH.exists():
                logger.warning("Guideline file not found", path=str(GUIDELINE_PATH))
                self._loaded = True
                return
            
            # Check document hash
            current_hash = self._compute_document_hash()
            cached_hash = self._load_cached_hash()
            
            if current_hash == cached_hash and self._load_cache():
                logger.info("Loaded chunks and indexes from cache")
                self._loaded = True
                return
            
            # Document changed or cache missing - re-parse
            logger.info("Cache invalid or missing, creating chunks...")
            
            with open(GUIDELINE_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Parse into two-level structure
            self._parse_document(content)
            
            # Generate embeddings and BM25 index
            self._generate_indexes()
            
            # Save cache
            self._save_cache()
            self._save_document_hash(current_hash)
            
            logger.info(
                "Guideline loaded and indexed",
                containers=len(self._section_containers),
                rule_chunks=len(self._rule_chunks),
            )
            self._loaded = True
            
        except Exception as e:
            logger.exception("Failed to load guideline", error=str(e))
            self._loaded = True
    
    def _compute_document_hash(self) -> str:
        """Compute MD5 hash of final.md for cache invalidation."""
        if not GUIDELINE_PATH.exists():
            return ""
        
        with open(GUIDELINE_PATH, "rb") as f:
            content = f.read()
        
        return hashlib.md5(content).hexdigest()
    
    def _load_cached_hash(self) -> str | None:
        """Load cached document hash."""
        if not CACHE_DOCUMENT_HASH.exists():
            return None
        
        try:
            with open(CACHE_DOCUMENT_HASH, "r") as f:
                return f.read().strip()
        except Exception:
            return None
    
    def _save_document_hash(self, hash_value: str) -> None:
        """Save document hash to cache."""
        try:
            with open(CACHE_DOCUMENT_HASH, "w") as f:
                f.write(hash_value)
        except Exception as e:
            logger.warning("Failed to save document hash", error=str(e))
    
    def _parse_document(self, content: str) -> None:
        """
        Parse document into two-level structure.
        
        Level 1: Section containers (navigation only)
        Level 2: Rule chunks (retrieval + evidence)
        """
        logger.info("Parsing document into two-level structure")
        
        self._section_containers = []
        self._rule_chunks = []
        
        # Split by major sections (## headings)
        lines = content.split("\n")
        current_container: SectionContainer | None = None
        current_section_text = []
        container_id_map: dict[str, str] = {}  # section_path -> container_id
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for major section (## heading)
            if line.startswith("## ") and not line.startswith("###"):
                # Save previous container if exists
                if current_container and current_section_text:
                    self._extract_rule_chunks(
                        current_container,
                        "\n".join(current_section_text),
                        container_id_map,
                    )
                
                # Create new section container
                section_title = line[3:].strip()
                cancer_site = self._extract_cancer_site(section_title)
                section_path = f"NG12 > {section_title}"
                
                container_id = str(uuid4())
                current_container = SectionContainer(
                    container_id=container_id,
                    title=section_title,
                    section_path=section_path,
                    cancer_site=cancer_site,
                    children=[],
                )
                self._section_containers.append(current_container)
                container_id_map[section_path] = container_id
                current_section_text = [line]
                
            elif current_container:
                current_section_text.append(line)
            
            i += 1
        
        # Process last container
        if current_container and current_section_text:
            self._extract_rule_chunks(
                current_container,
                "\n".join(current_section_text),
                container_id_map,
            )
        
        logger.info(
            "Document parsed",
            containers=len(self._section_containers),
            rule_chunks=len(self._rule_chunks),
        )
    
    def _extract_cancer_site(self, section_title: str) -> str | None:
        """Extract cancer site from section title."""
        # Simple extraction - can be enhanced
        title_lower = section_title.lower()
        
        cancer_sites = [
            "lung",
            "gastrointestinal",
            "colorectal",
            "breast",
            "gynaecological",
            "urological",
            "skin",
            "head and neck",
            "brain",
            "haematological",
            "sarcoma",
            "childhood",
        ]
        
        for site in cancer_sites:
            if site in title_lower:
                return site
        
        return None
    
    def _extract_rule_chunks(
        self,
        container: SectionContainer,
        section_text: str,
        container_id_map: dict[str, str],
    ) -> None:
        """
        Extract rule chunks from a section.
        
        Each rule (identified by rule_id pattern) becomes a separate chunk.
        """
        # Find all rules in the section
        rule_pattern = r'(\d+\.\d+\.\d+)'
        rule_matches = list(re.finditer(rule_pattern, section_text))
        
        if not rule_matches:
            # No rules found - create a single chunk for the section
            chunk_id = str(uuid4())
            inherited_metadata = InheritedMetadata(
                cancer_site=container.cancer_site,
                section_path=container.section_path,
                guideline_version="NG12",
                source_doc="final.md",
                source_url="https://www.nice.org.uk/guidance/ng12",
                source_page=None,
            )
            
            local_metadata, audit_metadata = self.metadata_extractor.extract_local_metadata(
                section_text,
                inherited_metadata,
            )
            
            metadata_quality = self.metadata_extractor.assign_metadata_quality(local_metadata)
            
            chunk_metadata = NG12ChunkMetadata(
                inherited=inherited_metadata,
                local=local_metadata,
                audit=audit_metadata,
                metadata_quality=metadata_quality,
            )
            
            chunk = NG12Chunk(
                chunk_id=chunk_id,
                text=section_text,
                metadata=chunk_metadata,
                verbatim_source=section_text,
                parent_container_id=container.container_id,
            )
            
            self._rule_chunks.append(chunk)
            container.children.append(chunk_id)
            return
        
        # Extract each rule as a separate chunk
        for idx, match in enumerate(rule_matches):
            rule_id = match.group(1)
            rule_start = match.start()
            
            # Find end of rule (next rule or end of section)
            if idx + 1 < len(rule_matches):
                rule_end = rule_matches[idx + 1].start()
            else:
                rule_end = len(section_text)
            
            rule_text = section_text[rule_start:rule_end].strip()
            
            # Create chunk for this rule
            chunk_id = str(uuid4())
            inherited_metadata = InheritedMetadata(
                cancer_site=container.cancer_site,
                section_path=container.section_path,
                guideline_version="NG12",
                source_doc="final.md",
                source_url="https://www.nice.org.uk/guidance/ng12",
                source_page=None,
            )
            
            local_metadata, audit_metadata = self.metadata_extractor.extract_local_metadata(
                rule_text,
                inherited_metadata,
            )
            
            # Ensure rule_id is set
            if not local_metadata.rule_id:
                local_metadata.rule_id = rule_id
            
            metadata_quality = self.metadata_extractor.assign_metadata_quality(local_metadata)
            
            chunk_metadata = NG12ChunkMetadata(
                inherited=inherited_metadata,
                local=local_metadata,
                audit=audit_metadata,
                metadata_quality=metadata_quality,
            )
            
            chunk = NG12Chunk(
                chunk_id=chunk_id,
                text=rule_text,
                metadata=chunk_metadata,
                verbatim_source=rule_text,
                parent_container_id=container.container_id,
            )
            
            self._rule_chunks.append(chunk)
            container.children.append(chunk_id)
    
    def _generate_indexes(self) -> None:
        """Generate embeddings and BM25 index for rule chunks only."""
        logger.info("Generating indexes for rule chunks", count=len(self._rule_chunks))
        
        # Generate embeddings
        if not self._embedding_model:
            logger.info("Loading embedding model", model=self.settings.embedding_model)
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
        
        chunk_texts = [chunk.text for chunk in self._rule_chunks]
        self._embeddings = self._embedding_model.encode(chunk_texts, show_progress_bar=True)
        self._embeddings = np.array(self._embeddings)
        
        logger.info("Embeddings generated", shape=self._embeddings.shape)
        
        # Generate BM25 index
        tokenized_texts = [text.lower().split() for text in chunk_texts]
        self._bm25_index = BM25Okapi(
            tokenized_texts,
            k1=self.settings.bm25_k1,
            b=self.settings.bm25_b,
        )
        
        logger.info("BM25 index generated")
    
    def _load_cache(self) -> bool:
        """Load chunks and indexes from cache."""
        try:
            if not all([
                CACHE_CHUNKS.exists(),
                CACHE_CONTAINERS.exists(),
                CACHE_EMBEDDINGS.exists(),
                CACHE_BM25.exists(),
            ]):
                return False
            
            # Load section containers
            with open(CACHE_CONTAINERS, "r") as f:
                containers_data = json.load(f)
            self._section_containers = [
                SectionContainer(**c) for c in containers_data
            ]
            
            # Load rule chunks
            with open(CACHE_CHUNKS, "r") as f:
                chunks_data = json.load(f)
            self._rule_chunks = [NG12Chunk(**c) for c in chunks_data]
            
            # Load embeddings
            with open(CACHE_EMBEDDINGS, "rb") as f:
                self._embeddings = pickle.load(f)
            
            # Load BM25 index
            with open(CACHE_BM25, "rb") as f:
                self._bm25_index = pickle.load(f)
            
            # Load embedding model
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
            
            logger.info("Cache loaded successfully")
            return True
            
        except Exception as e:
            logger.warning("Failed to load cache", error=str(e))
            return False
    
    def _save_cache(self) -> None:
        """Save chunks and indexes to cache."""
        try:
            # Save section containers
            with open(CACHE_CONTAINERS, "w") as f:
                json.dump([c.model_dump() for c in self._section_containers], f)
            
            # Save rule chunks
            with open(CACHE_CHUNKS, "w") as f:
                json.dump([c.model_dump() for c in self._rule_chunks], f)
            
            # Save embeddings
            with open(CACHE_EMBEDDINGS, "wb") as f:
                pickle.dump(self._embeddings, f)
            
            # Save BM25 index
            with open(CACHE_BM25, "wb") as f:
                pickle.dump(self._bm25_index, f)
            
            logger.info("Cache saved successfully")
            
        except Exception as e:
            logger.error("Failed to save cache", error=str(e))
    
    def retrieve(
        self,
        query: str,
        cancer_site: str | None = None,
        age: int | None = None,
        symptoms: list[str] | None = None,
        max_chunks: int | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant rule chunks using hybrid retrieval.
        
        Two-phase retrieval:
        1. Coarse routing by cancer_site (if provided)
        2. Fine retrieval with hybrid BM25 + embeddings
        
        Args:
            query: User query text.
            cancer_site: Optional cancer site filter.
            age: Optional age filter.
            symptoms: Optional symptom tags filter.
            max_chunks: Maximum number of chunks to return.
            
        Returns:
            RetrievalResult with retrieved chunks and scores.
        """
        if not self._loaded or not self._rule_chunks:
            logger.warning("Guideline not loaded, returning empty result")
            return RetrievalResult(
                candidate_sections=[],
                rule_chunks=[],
                retrieval_scores={},
            )
        
        max_chunks = max_chunks or self.settings.max_retrieved_chunks
        
        # Phase 1: Coarse routing by cancer_site
        candidate_chunks = self._rule_chunks
        if cancer_site:
            candidate_chunks = [
                chunk for chunk in self._rule_chunks
                if chunk.metadata.inherited.cancer_site
                and cancer_site.lower() in chunk.metadata.inherited.cancer_site.lower()
            ]
            logger.debug("Coarse routing filtered", cancer_site=cancer_site, count=len(candidate_chunks))
        
        if not candidate_chunks:
            candidate_chunks = self._rule_chunks  # Fallback to all chunks
        
        # Phase 2: Fine retrieval with hybrid BM25 + embeddings
        candidate_indices = [self._rule_chunks.index(c) for c in candidate_chunks]
        
        # BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = self._bm25_index.get_scores(query_tokens)
        bm25_scores = np.array([bm25_scores[i] for i in candidate_indices])
        
        # Normalize BM25 scores
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Embedding scores
        query_embedding = self._embedding_model.encode([query])[0]
        candidate_embeddings = self._embeddings[candidate_indices]
        embedding_scores = np.dot(candidate_embeddings, query_embedding)
        
        # Normalize embedding scores
        if embedding_scores.max() > 0:
            embedding_scores = embedding_scores / embedding_scores.max()
        
        # Combine scores
        combined_scores = (
            self.settings.score_weight_bm25 * bm25_scores +
            self.settings.score_weight_embedding * embedding_scores
        )
        
        # Apply metadata filters and boosts
        final_scores = self._apply_metadata_filters(
            candidate_chunks,
            combined_scores,
            age=age,
            symptoms=symptoms,
        )
        
        # Get top N chunks
        top_indices = np.argsort(final_scores)[::-1][:max_chunks]
        top_chunks = [candidate_chunks[i] for i in top_indices]
        
        logger.info(
            "Retrieval completed",
            query_preview=query[:50],
            chunks_retrieved=len(top_chunks),
        )
        
        return RetrievalResult(
            candidate_sections=[chunk.metadata.inherited.section_path for chunk in top_chunks],
            rule_chunks=top_chunks,
            retrieval_scores={
                "bm25_avg": float(bm25_scores[top_indices].mean()) if len(top_indices) > 0 else 0.0,
                "embedding_avg": float(embedding_scores[top_indices].mean()) if len(top_indices) > 0 else 0.0,
                "combined_avg": float(final_scores[top_indices].mean()) if len(top_indices) > 0 else 0.0,
            },
        )
    
    def _apply_metadata_filters(
        self,
        chunks: list[NG12Chunk],
        scores: np.ndarray,
        age: int | None = None,
        symptoms: list[str] | None = None,
    ) -> np.ndarray:
        """
        Apply metadata filters and boosts to scores.
        
        Hard filters: Age must match if explicit
        Soft boosts: Symptom matches boost score
        """
        final_scores = scores.copy()
        
        for i, chunk in enumerate(chunks):
            local_meta = chunk.metadata.local
            
            # Hard filter: Age
            if age is not None and local_meta.age_min is not None:
                if age < local_meta.age_min:
                    final_scores[i] = 0.0  # Hard filter out
                    continue
            if age is not None and local_meta.age_max is not None:
                if age > local_meta.age_max:
                    final_scores[i] = 0.0  # Hard filter out
                    continue
            
            # Soft boost: Symptoms
            if symptoms and local_meta.symptom_tags:
                matching_symptoms = set(s.lower() for s in symptoms) & set(
                    s.lower() for s in local_meta.symptom_tags
                )
                if matching_symptoms:
                    boost = len(matching_symptoms) / max(len(symptoms), len(local_meta.symptom_tags))
                    final_scores[i] *= (1.0 + boost * 0.2)  # 20% boost max
        
        return final_scores


# Singleton instance
_custom_guideline_service: CustomGuidelineService | None = None


def get_custom_guideline_service() -> CustomGuidelineService:
    """Get the custom guideline service singleton."""
    global _custom_guideline_service
    if _custom_guideline_service is None:
        _custom_guideline_service = CustomGuidelineService()
    return _custom_guideline_service
