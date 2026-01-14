"""
Document preprocessor for NG12 guidelines.

One-time preprocessing to split final.md by H1 headers and create searchable structure.
This preprocessing is done once and cached - only recomputed if document changes.
"""

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from rank_bm25 import BM25Okapi
from openai import AsyncOpenAI

from pydantic import BaseModel

from config.config import Settings, get_settings
from config.custom_config import CustomPipelineSettings, get_custom_settings
from config.logging_config import get_logger

logger = get_logger(__name__)

# Paths
GUIDELINE_PATH = Path(__file__).parent.parent.parent / "data" / "final.md"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_SECTIONS = CACHE_DIR / "langgraph_sections.json"
CACHE_EMBEDDINGS = CACHE_DIR / "langgraph_section_embeddings.pkl"
CACHE_BM25 = CACHE_DIR / "langgraph_bm25_index.pkl"
CACHE_TOC = CACHE_DIR / "langgraph_toc.json"
CACHE_DOCUMENT_HASH = CACHE_DIR / "langgraph_document_hash.txt"


class DocumentSection(BaseModel):
    """A section split by H1 header."""
    section_id: str
    section_title: str
    section_index: int
    content: str
    subsections: list["DocumentSubsection"]


class DocumentSubsection(BaseModel):
    """A subsection (H2/H3) within a section."""
    subsection_id: str
    subsection_path: str
    subsection_title: str
    content: str
    normalized_content: str  # Normalized for keyword matching


# Forward reference resolution
DocumentSection.model_rebuild()


class DocumentPreprocessor:
    """
    Preprocessor for NG12 guideline document.
    
    Splits document by H1 headers and creates searchable indexes.
    """
    
    def __init__(
        self,
        settings: CustomPipelineSettings | None = None,
    ):
        """
        Initialize the document preprocessor.
        
        Args:
            settings: Custom pipeline settings. Uses default if not provided.
        """
        self.settings = settings or get_custom_settings()
        self.main_settings = get_settings()
        
        # OpenAI client for embeddings
        self._openai_client: AsyncOpenAI | None = None
        
        # State
        self._sections: list[DocumentSection] = []
        self._toc: dict[str, Any] = {}
        self._embeddings: np.ndarray | None = None
        self._bm25_indexes: dict[str, BM25Okapi] = {}
        self._subsection_texts: list[str] = []
        self._subsection_map: dict[str, DocumentSubsection] = {}
        self._loaded = False
        
        # Load or create sections
        self._load_or_create_sections()
    
    def _load_or_create_sections(self) -> None:
        """Load sections from cache or create them."""
        if self._loaded:
            return
        
        try:
            logger.info(
                "Loading or creating sections",
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
                logger.info("Loaded sections and indexes from cache")
                self._loaded = True
                return
            
            # Document changed or cache missing - re-parse
            logger.info("Cache invalid or missing, creating sections...")
            
            with open(GUIDELINE_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Parse document
            self._parse_document(content)
            
            # Build indexes (BM25 only - embeddings computed lazily)
            self._build_indexes()
            
            # Save cache
            self._save_cache()
            self._save_document_hash(current_hash)
            
            logger.info(
                "Document preprocessed",
                sections=len(self._sections),
                total_subsections=sum(len(s.subsections) for s in self._sections),
            )
            
            self._loaded = True
            
        except Exception as e:
            logger.exception("Error loading or creating sections", error=str(e))
            raise
    
    def _compute_document_hash(self) -> str:
        """Compute MD5 hash of final.md for cache invalidation."""
        if not GUIDELINE_PATH.exists():
            return ""
        with open(GUIDELINE_PATH, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _load_cached_hash(self) -> str:
        """Load cached document hash."""
        if CACHE_DOCUMENT_HASH.exists():
            try:
                return CACHE_DOCUMENT_HASH.read_text().strip()
            except Exception:
                return ""
        return ""
    
    def _save_document_hash(self, hash_value: str) -> None:
        """Save document hash to cache."""
        try:
            CACHE_DOCUMENT_HASH.write_text(hash_value)
        except Exception as e:
            logger.warning("Failed to save document hash", error=str(e))
    
    def _parse_document(self, content: str) -> None:
        """Parse document by H1 headers and extract sections/subsections."""
        lines = content.split("\n")
        
        # Extract TOC from "# Contents" section
        self._extract_toc(content)
        
        # Find all H1 headers (^# [^#])
        h1_pattern = re.compile(r"^# ([^#].*)$")
        sections: list[tuple[int, str]] = []  # (line_index, title)
        
        for i, line in enumerate(lines):
            match = h1_pattern.match(line)
            if match:
                sections.append((i, match.group(1).strip()))
        
        logger.info("Found H1 sections", count=len(sections))
        
        # Parse each section
        self._sections = []
        for section_idx, (start_line, title) in enumerate(sections):
            # Determine end line (next H1 or end of file)
            end_line = sections[section_idx + 1][0] if section_idx + 1 < len(sections) else len(lines)
            
            # Extract section content
            section_lines = lines[start_line:end_line]
            section_content = "\n".join(section_lines)
            
            # Parse subsections (H2/H3)
            subsections = self._parse_subsections(section_content, title, section_idx)
            
            section = DocumentSection(
                section_id=f"section_{section_idx}",
                section_title=title,
                section_index=section_idx,
                content=section_content,
                subsections=subsections,
            )
            self._sections.append(section)
    
    def _extract_toc(self, content: str) -> None:
        """Extract table of contents from '# Contents' section."""
        toc_pattern = re.compile(r"^# Contents\n\n(.*?)(?=\n\n# |$)", re.DOTALL)
        match = toc_pattern.search(content)
        
        if match:
            toc_text = match.group(1)
            # Parse TOC entries (simplified - just store text for now)
            self._toc = {
                "text": toc_text,
                "entries": self._parse_toc_entries(toc_text),
            }
        else:
            self._toc = {"text": "", "entries": []}
        
        logger.info("TOC extracted", entries=len(self._toc.get("entries", [])))
    
    def _parse_toc_entries(self, toc_text: str) -> list[dict[str, Any]]:
        """Parse TOC entries into structured format."""
        entries = []
        lines = toc_text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith("*"):
                continue
            
            # Extract title and page number (simplified)
            # Format: "* Title ................................................................... page"
            parts = line.split(".")
            if len(parts) >= 2:
                title = parts[0].replace("*", "").strip()
                page_part = parts[-1].strip()
                page_num = None
                try:
                    page_num = int(page_part)
                except ValueError:
                    pass
                
                entries.append({"title": title, "page": page_num})
        
        return entries
    
    def _parse_subsections(self, section_content: str, section_title: str, section_index: int) -> list[DocumentSubsection]:
        """Parse H2/H3 subsections from section content, preserving hierarchy."""
        lines = section_content.split("\n")
        subsections: list[DocumentSubsection] = []
        
        current_subsection: list[str] = []
        current_title = ""
        current_level = 0
        current_h2_title = ""  # Track parent H2 for H3 subsections
        subsection_idx = 0
        
        h2_pattern = re.compile(r"^## (.*)$")
        h3_pattern = re.compile(r"^### (.*)$")
        
        for line in lines:
            h2_match = h2_pattern.match(line)
            h3_match = h3_pattern.match(line)
            
            if h2_match:
                # Save previous subsection
                if current_subsection and current_title:
                    subsection = self._create_subsection(
                        current_subsection,
                        current_title,
                        section_title,
                        section_index,
                        subsection_idx,
                        "h2" if current_level == 2 else "h3",
                        current_h2_title if current_level == 3 else "",
                    )
                    subsections.append(subsection)
                    subsection_idx += 1
                
                # Start new H2 subsection
                current_title = h2_match.group(1).strip()
                current_h2_title = current_title  # Update H2 parent
                current_level = 2
                current_subsection = [line]
                
            elif h3_match:
                # Save previous subsection (could be H2 or H3)
                if current_subsection and current_title:
                    subsection = self._create_subsection(
                        current_subsection,
                        current_title,
                        section_title,
                        section_index,
                        subsection_idx,
                        "h2" if current_level == 2 else "h3",
                        current_h2_title if current_level == 3 else "",
                    )
                    subsections.append(subsection)
                    subsection_idx += 1
                
                # Start new H3 subsection (keep current_h2_title as parent)
                current_title = h3_match.group(1).strip()
                current_level = 3
                current_subsection = [line]
                
            else:
                # Add to current subsection
                if current_subsection:
                    current_subsection.append(line)
        
        # Save last subsection
        if current_subsection and current_title:
            subsection = self._create_subsection(
                current_subsection,
                current_title,
                section_title,
                section_index,
                subsection_idx,
                "h2" if current_level == 2 else "h3",
                current_h2_title if current_level == 3 else "",
            )
            subsections.append(subsection)
        
        return subsections
    
    def _create_subsection(
        self,
        lines: list[str],
        title: str,
        section_title: str,
        section_index: int,
        subsection_idx: int,
        level: str,
        h2_parent: str = "",
    ) -> DocumentSubsection:
        """Create a DocumentSubsection from lines."""
        content = "\n".join(lines)
        
        # Normalize content for keyword matching
        normalized = self._normalize_text(content)
        
        # Create subsection path with hierarchy (H2 > H3 if H3)
        if level == "h3" and h2_parent:
            subsection_path = f"{section_title} > {h2_parent} > {title}"
        else:
            subsection_path = f"{section_title} > {title}"
        
        # Create subsection ID
        subsection_id = f"section_{section_index}_subsection_{subsection_idx}"
        
        return DocumentSubsection(
            subsection_id=subsection_id,
            subsection_path=subsection_path,
            subsection_title=title,
            content=content,
            normalized_content=normalized,
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for keyword matching (lowercase, remove punctuation, normalize spaces)."""
        # Lowercase
        normalized = text.lower()
        
        # Remove punctuation (keep spaces and alphanumeric)
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        
        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized)
        
        return normalized.strip()
    
    def _build_indexes(self) -> None:
        """Build search indexes (BM25 per section, embeddings per subsection)."""
        logger.info("Building search indexes...")
        
        # Build BM25 index per section
        self._subsection_texts = []
        self._subsection_map = {}
        
        for section in self._sections:
            subsection_texts = []
            for subsection in section.subsections:
                self._subsection_texts.append(subsection.content)
                self._subsection_map[subsection.subsection_id] = subsection
                subsection_texts.append(subsection.content)
            
            if subsection_texts:
                # Tokenize for BM25
                tokenized = [text.lower().split() for text in subsection_texts]
                self._bm25_indexes[section.section_id] = BM25Okapi(
                    tokenized,
                    k1=self.settings.bm25_k1,
                    b=self.settings.bm25_b,
                )
        
        # Note: Embeddings are computed lazily on first access via async method
        # This avoids blocking during initialization
        logger.info("Embeddings will be computed on first access", model=self.settings.embedding_model, count=len(self._subsection_texts))
        self._embeddings = None  # Will be computed async when needed
        
        logger.info("Indexes built", sections=len(self._bm25_indexes), embeddings="lazy")
    
    def _load_cache(self) -> bool:
        """Load sections and indexes from cache."""
        try:
            # Check if all required cache files exist
            if not CACHE_SECTIONS.exists() or not CACHE_EMBEDDINGS.exists():
                logger.info("Cache files missing, will rebuild")
                return False
            
            # Load sections
            logger.info("Loading sections from cache...")
            with open(CACHE_SECTIONS, "r", encoding="utf-8") as f:
                sections_data = json.load(f)
            
            # Reconstruct sections
            self._sections = []
            self._subsection_texts = []
            self._subsection_map = {}
            
            for section_data in sections_data:
                subsections = []
                for sub_data in section_data.get("subsections", []):
                    subsection = DocumentSubsection(
                        subsection_id=sub_data["subsection_id"],
                        subsection_path=sub_data["subsection_path"],
                        subsection_title=sub_data["subsection_title"],
                        content=sub_data["content"],
                        normalized_content=sub_data["normalized_content"],
                    )
                    subsections.append(subsection)
                    self._subsection_texts.append(subsection.content)
                    self._subsection_map[subsection.subsection_id] = subsection
                
                section = DocumentSection(
                    section_id=section_data["section_id"],
                    section_title=section_data["section_title"],
                    section_index=section_data["section_index"],
                    content="",  # Not needed for search, can reconstruct if needed
                    subsections=subsections,
                )
                self._sections.append(section)
            
            # Load TOC
            if CACHE_TOC.exists():
                with open(CACHE_TOC, "r", encoding="utf-8") as f:
                    self._toc = json.load(f)
            else:
                self._toc = {"text": "", "entries": []}
            
            # Load embeddings and validate dimensions match current model
            logger.info("Loading embeddings from cache...")
            with open(CACHE_EMBEDDINGS, "rb") as f:
                cached_embeddings = pickle.load(f)
            
            # Validate embedding dimensions match current model
            # text-embedding-3-small produces 1536-dimensional embeddings
            # Old models (all-MiniLM-L6-v2) produce 384-dimensional embeddings
            if cached_embeddings is not None and len(cached_embeddings.shape) == 2:
                # Determine expected dimension based on model
                if "text-embedding-3-small" in self.settings.embedding_model:
                    expected_dim = 1536
                elif "text-embedding-3-large" in self.settings.embedding_model:
                    expected_dim = 3072
                else:
                    # Default for older models
                    expected_dim = 384
                
                if cached_embeddings.shape[1] != expected_dim:
                    logger.warning(
                        "Embedding dimension mismatch - cache invalidated",
                        cached_dim=cached_embeddings.shape[1],
                        expected_dim=expected_dim,
                        model=self.settings.embedding_model,
                    )
                    # Cache is invalid - will need to recompute
                    return False
            
            self._embeddings = cached_embeddings
            
            # Rebuild BM25 indexes (fast, doesn't need caching)
            logger.info("Rebuilding BM25 indexes...")
            self._bm25_indexes = {}
            for section in self._sections:
                subsection_texts = [sub.content for sub in section.subsections]
                if subsection_texts:
                    tokenized_texts = [text.lower().split() for text in subsection_texts]
                    self._bm25_indexes[section.section_id] = BM25Okapi(
                        tokenized_texts,
                        k1=self.settings.bm25_k1,
                        b=self.settings.bm25_b,
                    )
            
            logger.info(
                "Cache loaded successfully",
                sections=len(self._sections),
                subsections=len(self._subsection_map),
                embeddings_shape=self._embeddings.shape if self._embeddings is not None else None,
            )
            
            return True
            
        except Exception as e:
            logger.warning("Failed to load cache", error=str(e), exc_info=True)
            return False
    
    def _save_cache(self) -> None:
        """Save sections and indexes to cache."""
        try:
            # Save sections (simplified - just save metadata for now)
            sections_data = [
                {
                    "section_id": s.section_id,
                    "section_title": s.section_title,
                    "section_index": s.section_index,
                    "subsection_count": len(s.subsections),
                    "subsections": [
                        {
                            "subsection_id": sub.subsection_id,
                            "subsection_path": sub.subsection_path,
                            "subsection_title": sub.subsection_title,
                            "content": sub.content,
                            "normalized_content": sub.normalized_content,
                        }
                        for sub in s.subsections
                    ],
                }
                for s in self._sections
            ]
            
            with open(CACHE_SECTIONS, "w", encoding="utf-8") as f:
                json.dump(sections_data, f, indent=2, ensure_ascii=False)
            
            # Save TOC
            with open(CACHE_TOC, "w", encoding="utf-8") as f:
                json.dump(self._toc, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            if self._embeddings is not None:
                with open(CACHE_EMBEDDINGS, "wb") as f:
                    pickle.dump(self._embeddings, f)
            
            logger.info("Cache saved")
            
        except Exception as e:
            logger.warning("Failed to save cache", error=str(e))
    
    # Public API
    def get_sections(self) -> list[DocumentSection]:
        """Get all sections."""
        return self._sections
    
    def get_section(self, section_id: str) -> DocumentSection | None:
        """Get a section by ID."""
        for section in self._sections:
            if section.section_id == section_id:
                return section
        return None
    
    def get_toc(self) -> dict[str, Any]:
        """Get table of contents."""
        return self._toc
    
    def get_subsection(self, subsection_id: str) -> DocumentSubsection | None:
        """Get a subsection by ID."""
        return self._subsection_map.get(subsection_id)
    
    def get_bm25_index(self, section_id: str) -> BM25Okapi | None:
        """Get BM25 index for a section."""
        return self._bm25_indexes.get(section_id)
    
    async def get_embeddings(self) -> np.ndarray | None:
        """
        Get all subsection embeddings, computing them if needed.
        
        This method is async because embedding computation requires async OpenAI API calls.
        """
        if self._embeddings is not None:
            return self._embeddings
        
        # Compute embeddings if not already computed
        await self._compute_embeddings()
        return self._embeddings
    
    async def _compute_embeddings(self) -> None:
        """Compute embeddings for all subsections using OpenAI API."""
        if self._embeddings is not None:
            return  # Already computed
        
        logger.info("Computing embeddings with OpenAI...", model=self.settings.embedding_model, count=len(self._subsection_texts))
        
        if not self.main_settings.openai_api_key:
            logger.error(
                "OpenAI API key not configured",
                has_key=bool(self.main_settings.openai_api_key),
                key_length=len(self.main_settings.openai_api_key) if self.main_settings.openai_api_key else 0,
            )
            raise ValueError("OpenAI API key is required for embeddings. Please set OPENAI_API_KEY in your .env file.")
        
        logger.debug("OpenAI API key loaded", key_length=len(self.main_settings.openai_api_key))
        
        if self._openai_client is None:
            # Embeddings must use OpenAI API, not DeepSeek
            self._openai_client = AsyncOpenAI(
                api_key=self.main_settings.openai_api_key,
                base_url="https://api.openai.com/v1",  # Explicitly use OpenAI API for embeddings
            )
        
        # Compute embeddings in batches (OpenAI has rate limits)
        batch_size = 100
        all_embeddings = []
        
        async def compute_batch(texts: list[str]) -> list[list[float]]:
            """Compute embeddings for a batch of texts."""
            response = await self._openai_client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        
        # Process in batches
        for i in range(0, len(self._subsection_texts), batch_size):
            batch = self._subsection_texts[i:i + batch_size]
            batch_embeddings = await compute_batch(batch)
            all_embeddings.extend(batch_embeddings)
            logger.info("Computed embeddings batch", batch=i // batch_size + 1, total_batches=(len(self._subsection_texts) + batch_size - 1) // batch_size)
        
        self._embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info("Embeddings computed", embeddings_shape=self._embeddings.shape)
        
        # Save embeddings to cache
        try:
            CACHE_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_EMBEDDINGS, "wb") as f:
                pickle.dump(self._embeddings, f)
            logger.info("Embeddings saved to cache")
        except Exception as e:
            logger.warning("Failed to save embeddings to cache", error=str(e))
    
    def get_embedding_model_name(self) -> str:
        """Get embedding model name."""
        return self.settings.embedding_model
    
    def get_all_subsections(self) -> list[DocumentSubsection]:
        """Get all subsections across all sections."""
        subsections = []
        for section in self._sections:
            subsections.extend(section.subsections)
        return subsections
    
    def search_keywords(self, query: str) -> dict[str, dict[str, Any]]:
        """
        Search for keywords across all subsections (normalized matching).
        
        Args:
            query: Search query
            
        Returns:
            Dict mapping subsection_id to match info (keywords found, match counts)
        """
        normalized_query = self._normalize_text(query)
        query_words = set(normalized_query.split())
        
        matches: dict[str, dict[str, Any]] = {}
        
        for subsection in self.get_all_subsections():
            normalized_content = subsection.normalized_content
            found_keywords = []
            keyword_counts = {}
            
            for word in query_words:
                if word in normalized_content:
                    count = normalized_content.count(word)
                    found_keywords.append(word)
                    keyword_counts[word] = count
            
            if found_keywords:
                matches[subsection.subsection_id] = {
                    "keywords": found_keywords,
                    "counts": keyword_counts,
                    "total_matches": sum(keyword_counts.values()),
                }
        
        return matches


# Singleton instance
_document_preprocessor: DocumentPreprocessor | None = None


def get_document_preprocessor() -> DocumentPreprocessor:
    """Get the document preprocessor singleton."""
    global _document_preprocessor
    if _document_preprocessor is None:
        _document_preprocessor = DocumentPreprocessor()
    return _document_preprocessor
