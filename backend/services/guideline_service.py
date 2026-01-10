"""
Guideline retrieval service for NICE NG12.

Classic RAG pipeline:
1. Chunk document by half-page (~750 chars) - done once, cached locally
2. Bag-of-words vectorization - all chunks vectorized and cached
3. Cosine similarity retrieval - query vectorized, compared with all chunks
4. Reranker - reranks top candidates using multiple factors

Chunks and vectors are stored in data/.cache/ and loaded on startup.
Cache is invalidated if the source document changes.
"""

import hashlib
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)

# Paths
GUIDELINE_PATH = Path(__file__).parent.parent / "data" / "final.md"
CACHE_DIR = Path(__file__).parent.parent / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache file paths
CHUNKS_CACHE_FILE = CACHE_DIR / "guideline_chunks.json"
VECTORS_CACHE_FILE = CACHE_DIR / "guideline_vectors.pkl"
VOCABULARY_CACHE_FILE = CACHE_DIR / "guideline_vocabulary.json"

# Half-page chunk size (approximately 750 characters)
HALF_PAGE_CHUNK_SIZE = 750


class GuidelineService:
    """
    Service for retrieving relevant sections from the NICE NG12 guideline.
    
    Provides traceable artifacts showing exactly what guideline text was used.
    """
    
    def __init__(self):
        self._guideline_content: str | None = None
        self._sections: dict[str, str] = {}
        self._chunks: list[dict[str, Any]] = []  # Store all chunks with metadata
        self._vocabulary: set[str] = set()  # All unique words in document
        self._vocab_list: list[str] = []  # Sorted vocabulary list for consistent indexing
        self._chunk_vectors: list[np.ndarray] = []  # Bag-of-words vectors for each chunk
        self._loaded = False
    
    def _load_guideline(self) -> None:
        """Load the guideline file into memory."""
        if self._loaded:
            return
        
        try:
            logger.info("Attempting to load guideline", path=str(GUIDELINE_PATH), exists=GUIDELINE_PATH.exists())
            
            if not GUIDELINE_PATH.exists():
                logger.warning("Guideline file not found", path=str(GUIDELINE_PATH), absolute=str(GUIDELINE_PATH.resolve()))
                self._guideline_content = ""
                self._loaded = True
                return
            
            with open(GUIDELINE_PATH, "r", encoding="utf-8") as f:
                self._guideline_content = f.read()
            
            # Parse sections for faster lookup
            self._parse_sections()
            
            # Try to load from cache, otherwise chunk and build vectors
            if not self._load_cache():
                logger.info("Cache not found or invalid, creating chunks and vectors...")
                self._chunk_document()
                self._build_vocabulary()
                self._build_chunk_vectors()
                self._save_cache()
            else:
                logger.info("Loaded chunks and vectors from cache")
            
            logger.info(
                "Guideline loaded successfully",
                path=str(GUIDELINE_PATH),
                length=len(self._guideline_content),
                sections=len(self._sections),
                chunks=len(self._chunks),
                vocabulary_size=len(self._vocabulary),
            )
            self._loaded = True
            
        except Exception as e:
            logger.error("Failed to load guideline", error=str(e), exc_info=True)
            self._guideline_content = ""
            self._loaded = True
    
    def _parse_sections(self) -> None:
        """Parse the guideline into sections by heading."""
        if not self._guideline_content:
            return
        
        lines = self._guideline_content.split("\n")
        current_section = "Introduction"
        current_content: list[str] = []
        
        for line in lines:
            # Detect section headers (## or ###)
            if line.startswith("##"):
                # Save previous section
                if current_content:
                    self._sections[current_section] = "\n".join(current_content).strip()
                
                # Start new section
                current_section = line.replace("#", "").strip()
                current_content = [line]
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            self._sections[current_section] = "\n".join(current_content).strip()
    
    def _chunk_document(self) -> None:
        """
        Chunk the guideline document by half-page (~750 characters).
        Chunks are created once and stored locally.
        """
        if not self._guideline_content:
            return
        
        self._chunks = []
        lines = self._guideline_content.split("\n")
        current_section = "Introduction"
        current_chunk = ""
        chunk_idx = 0
        
        for line in lines:
            # Detect section headers
            if line.startswith("##"):
                # Save current chunk if it has content
                if current_chunk.strip():
                    self._chunks.append({
                        "chunk_id": chunk_idx,
                        "section": current_section,
                        "text": current_chunk.strip(),
                        "char_count": len(current_chunk.strip()),
                    })
                    chunk_idx += 1
                    current_chunk = ""
                
                # Update section
                current_section = line.replace("#", "").strip()
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
                
                # If chunk exceeds half-page size, split it
                if len(current_chunk) >= HALF_PAGE_CHUNK_SIZE:
                    # Try to split at sentence boundary
                    sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                    if len(sentences) > 1:
                        # Save first part
                        first_part = sentences[0] + " "
                        if len(first_part.strip()) >= 100:  # Minimum chunk size
                            self._chunks.append({
                                "chunk_id": chunk_idx,
                                "section": current_section,
                                "text": first_part.strip(),
                                "char_count": len(first_part.strip()),
                            })
                            chunk_idx += 1
                            # Continue with remaining sentences
                            current_chunk = " ".join(sentences[1:])
                        else:
                            # Sentence too short, just save the whole chunk
                            self._chunks.append({
                                "chunk_id": chunk_idx,
                                "section": current_section,
                                "text": current_chunk.strip(),
                                "char_count": len(current_chunk.strip()),
                            })
                            chunk_idx += 1
                            current_chunk = ""
                    else:
                        # No sentence boundary, save as-is
                        self._chunks.append({
                            "chunk_id": chunk_idx,
                            "section": current_section,
                            "text": current_chunk.strip(),
                            "char_count": len(current_chunk.strip()),
                        })
                        chunk_idx += 1
                        current_chunk = ""
        
        # Save final chunk
        if current_chunk.strip():
            self._chunks.append({
                "chunk_id": chunk_idx,
                "section": current_section,
                "text": current_chunk.strip(),
                "char_count": len(current_chunk.strip()),
            })
        
        logger.info(f"Chunked document into {len(self._chunks)} chunks (half-page size: {HALF_PAGE_CHUNK_SIZE})")
    
    def _build_vocabulary(self) -> None:
        """Build vocabulary (set of unique words) from all chunks."""
        if not self._chunks:
            return
        
        self._vocabulary = set()
        
        for chunk in self._chunks:
            text = chunk["text"].lower()
            # Extract words (alphanumeric, at least 2 chars)
            words = re.findall(r'\b[a-z0-9]{2,}\b', text)
            self._vocabulary.update(words)
        
        # Create sorted list for consistent indexing
        self._vocab_list = sorted(self._vocabulary)
        
        logger.info(f"Built vocabulary: {len(self._vocabulary)} unique words")
    
    def _text_to_bow_vector(self, text: str) -> np.ndarray:
        """
        Convert text to bag-of-words vector.
        
        Uses consistent vocabulary ordering to ensure vector indices match.
        
        Args:
            text: Input text.
            
        Returns:
            Bag-of-words vector as numpy array (L2 normalized).
        """
        if not self._vocab_list:
            # Fallback: build vocab list if not available
            self._vocab_list = sorted(self._vocabulary) if self._vocabulary else []
        
        if not self._vocab_list:
            # No vocabulary, return empty vector
            return np.array([])
        
        vocab_index = {word: idx for idx, word in enumerate(self._vocab_list)}
        
        # Tokenize text
        text_lower = text.lower()
        words = re.findall(r'\b[a-z0-9]{2,}\b', text_lower)
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Create vector
        vector = np.zeros(len(self._vocab_list))
        for word, count in word_counts.items():
            if word in vocab_index:
                vector[vocab_index[word]] = count
        
        # Normalize (L2 norm) - required for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _build_chunk_vectors(self) -> None:
        """Build bag-of-words vectors for all chunks."""
        if not self._chunks or not self._vocabulary:
            return
        
        # Ensure vocab_list is built
        if not self._vocab_list:
            self._vocab_list = sorted(self._vocabulary)
        
        self._chunk_vectors = []
        
        logger.info(f"Building {len(self._chunks)} chunk vectors with vocabulary size {len(self._vocab_list)}")
        
        for chunk in self._chunks:
            vector = self._text_to_bow_vector(chunk["text"])
            if len(vector) > 0:
                self._chunk_vectors.append(vector)
            else:
                # Empty vector, create zero vector of correct size
                self._chunk_vectors.append(np.zeros(len(self._vocab_list)))
        
        logger.info(f"Built all {len(self._chunk_vectors)} chunk vectors")
    
    def _save_cache(self) -> None:
        """Save chunks, vectors, and vocabulary to disk cache."""
        try:
            # Save chunks (JSON)
            chunks_data = []
            for chunk in self._chunks:
                chunks_data.append({
                    "chunk_id": chunk["chunk_id"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "char_count": chunk["char_count"],
                })
            
            with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            # Save vectors (pickle for numpy arrays)
            with open(VECTORS_CACHE_FILE, "wb") as f:
                pickle.dump(self._chunk_vectors, f)
            
            # Save vocabulary (JSON) - save as sorted list for consistent ordering
            if not self._vocab_list:
                self._vocab_list = sorted(self._vocabulary)
            
            with open(VOCABULARY_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._vocab_list, f, indent=2)
            
            # Save document hash to detect changes
            doc_hash = hashlib.md5(self._guideline_content.encode()).hexdigest()
            hash_file = CACHE_DIR / "guideline_hash.txt"
            with open(hash_file, "w") as f:
                f.write(doc_hash)
            
            logger.info(
                "Cache saved successfully",
                chunks_file=str(CHUNKS_CACHE_FILE),
                vectors_file=str(VECTORS_CACHE_FILE),
                vocabulary_file=str(VOCABULARY_CACHE_FILE),
            )
        except Exception as e:
            logger.warning("Failed to save cache", error=str(e))
    
    def _load_cache(self) -> bool:
        """
        Load chunks, vectors, and vocabulary from disk cache.
        
        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        try:
            # Check if cache files exist
            if not all([
                CHUNKS_CACHE_FILE.exists(),
                VECTORS_CACHE_FILE.exists(),
                VOCABULARY_CACHE_FILE.exists(),
            ]):
                logger.debug("Cache files not found")
                return False
            
            # Check if source document has changed (compare with cached hash)
            hash_file = CACHE_DIR / "guideline_hash.txt"
            if GUIDELINE_PATH.exists() and hash_file.exists():
                # Read current document to compute hash
                with open(GUIDELINE_PATH, "r", encoding="utf-8") as f:
                    current_content = f.read()
                
                current_hash = hashlib.md5(current_content.encode()).hexdigest()
                
                with open(hash_file, "r") as f:
                    cached_hash = f.read().strip()
                
                if current_hash != cached_hash:
                    logger.info("Source document changed, cache invalid - will rebuild")
                    return False
            
            # Load chunks
            with open(CHUNKS_CACHE_FILE, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            
            self._chunks = chunks_data
            
            # Load vectors
            with open(VECTORS_CACHE_FILE, "rb") as f:
                self._chunk_vectors = pickle.load(f)
            
            # Load vocabulary
            with open(VOCABULARY_CACHE_FILE, "r", encoding="utf-8") as f:
                self._vocab_list = json.load(f)
            
            self._vocabulary = set(self._vocab_list)
            
            logger.info(
                "Cache loaded successfully",
                chunks_count=len(self._chunks),
                vectors_count=len(self._chunk_vectors),
                vocabulary_size=len(self._vocabulary),
            )
            
            return True
            
        except Exception as e:
            logger.warning("Failed to load cache", error=str(e))
            return False
    
    def _rerank_chunks(
        self,
        chunks_with_scores: list[tuple[dict[str, Any], float]],
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Rerank chunks using a simple reranking algorithm.
        
        Reranking factors:
        - Original cosine similarity score
        - Exact phrase matches
        - Query term density in chunk
        - Section relevance
        
        Args:
            chunks_with_scores: List of (chunk, similarity_score) tuples
            query: Original query
            top_k: Number of chunks to return after reranking
            
        Returns:
            List of reranked chunks with updated relevance scores.
        """
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b[a-z0-9]{2,}\b', query_lower))
        
        reranked = []
        
        for chunk, cosine_score in chunks_with_scores:
            chunk_text = chunk["text"].lower()
            chunk_words = set(re.findall(r'\b[a-z0-9]{2,}\b', chunk_text))
            
            # Reranking score (start with cosine similarity)
            rerank_score = cosine_score
            
            # Boost for exact phrase match
            if query_lower in chunk_text:
                rerank_score += 0.3
            
            # Boost for query term density
            matching_terms = query_terms.intersection(chunk_words)
            if query_terms:
                term_density = len(matching_terms) / len(query_terms)
                rerank_score += term_density * 0.2
            
            # Boost for section name relevance
            section_lower = chunk["section"].lower()
            if any(term in section_lower for term in query_terms):
                rerank_score += 0.1
            
            reranked.append((chunk, rerank_score))
        
        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k with updated scores
        result = []
        for chunk, rerank_score in reranked[:top_k]:
            chunk_with_score = chunk.copy()
            chunk_with_score["relevance_score"] = float(rerank_score)
            result.append(chunk_with_score)
        
        return result
    
    def search(
        self,
        query: str,
        max_chunks: int = 3,
        chunk_size: int = 500,  # Legacy param, kept for compatibility but not used
    ) -> list[dict[str, Any]]:
        """
        Search the guideline using classic RAG pipeline:
        1. Bag-of-words vectorization of query
        2. Cosine similarity with all chunks
        3. Reranker for final ranking
        
        Args:
            query: Search query (keywords, symptoms, cancer types, etc.)
            max_chunks: Maximum number of chunks to return after reranking (capped at 3)
            chunk_size: Legacy parameter, not used (chunks are pre-computed)
            
        Returns:
            List of relevant chunks with metadata and relevance scores (max 3 chunks).
            Each artifact contains the full chunk data including chunk_id and char_count.
        """
        self._load_guideline()
        
        if not self._chunks or not self._chunk_vectors:
            logger.warning("No chunks or vectors available for search")
            return []
        
        if not self._vocabulary:
            logger.warning("Vocabulary not built")
            return []
        
        logger.debug(
            "Starting RAG search",
            query=query[:100],
            chunks_available=len(self._chunks),
            max_chunks=max_chunks,
        )
        
        # Step 1: Convert query to bag-of-words vector
        query_vector = self._text_to_bow_vector(query)
        
        if len(query_vector) == 0:
            logger.warning("Query vector is empty, vocabulary may not be built")
            return []
        
        # Step 2: Calculate cosine similarity with all chunks
        similarities: list[tuple[dict[str, Any], float]] = []
        
        for chunk, chunk_vector in zip(self._chunks, self._chunk_vectors):
            # Verify vector dimensions match
            if len(query_vector) != len(chunk_vector):
                logger.warning(
                    "Vector dimension mismatch",
                    query_dim=len(query_vector),
                    chunk_dim=len(chunk_vector),
                    chunk_id=chunk.get("chunk_id"),
                )
                continue
            
            # Calculate cosine similarity: dot product of normalized vectors
            # Returns value between -1 (opposite) and 1 (identical)
            try:
                # Cosine similarity = dot(a, b) / (||a|| * ||b||)
                # Since vectors are already L2 normalized, this simplifies to dot product
                dot_product = np.dot(query_vector, chunk_vector)
                
                # Handle NaN or edge cases
                if np.isnan(dot_product) or np.isinf(dot_product):
                    similarity = 0.0
                else:
                    # Clip to [0, 1] range (negative similarity means opposite direction, treat as 0)
                    similarity = max(0.0, float(dot_product))
                
                similarities.append((chunk, similarity))
            except Exception as e:
                logger.debug("Error calculating similarity", chunk_id=chunk.get("chunk_id"), error=str(e))
                continue
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Take top candidates for reranking (take more than max_chunks for reranking)
        top_candidates = similarities[:max(max_chunks * 3, 10)]
        
        # Step 4: Rerank (enforce max 3 chunks)
        max_chunks_enforced = min(max_chunks, 3)
        reranked_chunks = self._rerank_chunks(top_candidates, query, top_k=max_chunks_enforced)
        
        # Step 5: Format as artifacts (these ARE the chunks used)
        artifacts = []
        for chunk in reranked_chunks:
            # Include all chunk metadata for full traceability
            artifacts.append({
                "section": chunk.get("section", "Unknown"),
                "text": chunk.get("text", ""),
                "relevance_score": chunk.get("relevance_score", 0.0),
                "source": "NICE NG12",
                "source_url": "https://www.nice.org.uk/guidance/ng12",
                # Additional chunk metadata for traceability
                "chunk_id": chunk.get("chunk_id"),
                "char_count": chunk.get("char_count", len(chunk.get("text", ""))),
            })
        
        # If no artifacts found, return overview chunks (max 3)
        if not artifacts and self._chunks:
            logger.info("No relevant chunks found, returning overview")
            overview_chunks = self._chunks[:max_chunks_enforced]
            for chunk in overview_chunks:
                artifacts.append({
                    "section": chunk.get("section", "Overview"),
                    "text": chunk.get("text", ""),
                    "relevance_score": 0.0,
                    "source": "NICE NG12",
                    "source_url": "https://www.nice.org.uk/guidance/ng12",
                    "chunk_id": chunk.get("chunk_id"),
                    "char_count": chunk.get("char_count", len(chunk.get("text", ""))),
                })
        
        logger.info(
            "RAG search completed",
            query=query[:50],
            artifacts_found=len(artifacts),
            max_chunks_limit=3,
            top_similarity=similarities[0][1] if similarities else 0.0,
            chunks_searched=len(similarities),
            reranked_count=len(reranked_chunks),
        )
        
        return artifacts
    
    
    def format_artifacts_for_llm(self, artifacts: list[dict[str, Any]]) -> str:
        """
        Format artifacts as context for the LLM.
        
        Args:
            artifacts: List of artifact dictionaries.
            
        Returns:
            Formatted string for LLM context.
        """
        if not artifacts:
            return ""
        
        sections = []
        for i, artifact in enumerate(artifacts, 1):
            section_name = artifact.get("section", "Unknown")
            text = artifact.get("text", "")
            sections.append(f"### {section_name}\n{text}")
        
        return "\n\n".join(sections)


# Singleton instance
_guideline_service: GuidelineService | None = None


def get_guideline_service() -> GuidelineService:
    """Get the guideline service singleton."""
    global _guideline_service
    if _guideline_service is None:
        _guideline_service = GuidelineService()
    return _guideline_service
