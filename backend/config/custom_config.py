"""
Discrete configuration module for the Custom NG12 Assistant Pipeline.

All settings are configurable via environment variables with sensible defaults.
This module is separate from the main config to keep the custom pipeline discrete.
"""

import hashlib
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CustomPipelineSettings(BaseSettings):
    """
    Configuration settings for the Custom NG12 Assistant Pipeline.
    
    All settings can be overridden via environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file="../.env",  # Load from project root
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
        env_prefix="",  # No prefix - use field names as-is
    )
    
    # Confidence and Thresholds
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for recommendations (0-1)"
    )
    
    # Embedding Model
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name (e.g., text-embedding-3-small, text-embedding-3-large)"
    )
    
    # BM25 Parameters
    bm25_k1: float = Field(
        default=1.5,
        ge=0.0,
        description="BM25 k1 parameter (term frequency saturation)"
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b parameter (length normalization)"
    )
    
    # Hybrid Retrieval Score Weights
    score_weight_bm25: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="BM25 weight in hybrid retrieval (0-1)"
    )
    score_weight_embedding: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Embedding weight in hybrid retrieval (0-1)"
    )
    
    # Symptom Vocabulary (can be loaded from file or env)
    symptom_vocabulary: list[str] = Field(
        default_factory=lambda: [
            "dysphagia",
            "weight loss",
            "rectal bleeding",
            "abdominal pain",
            "haemoptysis",
            "cough",
            "fatigue",
            "thrombocytosis",
            "dyspepsia",
            "heartburn",
            "reflux",
            "change in bowel habit",
            "constipation",
            "diarrhoea",
            "bloating",
            "nausea",
            "vomiting",
            "chest pain",
            "shortness of breath",
            "hoarseness",
            "lump",
            "mass",
            "bleeding",
            "pain",
        ],
        description="Controlled vocabulary for symptom tagging"
    )
    
    # Retrieval Parameters
    max_retrieved_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to retrieve"
    )
    
    # LLM Parameters for Custom Pipeline
    llm_temperature_intent: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for intent classification (low for determinism)"
    )
    llm_temperature_extraction: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for field extraction (low for determinism)"
    )
    llm_temperature_response: float = Field(
        default=1.0,  # Balanced temperature for general conversation and data analysis
        ge=0.0,
        le=2.0,
        description="LLM temperature for final response generation (1.0 = balanced, 1.3 = more conversational)"
    )
    
    # Confidence Calculation Weights
    confidence_weight_retrieval: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for retrieval_strength in confidence calculation"
    )
    confidence_weight_constraint: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for constraint_match in confidence calculation"
    )
    confidence_weight_specificity: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for evidence_specificity in confidence calculation"
    )
    confidence_weight_coverage: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for coverage in confidence calculation"
    )
    confidence_weight_metadata: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for metadata_quality in confidence calculation"
    )
    confidence_weight_consensus: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for action_consensus in confidence calculation"
    )
    
    # Age Threshold Options for Intake Questions
    # Note: Complex structure, not loaded from env vars
    age_thresholds: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {"label": "Under 18", "min": 0, "max": 17},
            {"label": "18-54", "min": 18, "max": 54},
            {"label": "55-64", "min": 55, "max": 64},
            {"label": "65+", "min": 65, "max": 150},
        ],
        description="Age threshold options for intake questions",
    )
    
    @field_validator("age_thresholds", mode="before")
    @classmethod
    def validate_age_thresholds(cls, v: Any) -> Any:
        """Ensure age_thresholds maintains correct structure and isn't parsed incorrectly from env."""
        # If it's already a list of dicts with correct structure, return as-is
        if isinstance(v, list) and all(
            isinstance(item, dict) and "label" in item and "min" in item and "max" in item
            for item in v
        ):
            return v
        # If somehow malformed (e.g., from env var parsing), return default
        return [
            {"label": "Under 18", "min": 0, "max": 17},
            {"label": "18-54", "min": 18, "max": 54},
            {"label": "55-64", "min": 55, "max": 64},
            {"label": "65+", "min": 65, "max": 150},
        ]
    
    # Safety Gate Keywords
    emergency_keywords: list[str] = Field(
        default_factory=lambda: [
            "diagnose",
            "diagnosis",
            "treatment",
            "prescribe",
            "medication",
            "drug",
            "dose",
            "dosage",
            "therapy",
            "chemotherapy",
            "radiotherapy",
            "surgery",
            "operate",
        ],
        description="Keywords that trigger safety gate red flags"
    )
    
    # Configuration Versioning
    config_version: str = Field(
        default="1.0.0",
        description="Configuration version for traceability in logs"
    )
    
    # Document Preprocessing Parameters
    document_path: str = Field(
        default="data/final.md",
        description="Path to the NG12 guideline document"
    )
    
    # Section/Subsection Selection Parameters
    max_sections_to_select: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum number of sections to select in Node 1"
    )
    max_subsections_to_select: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum number of subsections to select in Node 1"
    )
    
    # Search Parameters
    top_k_per_section: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Top K subsections to return per section in Node 2"
    )
    search_bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 score in search (cosine weight = 1 - this)"
    )
    selected_subsection_boost: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Multiplier boost for subsections selected by Node 1"
    )
    
    # Reranking Parameters
    top_n_chunks: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Top N chunks to return after reranking"
    )
    rerank_bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 score in reranking (cosine weight = 1 - this)"
    )
    
    # Input Validation Parameters
    min_query_length: int = Field(
        default=3,
        ge=1,
        description="Minimum query length after trimming"
    )
    
    # LLM Parameters for Section Selection
    llm_temperature_section_selection: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for section/subsection selection (low for determinism)"
    )
    
    def validate_weights(self) -> bool:
        """Validate that confidence weights sum to approximately 1.0."""
        total = (
            self.confidence_weight_retrieval +
            self.confidence_weight_constraint +
            self.confidence_weight_specificity +
            self.confidence_weight_coverage +
            self.confidence_weight_metadata +
            self.confidence_weight_consensus
        )
        return 0.95 <= total <= 1.05  # Allow small floating point differences
    
    def get_config_hash(self) -> str:
        """
        Compute a hash of key configuration values for versioning.
        
        This provides a deterministic version identifier that changes when
        configuration values change, useful for traceability.
        """
        # Hash key configuration values
        key_values = {
            "confidence_threshold": self.confidence_threshold,
            "embedding_model": self.embedding_model,
            "max_retrieved_chunks": self.max_retrieved_chunks,
            "max_sections_to_select": self.max_sections_to_select,
            "max_subsections_to_select": self.max_subsections_to_select,
            "top_k_per_section": self.top_k_per_section,
            "top_n_chunks": self.top_n_chunks,
            "search_bm25_weight": self.search_bm25_weight,
            "rerank_bm25_weight": self.rerank_bm25_weight,
        }
        config_str = str(sorted(key_values.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_safe_config_dict(self) -> dict:
        """Return configuration dict with sensitive values redacted for logging."""
        return self.model_dump()


@lru_cache
def get_custom_settings() -> CustomPipelineSettings:
    """
    Get cached custom pipeline settings.
    
    Settings are loaded once and cached for the application lifetime.
    """
    settings = CustomPipelineSettings()
    if not settings.validate_weights():
        raise ValueError(
            "Confidence weights must sum to approximately 1.0. "
            f"Current sum: {sum([settings.confidence_weight_retrieval, settings.confidence_weight_constraint, settings.confidence_weight_specificity, settings.confidence_weight_coverage, settings.confidence_weight_metadata, settings.confidence_weight_consensus])}"
        )
    return settings
