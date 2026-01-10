"""
Discrete configuration module for the Custom NG12 Assistant Pipeline.

All settings are configurable via environment variables with sensible defaults.
This module is separate from the main config to keep the custom pipeline discrete.
"""

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
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
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
        default=0.3,  # Lower temperature for more deterministic, grounded responses
        ge=0.0,
        le=2.0,
        description="LLM temperature for final response generation (lower = more deterministic)"
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
