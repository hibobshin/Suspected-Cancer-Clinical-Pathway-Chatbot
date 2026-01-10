"""
Pydantic models for the Custom NG12 Assistant Pipeline.

All models enforce strict validation and fail-closed behavior.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================================
# Stage 1: Intent + Safety Gate
# ============================================================================

class IntentType(str, Enum):
    """Intent classification types."""
    GUIDELINE_LOOKUP = "guideline_lookup"
    CASE_TRIAGE = "case_triage"
    DOCUMENTATION = "documentation"


class IntentClassification(BaseModel):
    """Intent classification result."""
    intent: IntentType = Field(..., description="Classified intent type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Brief reasoning for classification")


class SafetyCheck(BaseModel):
    """Safety check result."""
    has_red_flags: bool = Field(..., description="Whether red flags were detected")
    red_flags_list: list[str] = Field(default_factory=list, description="List of detected red flags")
    escalation_message: str | None = Field(
        default=None,
        description="Escalation message if red flags found"
    )


class SafetyGateResult(BaseModel):
    """Safety gate result."""
    passed: bool = Field(..., description="Whether safety gate passed")
    escalation_message: str | None = Field(
        default=None,
        description="Escalation message if gate failed"
    )
    intent_classification: IntentClassification | None = Field(
        default=None,
        description="Intent classification if gate passed"
    )


# ============================================================================
# Stage 2: Structured Intake
# ============================================================================

class CaseFields(BaseModel):
    """Case fields extracted from conversation."""
    age: int | None = Field(default=None, ge=0, le=150, description="Patient age")
    sex: str | None = Field(default=None, description="Patient sex")
    symptoms: list[str] = Field(default_factory=list, description="List of symptoms")
    symptom_duration: str | None = Field(default=None, description="Symptom duration")
    key_triggers: list[str] = Field(default_factory=list, description="Key trigger factors")
    missing_fields: list[str] = Field(default_factory=list, description="Fields that are missing or ambiguous")


class IntakeOption(BaseModel):
    """Option for an intake question."""
    option_id: str = Field(..., description="Unique option identifier")
    label: str = Field(..., description="Display label for the option")
    value: str = Field(..., description="Value for this option")
    pathway_correlation: str | None = Field(
        default=None,
        description="NG12 pathway correlation (section reference)"
    )


class IntakeQuestion(BaseModel):
    """Intake question for structured data collection."""
    question_id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="Question text to display")
    question_type: Literal["single_choice", "multi_choice", "text", "number"] = Field(
        ...,
        description="Type of question"
    )
    options: list[IntakeOption] | None = Field(
        default=None,
        description="Options for choice questions"
    )
    required: bool = Field(default=True, description="Whether this question is required")


class IntakeResult(BaseModel):
    """Result of intake validation."""
    fields_collected: CaseFields = Field(..., description="Collected case fields")
    is_complete: bool = Field(..., description="Whether all required fields are collected")
    current_question: IntakeQuestion | None = Field(
        default=None,
        description="Current question to ask (if not complete)"
    )
    follow_up_questions: list[IntakeQuestion] = Field(
        default_factory=list,
        description="Additional follow-up questions"
    )


# ============================================================================
# Stage 3: Structured Retrieval
# ============================================================================

class ActionType(str, Enum):
    """Action type for recommendations."""
    TWO_WW_REFERRAL = "2WW_REFERRAL"
    URGENT_TEST = "URGENT_TEST"
    ROUTINE_WORKUP = "ROUTINE_WORKUP"
    SAFETY_NET = "SAFETY_NET"
    INFO_ONLY = "INFO_ONLY"
    URGENT_CXR = "URGENT_CXR"
    CONSIDER_URGENT_CXR = "CONSIDER_URGENT_CXR"


class TriggerType(str, Enum):
    """Trigger type for recommendations."""
    SYMPTOM = "SYMPTOM"
    SIGN = "SIGN"
    TEST_RESULT = "TEST_RESULT"
    INCIDENTAL_FINDING = "INCIDENTAL_FINDING"


class MetadataQuality(str, Enum):
    """Metadata quality level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class InheritedMetadata(BaseModel):
    """Metadata inherited from parent sections."""
    cancer_site: str | None = Field(default=None, description="Cancer site from parent section")
    section_path: str = Field(..., description="Section path (e.g., 'NG12 > Lung cancer')")
    guideline_version: str = Field(..., description="Guideline version")
    source_doc: str = Field(..., description="Source document name")
    source_url: str | None = Field(default=None, description="Source document URL")
    source_page: int | None = Field(default=None, description="Page number if available")


class LocalMetadata(BaseModel):
    """Metadata extracted locally from chunk text."""
    rule_id: str | None = Field(default=None, description="Rule ID (e.g., '1.1.1')")
    age_min: int | None = Field(default=None, ge=0, description="Minimum age")
    age_max: int | None = Field(default=None, ge=0, description="Maximum age")
    age_text: str | None = Field(default=None, description="Verbatim age text")
    symptom_tags: list[str] = Field(default_factory=list, description="Symptom tags from controlled vocabulary")
    action_type: ActionType | None = Field(default=None, description="Action type")
    trigger_type: TriggerType | None = Field(default=None, description="Trigger type")


class AuditMetadata(BaseModel):
    """Audit metadata for traceability."""
    source_heading: str = Field(..., description="Exact heading text")
    source_offsets: dict[str, int] | None = Field(
        default=None,
        description="Character offsets (start/end) if available"
    )
    age_text: str | None = Field(default=None, description="Verbatim age phrase")
    extraction_notes: str | None = Field(
        default=None,
        description="Notes about extraction (e.g., 'age ambiguous', 'rule_id not found')"
    )


class NG12ChunkMetadata(BaseModel):
    """Complete metadata for an NG12 chunk."""
    inherited: InheritedMetadata = Field(..., description="Inherited metadata")
    local: LocalMetadata = Field(..., description="Local metadata")
    audit: AuditMetadata = Field(..., description="Audit metadata")
    metadata_quality: MetadataQuality = Field(..., description="Metadata quality level")


class SectionContainer(BaseModel):
    """Section container for navigation (Level 1)."""
    container_id: str = Field(..., description="Unique container identifier")
    title: str = Field(..., description="Section title")
    section_path: str = Field(..., description="Section path")
    cancer_site: str | None = Field(default=None, description="Cancer site")
    children: list[str] = Field(
        default_factory=list,
        description="List of rule chunk IDs belonging to this container"
    )


class NG12Chunk(BaseModel):
    """NG12 rule chunk for retrieval (Level 2)."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Verbatim chunk text")
    metadata: NG12ChunkMetadata = Field(..., description="Chunk metadata")
    verbatim_source: str = Field(..., description="Verbatim source text")
    parent_container_id: str | None = Field(
        default=None,
        description="Parent section container ID"
    )


class RetrievalResult(BaseModel):
    """Result from structured retrieval."""
    candidate_sections: list[str] = Field(
        default_factory=list,
        description="Candidate section paths"
    )
    rule_chunks: list[NG12Chunk] = Field(..., description="Retrieved rule chunks")
    retrieval_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Retrieval scores (BM25, embedding, combined)"
    )


# ============================================================================
# Stage 4: Evidence Extraction + Confidence + Final Output
# ============================================================================

class VerbatimEvidence(BaseModel):
    """Verbatim evidence extracted from chunks."""
    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Verbatim evidence text")
    section_path: str = Field(..., description="Section path")
    rule_id: str | None = Field(default=None, description="Rule ID if available")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata_quality: MetadataQuality = Field(..., description="Metadata quality")


class ConfidenceFactors(BaseModel):
    """Individual confidence factors."""
    retrieval_strength: float = Field(..., ge=0.0, le=1.0, description="Retrieval strength (0-1)")
    constraint_match: float = Field(..., ge=0.0, le=1.0, description="Constraint match percentage (0-1)")
    evidence_specificity: float = Field(..., ge=0.0, le=1.0, description="Evidence specificity (0-1)")
    coverage: float = Field(..., ge=0.0, le=1.0, description="Coverage score (0-1)")
    metadata_quality_score: float = Field(..., ge=0.0, le=1.0, description="Metadata quality score (0-1)")
    action_consensus: float = Field(..., ge=0.0, le=1.0, description="Action consensus (0-1)")


class ConfidenceScore(BaseModel):
    """Overall confidence score."""
    overall: float = Field(..., ge=0.0, le=1.0, description="Overall confidence (0-1)")
    factors: ConfidenceFactors = Field(..., description="Individual confidence factors")
    threshold_met: bool = Field(..., description="Whether confidence threshold is met")


class Citation(BaseModel):
    """Citation in the response."""
    rule_id: str | None = Field(default=None, description="Rule ID")
    section_path: str = Field(..., description="Section path")
    evidence_text: str = Field(..., description="Verbatim evidence text excerpt")


class StructuredResponse(BaseModel):
    """Structured final response."""
    answer: str = Field(..., description="Answer text")
    citations: list[Citation] = Field(default_factory=list, description="Validated citations")
    evidence: list[VerbatimEvidence] = Field(..., description="Verbatim evidence")
    confidence: ConfidenceScore = Field(..., description="Confidence score")
    referral_note_draft: str | None = Field(
        default=None,
        description="Draft referral note if applicable"
    )


# ============================================================================
# Memory Support
# ============================================================================

class ConversationMemory(BaseModel):
    """Conversation memory for persistence."""
    conversation_id: str = Field(..., description="Conversation ID")
    intake_state: dict = Field(default_factory=dict, description="Intake state")
    collected_fields: CaseFields | None = Field(
        default=None,
        description="Collected case fields"
    )
    previous_intent: IntentType | None = Field(
        default=None,
        description="Previous intent classification"
    )
    retrieval_history: list[RetrievalResult] = Field(
        default_factory=list,
        description="Retrieval history"
    )
