"""
Pydantic models for the Document-Native Rule Engine.

This module defines the structured representation of NG12 rules,
extracted facts, match results, and conversation state.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field


# ============================================================================
# Action Types (from NG12 guideline)
# ============================================================================

class ActionType(str, Enum):
    """Types of actions specified in NG12 rules."""
    REFER_SUSPECTED_CANCER = "refer_suspected_cancer_pathway"
    OFFER_URGENT = "offer_urgent"
    CONSIDER_URGENT = "consider_urgent"
    CONSIDER_NON_URGENT = "consider_non_urgent"
    OFFER_VERY_URGENT = "offer_very_urgent"
    REFER_IMMEDIATE = "refer_immediate"
    CONSIDER_REFERRAL = "consider_referral"


class LogicOperator(str, Enum):
    """Logical operators for condition groups."""
    AND = "and"
    OR = "or"


# ============================================================================
# Condition Models (for parsing rule logic)
# ============================================================================

class AtomicCondition(BaseModel):
    """Leaf node - single checkable condition."""
    type: Literal["symptom", "finding", "history", "age"] = Field(
        ..., description="Type of condition"
    )
    value: str = Field(..., description="The condition value (e.g., 'haemoptysis')")
    qualifier: str | None = Field(
        default=None, 
        description="Qualifier like 'unexplained', 'persistent'"
    )

    def __hash__(self):
        return hash((self.type, self.value, self.qualifier))


class CountCondition(BaseModel):
    """Threshold condition - N or more from a list."""
    type: Literal["count_gte", "count_any"] = Field(
        ..., description="count_gte for 'N or more', count_any for 'any'"
    )
    threshold: int = Field(..., ge=1, description="Minimum count required")
    options: list[AtomicCondition] = Field(
        ..., description="List of options to count matches from"
    )


class CompositeCondition(BaseModel):
    """Logic node - combines conditions with AND/OR."""
    type: Literal["and", "or"] = Field(..., description="Logical operator")
    children: list[Union["AtomicCondition", "CountCondition", "CompositeCondition"]] = Field(
        ..., description="Child conditions"
    )


# Union type for any condition
Condition = AtomicCondition | CountCondition | CompositeCondition

# Update forward refs for recursive model
CompositeCondition.model_rebuild()


# ============================================================================
# Age Constraint
# ============================================================================

class AgeConstraint(BaseModel):
    """Age constraint for a rule."""
    min_age: int | None = Field(default=None, ge=0, description="Minimum age")
    max_age: int | None = Field(default=None, ge=0, description="Maximum age")
    text: str = Field(..., description="Verbatim age text (e.g., 'aged 40 and over')")


# ============================================================================
# NG12 Rule
# ============================================================================

class NG12Rule(BaseModel):
    """Structured representation of a single NG12 rule."""
    rule_id: str = Field(..., description="Rule ID (e.g., '1.1.1')")
    cancer_site: str = Field(..., description="Cancer site (e.g., 'lung')")
    cancer_type: str | None = Field(default=None, description="Specific cancer type")
    section_path: str = Field(
        ..., 
        description="Full section path (e.g., 'NG12 > Lung and pleural cancers > Lung cancer')"
    )
    action: ActionType = Field(..., description="Action type")
    action_text: str = Field(..., description="Verbatim action text from guideline")
    age_constraint: AgeConstraint | None = Field(
        default=None, description="Age constraint if specified"
    )
    conditions: CompositeCondition | AtomicCondition | CountCondition | None = Field(
        default=None, description="Parsed condition tree"
    )
    verbatim_text: str = Field(..., description="Full verbatim rule text for citation")
    source_year: str = Field(..., description="Source year (e.g., '2015, amended 2025')")
    char_start: int = Field(..., ge=0, description="Start position in document")
    char_end: int = Field(..., ge=0, description="End position in document")

    def __hash__(self):
        return hash(self.rule_id)


# ============================================================================
# Extracted Facts (from user query)
# ============================================================================

class ExtractedFacts(BaseModel):
    """Facts extracted from a user query."""
    age: int | None = Field(default=None, ge=0, le=150, description="Patient age")
    age_term: str | None = Field(
        default=None, 
        description="Verbatim age term if not numeric (e.g., 'elderly')"
    )
    gender: Literal["male", "female"] | None = Field(default=None, description="Patient gender")
    symptoms: list[str] = Field(default_factory=list, description="Normalized symptom list")
    symptoms_raw: list[str] = Field(
        default_factory=list, 
        description="Raw symptom phrases from query (for audit)"
    )
    findings: list[str] = Field(default_factory=list, description="Clinical/test findings")
    history: list[str] = Field(default_factory=list, description="Patient history items")
    raw_query: str = Field(default="", description="Original query text")


# ============================================================================
# Match Result
# ============================================================================

class MatchResult(BaseModel):
    """Result of matching facts against a rule."""
    rule: NG12Rule = Field(..., description="The matched rule")
    match_type: Literal["full", "partial", "age_only", "symptom_only", "no_match"] = Field(
        ..., description="Type of match"
    )
    matched_conditions: list[str] = Field(
        default_factory=list, description="List of matched conditions"
    )
    unmatched_conditions: list[str] = Field(
        default_factory=list, description="List of unmatched conditions"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Match confidence (0-1)")


# ============================================================================
# Intake Models
# ============================================================================

class IntakeField(BaseModel):
    """Field for structured intake form."""
    id: str = Field(..., description="Field identifier")
    label: str = Field(..., description="Display label")
    field_type: Literal["number", "select", "multiselect", "text"] = Field(
        ..., description="Input type"
    )
    required: bool = Field(default=True, description="Whether field is required")
    options: list[str] | None = Field(default=None, description="Options for select/multiselect")
    context: str = Field(..., description="Why this field is needed")


class IntakeRequest(BaseModel):
    """Request for additional information."""
    type: Literal["intake_required"] = "intake_required"
    reason: str = Field(..., description="Why intake is needed")
    fields: list[IntakeField] = Field(..., description="Fields to collect")
    partial_assessment: str | None = Field(
        default=None, description="What can be said so far"
    )


# ============================================================================
# Conversation State
# ============================================================================

class ConversationTurn(BaseModel):
    """Single turn in a conversation."""
    turn_id: int = Field(..., description="Turn number")
    timestamp: datetime = Field(default_factory=datetime.now, description="Turn timestamp")
    user_message: str = Field(..., description="User's message")
    extracted_facts: ExtractedFacts = Field(..., description="Facts extracted this turn")
    matches: list[MatchResult] = Field(default_factory=list, description="Rules matched")
    response: str = Field(..., description="System response")
    response_type: Literal["answer", "clarification", "intake_form", "fail_closed"] = Field(
        ..., description="Type of response"
    )


class ConversationState(BaseModel):
    """Accumulated state across a conversation."""
    conversation_id: str = Field(..., description="Unique conversation ID")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    accumulated_facts: ExtractedFacts = Field(
        default_factory=ExtractedFacts, description="Merged facts from all turns"
    )
    asked_fields: set[str] = Field(
        default_factory=set, description="Fields already asked about"
    )
    previous_matches: list[MatchResult] = Field(
        default_factory=list, description="Matches from previous turns"
    )
    pending_intake: IntakeRequest | None = Field(
        default=None, description="Pending intake request"
    )
    turns: list[ConversationTurn] = Field(
        default_factory=list, description="Conversation history"
    )

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Rule Engine Response
# ============================================================================

class Artifact(BaseModel):
    """Citation artifact for traceability."""
    section: str = Field(..., description="Section path")
    text: str = Field(..., description="Relevant text snippet")
    rule_id: str | None = Field(default=None, description="Rule ID if applicable")
    relevance_score: float = Field(default=1.0, description="Relevance score")
    char_start: int | None = Field(default=None, description="Start position for highlighting")
    char_end: int | None = Field(default=None, description="End position for highlighting")


class RuleEngineResponse(BaseModel):
    """Response from the rule engine."""
    response: str | IntakeRequest = Field(..., description="Response text or intake request")
    response_type: Literal["answer", "clarification", "intake_form", "fail_closed"] = Field(
        ..., description="Type of response"
    )
    facts: ExtractedFacts = Field(..., description="Extracted/accumulated facts")
    matches: list[MatchResult] = Field(default_factory=list, description="Matched rules")
    artifacts: list[Artifact] = Field(default_factory=list, description="Citation artifacts")
    conversation_id: str | None = Field(default=None, description="Conversation ID if using memory")
