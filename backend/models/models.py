"""
Pydantic models for API request/response validation.

All models are explicit, documented, and enforce strict validation.
Invalid inputs fail closed with descriptive error messages.
"""

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Valid roles for chat messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PathwayRouteType(str, Enum):
    """Available pathway route types for the chatbot."""
    CANCER_RECOGNITION = "cancer_recognition"
    SYMPTOM_TRIAGE = "symptom_triage"
    REFERRAL_GUIDANCE = "referral_guidance"
    GRAPH_RAG = "graph_rag"
    CUSTOM = "custom"


class ChatMessage(BaseModel):
    """
    A single message in a chat conversation.
    
    Attributes:
        role: Who sent the message (user, assistant, system).
        content: The message text content.
        timestamp: When the message was created.
    """
    role: MessageRole = Field(..., description="Message sender role")
    content: str = Field(..., min_length=1, max_length=10000, description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Message content cannot be empty or whitespace only")
        return v.strip()


class ConversationContext(BaseModel):
    """
    Context for a chat conversation including history and metadata.
    
    Attributes:
        conversation_id: Unique identifier for the conversation.
        messages: List of previous messages in the conversation.
        user_context: Optional additional context about the user query.
    """
    conversation_id: UUID = Field(default_factory=uuid4, description="Unique conversation ID")
    messages: list[ChatMessage] = Field(default_factory=list, description="Message history")
    user_context: str | None = Field(
        default=None,
        max_length=2000,
        description="Additional context for the query"
    )


class ChatRequest(BaseModel):
    """
    Request to send a message to the healthcare assistant.
    
    Attributes:
        message: The user's current message.
        route_type: The pathway route to use for this request.
        conversation_id: Optional ID to continue an existing conversation.
        context: Optional conversation context with history.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User message to the assistant"
    )
    route_type: PathwayRouteType = Field(
        default=PathwayRouteType.CANCER_RECOGNITION,
        description="Pathway route type for this chat"
    )
    conversation_id: UUID | None = Field(
        default=None,
        description="Existing conversation ID to continue"
    )
    context: ConversationContext | None = Field(
        default=None,
        description="Full conversation context"
    )
    
    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and clean the message."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty")
        return cleaned


class ResponseType(str, Enum):
    """Type of response from the assistant."""
    ANSWER = "answer"  # Direct answer to an in-scope question
    CLARIFICATION = "clarification"  # Probing for more information
    REFUSAL = "refusal"  # Out-of-scope, fail-closed response
    ERROR = "error"  # System error


class Citation(BaseModel):
    """
    A citation reference in an assistant response.
    
    Attributes:
        statement_id: The quality statement ID (e.g., QS124-S1).
        section: The section being cited.
        text: The relevant cited text.
    """
    statement_id: str = Field(..., description="Quality statement ID")
    section: str = Field(..., description="Section name")
    text: str | None = Field(default=None, description="Cited text excerpt")


class Artifact(BaseModel):
    """
    Source artifact showing the guideline chunk used for traceability.
    
    These artifacts are the actual chunks retrieved from the RAG pipeline.
    
    Attributes:
        section: The guideline section name.
        text: The actual guideline text chunk used.
        source: Source identifier (e.g., "NICE NG12").
        source_url: URL to the source document.
        relevance_score: Relevance score for this chunk.
        chunk_id: Optional chunk identifier for tracking.
        char_count: Character count of the chunk text.
        rule_id: Optional rule ID (e.g., "1.3.1") for section highlighting.
    """
    section: str = Field(..., description="Section name")
    text: str = Field(..., description="Guideline text chunk used")
    source: str = Field(default="NICE NG12", description="Source identifier")
    source_url: str = Field(
        default="https://www.nice.org.uk/guidance/ng12",
        description="URL to source document"
    )
    relevance_score: float = Field(default=0.0, description="Relevance score")
    chunk_id: str | None = Field(default=None, description="Chunk identifier")
    char_count: int | None = Field(default=None, description="Character count of chunk")
    rule_id: str | None = Field(default=None, description="Rule ID (e.g., '1.3.1') for section highlighting")


class ChatResponse(BaseModel):
    """
    Response from the healthcare assistant.
    
    Attributes:
        conversation_id: The conversation this response belongs to.
        message: The assistant's response message.
        response_type: Classification of the response type.
        citations: Any guideline citations included.
        artifacts: Source artifacts showing guideline text used (for traceability).
        follow_up_questions: Suggested follow-up questions.
        processing_time_ms: Time taken to generate the response.
    """
    conversation_id: UUID = Field(..., description="Conversation ID")
    message: str = Field(..., description="Assistant response")
    response_type: ResponseType = Field(..., description="Type of response")
    citations: list[Citation] = Field(default_factory=list, description="Cited references")
    artifacts: list[Artifact] = Field(
        default_factory=list,
        description="Source artifacts showing guideline text used (custom routes only)"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Suggested follow-up questions"
    )
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    query_type: str | None = Field(default=None, description="Query classification: 'general' or 'clinical'")


class HealthStatus(str, Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """
    Health check response for monitoring.
    
    Attributes:
        status: Overall system health status.
        version: Application version.
        environment: Deployment environment.
        checks: Individual component health checks.
    """
    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Deployment environment")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    checks: dict[str, bool] = Field(default_factory=dict, description="Component health checks")


class ErrorResponse(BaseModel):
    """
    Standardized error response.
    
    Attributes:
        error: Error type/code.
        message: Human-readable error message.
        details: Additional error details.
        request_id: Request ID for tracing.
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict | None = Field(default=None, description="Additional details")
    request_id: str | None = Field(default=None, description="Request ID for tracing")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class PathwayRouteInfo(BaseModel):
    """
    Information about a pathway route.
    
    Attributes:
        route_type: The route type identifier.
        name: Human-readable name.
        description: Brief description of what this route does.
        welcome_message: Initial message shown to users.
        example_prompts: Example questions for this route.
    """
    route_type: PathwayRouteType = Field(..., description="Route type identifier")
    name: str = Field(..., description="Route name")
    description: str = Field(..., description="Route description")
    welcome_message: str = Field(..., description="Welcome message")
    example_prompts: list[str] = Field(default_factory=list, description="Example prompts")


class PathwayRoutesResponse(BaseModel):
    """Response containing all available pathway routes."""
    routes: list[PathwayRouteInfo] = Field(..., description="Available routes")
    current_route: PathwayRouteType | None = Field(
        default=None,
        description="Currently selected route"
    )
