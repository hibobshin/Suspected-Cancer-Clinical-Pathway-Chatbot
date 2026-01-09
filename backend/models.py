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
        conversation_id: Optional ID to continue an existing conversation.
        context: Optional conversation context with history.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User message to the assistant"
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


class ChatResponse(BaseModel):
    """
    Response from the healthcare assistant.
    
    Attributes:
        conversation_id: The conversation this response belongs to.
        message: The assistant's response message.
        response_type: Classification of the response type.
        citations: Any guideline citations included.
        follow_up_questions: Suggested follow-up questions.
        processing_time_ms: Time taken to generate the response.
    """
    conversation_id: UUID = Field(..., description="Conversation ID")
    message: str = Field(..., description="Assistant response")
    response_type: ResponseType = Field(..., description="Type of response")
    citations: list[Citation] = Field(default_factory=list, description="Cited references")
    follow_up_questions: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Suggested follow-up questions"
    )
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


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
