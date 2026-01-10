"""
Custom chat service for flexible, configurable responses.

This service provides a customizable implementation that can be extended
with different retrieval strategies, LLM configurations, or business logic.
"""

import time
from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

from openai import AsyncOpenAI, OpenAIError

from config.config import Settings, get_settings
from config.logging_config import get_logger
from models.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationContext,
    MessageRole,
    PathwayRouteType,
    ResponseType,
)

logger = get_logger(__name__)


class CustomChatService:
    """
    Service for processing chat messages with custom implementation.
    
    This service can be configured and extended for specific use cases.
    Currently uses direct LLM responses without retrieval augmentation.
    """
    
    def __init__(self, settings: Settings | None = None):
        """
        Initialize the custom chat service.
        
        Args:
            settings: Application settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client (for DeepSeek API)."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        return self._client
    
    async def process_message(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Process a chat message and return a response.
        
        Args:
            request: Chat request with message and context.
            
        Returns:
            ChatResponse with the assistant's response.
        """
        start_time = time.perf_counter()
        conversation_id = request.conversation_id or uuid4()
        
        logger.info(
            "Processing custom chat message",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
            route_type=request.route_type.value if request.route_type else None,
        )
        
        try:
            # Build messages for LLM
            messages = self._build_messages(request)
            
            # Call LLM
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            
            assistant_message = response.choices[0].message.content or ""
            
            processing_time = int((time.perf_counter() - start_time) * 1000)
            
            return ChatResponse(
                conversation_id=conversation_id,
                message=assistant_message,
                response_type=ResponseType.ANSWER,
                citations=[],
                artifacts=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
            
        except OpenAIError as e:
            logger.error("OpenAI API error in custom chat", error=str(e))
            processing_time = int((time.perf_counter() - start_time) * 1000)
            return ChatResponse(
                conversation_id=conversation_id,
                message=f"I encountered an error processing your request: {str(e)}",
                response_type=ResponseType.ERROR,
                citations=[],
                artifacts=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
        except Exception as e:
            logger.exception("Unexpected error in custom chat", error=str(e))
            processing_time = int((time.perf_counter() - start_time) * 1000)
            return ChatResponse(
                conversation_id=conversation_id,
                message="I encountered an unexpected error. Please try again.",
                response_type=ResponseType.ERROR,
                citations=[],
                artifacts=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
    
    async def process_message_stream(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat message and stream the response.
        
        Args:
            request: Chat request with message and context.
            
        Yields:
            SSE-formatted event strings.
        """
        conversation_id = request.conversation_id or uuid4()
        
        logger.info(
            "Processing custom chat message stream",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
        )
        
        try:
            import json
            
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'conversation_id': str(conversation_id)})}\n\n"
            
            # Build messages
            messages = self._build_messages(request)
            
            # Stream from LLM
            stream = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True,
            )
            
            # Stream chunks
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
            
            # Send done event
            yield f"data: {json.dumps({'type': 'done', 'response_type': 'answer'})}\n\n"
            
        except Exception as e:
            logger.exception("Error in custom chat stream", error=str(e))
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    def _build_messages(self, request: ChatRequest) -> list[dict[str, str]]:
        """
        Build messages list for LLM from request.
        
        Args:
            request: Chat request.
            
        Returns:
            List of message dictionaries.
        """
        messages = []
        
        # System prompt
        system_prompt = (
            "You are a helpful healthcare assistant. "
            "Provide clear, concise, and accurate responses to healthcare-related questions. "
            "If you don't know something, say so rather than guessing."
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if available
        if request.context and request.context.messages:
            for msg in request.context.messages:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
        return messages


# Singleton instance
_custom_chat_service: CustomChatService | None = None


def get_custom_chat_service() -> CustomChatService:
    """Get the custom chat service singleton."""
    global _custom_chat_service
    if _custom_chat_service is None:
        _custom_chat_service = CustomChatService()
    return _custom_chat_service
