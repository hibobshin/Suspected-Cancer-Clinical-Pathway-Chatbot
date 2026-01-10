"""
RAG chat service for guideline-based responses.

Handles RAG routes (cancer_recognition, symptom_triage, referral_guidance)
using NICE NG12 guideline retrieval with classic RAG pipeline and traceable artifacts.
"""

import time
from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

from openai import AsyncOpenAI, OpenAIError

from config.config import Settings, get_settings
from config.logging_config import get_logger
from models.models import (
    Artifact,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationContext,
    MessageRole,
    PathwayRouteType,
    ResponseType,
)
from services.guideline_service import get_guideline_service
from services.pathway_routes import get_route_system_prompt, PathwayRouteType as RouteType

logger = get_logger(__name__)


class RagChatService:
    """
    Service for processing chat messages using RAG (Retrieval-Augmented Generation).
    
    Uses NICE NG12 guideline file with classic RAG pipeline:
    - Bag-of-words vectorization
    - Cosine similarity retrieval
    - Reranking
    - Provides traceable artifacts
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
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
            )
        return self._client
    
    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """
        Process a user message using custom guideline retrieval.
        
        Args:
            request: The chat request containing the user message.
            
        Returns:
            ChatResponse with the assistant's response and artifacts.
        """
        start_time = time.perf_counter()
        conversation_id = request.conversation_id or uuid4()
        
        logger.info(
            "Processing RAG chat message",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
            route_type=request.route_type.value if request.route_type else None,
        )
        
        try:
            # Retrieve guideline artifacts
            guideline_artifacts: list[Artifact] = []
            guideline_context = ""
            
            try:
                guideline_service = get_guideline_service()
                artifacts_data = guideline_service.search(
                    request.message,
                    max_chunks=3,
                    chunk_size=500,
                )
                guideline_artifacts = [
                    Artifact(**artifact) for artifact in artifacts_data
                ]
                guideline_context = guideline_service.format_artifacts_for_llm(artifacts_data)
                logger.info(
                    "Guideline artifacts retrieved",
                    count=len(guideline_artifacts),
                    context_length=len(guideline_context),
                )
            except Exception as e:
                logger.warning("Guideline retrieval failed", error=str(e))
            
            # Build conversation history with guideline context
            messages = self._build_messages(request, guideline_context=guideline_context)
            
            # Check if API key is configured
            if not self.settings.deepseek_api_key:
                return self._mock_response(request, conversation_id, start_time, guideline_artifacts)
            
            # Call LLM (DeepSeek)
            response = await self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
            )
            
            assistant_message = response.choices[0].message.content or ""
            
            # Classify response type
            response_type = self._classify_response(assistant_message)
            
            # Extract citations
            citations = self._extract_citations(assistant_message)
            
            # Generate follow-up suggestions
            follow_ups = self._generate_follow_ups(response_type, assistant_message)
            
            processing_time = int((time.perf_counter() - start_time) * 1000)
            
            logger.info(
                "RAG chat response generated",
                conversation_id=str(conversation_id),
                response_type=response_type.value,
                citations_count=len(citations),
                artifacts_count=len(guideline_artifacts),
                processing_time_ms=processing_time,
            )
            
            return ChatResponse(
                conversation_id=conversation_id,
                message=assistant_message,
                response_type=response_type,
                citations=citations,
                artifacts=guideline_artifacts,
                follow_up_questions=follow_ups,
                processing_time_ms=processing_time,
            )
            
        except OpenAIError as e:
            processing_time = int((time.perf_counter() - start_time) * 1000)
            logger.error(
                "LLM API error in custom chat",
                error=str(e),
                conversation_id=str(conversation_id),
            )
            return ChatResponse(
                conversation_id=conversation_id,
                message=f"I apologize, but I encountered an error: {str(e)}",
                response_type=ResponseType.ERROR,
                citations=[],
                artifacts=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
        except Exception as e:
            processing_time = int((time.perf_counter() - start_time) * 1000)
            logger.exception(
                "Unexpected error in custom chat",
                error=str(e),
                conversation_id=str(conversation_id),
            )
            return ChatResponse(
                conversation_id=conversation_id,
                message="I apologize, but an unexpected error occurred. Please try again.",
                response_type=ResponseType.ERROR,
                citations=[],
                artifacts=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
    
    async def process_message_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message and stream the response using custom guideline retrieval.
        
        Yields Server-Sent Events (SSE) formatted strings.
        
        Args:
            request: The chat request containing the user message.
            
        Yields:
            SSE formatted strings with response chunks.
        """
        start_time = time.perf_counter()
        conversation_id = request.conversation_id or uuid4()
        full_response = ""
        
        logger.info(
            "Processing streaming custom chat message",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
            route_type=request.route_type.value if request.route_type else None,
        )
        
        # Send initial event with conversation ID
        import json
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': str(conversation_id)})}\n\n"
        
        try:
            # Retrieve guideline artifacts
            guideline_artifacts: list[Artifact] = []
            guideline_context = ""
            
            try:
                guideline_service = get_guideline_service()
                artifacts_data = guideline_service.search(
                    request.message,
                    max_chunks=3,
                    chunk_size=500,
                )
                guideline_artifacts = [
                    Artifact(**artifact) for artifact in artifacts_data
                ]
                guideline_context = guideline_service.format_artifacts_for_llm(artifacts_data)
                logger.info(
                    "Guideline artifacts retrieved for stream",
                    count=len(guideline_artifacts),
                    context_length=len(guideline_context),
                )
            except Exception as e:
                logger.warning("Guideline retrieval failed in stream", error=str(e))
            
            # Build conversation history
            messages = self._build_messages(request, guideline_context=guideline_context)
            
            # Check if API key is configured
            if not self.settings.deepseek_api_key:
                # Mock streaming response
                mock_text = "I'm running in demo mode. Please configure your DeepSeek API key for full functionality."
                for char in mock_text:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': char})}\n\n"
                    full_response += char
                
                yield f"data: {json.dumps({'type': 'done', 'response_type': 'answer', 'citations': [], 'artifacts': []})}\n\n"
                return
            
            # Call LLM with streaming
            stream = await self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
            
            # After streaming completes, classify and extract citations
            response_type = self._classify_response(full_response)
            citations = self._extract_citations(full_response)
            
            processing_time = int((time.perf_counter() - start_time) * 1000)
            
            logger.info(
                "Streaming custom response completed",
                conversation_id=str(conversation_id),
                response_type=response_type.value,
                citations_count=len(citations),
                artifacts_count=len(guideline_artifacts),
                processing_time_ms=processing_time,
            )
            
            # Send completion event with metadata
            yield f"data: {json.dumps({'type': 'done', 'response_type': response_type.value, 'citations': [c.model_dump() for c in citations], 'artifacts': [a.model_dump() for a in guideline_artifacts], 'processing_time_ms': processing_time})}\n\n"
            
        except OpenAIError as e:
            logger.error(
                "LLM API error during custom streaming",
                error=str(e),
                conversation_id=str(conversation_id),
            )
            yield f"data: {json.dumps({'type': 'error', 'message': 'I apologize, but I experienced a technical issue. Please try again.'})}\n\n"
            
        except Exception as e:
            logger.exception(
                "Unexpected error during custom streaming",
                error=str(e),
                conversation_id=str(conversation_id),
            )
            yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred. Please try again.'})}\n\n"
    
    def _build_messages(
        self, request: ChatRequest, guideline_context: str = ""
    ) -> list[dict]:
        """
        Build the message list for the LLM API call.
        
        Uses route-specific system prompt and injects guideline context.
        
        Args:
            request: The chat request.
            guideline_context: Retrieved guideline context.
            
        Returns:
            List of message dicts for the API.
        """
        # Get route-specific system prompt
        try:
            route_type = RouteType(request.route_type.value)
            system_prompt = get_route_system_prompt(route_type)
        except (ValueError, AttributeError):
            # Fallback to default prompt
            from pathway_routes import CANCER_RECOGNITION_PROMPT
            system_prompt = CANCER_RECOGNITION_PROMPT
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add guideline context if available
        if guideline_context:
            messages.append({
                "role": "system",
                "content": f"## Retrieved Context from NICE NG12 Guideline\n\n{guideline_context}"
            })
        
        # Add conversation history if provided
        if request.context and request.context.messages:
            for msg in request.context.messages[-10:]:  # Limit to last 10 messages
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        # Add current message
        messages.append({"role": "user", "content": request.message})
        
        return messages
    
    def _classify_response(self, response: str) -> ResponseType:
        """Classify the response type based on content."""
        response_lower = response.lower()
        
        refusal_patterns = [
            "outside the scope",
            "cannot provide",
            "i cannot",
            "not covered",
            "does not cover",
            "out of scope",
        ]
        if any(pattern in response_lower for pattern in refusal_patterns):
            return ResponseType.REFUSAL
        
        clarification_patterns = [
            "before i can",
            "i need to clarify",
            "could you provide",
            "can you confirm",
            "what is the patient's age",
        ]
        if any(pattern in response_lower for pattern in clarification_patterns):
            return ResponseType.CLARIFICATION
        
        return ResponseType.ANSWER
    
    def _extract_citations(self, response: str) -> list[Citation]:
        """Extract citations from the response (NG12 or QS124)."""
        import re
        
        citations = []
        
        # Pattern: [NG12 1.3.1] or [NG12 1.3.1: Section Name]
        ng12_pattern = r'\[NG12\s+([\d.]+)(?::\s*([^\]]+))?\]'
        ng12_matches = re.findall(ng12_pattern, response)
        
        for section_ref, section_name in ng12_matches:
            citations.append(Citation(
                statement_id=f"NG12 {section_ref}",
                section=section_name.strip() if section_name else f"Section {section_ref}",
                text=None,
            ))
        
        # Pattern: [QS124-S1: Section Name]
        qs124_pattern = r'\[QS124-(S\d+):\s*([^\]]+)\]'
        qs124_matches = re.findall(qs124_pattern, response)
        
        for statement_num, section in qs124_matches:
            citations.append(Citation(
                statement_id=f"QS124-{statement_num}",
                section=section.strip(),
                text=None,
            ))
        
        # Deduplicate
        seen = set()
        unique_citations = []
        for c in citations:
            key = (c.statement_id, c.section)
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)
        
        return unique_citations
    
    def _generate_follow_ups(self, response_type: ResponseType, response: str) -> list[str]:
        """Generate follow-up question suggestions."""
        if response_type == ResponseType.CLARIFICATION:
            return [
                "What additional information do you need?",
                "Can you provide more details?",
            ]
        elif response_type == ResponseType.REFUSAL:
            return [
                "What types of questions can you help with?",
                "What is within your scope?",
            ]
        return []
    
    def _mock_response(
        self, 
        request: ChatRequest, 
        conversation_id: UUID, 
        start_time: float,
        artifacts: list[Artifact] = None
    ) -> ChatResponse:
        """Generate a mock response for demo mode."""
        processing_time = int((time.perf_counter() - start_time) * 1000)
        return ChatResponse(
            conversation_id=conversation_id,
            message="I'm running in demo mode. Please configure your DeepSeek API key for full functionality.",
            response_type=ResponseType.ANSWER,
            citations=[],
            artifacts=artifacts or [],
            follow_up_questions=[],
            processing_time_ms=processing_time,
        )


# Singleton instance
_rag_chat_service: RagChatService | None = None


def get_rag_chat_service() -> RagChatService:
    """Get the RAG chat service singleton."""
    global _rag_chat_service
    if _rag_chat_service is None:
        _rag_chat_service = RagChatService()
    return _rag_chat_service
