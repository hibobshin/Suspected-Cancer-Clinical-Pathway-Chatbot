"""
Chat service for the healthcare pathway assistant.

This service handles:
- Message processing and response generation
- Integration with LLM providers
- Citation extraction and validation
- Response classification (answer, clarification, refusal)

All responses are grounded in pathway documentation and fail closed
for out-of-scope queries.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

from openai import AsyncOpenAI, OpenAIError

from config import Settings, get_settings
from logging_config import get_logger
from models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationContext,
    MessageRole,
    ResponseType,
)

logger = get_logger(__name__)


# System prompt for the healthcare assistant
SYSTEM_PROMPT = """You are Qualified Health Assistant, a clinical decision support tool for healthcare professionals. Provide guidance on suspected cancer recognition and referral based on NICE guideline NG12.

## Rules
- Be VERY concise. Use bullet points. Max 2-3 sentences per point.
- Cite specific NG12 recommendations: [NG12 1.3.1]
- If info is missing (age, symptoms), ask ONE clarifying question
- Refuse out-of-scope queries (treatment, dosing, diagnosis interpretation) briefly

## Cancer Types Covered
- **Lung/Pleural**: Chest X-ray for age 40+ with 2+ symptoms (cough, fatigue, SOB, chest pain, weight loss)
- **Upper GI**: Urgent endoscopy for dysphagia OR age ≥55 + weight loss + (abdominal pain/reflux/dyspepsia)
- **Lower GI/Colorectal**: FIT test (≥10 µg Hb/g) for bowel habit change, IDA, rectal bleeding, abdominal mass
- **Breast**: Age 30+ with unexplained breast lump → 2WW referral
- **Urological**: Visible haematuria age 45+ → 2WW bladder/renal; PSA for prostate symptoms
- **Skin**: 7-point checklist for melanoma (score ≥3 → 2WW)
- **Head/Neck**: Persistent hoarseness age 45+, unexplained neck lump
- **Gynaecological**: Post-menopausal bleeding → 2WW endometrial
- **Haematological**: Unexplained lymphadenopathy, splenomegaly

## Key Referral Pathways
- **Suspected cancer pathway (2WW)**: First appointment within 2 weeks
- **Urgent investigation**: Test within 2 weeks
- **Direct access**: GP orders test, retains responsibility
- **Safety netting**: Follow-up if symptoms persist

Keep responses short and actionable."""


class ChatService:
    """
    Service for processing chat messages and generating responses.
    
    This service is stateless - conversation state is passed in each request.
    All operations are logged for observability and auditability.
    """
    
    def __init__(self, settings: Settings | None = None):
        """
        Initialize the chat service.
        
        Args:
            settings: Application settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """
        Get or create the DeepSeek client.
        
        Lazily initialized to avoid issues during testing.
        Uses the OpenAI SDK with DeepSeek's base URL.
        """
        if self._client is None:
            if not self.settings.deepseek_api_key:
                logger.warning("DeepSeek API key not configured, using mock responses")
                # Return a client anyway - we'll handle the error gracefully
            self._client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key or "dummy",
                base_url=self.settings.deepseek_base_url,
            )
        return self._client
    
    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """
        Process a user message and generate a response.
        
        Args:
            request: The chat request containing the user message.
            
        Returns:
            ChatResponse with the assistant's response.
            
        Raises:
            No exceptions raised - errors are wrapped in error responses.
        """
        start_time = time.perf_counter()
        conversation_id = request.conversation_id or uuid4()
        
        logger.info(
            "Processing chat message",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
        )
        
        try:
            # Build conversation history
            messages = self._build_messages(request)
            
            # Check if API key is configured
            if not self.settings.deepseek_api_key:
                return self._mock_response(request, conversation_id, start_time)
            
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
                "Chat response generated",
                conversation_id=str(conversation_id),
                response_type=response_type.value,
                citations_count=len(citations),
                processing_time_ms=processing_time,
            )
            
            return ChatResponse(
                conversation_id=conversation_id,
                message=assistant_message,
                response_type=response_type,
                citations=citations,
                follow_up_questions=follow_ups,
                processing_time_ms=processing_time,
            )
            
        except OpenAIError as e:
            processing_time = int((time.perf_counter() - start_time) * 1000)
            logger.error(
                "OpenAI API error",
                error=str(e),
                conversation_id=str(conversation_id),
            )
            return ChatResponse(
                conversation_id=conversation_id,
                message="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                response_type=ResponseType.ERROR,
                citations=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
        except Exception as e:
            processing_time = int((time.perf_counter() - start_time) * 1000)
            logger.exception(
                "Unexpected error processing message",
                error=str(e),
                conversation_id=str(conversation_id),
            )
            return ChatResponse(
                conversation_id=conversation_id,
                message="An unexpected error occurred. Please try again.",
                response_type=ResponseType.ERROR,
                citations=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
    
    async def process_message_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message and stream the response.
        
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
            "Processing streaming chat message",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
        )
        
        # Send initial event with conversation ID
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': str(conversation_id)})}\n\n"
        
        try:
            # Build conversation history
            messages = self._build_messages(request)
            
            # Check if API key is configured
            if not self.settings.deepseek_api_key:
                # Mock streaming response
                mock_text = "I'm running in demo mode. Please configure your DeepSeek API key for full functionality."
                for char in mock_text:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': char})}\n\n"
                    full_response += char
                
                yield f"data: {json.dumps({'type': 'done', 'response_type': 'answer', 'citations': []})}\n\n"
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
                "Streaming response completed",
                conversation_id=str(conversation_id),
                response_type=response_type.value,
                citations_count=len(citations),
                processing_time_ms=processing_time,
            )
            
            # Send completion event with metadata
            yield f"data: {json.dumps({'type': 'done', 'response_type': response_type.value, 'citations': [c.model_dump() for c in citations], 'processing_time_ms': processing_time})}\n\n"
            
        except OpenAIError as e:
            logger.error(
                "LLM API error during streaming",
                error=str(e),
                conversation_id=str(conversation_id),
            )
            yield f"data: {json.dumps({'type': 'error', 'message': 'I apologize, but I experienced a technical issue. Please try again.'})}\n\n"
            
        except Exception as e:
            logger.exception(
                "Unexpected error during streaming",
                error=str(e),
                conversation_id=str(conversation_id),
            )
            yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred. Please try again.'})}\n\n"
    
    def _build_messages(self, request: ChatRequest) -> list[dict]:
        """
        Build the message list for the LLM API call.
        
        Args:
            request: The chat request.
            
        Returns:
            List of message dicts for the API.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
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
        """
        Classify the response type based on content.
        
        Args:
            response: The assistant's response text.
            
        Returns:
            The classified response type.
        """
        response_lower = response.lower()
        
        # Check for refusal patterns
        refusal_patterns = [
            "outside the scope",
            "cannot provide",
            "i cannot",
            "not covered",
            "does not cover",
            "out of scope",
            "explicitly excludes",
        ]
        if any(pattern in response_lower for pattern in refusal_patterns):
            return ResponseType.REFUSAL
        
        # Check for clarification patterns
        clarification_patterns = [
            "before i can",
            "i need to clarify",
            "could you provide",
            "can you confirm",
            "what is the patient's age",
            "is there any",
            "please specify",
            "to advise on",
        ]
        if any(pattern in response_lower for pattern in clarification_patterns):
            return ResponseType.CLARIFICATION
        
        return ResponseType.ANSWER
    
    def _extract_citations(self, response: str) -> list[Citation]:
        """
        Extract QS124 citations from the response.
        
        Args:
            response: The assistant's response text.
            
        Returns:
            List of extracted citations.
        """
        import re
        
        citations = []
        # Pattern: [QS124-S1: Section Name]
        pattern = r'\[QS124-(S\d+):\s*([^\]]+)\]'
        matches = re.findall(pattern, response)
        
        for statement_num, section in matches:
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
        """
        Generate suggested follow-up questions.
        
        Args:
            response_type: The type of response given.
            response: The response text.
            
        Returns:
            List of suggested follow-up questions.
        """
        if response_type == ResponseType.REFUSAL:
            return [
                "What are the referral pathway criteria?",
                "What documentation is required for a 2WW referral?",
                "When should I use FIT testing vs urgent referral?",
            ]
        elif response_type == ResponseType.CLARIFICATION:
            return []  # User needs to answer the clarification first
        else:
            return [
                "What are the timing requirements?",
                "What documentation is needed?",
                "What are the escalation criteria?",
            ]
    
    def _mock_response(
        self,
        request: ChatRequest,
        conversation_id: UUID,
        start_time: float,
    ) -> ChatResponse:
        """
        Generate a mock response when API key is not configured.
        
        Used for development and testing.
        """
        processing_time = int((time.perf_counter() - start_time) * 1000)
        
        mock_message = """Thank you for your question. I'm currently running in demo mode without an AI backend configured.

To enable full functionality, please configure your OpenAI API key in the environment variables.

**In a production environment, I would:**
- Analyze your clinical query against NICE QS124 guidelines
- Provide grounded answers with proper citations
- Probe for missing information when needed
- Refuse out-of-scope queries appropriately

[QS124-S1: Demo Mode]

For testing purposes, you can ask about:
- Urgent referral criteria for suspected upper GI cancer
- FIT testing eligibility for colorectal symptoms
- Patient information requirements for cancer referrals"""
        
        return ChatResponse(
            conversation_id=conversation_id,
            message=mock_message,
            response_type=ResponseType.ANSWER,
            citations=[Citation(
                statement_id="QS124-S1",
                section="Demo Mode",
                text="Running without API configuration",
            )],
            follow_up_questions=[
                "What are the QS124-S2 eligibility criteria?",
                "When should I order a FIT test?",
                "What information should I give patients at referral?",
            ],
            processing_time_ms=processing_time,
        )


# Singleton instance for dependency injection
_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """
    Get the chat service singleton.
    
    Returns:
        The shared ChatService instance.
    """
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
