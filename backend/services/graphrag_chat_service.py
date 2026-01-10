"""
GraphRAG chat service for vendor solution.

Handles GraphRAG route using ArangoDB's GraphRAG retriever endpoint directly.
Returns the retrieved context as the response without LLM processing.
"""

import json
import time
from collections.abc import AsyncGenerator
from uuid import uuid4

from config.config import Settings, get_settings
from config.logging_config import get_logger
from models.models import (
    Citation,
    ChatRequest,
    ChatResponse,
    ResponseType,
)
from services.graphrag_service import get_graphrag_service

logger = get_logger(__name__)


class GraphRAGChatService:
    """
    Service for processing chat messages using GraphRAG retrieval.
    
    Uses ArangoDB's GraphRAG retriever directly - returns formatted context
    without LLM processing.
    """
    
    def __init__(self, settings: Settings | None = None):
        """
        Initialize the GraphRAG chat service.
        
        Args:
            settings: Application settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
    
    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """
        Process a user message using GraphRAG retrieval.
        
        Args:
            request: The chat request containing the user message.
            
        Returns:
            ChatResponse with the assistant's response.
        """
        start_time = time.perf_counter()
        conversation_id = request.conversation_id or uuid4()
        
        logger.info(
            "Processing GraphRAG chat message",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
        )
        
        try:
            # Fetch GraphRAG context directly from ArangoDB
            try:
                graphrag_service = get_graphrag_service()
                graphrag_context, retrieval_result = await graphrag_service.query_with_context(
                    request.message, top_k=5
                )
                logger.info("GraphRAG context retrieved", context_length=len(graphrag_context))
                
                # Use the formatted context as the response
                assistant_message = graphrag_context if graphrag_context else "No relevant information found in the knowledge graph."
                
            except Exception as e:
                logger.warning("GraphRAG retrieval failed", error=str(e))
                assistant_message = f"I encountered an error retrieving information from the knowledge graph: {str(e)}"
            
            # Classify response type
            response_type = ResponseType.ANSWER if assistant_message and "error" not in assistant_message.lower() else ResponseType.ERROR
            
            # Extract citations (if any in the response)
            citations = self._extract_citations(assistant_message)
            
            # Generate follow-up suggestions
            follow_ups = self._generate_follow_ups(response_type, assistant_message)
            
            processing_time = int((time.perf_counter() - start_time) * 1000)
            
            logger.info(
                "GraphRAG chat response generated",
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
                artifacts=[],  # GraphRAG doesn't provide artifacts
                follow_up_questions=follow_ups,
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            processing_time = int((time.perf_counter() - start_time) * 1000)
            logger.exception(
                "Unexpected error in GraphRAG chat",
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
        Process a user message and stream the response using GraphRAG.
        
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
            "Processing streaming GraphRAG chat message",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
        )
        
        # Send initial event with conversation ID
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': str(conversation_id)})}\n\n"
        
        try:
            # Fetch GraphRAG context directly from ArangoDB
            try:
                graphrag_service = get_graphrag_service()
                graphrag_context, retrieval_result = await graphrag_service.query_with_context(
                    request.message, top_k=5
                )
                logger.info("GraphRAG context retrieved for stream", context_length=len(graphrag_context))
                
                # Stream the formatted context as the response
                full_response = graphrag_context if graphrag_context else "No relevant information found in the knowledge graph."
                
            except Exception as e:
                logger.warning("GraphRAG retrieval failed in stream", error=str(e))
                full_response = f"I encountered an error retrieving information from the knowledge graph: {str(e)}"
            
            # Stream the response character by character for smooth UX
            for char in full_response:
                yield f"data: {json.dumps({'type': 'chunk', 'content': char})}\n\n"
            
            # After streaming completes, classify and extract citations
            response_type = ResponseType.ANSWER if full_response and "error" not in full_response.lower() else ResponseType.ERROR
            citations = self._extract_citations(full_response)
            
            processing_time = int((time.perf_counter() - start_time) * 1000)
            
            logger.info(
                "Streaming GraphRAG response completed",
                conversation_id=str(conversation_id),
                response_type=response_type.value,
                citations_count=len(citations),
                processing_time_ms=processing_time,
            )
            
            # Send completion event with metadata
            yield f"data: {json.dumps({'type': 'done', 'response_type': response_type.value, 'citations': [c.model_dump() for c in citations], 'artifacts': [], 'processing_time_ms': processing_time})}\n\n"
            
        except Exception as e:
            logger.exception(
                "Unexpected error during GraphRAG streaming",
                error=str(e),
                conversation_id=str(conversation_id),
            )
            yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred. Please try again.'})}\n\n"
    
    def _classify_response(self, response: str) -> ResponseType:
        """Classify the response type based on content."""
        response_lower = response.lower()
        
        refusal_patterns = [
            "outside the scope",
            "cannot provide",
            "i cannot",
            "not covered",
        ]
        if any(pattern in response_lower for pattern in refusal_patterns):
            return ResponseType.REFUSAL
        
        clarification_patterns = [
            "before i can",
            "i need to clarify",
            "could you provide",
        ]
        if any(pattern in response_lower for pattern in clarification_patterns):
            return ResponseType.CLARIFICATION
        
        return ResponseType.ANSWER
    
    def _extract_citations(self, response: str) -> list[Citation]:
        """Extract citations from the response."""
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
    


# Singleton instance
_graphrag_chat_service: GraphRAGChatService | None = None


def get_graphrag_chat_service() -> GraphRAGChatService:
    """Get the GraphRAG chat service singleton."""
    global _graphrag_chat_service
    if _graphrag_chat_service is None:
        _graphrag_chat_service = GraphRAGChatService()
    return _graphrag_chat_service
