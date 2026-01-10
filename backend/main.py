"""
Qualified Health API - Healthcare Pathway Assistant

A clinical decision support API for suspected cancer recognition and referral
based on NICE Quality Standard QS124.

This API provides:
- Chat interface for pathway queries
- Grounded responses with citations
- Fail-closed behavior for out-of-scope queries
- Comprehensive logging and observability
"""

import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from services.rag_chat_service import get_rag_chat_service
from services.graphrag_chat_service import get_graphrag_chat_service
from services.custom_chat_service import get_custom_chat_service
from services.custom_guideline_service import get_custom_guideline_service
from config.config import Settings, get_settings
from config.logging_config import configure_logging, get_logger, log_request_context
from models.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    PathwayRouteInfo,
    PathwayRoutesResponse,
    PathwayRouteType,
)
from services.pathway_routes import get_all_routes, get_route_by_type
from pathlib import Path
import re

# Configure logging before anything else
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events with proper logging.
    """
    settings = get_settings()
    
    # Startup
    logger.info(
        "Application starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        config=settings.get_safe_config_dict(),
    )
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Optional settings override for testing.
        
    Returns:
        Configured FastAPI application.
    """
    settings = settings or get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description=__doc__,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests with timing and context."""
        request_id = str(uuid4())
        start_time = time.perf_counter()
        
        # Bind request context for all logs in this request
        log_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        
        # Add request ID to response headers
        response = await call_next(request)
        
        processing_time = int((time.perf_counter() - start_time) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time-Ms"] = str(processing_time)
        
        logger.info(
            "Request completed",
            status_code=response.status_code,
            processing_time_ms=processing_time,
        )
        
        return response
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with structured response."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"HTTP_{exc.status_code}",
                message=exc.detail,
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(mode="json"),
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.exception("Unhandled exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="INTERNAL_ERROR",
                message="An unexpected error occurred",
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(mode="json"),
        )
    
    # Register routes
    register_routes(app)
    
    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        settings = get_settings()
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "operational",
            "docs": "/docs" if settings.debug else "disabled",
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
        """
        Health check endpoint for monitoring.
        
        Returns system health status and component checks.
        """
        checks = {
            "api": True,
            "deepseek_configured": bool(settings.deepseek_api_key),
        }
        
        # Determine overall status
        if all(checks.values()):
            status = HealthStatus.HEALTHY
        elif checks["api"]:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        return HealthResponse(
            status=status,
            version=settings.app_version,
            environment=settings.environment,
            checks=checks,
        )
    
    @app.get("/api/v1/routes", response_model=PathwayRoutesResponse, tags=["Routes"])
    async def get_routes() -> PathwayRoutesResponse:
        """
        Get all available pathway routes.
        
        Returns the list of chatbot modes the user can toggle between:
        - Cancer Recognition: Identify symptoms by cancer site
        - Symptom Triage: Evaluate symptoms and determine investigations
        - Referral Guidance: Determine correct referral pathway
        """
        routes = get_all_routes()
        route_infos = [
            PathwayRouteInfo(
                route_type=PathwayRouteType(route.route_type.value),
                name=route.name,
                description=route.description,
                welcome_message=route.welcome_message,
                example_prompts=route.example_prompts,
            )
            for route in routes
        ]
        return PathwayRoutesResponse(routes=route_infos)
    
    @app.get("/api/v1/routes/{route_type}", response_model=PathwayRouteInfo, tags=["Routes"])
    async def get_route(route_type: PathwayRouteType) -> PathwayRouteInfo:
        """
        Get details for a specific pathway route.
        
        Args:
            route_type: The route type to retrieve.
        """
        from services.pathway_routes import PathwayRouteType as RouteType
        
        route = get_route_by_type(RouteType(route_type.value))
        if not route:
            raise HTTPException(status_code=404, detail="Route not found")
        
        return PathwayRouteInfo(
            route_type=route_type,
            name=route.name,
            description=route.description,
            welcome_message=route.welcome_message,
            example_prompts=route.example_prompts,
        )
    
    @app.post("/api/v1/chat/rag", response_model=ChatResponse, tags=["Chat"])
    async def chat_rag(
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Send a message to the RAG (Retrieval-Augmented Generation) healthcare pathway assistant.
        
        Uses NICE NG12 guideline retrieval with classic RAG pipeline and traceable artifacts.
        Supports RAG routes: cancer_recognition, symptom_triage, referral_guidance.
        
        **Features:**
        - Grounded in NICE NG12 guideline content
        - Classic RAG: Bag-of-words, cosine similarity, reranking
        - Returns traceable artifacts showing source text
        - Route-specific system prompts
        
        **Example queries:**
        - "A 58-year-old patient has weight loss and heartburn. What referral pathway?"
        - "Should I order a FIT test for abdominal pain?"
        - "What documentation is needed for a 2WW referral?"
        """
        logger.info("RAG chat request received", message_preview=request.message[:100])
        
        rag_service = get_rag_chat_service()
        response = await rag_service.process_message(request)
        
        return response
    
    @app.post("/api/v1/chat/rag/stream", tags=["Chat"])
    async def chat_rag_stream(
        request: ChatRequest,
    ) -> StreamingResponse:
        """
        Stream a response from the RAG healthcare pathway assistant.
        
        Uses NICE NG12 guideline retrieval with classic RAG pipeline and traceable artifacts.
        """
        logger.info("RAG chat stream request received", message_preview=request.message[:100])
        
        rag_service = get_rag_chat_service()
        return StreamingResponse(
            rag_service.process_message_stream(request),
            media_type="text/event-stream",
        )
    
    @app.post("/api/v1/chat/custom", response_model=ChatResponse, tags=["Chat"])
    async def chat_custom(
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Send a message to the custom healthcare pathway assistant.
        
        Custom implementation with flexible configuration.
        """
        logger.info("Custom chat request received", message_preview=request.message[:100])
        
        custom_service = get_custom_chat_service()
        response = await custom_service.process_message(request)
        
        return response
    
    @app.post("/api/v1/chat/custom/stream", tags=["Chat"])
    async def chat_custom_stream(
        request: ChatRequest,
    ) -> StreamingResponse:
        """
        Stream a response from the custom healthcare pathway assistant.
        """
        logger.info("Custom chat stream request received", message_preview=request.message[:100])
        
        custom_service = get_custom_chat_service()
        return StreamingResponse(
            custom_service.process_message_stream(request),
            media_type="text/event-stream",
        )
    
    @app.post("/api/v1/chat/graphrag", response_model=ChatResponse, tags=["Chat"])
    async def chat_graphrag(
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Send a message to the GraphRAG-powered healthcare assistant.
        
        Uses ArangoDB GraphRAG retriever for context-aware responses from knowledge graph.
        
        **Features:**
        - Knowledge graph retrieval
        - Entity and relationship queries
        - Community summaries
        
        **Example queries:**
        - "What entities relate to colorectal cancer?"
        - "Show connections between FIT testing and referral"
        - "What does the graph say about 2WW pathways?"
        """
        logger.info("GraphRAG chat request received", message_preview=request.message[:100])
        
        graphrag_service = get_graphrag_chat_service()
        response = await graphrag_service.process_message(request)
        
        return response
    
    @app.post("/api/v1/chat/graphrag/stream", tags=["Chat"])
    async def chat_graphrag_stream(
        request: ChatRequest,
    ) -> StreamingResponse:
        """
        Stream a response from the GraphRAG-powered healthcare assistant.
        
        Uses ArangoDB GraphRAG retriever for context-aware responses.
        """
        logger.info("GraphRAG chat stream request received", message_preview=request.message[:100])
        
        graphrag_service = get_graphrag_chat_service()
        return StreamingResponse(
            graphrag_service.process_message_stream(request),
            media_type="text/event-stream",
        )
    
    # Legacy endpoint for backwards compatibility (routes to RAG)
    @app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Legacy endpoint - routes to RAG chat service.
        
        Use /api/v1/chat/rag, /api/v1/chat/custom, or /api/v1/chat/graphrag instead.
        """
        logger.info("Legacy chat request received, routing to RAG", message_preview=request.message[:100])
        
        rag_service = get_rag_chat_service()
        response = await rag_service.process_message(request)
        
        return response
    
    @app.post("/api/v1/chat/stream", tags=["Chat"])
    async def chat_stream(
        request: ChatRequest,
    ) -> StreamingResponse:
        """
        Legacy streaming endpoint - routes to custom chat service.
        
        Use /api/v1/chat/custom/stream or /api/v1/chat/graphrag/stream instead.
        
        Returns a Server-Sent Events (SSE) stream with the following event types:
        - `start`: Initial event with conversation_id
        - `chunk`: Text chunk from the assistant
        - `done`: Final event with response_type and citations
        - `error`: Error event if something goes wrong
        """
        logger.info("Legacy streaming chat request received, routing to RAG", message_preview=request.message[:100])
        
        rag_service = get_rag_chat_service()
        return StreamingResponse(
            rag_service.process_message_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    @app.get("/api/v1/document/section", tags=["Document"])
    async def get_document_section(
        rule_id: str | None = None,
        section_path: str | None = None,
    ):
        """
        Get a document section with highlighting support.
        
        Returns the full document or a specific section, with information
        about where to highlight based on rule_id or section_path.
        
        Args:
            rule_id: Rule ID to highlight (e.g., "1.3.1")
            section_path: Section path to find (e.g., "NG12 > Upper gastrointestinal tract cancers")
        """
        try:
            # Load the document
            doc_path = Path(__file__).parent.parent / "data" / "final.md"
            if not doc_path.exists():
                raise HTTPException(status_code=404, detail="Document not found")
            
            with open(doc_path, "r", encoding="utf-8") as f:
                document_text = f.read()
            
            # If no rule_id or section_path, return full document
            if not rule_id and not section_path:
                return {
                    "document": document_text,
                    "highlight_rule_id": None,
                    "highlight_section": None,
                    "highlight_start": None,
                    "highlight_end": None,
                }
            
            # Find the section to highlight
            highlight_start = None
            highlight_end = None
            highlight_section = None
            
            if rule_id:
                # Find rule ID in document (e.g., "1.3.1" or "recommendation 1.3.1")
                # Pattern: Look for numbered headings or explicit rule references
                patterns = [
                    rf"^#+\s+{re.escape(rule_id)}\b",  # Heading with rule ID
                    rf"^#+\s+.*{re.escape(rule_id)}\b",  # Heading containing rule ID
                    rf"\b{re.escape(rule_id)}\b",  # Any mention of rule ID
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, document_text, re.MULTILINE)
                    if match:
                        highlight_start = match.start()
                        # Find the end of this section (next heading of same or higher level)
                        remaining = document_text[highlight_start:]
                        # Find next heading
                        next_heading = re.search(r"^#+\s+", remaining[100:], re.MULTILINE)
                        if next_heading:
                            highlight_end = highlight_start + 100 + next_heading.start()
                        else:
                            highlight_end = highlight_start + min(2000, len(remaining))
                        highlight_section = rule_id
                        break
            
            elif section_path:
                # Find section by path (e.g., "NG12 > Upper gastrointestinal tract cancers")
                section_title = section_path.split(">")[-1].strip()
                pattern = rf"^#+\s+{re.escape(section_title)}"
                match = re.search(pattern, document_text, re.MULTILINE | re.IGNORECASE)
                if match:
                    highlight_start = match.start()
                    remaining = document_text[highlight_start:]
                    next_heading = re.search(r"^#+\s+", remaining[500:], re.MULTILINE)
                    if next_heading:
                        highlight_end = highlight_start + 500 + next_heading.start()
                    else:
                        highlight_end = highlight_start + min(5000, len(remaining))
                    highlight_section = section_title
            
            return {
                "document": document_text,
                "highlight_rule_id": rule_id,
                "highlight_section": highlight_section,
                "highlight_start": highlight_start,
                "highlight_end": highlight_end,
            }
            
        except Exception as e:
            logger.exception("Error fetching document section", error=str(e))
            raise HTTPException(status_code=500, detail=f"Error fetching document: {str(e)}")
    
    @app.get("/api/v1/conversations/{conversation_id}", tags=["Chat"])
    async def get_conversation(conversation_id: str):
        """
        Retrieve a conversation by ID.
        
        Note: In this stateless implementation, conversation history
        must be maintained client-side and passed with each request.
        """
        # Placeholder - in production, would retrieve from storage
        return {
            "conversation_id": conversation_id,
            "message": "Conversation retrieval requires persistent storage integration",
            "note": "Pass conversation context with each chat request for continuity",
        }


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
