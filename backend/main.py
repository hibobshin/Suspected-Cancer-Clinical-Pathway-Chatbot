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

from chat_service import ChatService, get_chat_service
from config import Settings, get_settings
from logging_config import configure_logging, get_logger, log_request_context
from models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
)

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
    
    @app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(
        request: ChatRequest,
        chat_service: ChatService = Depends(get_chat_service),
    ) -> ChatResponse:
        """
        Send a message to the healthcare pathway assistant.
        
        The assistant provides guidance on suspected cancer recognition and
        referral based on NICE QS124 guidelines.
        
        **Behaviors:**
        - **In-scope queries**: Grounded answers with citations
        - **Under-specified queries**: Probing questions for missing info
        - **Out-of-scope queries**: Polite refusal with alternatives
        
        **Example queries:**
        - "A 58-year-old patient has weight loss and heartburn. What referral pathway?"
        - "Should I order a FIT test for abdominal pain?"
        - "What documentation is needed for a 2WW referral?"
        """
        logger.info("Chat request received", message_preview=request.message[:100])
        
        response = await chat_service.process_message(request)
        
        return response
    
    @app.post("/api/v1/chat/stream", tags=["Chat"])
    async def chat_stream(
        request: ChatRequest,
        chat_service: ChatService = Depends(get_chat_service),
    ) -> StreamingResponse:
        """
        Send a message and receive a streaming response.
        
        Returns a Server-Sent Events (SSE) stream with the following event types:
        - `start`: Initial event with conversation_id
        - `chunk`: Text chunk from the assistant
        - `done`: Final event with response_type and citations
        - `error`: Error event if something goes wrong
        """
        logger.info("Streaming chat request received", message_preview=request.message[:100])
        
        return StreamingResponse(
            chat_service.process_message_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
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
