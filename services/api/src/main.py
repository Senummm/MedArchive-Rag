"""
MedArchive RAG API Service

FastAPI application for clinical decision support queries.
Provides sub-second retrieval with verifiable citations.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.models import HealthResponse
from shared.utils import get_settings, setup_logging

# Initialize configuration and logging
settings = get_settings()
logger = setup_logging(
    service_name="api",
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_output=settings.log_output,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager for startup and shutdown events.

    Handles initialization of:
    - Vector database connections
    - Embedding models
    - LLM clients
    """
    logger.info(
        "Starting MedArchive API",
        extra={
            "version": settings.app_version,
            "environment": settings.environment,
            "qdrant_url": settings.qdrant_url,
        },
    )

    # TODO: Initialize connections in Phase 2-4
    # - Qdrant client
    # - Embedding model (sentence-transformers)
    # - Groq client
    # - Reranker model

    yield  # Application runs here

    # Cleanup on shutdown
    logger.info("Shutting down MedArchive API")
    # TODO: Close connections gracefully


# Create FastAPI application
app = FastAPI(
    title="MedArchive RAG API",
    description="Clinical decision support system with zero-hallucination guarantees",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,  # Disable docs in production
    redoc_url="/redoc" if settings.is_development else None,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # type: ignore
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check Endpoint
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns service status and dependency health.
    """
    dependencies = {
        "qdrant": False,  # TODO: Check Qdrant connection in Phase 3
        "groq": False,  # TODO: Check Groq API in Phase 4
    }

    return HealthResponse(
        status="healthy",
        service="api",
        version=settings.app_version,
        dependencies=dependencies,
    )


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "service": "MedArchive RAG API",
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs" if settings.is_development else "disabled in production",
    }


# =============================================================================
# Error Handlers
# =============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Global exception handler to prevent leaking stack traces in production."""
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        extra={
            "path": str(request.url),
            "method": request.method,
        },
    )

    if settings.is_production:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please contact support."},
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "type": type(exc).__name__,
            },
        )


# =============================================================================
# TODO: Import and register route modules in Phase 4
# =============================================================================
# from services.api.src.routes import query
# app.include_router(query.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
