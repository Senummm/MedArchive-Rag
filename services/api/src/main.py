"""
MedArchive RAG API Service

FastAPI application for clinical decision support queries.
Provides sub-second retrieval with verifiable citations.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from shared.models import HealthResponse, QueryRequest, QueryResponse, Citation, RetrievalResult
from shared.utils import get_settings, setup_logging

from .retrieval import Retriever, Reranker
from .llm import LLMService
from .citations import CitationExtractor

# Initialize configuration and logging
settings = get_settings()
logger = setup_logging(
    service_name="api",
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_output=settings.log_output,
)

# Global service instances (initialized in lifespan)
retriever: Retriever = None
reranker: Reranker = None
llm_service: LLMService = None
citation_extractor: CitationExtractor = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager for startup and shutdown events.

    Handles initialization of:
    - Vector database connections
    - Embedding models
    - LLM clients
    """
    global retriever, reranker, llm_service, citation_extractor
    
    logger.info(
        "Starting MedArchive API",
        extra={
            "version": settings.app_version,
            "environment": settings.environment,
            "qdrant_url": settings.qdrant_url,
        },
    )

    try:
        # Initialize retrieval service
        logger.info("Initializing retriever...")
        retriever = Retriever(
            qdrant_url=settings.qdrant_url,
            collection_name="medical_documents",
        )

        # Initialize reranker
        logger.info("Initializing reranker...")
        reranker = Reranker()

        # Initialize LLM service
        logger.info("Initializing LLM service...")
        llm_service = LLMService(
            api_key=settings.groq_api_key,
            model=settings.groq_model_name,
        )

        # Initialize citation extractor
        logger.info("Initializing citation extractor...")
        citation_extractor = CitationExtractor()

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield  # Application runs here

    # Cleanup on shutdown
    logger.info("Shutting down MedArchive API")
    # Services cleanup (models unload automatically)


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
        "qdrant": False,
        "groq": False,
        "retriever": False,
        "reranker": False,
    }

    # Check Qdrant
    try:
        if retriever:
            stats = retriever.get_collection_stats()
            dependencies["qdrant"] = bool(stats)
            dependencies["retriever"] = True
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")

    # Check LLM service
    dependencies["groq"] = llm_service is not None

    # Check reranker
    dependencies["reranker"] = reranker is not None

    return HealthResponse(
        status="healthy" if all(dependencies.values()) else "degraded",
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
# Query Endpoints
# =============================================================================
@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the MedArchive RAG system.

    Performs:
    1. Semantic search with embeddings
    2. Optional reranking with cross-encoder
    3. LLM answer generation with citations
    
    Returns answer with verifiable source citations.
    """
    start_time = time.time()
    
    logger.info(f"Processing query: '{request.query}'")

    try:
        # Step 1: Semantic retrieval
        search_results = retriever.search(
            query=request.query,
            top_k=request.top_k * 2,  # Retrieve more for reranking
            score_threshold=getattr(request, 'score_threshold', 0.5),
            filters=request.filters,
        )

        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
  Import note: Routes are now inline in main.py for Phase 3
# Can be refactored to separate route modules in future phases
# =============================================================================
        if request.enable_reranking and reranker:
            logger.info(f"Reranking {len(search_results)} results")
            search_results = reranker.rerank(
                query=request.query,
                results=search_results,
                top_k=request.top_k,
            )
        else:
            # Just take top_k
            search_results = search_results[:request.top_k]

        # Step 3: Generate answer
        logger.info(f"Generating answer with {len(search_results)} context chunks")
        answer = await llm_service.generate_answer(
            query=request.query,
            context_chunks=search_results,
            stream=False,
        )

        # Step 4: Extract citations
        citations = citation_extractor.extract_citations(
            answer=answer,
            search_results=search_results,
        )

        # Step 5: Build response
        latency_ms = (time.time() - start_time) * 1000

        # Convert search_results to RetrievalResult
        retrieval_results = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text,
                score=r.score,
                metadata={
                    "source_file": r.source_file,
                    "page_numbers": r.page_numbers,
                    "section_path": r.section_path,
                    "chunk_index": r.chunk_index,
                },
            )
            for r in search_results
        ]

        response = QueryResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieval_results,
            latency_ms=latency_ms,
            model_used=settings.groq_model_name,
        )

        logger.info(f"Query completed in {latency_ms:.1f}ms")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


@app.post("/api/v1/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Query with streaming response.

    Same as /query but streams the LLM answer as it's generated.
    """
    start_time = time.time()
    
    logger.info(f"Processing streaming query: '{request.query}'")

    try:
        # Retrieval and reranking (same as regular query)
        search_results = retriever.search(
            query=request.query,
            top_k=request.top_k * 2,
            score_threshold=getattr(request, 'score_threshold', 0.5),
            filters=request.filters,
        )

        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for your query.",
            )

        if request.enable_reranking and reranker:
            search_results = reranker.rerank(
                query=request.query,
                results=search_results,
                top_k=request.top_k,
            )
        else:
            search_results = search_results[:request.top_k]

        # Stream answer
        async def generate():
            full_answer = ""
            async for chunk in llm_service.generate_answer_stream(
                query=request.query,
                context_chunks=search_results,
            ):
                full_answer += chunk
                yield chunk

            # Send citations after answer completes
            citations = citation_extractor.extract_citations(
                answer=full_answer,
                search_results=search_results,
            )
            
            # Send citation data as JSON at the end
            import json
            yield "\n\n---CITATIONS---\n"
            yield json.dumps([c.dict() for c in citations])

        return StreamingResponse(generate(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Streaming query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming query failed: {str(e)}",
        )


@app.get("/api/v1/stats", tags=["Monitoring"])
async def get_stats():
    """
    Get system statistics.

    Returns collection info and service status.
    """
    try:
        stats = {
            "collection": retriever.get_collection_stats() if retriever else {},
            "services": {
                "retriever": retriever is not None,
                "reranker": reranker is not None,
                "llm": llm_service is not None,
                "citations": citation_extractor is not None,
            },
        }
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


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
