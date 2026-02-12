"""
MedArchive RAG API Service

FastAPI application for clinical decision support queries.
Provides sub-second retrieval with verifiable citations.
"""

import time
import uuid
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from shared.models import HealthResponse, QueryRequest, QueryResponse, Citation, RetrievalResult, ChatRequest, ChatResponse
from shared.utils import get_settings, setup_logging

from .retrieval import Retriever, Reranker
from .llm import LLMService
from .citations import CitationExtractor
from .observability import init_tracer, get_tracer
from .conversation import SessionManager

# =============================================================================
# Request/Response Models
# =============================================================================

class FeedbackRequest(BaseModel):
    """User feedback on query response."""
    trace_id: str = Field(..., description="Trace ID from query response")
    feedback: Literal["thumbs_up", "thumbs_down"] = Field(..., description="User feedback")
    comment: str | None = Field(None, description="Optional user comment")


class FeedbackResponse(BaseModel):
    """Feedback submission confirmation."""
    success: bool
    message: str


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
phoenix_tracer = None  # Phoenix observability tracer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager for startup and shutdown events.

    Handles initialization of:
    - Vector database connections
    - Embedding models
    - LLM clients
    - Observability tracing
    """
    global retriever, reranker, llm_service, citation_extractor, phoenix_tracer, session_manager

    logger.info(
        "Starting MedArchive API",
        extra={
            "version": settings.app_version,
            "environment": settings.environment,
            "qdrant_url": settings.qdrant_url,
        },
    )

    try:
        # Initialize session manager
        logger.info("Initializing session manager...")
        session_manager = SessionManager()

        # Initialize retrieval service
        logger.info("Initializing retriever...")
        retriever = Retriever(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            api_key=settings.qdrant_api_key,
        )

        # Initialize reranker
        logger.info("Initializing reranker...")
        reranker = Reranker()

        # Initialize LLM service
        logger.info("Initializing LLM service...")
        llm_service = LLMService(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )

        # Initialize citation extractor
        logger.info("Initializing citation extractor...")
        citation_extractor = CitationExtractor()

        # Initialize Phoenix tracer (optional)
        logger.info("Initializing Phoenix tracer...")
        try:
            phoenix_tracer = init_tracer(project_name="medarchive-rag")
            logger.info("Phoenix tracer initialized successfully")
        except Exception as e:
            logger.warning(f"Phoenix tracer not available: {e}")
            phoenix_tracer = None

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
    trace_id = str(uuid.uuid4())  # Generate unique trace ID

    logger.info(f"Processing query: '{request.query}' [trace_id={trace_id}]")

    try:
        # Step 1: Semantic retrieval ("Wide Net" - prioritize Recall)
        # Retrieve top 50 chunks to ensure we don't miss relevant context
        search_results = retriever.search(
            query=request.query,
            top_k=50,  # Fixed: retrieve 50 candidates
            score_threshold=0.3,  # Lower threshold for broad recall
            filters=request.filters,
        )

        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for your query.",
            )

        # Step 2: Rerank with BGE-Reranker-v2-m3 ("Filter" - prioritize Precision)
        # Takes top 50 and re-orders based on query specificity, keeps top 5
        if request.enable_reranking and reranker:
            logger.info(f"Reranking {len(search_results)} results with BGE-Reranker-v2-m3")
            search_results = reranker.rerank(
                query=request.query,
                results=search_results,
                top_k=5,  # Fixed: return top 5 most relevant chunks
            )
        else:
            # Fallback: just take top 5 without reranking
            search_results = search_results[:5]

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
            model_used=settings.groq_model,
            trace_id=trace_id,
        )

        # Log to Phoenix tracer if available
        if phoenix_tracer:
            try:
                phoenix_tracer.trace_query(
                    trace_id=trace_id,
                    query=request.query,
                    answer=answer,
                    retrieved_chunks=[r.text for r in search_results],
                    latency_ms=latency_ms,
                    metadata={
                        "model": settings.groq_model,
                        "num_citations": len(citations),
                        "reranking_enabled": request.enable_reranking,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log trace to Phoenix: {e}")

        logger.info(f"Query completed in {latency_ms:.1f}ms [trace_id={trace_id}]")
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
    trace_id = str(uuid.uuid4())  # Generate unique trace ID

    logger.info(f"Processing streaming query: '{request.query}' [trace_id={trace_id}]")

    try:
        # Retrieval and reranking (same as regular query: 50→5 pattern)
        search_results = retriever.search(
            query=request.query,
            top_k=50,
            score_threshold=0.3,
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
                top_k=5,
            )
        else:
            search_results = search_results[:5]

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

            # Log to Phoenix tracer if available
            if phoenix_tracer:
                try:
                    latency_ms = (time.time() - start_time) * 1000
                    phoenix_tracer.trace_query(
                        trace_id=trace_id,
                        query=request.query,
                        answer=full_answer,
                        retrieved_chunks=[r.text for r in search_results],
                        latency_ms=latency_ms,
                        metadata={
                            "model": settings.groq_model,
                            "num_citations": len(citations),
                            "streaming": True,
                            "reranking_enabled": request.enable_reranking,
                        },
                    )
                    logger.info(f"Streaming query completed in {latency_ms:.1f}ms [trace_id={trace_id}]")
                except Exception as e:
                    logger.warning(f"Failed to log trace to Phoenix: {e}")

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
# Feedback Collection Endpoint
# =============================================================================
@app.post("/feedback", response_model=FeedbackResponse, tags=["Observability"])
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Collect user feedback on query responses for continuous improvement.

    Feedback is logged to Phoenix tracer for analysis and failure pattern detection.
    """
    try:
        if not phoenix_tracer:
            return FeedbackResponse(
                success=False,
                message="Feedback collection not available (Phoenix tracer not initialized)",
            )

        # Log feedback
        phoenix_tracer.log_feedback(
            trace_id=request.trace_id,
            feedback=request.feedback,
            comment=request.comment,
        )

        logger.info(
            "Feedback received",
            extra={
                "trace_id": request.trace_id,
                "feedback": request.feedback,
                "has_comment": request.comment is not None,
            },
        )

        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
        )

    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        return FeedbackResponse(
            success=False,
            message=f"Failed to record feedback: {str(e)}",
        )


# =============================================================================
# Chat & Conversation Endpoints
# =============================================================================

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Conversational endpoint with follow-up support.

    Maintains conversation history and provides contextual answers.
    Includes suggested follow-up questions.
    """
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    try:
        # Get or create session
        session = session_manager.get_or_create_session(request.session_id)

        # Add user message to history
        session.add_message("user", request.message)

        # Build context from conversation history
        conversation_context = session.get_context(max_turns=request.max_context_turns)

        # Step 1: Retrieve relevant documents
        search_results = retriever.search(
            query=request.message,
            top_k=50,
            score_threshold=0.3,
        )

        # Step 2: Rerank if enabled
        if request.enable_reranking and search_results:
            search_results = reranker.rerank(
                query=request.message,
                results=search_results,
                top_k=5,
            )

        # Step 3: Generate answer with conversation context
        answer = await llm_service.generate_answer_with_history(
            query=request.message,
            context_chunks=search_results,
            conversation_history=conversation_context,
            stream=False,
        )

        # Add assistant message to history
        session.add_message("assistant", answer, trace_id=trace_id)

        # Step 4: Extract citations
        citations = citation_extractor.extract_citations(
            answer=answer,
            search_results=search_results,
        )

        # Step 5: Generate suggested follow-up questions
        suggested_questions = _generate_suggested_questions(
            current_query=request.message,
            answer=answer,
            search_results=search_results,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Log to Phoenix tracer if available
        if phoenix_tracer:
            try:
                phoenix_tracer.trace_query(
                    trace_id=trace_id,
                    query=request.message,
                    answer=answer,
                    retrieved_chunks=[r.text for r in search_results],
                    latency_ms=latency_ms,
                    metadata={
                        "model": settings.groq_model,
                        "session_id": str(session.session_id),
                        "conversation_turns": len(session.messages) // 2,
                        "suggested_questions": len(suggested_questions),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log trace to Phoenix: {e}")

        logger.info(f"Chat query completed in {latency_ms:.1f}ms [session={session.session_id}]")

        return ChatResponse(
            session_id=session.session_id,
            message=answer,
            citations=citations,
            suggested_questions=suggested_questions,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )

    except Exception as e:
        logger.error(f"Chat query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat query failed: {str(e)}",
        )


def _generate_suggested_questions(
    current_query: str,
    answer: str,
    search_results: list,
    max_suggestions: int = 3,
) -> list[str]:
    """
    Generate relevant follow-up questions based on current conversation.

    Uses a simple heuristic approach based on document context.
    In production, this could use an LLM to generate more intelligent suggestions.
    """
    suggestions = []

    # Extract topics from current answer
    if "treatment" in current_query.lower() or "treatment" in answer.lower():
        suggestions.extend([
            "What are the potential side effects?",
            "Are there any contraindications?",
            "What is the typical duration of treatment?",
        ])
    elif "dosage" in current_query.lower() or "dose" in answer.lower():
        suggestions.extend([
            "How should this be adjusted for renal impairment?",
            "What about pediatric dosing?",
            "Are there any drug interactions?",
        ])
    elif "diagnosis" in current_query.lower():
        suggestions.extend([
            "What are the differential diagnoses?",
            "What diagnostic tests are recommended?",
            "What is the treatment approach?",
        ])
    else:
        # Generic follow-ups based on context
        if search_results:
            suggestions.extend([
                "Can you explain this in more detail?",
                "What are the clinical implications?",
                "Are there any recent guidelines?",
            ])

    return suggestions[:max_suggestions]


@app.websocket("/api/v1/chat/ws")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming chat.

    Provides character-by-character streaming with typing indicators.
    """
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            message = message_data.get("message", "").strip()
            session_id = message_data.get("session_id")
            enable_reranking = message_data.get("enable_reranking", True)
            max_context_turns = message_data.get("max_context_turns", 5)

            if not message:
                await websocket.send_json({
                    "type": "error",
                    "message": "Message cannot be empty"
                })
                continue

            start_time = time.time()
            trace_id = str(uuid.uuid4())

            try:
                # Get or create session
                session = session_manager.get_or_create_session(
                    uuid.UUID(session_id) if session_id else None
                )

                # Send session ID to client
                await websocket.send_json({
                    "type": "session",
                    "session_id": str(session.session_id)
                })

                # Add user message to history
                session.add_message("user", message)

                # Build context from conversation history
                conversation_context = session.get_context(max_turns=max_context_turns)

                # Send typing indicator
                await websocket.send_json({"type": "typing", "status": "retrieving"})

                # Step 1: Retrieve relevant documents
                search_results = retriever.search(
                    query=message,
                    top_k=50,
                    score_threshold=0.3,
                )

                # Step 2: Rerank if enabled
                if enable_reranking and search_results:
                    await websocket.send_json({"type": "typing", "status": "reranking"})
                    search_results = reranker.rerank(
                        query=message,
                        results=search_results,
                        top_k=5,
                    )

                # Step 3: Generate answer with streaming
                await websocket.send_json({"type": "typing", "status": "generating"})

                # Build context
                context = llm_service._build_context(search_results)
                messages = [
                    {"role": "system", "content": llm_service._get_system_prompt()}
                ]

                # Add conversation history
                if conversation_context:
                    for turn in conversation_context[-4:]:
                        messages.append({
                            "role": turn["role"],
                            "content": turn["content"],
                        })

                # Add current query
                current_prompt = f"""Context from medical documents:

{context}

Question: {message}

Please provide a comprehensive, accurate answer based on the context above. If the context doesn't contain enough information to answer the question, clearly state that."""

                messages.append({"role": "user", "content": current_prompt})

                # Stream the response
                full_answer = ""
                async for chunk in llm_service.generate_answer_stream_from_messages(messages):
                    full_answer += chunk
                    await websocket.send_json({
                        "type": "token",
                        "content": chunk
                    })

                # Add assistant message to history
                session.add_message("assistant", full_answer, trace_id=trace_id)

                # Extract citations
                citations = citation_extractor.extract_citations(
                    answer=full_answer,
                    search_results=search_results,
                )

                # Generate suggested follow-up questions
                suggested_questions = _generate_suggested_questions(
                    current_query=message,
                    answer=full_answer,
                    search_results=search_results,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Send completion with metadata
                await websocket.send_json({
                    "type": "complete",
                    "citations": [
                        {
                            "document_id": str(c.document_id),
                            "document_title": c.document_title,
                            "page_numbers": c.page_numbers,
                            "text_snippet": c.text_snippet,
                            "relevance_score": c.relevance_score,
                        }
                        for c in citations
                    ],
                    "suggested_questions": suggested_questions,
                    "latency_ms": latency_ms,
                    "trace_id": trace_id,
                })

                # Log to Phoenix tracer if available
                if phoenix_tracer:
                    try:
                        phoenix_tracer.trace_query(
                            trace_id=trace_id,
                            query=message,
                            answer=full_answer,
                            retrieved_chunks=[r.text for r in search_results],
                            latency_ms=latency_ms,
                            metadata={
                                "model": settings.groq_model,
                                "session_id": str(session.session_id),
                                "conversation_turns": len(session.messages) // 2,
                                "streaming": True,
                            },
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log trace to Phoenix: {e}")

                logger.info(f"WebSocket chat completed in {latency_ms:.1f}ms [session={session.session_id}]")

            except Exception as e:
                logger.error(f"WebSocket chat failed: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Chat failed: {str(e)}"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


@app.delete("/api/v1/chat/{session_id}", tags=["Chat"])
async def delete_session(session_id: uuid.UUID):
    """Delete a conversation session and its history."""
    success = session_manager.delete_session(session_id)
    if success:
        return {"message": f"Session {session_id} deleted successfully"}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Session {session_id} not found",
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
# Static Files (Must be last to not override API routes)
# =============================================================================
import os
# __file__ is services/api/src/main.py
# Go up 3 levels to workspace root: src -> api -> services -> root
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
static_dir = os.path.join(workspace_root, "static")
static_dir = os.path.abspath(static_dir)  # Resolve to absolute path

if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    logger.info(f"Mounted static files from {static_dir}")
else:
    logger.warning(f"Static directory not found: {static_dir}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
