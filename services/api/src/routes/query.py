"""
Query routes for RAG retrieval and generation.

Handles user queries with two-stage retrieval and LLM generation.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from shared.models import QueryRequest, QueryResponse
from shared.utils import get_settings, setup_logging

router = APIRouter(prefix="/query", tags=["Query"])
settings = get_settings()
logger = setup_logging("api.query", settings.log_level)


@router.post("/", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system for clinical decision support.

    Process:
    1. Embed the query using dense vector (BAAI/bge-large-en-v1.5)
    2. Retrieve top-K chunks from Qdrant (Stage 1: Wide Net)
    3. Rerank using BGE-Reranker-v2-m3 (Stage 2: Filter)
    4. Generate response with Groq Llama-3.3-70B
    5. Return answer with verifiable citations

    Args:
        request: Query request with filters and retrieval settings

    Returns:
        QueryResponse with generated answer and citations

    Raises:
        HTTPException: If query processing fails
    """
    logger.info(
        "Received query",
        extra={
            "query": request.query,
            "top_k": request.top_k,
            "enable_reranking": request.enable_reranking,
        },
    )

    # TODO Phase 3: Implement retrieval from Qdrant
    # TODO Phase 4: Implement two-stage retrieval with reranking
    # TODO Phase 4: Implement LLM generation with Groq

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Query endpoint not yet implemented. Complete in Phase 3-4.",
    )


@router.post("/stream")
async def query_rag_system_stream(request: QueryRequest) -> StreamingResponse:
    """
    Query the RAG system with streaming response.

    Same as /query but streams the LLM response token-by-token
    for better perceived latency (280+ tokens/sec with Groq).

    Args:
        request: Query request

    Returns:
        StreamingResponse with server-sent events
    """
    logger.info("Received streaming query", extra={"query": request.query})

    # TODO Phase 4: Implement streaming with FastAPI StreamingResponse
    # Use Groq's streaming API for real-time token generation

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Streaming endpoint not yet implemented. Complete in Phase 4.",
    )
