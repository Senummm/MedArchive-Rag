"""
Integration test for Phase 4-5 evaluation flow.

Tests the complete pipeline:
1. Query with 50→5 retrieval pattern
2. RAGAS evaluation
3. Phoenix tracing
4. Feedback collection
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from services.api.src.retrieval import Retriever, Reranker
from services.api.src.llm import LLMService
from services.api.src.evaluation import RAGASEvaluator
from services.api.src.observability import PhoenixTracer


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing."""
    retriever = Mock(spec=Retriever)

    # Mock 50 search results
    mock_results = []
    for i in range(50):
        result = Mock()
        result.chunk_id = f"chunk-{i}"
        result.document_id = f"doc-{i % 10}"
        result.text = f"Medical text chunk {i}"
        result.score = 0.9 - (i * 0.01)  # Decreasing scores
        result.source_file = "test_guideline.pdf"
        result.page_numbers = [i % 20]
        result.section_path = ["Section", "Subsection"]
        result.chunk_index = i
        mock_results.append(result)

    retriever.search.return_value = mock_results
    return retriever


@pytest.fixture
def mock_reranker():
    """Mock reranker for testing."""
    reranker = Mock(spec=Reranker)

    def rerank_side_effect(query, results, top_k=5):
        # Just return top 5
        return results[:top_k]

    reranker.rerank.side_effect = rerank_side_effect
    return reranker


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    llm = Mock(spec=LLMService)
    llm.generate_answer.return_value = "The first-line treatment for hypertension includes ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics."
    return llm


@pytest.mark.asyncio
async def test_full_query_pipeline(mock_retriever, mock_reranker, mock_llm_service):
    """Test complete query pipeline with 50→5 pattern."""
    query = "What is the first-line treatment for hypertension?"

    # Step 1: Retrieve 50 chunks
    search_results = mock_retriever.search(query, top_k=50, score_threshold=0.3)
    assert len(search_results) == 50

    # Step 2: Rerank to top 5
    reranked_results = mock_reranker.rerank(query, search_results, top_k=5)
    assert len(reranked_results) == 5

    # Step 3: Generate answer
    answer = await mock_llm_service.generate_answer(query, reranked_results, stream=False)
    assert "hypertension" in answer.lower()


@pytest.mark.asyncio
async def test_ragas_evaluation_pipeline():
    """Test RAGAS evaluation on query results."""
    evaluator = RAGASEvaluator()

    # Mock query result
    question = "What is the dose of amoxicillin?"
    answer = "500mg three times daily for 7 days"
    contexts = [
        "Amoxicillin dosage: 500mg TID for 7-10 days",
        "Standard adult dose is 500mg every 8 hours",
    ]

    with patch("services.api.src.evaluation.ragas_evaluator.RAGAS_AVAILABLE", False):
        scores = evaluator.evaluate_single(question, answer, contexts)

        # Should return default scores when RAGAS not available
        assert scores["faithfulness"] == 1.0
        assert scores["answer_relevancy"] == 1.0


def test_phoenix_tracing_pipeline():
    """Test Phoenix tracing integration."""
    tracer = PhoenixTracer(project_name="integration-test")

    trace_id = "test-trace-123"
    query = "What is the first-line treatment for hypertension?"
    answer = "ACE inhibitors or ARBs"
    chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

    # Trace query
    tracer.trace_query(
        trace_id=trace_id,
        query=query,
        answer=answer,
        retrieved_chunks=chunks,
        latency_ms=250.0,
    )

    # Log feedback
    tracer.log_feedback(
        trace_id=trace_id,
        feedback="thumbs_up",
    )

    # Get trace
    trace = tracer.get_trace(trace_id)
    assert trace is not None
    assert trace["query"] == query
    assert trace["feedback"] == "thumbs_up"


@pytest.mark.asyncio
async def test_end_to_end_with_evaluation(mock_retriever, mock_reranker, mock_llm_service):
    """Test complete pipeline with evaluation and tracing."""
    query = "What is the pediatric dose of ibuprofen?"

    # Initialize services
    evaluator = RAGASEvaluator()
    tracer = PhoenixTracer(project_name="e2e-test")

    trace_id = "e2e-trace-001"

    # Query pipeline
    search_results = mock_retriever.search(query, top_k=50, score_threshold=0.3)
    assert len(search_results) == 50

    reranked = mock_reranker.rerank(query, search_results, top_k=5)
    assert len(reranked) == 5

    answer = await mock_llm_service.generate_answer(query, reranked, stream=False)

    # Evaluate
    with patch("services.api.src.evaluation.ragas_evaluator.RAGAS_AVAILABLE", False):
        scores = evaluator.evaluate_single(
            question=query,
            answer=answer,
            contexts=[r.text for r in reranked],
        )
        assert "faithfulness" in scores

    # Trace
    tracer.trace_query(
        trace_id=trace_id,
        query=query,
        answer=answer,
        retrieved_chunks=[r.text for r in reranked],
        latency_ms=180.0,
        metadata=scores,
    )

    # Verify trace exists
    trace = tracer.get_trace(trace_id)
    assert trace is not None
    assert "metadata" in trace


def test_safety_threshold_check():
    """Test safety threshold validation."""
    evaluator = RAGASEvaluator()

    # Safe scores
    safe_scores = {
        "faithfulness": 0.97,
        "answer_relevancy": 0.85,
        "context_precision": 0.75,
    }
    assert evaluator.check_safety_threshold(safe_scores) is True

    # Unsafe scores (low faithfulness)
    unsafe_scores = {
        "faithfulness": 0.92,  # Below 0.95
        "answer_relevancy": 0.85,
        "context_precision": 0.75,
    }
    assert evaluator.check_safety_threshold(unsafe_scores) is False


def test_failure_analysis():
    """Test failure pattern analysis from feedback."""
    tracer = PhoenixTracer(project_name="failure-test")

    # Add some traces with mixed feedback
    for i in range(10):
        trace_id = f"trace-{i}"
        feedback = "thumbs_down" if i % 3 == 0 else "thumbs_up"

        tracer.trace_query(
            trace_id=trace_id,
            query=f"Query {i}",
            answer=f"Answer {i}",
            retrieved_chunks=[f"Chunk {i}"],
            latency_ms=100.0,
        )

        tracer.log_feedback(
            trace_id=trace_id,
            feedback=feedback,
            comment="Issue detected" if feedback == "thumbs_down" else None,
        )

    # Analyze failures
    failures = tracer.analyze_failures()

    # Should have ~4 failures (0, 3, 6, 9)
    assert len(failures) >= 3
    assert all(f["feedback"] == "thumbs_down" for f in failures)
