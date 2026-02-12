"""
Unit tests for Phoenix tracer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from services.api.src.observability import PhoenixTracer, init_tracer, get_tracer


class TestPhoenixTracer:
    """Test suite for Phoenix tracing."""

    def test_tracer_initialization(self):
        """Test PhoenixTracer can be initialized."""
        tracer = PhoenixTracer(project_name="test-project")
        assert tracer is not None
        assert tracer.project_name == "test-project"

    def test_tracer_graceful_degradation_without_phoenix(self):
        """Test tracer works without Phoenix installed."""
        with patch("services.api.src.observability.phoenix_tracer.PHOENIX_AVAILABLE", False):
            tracer = PhoenixTracer(project_name="test")

            # Should not raise errors
            tracer.trace_query(
                trace_id="test-123",
                query="What is X?",
                answer="X is Y",
                retrieved_chunks=["chunk1"],
                latency_ms=100.0,
            )

            tracer.log_feedback(
                trace_id="test-123",
                feedback="thumbs_up",
            )

    @patch("services.api.src.observability.phoenix_tracer.PHOENIX_AVAILABLE", True)
    def test_trace_query_logs_correctly(self):
        """Test trace_query logs data correctly."""
        tracer = PhoenixTracer(project_name="test")

        # Mock the traces list
        tracer.traces = {}

        tracer.trace_query(
            trace_id="trace-abc",
            query="What is hypertension?",
            answer="High blood pressure",
            retrieved_chunks=["Chunk 1", "Chunk 2"],
            latency_ms=150.5,
            metadata={"model": "llama-3.3-70b"},
        )

        # Verify trace was stored
        assert "trace-abc" in tracer.traces
        trace = tracer.traces["trace-abc"]
        assert trace["query"] == "What is hypertension?"
        assert trace["answer"] == "High blood pressure"
        assert len(trace["retrieved_chunks"]) == 2
        assert trace["latency_ms"] == 150.5
        assert trace["metadata"]["model"] == "llama-3.3-70b"

    def test_log_feedback(self):
        """Test feedback logging."""
        tracer = PhoenixTracer(project_name="test")
        tracer.traces = {
            "trace-123": {
                "query": "Test query",
                "answer": "Test answer",
            }
        }

        tracer.log_feedback(
            trace_id="trace-123",
            feedback="thumbs_down",
            comment="Answer was incorrect",
        )

        # Verify feedback was added
        assert "feedback" in tracer.traces["trace-123"]
        assert tracer.traces["trace-123"]["feedback"] == "thumbs_down"
        assert tracer.traces["trace-123"]["feedback_comment"] == "Answer was incorrect"

    def test_get_trace(self):
        """Test retrieving a specific trace."""
        tracer = PhoenixTracer(project_name="test")
        tracer.traces = {
            "trace-999": {
                "query": "Specific query",
                "answer": "Specific answer",
            }
        }

        trace = tracer.get_trace("trace-999")
        assert trace is not None
        assert trace["query"] == "Specific query"

        # Non-existent trace
        trace = tracer.get_trace("non-existent")
        assert trace is None

    def test_analyze_failures(self):
        """Test failure pattern analysis."""
        tracer = PhoenixTracer(project_name="test")
        tracer.traces = {
            "trace-1": {
                "query": "Query 1",
                "answer": "Answer 1",
                "feedback": "thumbs_down",
                "retrieved_chunks": ["chunk"],
            },
            "trace-2": {
                "query": "Query 2",
                "answer": "Answer 2",
                "feedback": "thumbs_up",
                "retrieved_chunks": ["chunk"],
            },
            "trace-3": {
                "query": "Query 3",
                "answer": "Answer 3",
                "feedback": "thumbs_down",
                "feedback_comment": "Wrong dosage",
                "retrieved_chunks": ["chunk"],
            },
        }

        failures = tracer.analyze_failures()

        # Should return 2 failures
        assert len(failures) == 2
        assert failures[0]["trace_id"] in ["trace-1", "trace-3"]

    def test_trace_retrieval(self):
        """Test trace_retrieval method."""
        tracer = PhoenixTracer(project_name="test")

        # Should not raise errors
        tracer.trace_retrieval(
            trace_id="test-retrieve",
            query="Test",
            retrieved_chunks=["chunk1", "chunk2"],
            latency_ms=50.0,
        )

    def test_trace_generation(self):
        """Test trace_generation method."""
        tracer = PhoenixTracer(project_name="test")

        # Should not raise errors
        tracer.trace_generation(
            trace_id="test-gen",
            query="Test",
            answer="Generated answer",
            latency_ms=200.0,
        )

    def test_init_tracer_singleton(self):
        """Test init_tracer creates singleton."""
        tracer1 = init_tracer(project_name="singleton-test")
        tracer2 = get_tracer()

        # Should return same instance
        assert tracer1 is tracer2

    def test_get_tracer_before_init_returns_none(self):
        """Test get_tracer returns None if not initialized."""
        # Reset global tracer
        import services.api.src.observability.phoenix_tracer as pt_module
        pt_module._tracer_instance = None

        tracer = get_tracer()
        assert tracer is None
