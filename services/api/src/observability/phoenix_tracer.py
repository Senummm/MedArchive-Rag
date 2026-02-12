"""
Arize Phoenix observability and tracing for RAG system.

Provides:
- Request/response tracing
- User feedback collection
- Performance monitoring
- Debug trace analysis
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID, uuid4
import json
import logging

try:
    from phoenix.trace import trace, using_project
    from phoenix.trace.exporter import HttpExporter
    import phoenix as px
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

logger = logging.getLogger(__name__)


class PhoenixTracer:
    """
    Arize Phoenix tracing for RAG system observability.

    Captures:
    - Query inputs and outputs
    - Retrieval results and scores
    - LLM prompts and completions
    - User feedback (thumbs up/down)
    - Latency metrics
    """

    def __init__(
        self,
        project_name: str = "medarchive-rag",
        phoenix_host: str = "http://localhost:6006",
        enable_tracing: bool = True,
    ):
        """
        Initialize Phoenix tracer.

        Args:
            project_name: Project name in Phoenix
            phoenix_host: Phoenix collector URL
            enable_tracing: Whether to enable tracing (disable in dev if needed)
        """
        self.project_name = project_name
        self.phoenix_host = phoenix_host
        self.enable_tracing = enable_tracing

        if not PHOENIX_AVAILABLE:
            logger.warning("Phoenix not installed. Run: pip install arize-phoenix")
            self.enable_tracing = False
            return

        if self.enable_tracing:
            try:
                # Initialize Phoenix
                px.launch_app()
                logger.info(f"Phoenix tracer initialized for project '{project_name}'")
                logger.info(f"Phoenix UI available at: {phoenix_host}")
            except Exception as e:
                logger.warning(f"Failed to initialize Phoenix: {e}")
                self.enable_tracing = False

    def trace_query(
        self,
        trace_id: UUID,
        query: str,
        search_results: List[Dict[str, Any]],
        answer: str,
        citations: List[Dict[str, Any]],
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Trace a complete query execution.

        Args:
            trace_id: Unique trace identifier
            query: User's query
            search_results: Retrieved chunks with scores
            answer: Generated answer
            citations: Extracted citations
            latency_ms: Total query latency
            metadata: Additional metadata (user_id, filters, etc.)
        """
        if not self.enable_tracing:
            return

        try:
            trace_data = {
                "trace_id": str(trace_id),
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "retrieval": {
                    "num_results": len(search_results),
                    "top_score": search_results[0]["score"] if search_results else 0,
                    "results": [
                        {
                            "text": r["text"][:200],  # Truncate for storage
                            "score": r["score"],
                            "source": r.get("source_file"),
                            "pages": r.get("page_numbers"),
                        }
                        for r in search_results
                    ],
                },
                "generation": {
                    "answer": answer,
                    "num_citations": len(citations),
                    "citations": citations,
                },
                "metrics": {
                    "latency_ms": latency_ms,
                },
                "metadata": metadata or {},
            }

            # Log trace (in production, this would send to Phoenix)
            logger.info(f"Trace logged: {trace_id}", extra={"trace": trace_data})

        except Exception as e:
            logger.error(f"Failed to trace query: {e}")

    def log_feedback(
        self,
        trace_id: UUID,
        feedback_type: str,  # "thumbs_up" or "thumbs_down"
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Log user feedback for a query.

        Args:
            trace_id: Trace ID to attach feedback to
            feedback_type: "thumbs_up" or "thumbs_down"
            comment: Optional user comment
            user_id: Optional user identifier
        """
        if not self.enable_tracing:
            return

        try:
            feedback_data = {
                "trace_id": str(trace_id),
                "timestamp": datetime.utcnow().isoformat(),
                "feedback_type": feedback_type,
                "comment": comment,
                "user_id": user_id,
            }

            logger.info(
                f"Feedback logged: {trace_id} - {feedback_type}",
                extra={"feedback": feedback_data},
            )

            # In production: send to Phoenix for analysis
            # phoenix.log_feedback(trace_id, feedback_data)

        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")

    def trace_retrieval(
        self,
        query: str,
        results: List[Dict[str, Any]],
        latency_ms: float,
    ) -> None:
        """
        Trace retrieval stage separately.

        Args:
            query: Search query
            results: Retrieved results
            latency_ms: Retrieval latency
        """
        if not self.enable_tracing:
            return

        try:
            trace_data = {
                "stage": "retrieval",
                "query": query,
                "num_results": len(results),
                "latency_ms": latency_ms,
                "top_scores": [r["score"] for r in results[:5]],
            }

            logger.debug("Retrieval traced", extra={"trace": trace_data})

        except Exception as e:
            logger.error(f"Failed to trace retrieval: {e}")

    def trace_generation(
        self,
        query: str,
        context: str,
        answer: str,
        latency_ms: float,
        model: str,
    ) -> None:
        """
        Trace LLM generation stage.

        Args:
            query: User query
            context: Provided context
            answer: Generated answer
            latency_ms: Generation latency
            model: Model identifier
        """
        if not self.enable_tracing:
            return

        try:
            trace_data = {
                "stage": "generation",
                "query": query,
                "context_length": len(context),
                "answer_length": len(answer),
                "latency_ms": latency_ms,
                "model": model,
            }

            logger.debug("Generation traced", extra={"trace": trace_data})

        except Exception as e:
            logger.error(f"Failed to trace generation: {e}")

    def get_trace(self, trace_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a trace by ID for debugging.

        Args:
            trace_id: Trace identifier

        Returns:
            Trace data dictionary or None if not found
        """
        if not self.enable_tracing:
            return None

        try:
            # In production: fetch from Phoenix
            # return phoenix.get_trace(trace_id)
            logger.info(f"Trace retrieval requested: {trace_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve trace: {e}")
            return None

    def analyze_failures(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze traces with negative feedback ("thumbs down").

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            List of failed traces with analysis
        """
        if not self.enable_tracing:
            return []

        try:
            # In production: query Phoenix for thumbs_down feedback
            # failures = phoenix.query_traces(
            #     feedback_type="thumbs_down",
            #     start_date=start_date,
            #     end_date=end_date,
            # )

            logger.info("Failure analysis requested")
            return []

        except Exception as e:
            logger.error(f"Failed to analyze failures: {e}")
            return []


# Global tracer instance
_tracer: Optional[PhoenixTracer] = None


def get_tracer() -> PhoenixTracer:
    """Get global Phoenix tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = PhoenixTracer()
    return _tracer


def init_tracer(
    project_name: str = "medarchive-rag",
    phoenix_host: str = "http://localhost:6006",
    enable_tracing: bool = True,
) -> PhoenixTracer:
    """
    Initialize global Phoenix tracer.

    Args:
        project_name: Project name
        phoenix_host: Phoenix server URL
        enable_tracing: Enable/disable tracing

    Returns:
        Initialized tracer
    """
    global _tracer
    _tracer = PhoenixTracer(project_name, phoenix_host, enable_tracing)
    return _tracer
