"""
Integration tests for the API query endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from services.api.src.main import app
from shared.models import SearchResult


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_services(monkeypatch):
    """Mock all service dependencies."""
    # Mock retriever
    mock_retriever = Mock()
    mock_search_results = [
        SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="Amoxicillin dosing: 500mg three times daily",
            score=0.95,
            source_file="antibiotics.pdf",
            page_numbers=[15],
            section_path="Antibiotics > Penicillins",
            chunk_index=0,
        ),
    ]
    mock_retriever.search.return_value = mock_search_results
    mock_retriever.get_collection_stats.return_value = {
        "vectors_count": 1000,
        "status": "green",
    }

    # Mock reranker
    mock_reranker = Mock()
    mock_reranker.rerank.return_value = mock_search_results

    # Mock LLM service
    mock_llm = Mock()
    mock_llm.generate_answer = AsyncMock(
        return_value="Based on [Source 1], Amoxicillin dosing is 500mg three times daily."
    )

    # Mock citation extractor
    mock_extractor = Mock()
    mock_extractor.extract_citations.return_value = []

    # Patch the global service variables
    monkeypatch.setattr("services.api.src.main.retriever", mock_retriever)
    monkeypatch.setattr("services.api.src.main.reranker", mock_reranker)
    monkeypatch.setattr("services.api.src.main.llm_service", mock_llm)
    monkeypatch.setattr("services.api.src.main.citation_extractor", mock_extractor)

    return {
        "retriever": mock_retriever,
        "reranker": mock_reranker,
        "llm": mock_llm,
        "extractor": mock_extractor,
    }


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "MedArchive RAG API"
        assert "version" in data

    def test_health_endpoint(self, client, mock_services):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "api"
        assert "dependencies" in data

    def test_query_endpoint_success(self, client, mock_services):
        """Test successful query processing."""
        query_data = {
            "query": "What is the dosage for amoxicillin?",
            "top_k": 5,
            "enable_reranking": True,
        }

        response = client.post("/api/v1/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "retrieved_chunks" in data
        assert data["query"] == query_data["query"]

    def test_query_endpoint_calls_services(self, client, mock_services):
        """Test that query endpoint calls all services correctly."""
        query_data = {
            "query": "test query",
            "top_k": 3,
            "enable_reranking": True,
        }

        response = client.post("/api/v1/query", json=query_data)

        # Verify service calls
        mock_services["retriever"].search.assert_called_once()
        mock_services["reranker"].rerank.assert_called_once()
        mock_services["llm"].generate_answer.assert_called_once()
        mock_services["extractor"].extract_citations.assert_called_once()

    def test_query_endpoint_no_reranking(self, client, mock_services):
        """Test query without reranking."""
        query_data = {
            "query": "test query",
            "top_k": 5,
            "enable_reranking": False,
        }

        response = client.post("/api/v1/query", json=query_data)

        assert response.status_code == 200
        # Reranker should not be called
        mock_services["reranker"].rerank.assert_not_called()

    def test_query_endpoint_with_filters(self, client, mock_services):
        """Test query with metadata filters."""
        query_data = {
            "query": "test query",
            "top_k": 5,
            "filters": {"source_file": "specific.pdf"},
        }

        response = client.post("/api/v1/query", json=query_data)

        assert response.status_code == 200
        # Verify filters were passed to retriever
        call_args = mock_services["retriever"].search.call_args
        assert call_args[1]["filters"] == {"source_file": "specific.pdf"}

    def test_query_endpoint_no_results(self, client, mock_services):
        """Test query when no results found."""
        mock_services["retriever"].search.return_value = []

        query_data = {
            "query": "nonexistent topic",
            "top_k": 5,
        }

        response = client.post("/api/v1/query", json=query_data)

        assert response.status_code == 404
        assert "No relevant documents found" in response.json()["detail"]

    def test_query_endpoint_invalid_request(self, client):
        """Test query with invalid request data."""
        query_data = {
            "query": "ab",  # Too short (min 3 chars)
        }

        response = client.post("/api/v1/query", json=query_data)

        assert response.status_code == 422  # Validation error

    def test_query_endpoint_service_error(self, client, mock_services):
        """Test error handling when service fails."""
        mock_services["retriever"].search.side_effect = Exception("Connection error")

        query_data = {
            "query": "test query",
        }

        response = client.post("/api/v1/query", json=query_data)

        assert response.status_code == 500

    def test_stats_endpoint(self, client, mock_services):
        """Test stats endpoint."""
        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert "collection" in data
        assert "services" in data
        assert data["collection"]["vectors_count"] == 1000

    def test_query_response_structure(self, client, mock_services):
        """Test that query response has correct structure."""
        query_data = {
            "query": "test query",
            "top_k": 5,
        }

        response = client.post("/api/v1/query", json=query_data)

        data = response.json()
        required_fields = [
            "query",
            "answer",
            "citations",
            "retrieved_chunks",
            "latency_ms",
            "model_used",
            "timestamp",
        ]
        for field in required_fields:
            assert field in data

    def test_query_latency_tracking(self, client, mock_services):
        """Test that latency is tracked."""
        query_data = {
            "query": "test query",
        }

        response = client.post("/api/v1/query", json=query_data)

        data = response.json()
        assert data["latency_ms"] > 0
        assert isinstance(data["latency_ms"], float)

    def test_query_with_top_k_limit(self, client, mock_services):
        """Test that top_k is respected."""
        # Create multiple mock results
        mock_results = [
            SearchResult(
                chunk_id=uuid4(),
                document_id=uuid4(),
                text=f"Text {i}",
                score=0.9,
                source_file="test.pdf",
                page_numbers=[i],
                chunk_index=i,
            )
            for i in range(10)
        ]
        mock_services["retriever"].search.return_value = mock_results

        query_data = {
            "query": "test query",
            "top_k": 3,
            "enable_reranking": False,
        }

        response = client.post("/api/v1/query", json=query_data)

        data = response.json()
        assert len(data["retrieved_chunks"]) == 3


@pytest.mark.integration
@pytest.mark.requires_api_keys
class TestRealServiceIntegration:
    """Integration tests with real services (requires API keys and Qdrant)."""

    def test_end_to_end_query(self):
        """Test end-to-end query with real services."""
        # This test requires:
        # - Running Qdrant instance
        # - Valid Groq API key
        # - Indexed documents
        pytest.skip("Requires real infrastructure")

    def test_streaming_query(self):
        """Test streaming query endpoint."""
        pytest.skip("Requires real infrastructure")
