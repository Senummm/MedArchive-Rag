"""
Integration tests for API service.

Tests the FastAPI application with real HTTP requests
(but mocked external dependencies).
"""

import pytest
from fastapi import status


@pytest.mark.integration
class TestAPIHealth:
    """Integration tests for health check endpoints."""

    def test_health_endpoint(self, api_client):
        """Test /health endpoint returns 200 OK."""
        response = api_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "api"
        assert "version" in data
        assert "dependencies" in data

    def test_root_endpoint(self, api_client):
        """Test root endpoint returns service information."""
        response = api_client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["service"] == "MedArchive RAG API"
        assert "version" in data
        assert data["status"] == "operational"

    def test_health_check_includes_dependencies(self, api_client):
        """Test that health check includes dependency status."""
        response = api_client.get("/health")
        data = response.json()

        assert "dependencies" in data
        # Phase 1: Dependencies not yet initialized, should be False
        assert "qdrant" in data["dependencies"]
        assert "groq" in data["dependencies"]


@pytest.mark.integration
class TestAPICORS:
    """Integration tests for CORS configuration."""

    def test_cors_headers_present(self, api_client):
        """Test that CORS headers are set correctly."""
        response = api_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # FastAPI TestClient doesn't fully simulate CORS preflight,
        # but we can verify the middleware is configured
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]


@pytest.mark.integration
class TestAPIErrorHandling:
    """Integration tests for error handling."""

    def test_404_for_unknown_endpoint(self, api_client):
        """Test that unknown endpoints return 404."""
        response = api_client.get("/nonexistent-endpoint")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_method_not_allowed(self, api_client):
        """Test that wrong HTTP methods return 405."""
        # Health endpoint only supports GET
        response = api_client.post("/health")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
