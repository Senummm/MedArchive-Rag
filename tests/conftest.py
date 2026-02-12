"""
Pytest configuration and shared fixtures for MedArchive RAG tests.

Provides reusable test fixtures for mocking services and test data.
"""

import sys
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Configuration Fixtures
# =============================================================================
@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing environment."""
    from shared.utils import Settings

    return Settings(
        environment="testing",
        log_level="DEBUG",
        api_host="localhost",
        api_port=8000,
        qdrant_url="http://localhost:6333",
        groq_api_key="test_groq_key",
        llamaparse_api_key="test_llamaparse_key",
        embedding_model="BAAI/bge-small-en-v1.5",  # Smaller model for tests
    )


# =============================================================================
# API Client Fixtures
# =============================================================================
@pytest.fixture
def api_client() -> Generator:
    """
    FastAPI test client.

    Provides a test client for making requests to the API without
    starting a real server.
    """
    from services.api.src.main import app

    with TestClient(app) as client:
        yield client


# =============================================================================
# Mock Service Fixtures
# =============================================================================
@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing without a real vector database."""
    mock_client = MagicMock()

    # Mock collection operations
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.create_collection.return_value = True
    mock_client.upsert.return_value = MagicMock(status="completed")

    # Mock search results
    mock_search_result = MagicMock()
    mock_search_result.id = str(uuid4())
    mock_search_result.score = 0.95
    mock_search_result.payload = {
        "text": "Amoxicillin pediatric dosing: 15mg/kg twice daily.",
        "document_title": "Test Formulary",
        "page_numbers": [12],
    }
    mock_client.search.return_value = [mock_search_result]

    return mock_client


@pytest.fixture
def mock_groq_client():
    """Mock Groq client for testing LLM generation."""
    mock_client = AsyncMock()

    # Mock chat completion
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="The pediatric dosage for Amoxicillin is 15mg/kg twice daily."
            )
        )
    ]
    mock_response.model = "llama-3.3-70b-versatile"
    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_embedding_model():
    """Mock sentence-transformers embedding model."""
    import numpy as np

    mock_model = MagicMock()

    # Return random embeddings
    def encode_fn(texts, **kwargs):
        if isinstance(texts, str):
            return np.random.rand(1024).astype(np.float32)
        return np.random.rand(len(texts), 1024).astype(np.float32)

    mock_model.encode.side_effect = encode_fn
    return mock_model


# =============================================================================
# Test Data Fixtures
# =============================================================================
@pytest.fixture
def sample_document_metadata():
    """Sample DocumentMetadata for testing."""
    from shared.models import DocumentMetadata, DocumentType

    return DocumentMetadata(
        title="Pediatric Antibiotic Formulary 2026",
        document_type=DocumentType.FORMULARY,
        source_path="/data/test_formulary.pdf",
        file_hash="5d41402abc4b2a76b9719d911017c592",
        department="Pediatrics",
        page_count=42,
    )


@pytest.fixture
def sample_chunk_metadata():
    """Sample ChunkMetadata for testing."""
    from shared.models import ChunkMetadata

    return ChunkMetadata(
        document_id=uuid4(),
        chunk_index=5,
        text="Amoxicillin pediatric dosing: 15mg/kg twice daily for children under 40kg.",
        section_path="Antibiotics > Beta-Lactams > Amoxicillin",
        heading="Pediatric Dosing",
        page_numbers=[12, 13],
        token_count=18,
    )


@pytest.fixture
def sample_query_request():
    """Sample QueryRequest for testing."""
    from shared.models import QueryRequest

    return QueryRequest(
        query="What is the pediatric dosage for Amoxicillin?",
        top_k=5,
        enable_reranking=True,
    )


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================
@pytest.fixture
def temp_document_store(tmp_path):
    """Temporary directory for test PDF files."""
    doc_store = tmp_path / "document_store"
    doc_store.mkdir()
    return doc_store


@pytest.fixture
def temp_vector_storage(tmp_path):
    """Temporary directory for vector database storage."""
    vector_store = tmp_path / "vector_storage"
    vector_store.mkdir()
    return vector_store


# =============================================================================
# Pytest Hooks
# =============================================================================
def pytest_configure(config):
    """Configure pytest environment."""
    # Set testing environment variable
    import os

    os.environ["ENVIRONMENT"] = "testing"
    os.environ["LOG_LEVEL"] = "DEBUG"


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location.

    - tests/unit/ -> @pytest.mark.unit
    - tests/integration/ -> @pytest.mark.integration
    """
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
