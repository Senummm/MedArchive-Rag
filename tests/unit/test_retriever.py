"""
Unit tests for the retrieval service.
"""

from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from services.api.src.retrieval.retriever import Retriever
from shared.models import SearchResult


class TestRetriever:
    """Test cases for the Retriever class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        return Mock()

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        model = Mock()
        model.encode.return_value = np.random.randn(1024).astype(np.float32)
        return model

    @pytest.fixture
    def retriever(self, mock_qdrant_client, mock_embedding_model, monkeypatch):
        """Create a retriever instance with mocked dependencies."""
        monkeypatch.setattr("services.api.src.retrieval.retriever.QdrantClient", lambda *args, **kwargs: mock_qdrant_client)
        monkeypatch.setattr("services.api.src.retrieval.retriever.SentenceTransformer", lambda *args, **kwargs: mock_embedding_model)

        return Retriever()

    def test_retriever_initialization(self, retriever):
        """Test that the retriever initializes correctly."""
        assert retriever.collection_name == "medical_documents"
        assert retriever.client is not None
        assert retriever.embedding_model is not None

    def test_embed_query(self, retriever, mock_embedding_model):
        """Test query embedding generation."""
        query = "What is the dosage for amoxicillin?"

        embedding = retriever.embed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        mock_embedding_model.encode.assert_called_once()

    def test_search_returns_results(self, retriever, mock_qdrant_client):
        """Test that search returns formatted results."""
        # Mock Qdrant response
        doc_id = uuid4()
        chunk_id = uuid4()

        mock_result = Mock()
        mock_result.id = str(chunk_id)
        mock_result.score = 0.95
        mock_result.payload = {
            "document_id": str(doc_id),
            "text": "Amoxicillin dosing: 500mg three times daily",
            "source_file": "antibiotics.pdf",
            "page_numbers": [15],
            "section_path": "Antibiotics > Penicillins",
            "chunk_index": 5,
        }

        mock_qdrant_client.search.return_value = [mock_result]

        results = retriever.search("amoxicillin dosage", top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score == 0.95
        assert results[0].text == "Amoxicillin dosing: 500mg three times daily"

    def test_search_with_filters(self, retriever, mock_qdrant_client):
        """Test search with metadata filters."""
        mock_qdrant_client.search.return_value = []

        retriever.search(
            "query",
            filters={"source_file": "specific.pdf"}
        )

        # Verify filter was passed
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]["query_filter"] is not None

    def test_search_empty_results(self, retriever, mock_qdrant_client):
        """Test search with no results."""
        mock_qdrant_client.search.return_value = []

        results = retriever.search("nonexistent query")

        assert len(results) == 0

    def test_search_score_threshold(self, retriever, mock_qdrant_client):
        """Test that score threshold is applied."""
        mock_qdrant_client.search.return_value = []

        retriever.search("query", score_threshold=0.8)

        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]["score_threshold"] == 0.8

    def test_get_collection_stats(self, retriever, mock_qdrant_client):
        """Test getting collection statistics."""
        mock_info = Mock()
        mock_info.vectors_count = 1000
        mock_info.points_count = 1000
        mock_info.status = "green"
        mock_info.segments_count = 1

        mock_qdrant_client.get_collection.return_value = mock_info

        stats = retriever.get_collection_stats()

        assert stats["vectors_count"] == 1000
        assert stats["status"] == "green"

    def test_batch_search(self, retriever, mock_qdrant_client):
        """Test batch searching."""
        mock_qdrant_client.search_batch.return_value = [[], []]

        queries = ["query1", "query2"]
        results = retriever.batch_search(queries)

        assert len(results) == 2
        mock_qdrant_client.search_batch.assert_called_once()

    def test_get_document_chunks(self, retriever, mock_qdrant_client):
        """Test retrieving all chunks for a document."""
        doc_id = uuid4()
        chunk_ids = [uuid4(), uuid4(), uuid4()]

        mock_points = []
        for i, chunk_id in enumerate(chunk_ids):
            point = Mock()
            point.id = str(chunk_id)
            point.payload = {
                "document_id": str(doc_id),
                "text": f"Chunk {i}",
                "source_file": "doc.pdf",
                "page_numbers": [i + 1],
                "chunk_index": i,
            }
            mock_points.append(point)

        mock_qdrant_client.scroll.return_value = (mock_points, None)

        chunks = retriever.get_document_chunks(doc_id)

        assert len(chunks) == 3
        assert chunks[0].chunk_index == 0
        assert chunks[2].chunk_index == 2

    def test_build_filter_source_file(self, retriever):
        """Test filter construction for source file."""
        filter_obj = retriever._build_filter({"source_file": "test.pdf"})

        assert filter_obj is not None
        assert len(filter_obj.must) == 1

    def test_build_filter_multiple_conditions(self, retriever):
        """Test filter with multiple conditions."""
        doc_id = uuid4()
        filter_obj = retriever._build_filter({
            "source_file": "test.pdf",
            "document_id": doc_id,
        })

        assert len(filter_obj.must) == 2

    def test_search_error_handling(self, retriever, mock_qdrant_client):
        """Test error handling during search."""
        mock_qdrant_client.search.side_effect = Exception("Connection error")

        with pytest.raises(Exception, match="Connection error"):
            retriever.search("query")
