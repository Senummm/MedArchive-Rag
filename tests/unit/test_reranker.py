"""
Unit tests for the reranker service.
"""

from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from services.api.src.retrieval.reranker import Reranker
from shared.models import SearchResult


class TestReranker:
    """Test cases for the Reranker class."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Create a mock cross-encoder model."""
        model = Mock()
        # Return scores in reverse order to test reranking
        model.predict.return_value = np.array([0.3, 0.7, 0.5])
        return model

    @pytest.fixture
    def reranker(self, mock_cross_encoder, monkeypatch):
        """Create a reranker with mocked model."""
        monkeypatch.setattr("services.api.src.retrieval.reranker.CrossEncoder", lambda *args: mock_cross_encoder)
        return Reranker()

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        results = []
        for i in range(3):
            result = SearchResult(
                chunk_id=uuid4(),
                document_id=uuid4(),
                text=f"Text chunk {i}",
                score=0.5 + i * 0.1,  # 0.5, 0.6, 0.7
                source_file="test.pdf",
                page_numbers=[i + 1],
                chunk_index=i,
            )
            results.append(result)
        return results

    def test_reranker_initialization(self, reranker):
        """Test that reranker initializes correctly."""
        assert reranker.model is not None
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_rerank_updates_scores(self, reranker, sample_results, mock_cross_encoder):
        """Test that reranking updates scores."""
        query = "test query"
        
        reranked = reranker.rerank(query, sample_results)

        # Scores should be updated from cross-encoder
        assert reranked[0].score != sample_results[0].score
        mock_cross_encoder.predict.assert_called_once()

    def test_rerank_sorts_by_score(self, reranker, sample_results):
        """Test that results are sorted by new scores."""
        query = "test query"
        
        # Mock returns [0.3, 0.7, 0.5], so order should be: 1, 2, 0
        reranked = reranker.rerank(query, sample_results)

        assert reranked[0].score == 0.7
        assert reranked[1].score == 0.5
        assert reranked[2].score == 0.3

    def test_rerank_with_top_k(self, reranker, sample_results):
        """Test reranking with top_k limit."""
        query = "test query"
        
        reranked = reranker.rerank(query, sample_results, top_k=2)

        assert len(reranked) == 2

    def test_rerank_empty_results(self, reranker):
        """Test reranking with empty results."""
        reranked = reranker.rerank("query", [])

        assert len(reranked) == 0

    def test_compute_score_single_pair(self, reranker, mock_cross_encoder):
        """Test computing score for single query-text pair."""
        mock_cross_encoder.predict.return_value = np.array([0.85])
        
        score = reranker.compute_score("query", "text")

        assert score == 0.85
        assert isinstance(score, float)

    def test_batch_rerank(self, reranker, sample_results):
        """Test batch reranking."""
        queries = ["query1", "query2"]
        results_list = [sample_results, sample_results[:2]]

        reranked_list = reranker.batch_rerank(queries, results_list)

        assert len(reranked_list) == 2
        assert len(reranked_list[0]) == 3
        assert len(reranked_list[1]) == 2

    def test_rerank_preserves_metadata(self, reranker, sample_results):
        """Test that reranking preserves all metadata."""
        query = "test query"
        original = sample_results[0]
        
        reranked = reranker.rerank(query, sample_results)

        # Find the corresponding result (may be reordered)
        matching = [r for r in reranked if r.chunk_id == original.chunk_id][0]
        
        assert matching.document_id == original.document_id
        assert matching.text == original.text
        assert matching.source_file == original.source_file
        assert matching.page_numbers == original.page_numbers

    def test_rerank_creates_new_objects(self, reranker, sample_results):
        """Test that reranking creates new SearchResult objects."""
        query = "test query"
        
        reranked = reranker.rerank(query, sample_results)

        # Objects should be different instances
        assert reranked[0] is not sample_results[0]
