"""
Reranking service using cross-encoder models.

Cross-encoders provide more accurate relevance scores than bi-encoders
by jointly encoding query and document.
"""

import logging
from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder

from shared.models import SearchResult
from shared.utils import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Reranking service using cross-encoder models.
    
    Cross-encoders are more accurate than bi-encoders (like the retrieval model)
    but are slower, so we use them as a second-stage reranker after initial retrieval.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.

        Args:
            model_name: Cross-encoder model name from sentence-transformers
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        logger.info(f"Initialized reranker with model '{model_name}'")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = None,
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original query text
            results: Initial search results from retriever
            top_k: Number of results to return after reranking (None = all)

        Returns:
            Reranked results with updated scores
        """
        if not results:
            return []

        logger.info(f"Reranking {len(results)} results")

        # Prepare query-document pairs
        pairs = [(query, result.text) for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Update results with new scores
        reranked_results = []
        for result, score in zip(results, scores):
            # Create new result with updated score
            reranked_result = SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                score=float(score),  # Cross-encoder score
                source_file=result.source_file,
                page_numbers=result.page_numbers,
                section_path=result.section_path,
                chunk_index=result.chunk_index,
            )
            reranked_results.append(reranked_result)

        # Sort by new scores (descending)
        reranked_results.sort(key=lambda r: r.score, reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]

        logger.info(f"Reranking complete, returning {len(reranked_results)} results")
        return reranked_results

    def compute_score(self, query: str, text: str) -> float:
        """
        Compute relevance score for a single query-text pair.

        Args:
            query: Query text
            text: Document text

        Returns:
            Relevance score (higher = more relevant)
        """
        score = self.model.predict([(query, text)])
        return float(score[0])

    def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[SearchResult]],
        top_k: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """
        Rerank multiple sets of results in batch.

        Args:
            queries: List of query strings
            results_list: List of result lists (one per query)
            top_k: Number of results to return per query

        Returns:
            List of reranked result lists
        """
        logger.info(f"Batch reranking for {len(queries)} queries")

        reranked_lists = []
        for query, results in zip(queries, results_list):
            reranked = self.rerank(query, results, top_k)
            reranked_lists.append(reranked)

        return reranked_lists
