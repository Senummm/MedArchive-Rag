"""
Document retrieval service for semantic search.

Handles vector search against Qdrant with filtering and result formatting.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
from sentence_transformers import SentenceTransformer

from shared.models import ChunkMetadata, QueryRequest, SearchResult
from shared.utils import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Retrieval service for semantic search against the vector database.

    Supports:
    - Semantic search with embeddings
    - Metadata filtering (source file, date range, etc.)
    - Configurable top-k retrieval
    - Score thresholding
    """

    def __init__(
        self,
        qdrant_url: str = "http://qdrant:6333",
        collection_name: str = "medical_documents",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
    ):
        """
        Initialize the retriever.

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Vector collection name
            embedding_model: Sentence transformer model name
        """
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(
            f"Initialized retriever with collection '{collection_name}' "
            f"and model '{embedding_model}'"
        )

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding vector for a query.

        Args:
            query: Query text

        Returns:
            Embedding vector (normalized)
        """
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Args:
            query: Search query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filters: Optional metadata filters (e.g., {"source_file": "guideline.pdf"})

        Returns:
            List of search results with scores and metadata
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k}, threshold={score_threshold})")

        # Generate query embedding
        query_embedding = self.embed_query(query)

        # Build Qdrant filter if provided
        qdrant_filter = None
        if filters:
            qdrant_filter = self._build_filter(filters)

        # Search
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
            )

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_result = SearchResult(
                    chunk_id=UUID(result.id),
                    document_id=UUID(result.payload["document_id"]),
                    text=result.payload["text"],
                    score=result.score,
                    source_file=result.payload["source_file"],
                    page_numbers=result.payload.get("page_numbers", []),
                    section_path=result.payload.get("section_path"),
                    chunk_index=result.payload["chunk_index"],
                )
                search_results.append(search_result)

            logger.info(f"Found {len(search_results)} results above threshold")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_with_query_request(self, query_request: QueryRequest) -> List[SearchResult]:
        """
        Search using a QueryRequest object.

        Args:
            query_request: Query request with parameters

        Returns:
            List of search results
        """
        return self.search(
            query=query_request.query,
            top_k=query_request.top_k,
            score_threshold=query_request.score_threshold,
            filters=query_request.filters,
        )

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from dictionary.

        Args:
            filters: Filter dictionary

        Returns:
            Qdrant Filter object
        """
        conditions = []

        if "source_file" in filters:
            conditions.append(
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=filters["source_file"]),
                )
            )

        if "document_id" in filters:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(filters["document_id"])),
                )
            )

        # Add more filter types as needed
        if "section_path" in filters:
            conditions.append(
                FieldCondition(
                    key="section_path",
                    match=MatchValue(value=filters["section_path"]),
                )
            )

        return Filter(must=conditions) if conditions else None

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "segments_count": info.segments_count,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        score_threshold: float = 0.5,
    ) -> List[List[SearchResult]]:
        """
        Perform batch search for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of results per query
            score_threshold: Minimum similarity score

        Returns:
            List of result lists (one per query)
        """
        logger.info(f"Batch search for {len(queries)} queries")

        # Generate embeddings for all queries
        query_embeddings = self.embedding_model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        # Build search requests
        search_requests = [
            SearchRequest(
                vector=embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
            )
            for embedding in query_embeddings
        ]

        # Batch search
        try:
            batch_results = self.client.search_batch(
                collection_name=self.collection_name,
                requests=search_requests,
            )

            # Convert to SearchResult objects
            all_results = []
            for results in batch_results:
                search_results = []
                for result in results:
                    search_result = SearchResult(
                        chunk_id=UUID(result.id),
                        document_id=UUID(result.payload["document_id"]),
                        text=result.payload["text"],
                        score=result.score,
                        source_file=result.payload["source_file"],
                        page_numbers=result.payload.get("page_numbers", []),
                        section_path=result.payload.get("section_path"),
                        chunk_index=result.payload["chunk_index"],
                    )
                    search_results.append(search_result)
                all_results.append(search_results)

            return all_results

        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            raise

    def get_document_chunks(self, document_id: UUID) -> List[SearchResult]:
        """
        Retrieve all chunks for a specific document.

        Args:
            document_id: Document UUID

        Returns:
            List of chunks for the document
        """
        logger.info(f"Retrieving chunks for document {document_id}")

        # Use scroll API to get all chunks
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                ),
                limit=1000,  # Max chunks per document
            )

            points, _ = results

            # Convert to SearchResult objects
            chunks = []
            for point in points:
                chunk = SearchResult(
                    chunk_id=UUID(point.id),
                    document_id=UUID(point.payload["document_id"]),
                    text=point.payload["text"],
                    score=1.0,  # No score for direct retrieval
                    source_file=point.payload["source_file"],
                    page_numbers=point.payload.get("page_numbers", []),
                    section_path=point.payload.get("section_path"),
                    chunk_index=point.payload["chunk_index"],
                )
                chunks.append(chunk)

            # Sort by chunk_index
            chunks.sort(key=lambda c: c.chunk_index)

            logger.info(f"Retrieved {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to retrieve document chunks: {e}")
            raise
