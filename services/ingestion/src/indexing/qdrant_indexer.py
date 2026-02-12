"""
Qdrant Indexer for Vector Storage.

Handles creation and management of Qdrant collections and indexing
of embedded chunks with metadata.
"""

from typing import List, Optional
from uuid import UUID

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from shared.models import ChunkMetadata
from shared.utils import get_settings, setup_logging

settings = get_settings()
logger = setup_logging("ingestion.indexer", settings.log_level)


class QdrantIndexer:
    """
    Qdrant vector database indexer.

    Manages collection creation, document indexing, and metadata storage
    in Qdrant for semantic search.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize Qdrant client and setup collection.

        Args:
            url: Qdrant server URL (defaults to settings)
            api_key: Qdrant API key for cloud (defaults to settings)
            collection_name: Collection name (defaults to settings)
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection_name

        logger.info(
            "Initializing Qdrant client",
            extra={
                "url": self.url,
                "collection": self.collection_name,
            },
        )

        try:
            # Initialize client
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=60,
            )

            # Verify connection
            collections = self.client.get_collections()
            logger.info(
                "Connected to Qdrant successfully",
                extra={"collections_count": len(collections.collections)},
            )

        except Exception as e:
            logger.error(
                "Failed to connect to Qdrant",
                exc_info=e,
                extra={"url": self.url},
            )
            raise

    def ensure_collection_exists(self, vector_size: int) -> None:
        """
        Create collection if it doesn't exist.

        Args:
            vector_size: Dimension of embedding vectors
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return

            # Create collection
            logger.info(
                f"Creating collection '{self.collection_name}'",
                extra={"vector_size": vector_size},
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

            logger.info(f"Collection '{self.collection_name}' created successfully")

        except Exception as e:
            logger.error(
                "Failed to create collection",
                exc_info=e,
                extra={"collection": self.collection_name},
            )
            raise

    def index_chunks(
        self,
        chunks: List[ChunkMetadata],
        embeddings: np.ndarray,
    ) -> int:
        """
        Index chunks with their embeddings into Qdrant.

        Args:
            chunks: List of ChunkMetadata objects
            embeddings: NumPy array of embeddings (shape: len(chunks) x embedding_dim)

        Returns:
            Number of chunks successfully indexed

        Raises:
            ValueError: If chunks and embeddings length mismatch
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) doesn't match embeddings count ({len(embeddings)})"
            )

        if not chunks:
            logger.warning("No chunks to index")
            return 0

        logger.info(
            f"Indexing {len(chunks)} chunks into Qdrant",
            extra={"collection": self.collection_name},
        )

        try:
            # Convert to Qdrant points
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point = PointStruct(
                    id=str(chunk.chunk_id),
                    vector=embedding.tolist(),
                    payload={
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "section_path": chunk.section_path,
                        "heading": chunk.heading,
                        "page_numbers": chunk.page_numbers,
                        "token_count": chunk.token_count,
                        "embedding_model": chunk.embedding_model,
                        "created_at": chunk.created_at.isoformat(),
                        # Add document metadata for citations
                        "document_title": getattr(chunk, 'document_title', None),
                        "source_file": getattr(chunk, 'source_file', None),
                    },
                )
                points.append(point)

            # Upsert points (batch operation)
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(
                f"Successfully indexed {len(chunks)} chunks",
                extra={
                    "collection": self.collection_name,
                    "total_chunks": len(chunks),
                },
            )

            return len(chunks)

        except Exception as e:
            logger.error(
                "Failed to index chunks",
                exc_info=e,
                extra={
                    "collection": self.collection_name,
                    "chunk_count": len(chunks),
                },
            )
            raise

    def delete_document_chunks(self, document_id: UUID) -> int:
        """
        Delete all chunks belonging to a document.

        Used when a document is updated to perform atomic replacement.

        Args:
            document_id: UUID of the document

        Returns:
            Number of chunks deleted
        """
        try:
            logger.info(
                f"Deleting chunks for document {document_id}",
                extra={"collection": self.collection_name},
            )

            # Delete by filter
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                ),
            )

            logger.info(f"Deleted chunks for document {document_id}")
            return 0  # Qdrant doesn't return count directly

        except Exception as e:
            logger.error(
                f"Failed to delete chunks for document {document_id}",
                exc_info=e,
            )
            raise

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.name,
                },
            }

        except Exception as e:
            logger.error(
                f"Failed to get collection info for '{self.collection_name}'",
                exc_info=e,
            )
            return {}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[dict]:
        """
        Search for similar chunks.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of search results with scores and payloads
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
            )

            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                }
                for result in results
            ]

        except Exception as e:
            logger.error(
                "Search failed",
                exc_info=e,
                extra={"collection": self.collection_name},
            )
            raise

    def count_documents(self) -> int:
        """
        Count total number of points in collection.

        Returns:
            Number of indexed chunks
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0
