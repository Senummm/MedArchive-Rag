"""
Unit tests for the Qdrant indexer module.
"""

import numpy as np
import pytest
from uuid import uuid4

from services.ingestion.src.indexing import QdrantIndexer
from shared.models import ChunkMetadata


class TestQdrantIndexer:
    """Test cases for the QdrantIndexer class."""

    @pytest.fixture
    def indexer(self, test_settings):
        """Create a Qdrant indexer instance for testing."""
        # Use test collection name
        return QdrantIndexer(collection_name="test_indexer_collection")

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunk metadata for testing."""
        doc_id = uuid4()
        chunks = []

        for i in range(3):
            chunk = ChunkMetadata(
                chunk_id=uuid4(),
                document_id=doc_id,
                text=f"Test chunk {i} content",
                chunk_index=i,
                source_file=f"test_doc.pdf",
                page_numbers=[i + 1],
                embedding_model="test-model",
                token_count=10 + i,
            )
            chunks.append(chunk)

        return chunks

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        return np.random.randn(3, 128).astype(np.float32)

    def test_indexer_initialization(self, indexer):
        """Test that the indexer initializes correctly."""
        assert indexer.collection_name == "test_indexer_collection"
        assert indexer.client is not None

    def test_ensure_collection_exists_creates_new(self, indexer):
        """Test creating a new collection."""
        # Delete if exists
        try:
            indexer.client.delete_collection(indexer.collection_name)
        except:
            pass

        # Create collection
        indexer.ensure_collection_exists(vector_size=128)

        # Verify it exists
        collections = indexer.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        assert indexer.collection_name in collection_names

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_ensure_collection_exists_idempotent(self, indexer):
        """Test that ensure_collection_exists is idempotent."""
        # Create collection twice
        indexer.ensure_collection_exists(vector_size=128)
        indexer.ensure_collection_exists(vector_size=128)

        # Should not raise error
        collections = indexer.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        assert indexer.collection_name in collection_names

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_index_chunks(self, indexer, sample_chunks, sample_embeddings):
        """Test indexing chunks."""
        indexer.ensure_collection_exists(vector_size=128)

        count = indexer.index_chunks(sample_chunks, sample_embeddings)

        assert count == 3

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_index_empty_chunks(self, indexer):
        """Test indexing empty list."""
        indexer.ensure_collection_exists(vector_size=128)

        count = indexer.index_chunks([], np.array([]))

        assert count == 0

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_index_chunks_mismatched_count(self, indexer, sample_chunks):
        """Test error when chunk and embedding counts mismatch."""
        indexer.ensure_collection_exists(vector_size=128)

        wrong_embeddings = np.random.randn(2, 128).astype(np.float32)

        with pytest.raises(ValueError, match="Mismatch between number of chunks"):
            indexer.index_chunks(sample_chunks, wrong_embeddings)

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_delete_document_chunks(self, indexer, sample_chunks, sample_embeddings):
        """Test deleting chunks by document ID."""
        indexer.ensure_collection_exists(vector_size=128)

        # Index chunks
        indexer.index_chunks(sample_chunks, sample_embeddings)

        # Delete by document ID
        doc_id = sample_chunks[0].document_id
        deleted_count = indexer.delete_document_chunks(doc_id)

        assert deleted_count == 3

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_delete_nonexistent_document(self, indexer):
        """Test deleting chunks for non-existent document."""
        indexer.ensure_collection_exists(vector_size=128)

        doc_id = uuid4()
        deleted_count = indexer.delete_document_chunks(doc_id)

        # Should return 0, not error
        assert deleted_count == 0

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_search(self, indexer, sample_chunks, sample_embeddings):
        """Test searching for similar chunks."""
        indexer.ensure_collection_exists(vector_size=128)

        # Index chunks
        indexer.index_chunks(sample_chunks, sample_embeddings)

        # Search with similar vector
        query_vector = sample_embeddings[0] + np.random.randn(128) * 0.01
        query_vector = query_vector.astype(np.float32)

        results = indexer.search(query_vector, top_k=2)

        assert len(results) <= 2
        assert all("id" in result for result in results)
        assert all("score" in result for result in results)
        assert all("payload" in result for result in results)

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_search_empty_collection(self, indexer):
        """Test searching in empty collection."""
        indexer.ensure_collection_exists(vector_size=128)

        query_vector = np.random.randn(128).astype(np.float32)
        results = indexer.search(query_vector, top_k=5)

        assert len(results) == 0

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_search_with_filter(self, indexer, sample_chunks, sample_embeddings):
        """Test searching with metadata filter."""
        indexer.ensure_collection_exists(vector_size=128)

        # Index chunks
        indexer.index_chunks(sample_chunks, sample_embeddings)

        # Search with filter
        query_vector = sample_embeddings[0].astype(np.float32)
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value="test_doc.pdf"),
                )
            ]
        )

        results = indexer.search(query_vector, top_k=5, query_filter=search_filter)

        assert all(result["payload"]["source_file"] == "test_doc.pdf" for result in results)

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_get_collection_info(self, indexer, sample_chunks, sample_embeddings):
        """Test getting collection information."""
        indexer.ensure_collection_exists(vector_size=128)
        indexer.index_chunks(sample_chunks, sample_embeddings)

        info = indexer.get_collection_info()

        assert info is not None
        assert "vectors_count" in info
        assert info["vectors_count"] == 3

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_get_collection_info_nonexistent(self, indexer):
        """Test getting info for non-existent collection."""
        # Ensure collection doesn't exist
        try:
            indexer.client.delete_collection(indexer.collection_name)
        except:
            pass

        info = indexer.get_collection_info()
        assert info is None

    def test_atomic_update(self, indexer, sample_chunks, sample_embeddings):
        """Test atomic document update (delete + insert)."""
        indexer.ensure_collection_exists(vector_size=128)

        # Index initial chunks
        doc_id = sample_chunks[0].document_id
        indexer.index_chunks(sample_chunks, sample_embeddings)

        # Create updated chunks with same document ID
        updated_chunks = []
        for i in range(2):  # Fewer chunks this time
            chunk = ChunkMetadata(
                chunk_id=uuid4(),
                document_id=doc_id,
                text=f"Updated chunk {i}",
                chunk_index=i,
                source_file="test_doc.pdf",
                page_numbers=[i + 1],
                embedding_model="test-model",
                token_count=15,
            )
            updated_chunks.append(chunk)

        updated_embeddings = np.random.randn(2, 128).astype(np.float32)

        # Atomic update: delete old + insert new
        indexer.delete_document_chunks(doc_id)
        indexer.index_chunks(updated_chunks, updated_embeddings)

        # Verify count
        info = indexer.get_collection_info()
        assert info["vectors_count"] == 2

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)

    def test_batch_indexing(self, indexer):
        """Test indexing large batch of chunks."""
        indexer.ensure_collection_exists(vector_size=128)

        # Create many chunks
        doc_id = uuid4()
        chunks = []
        for i in range(100):
            chunk = ChunkMetadata(
                chunk_id=uuid4(),
                document_id=doc_id,
                text=f"Chunk {i}",
                chunk_index=i,
                source_file="large_doc.pdf",
                page_numbers=[i // 10 + 1],
                embedding_model="test-model",
                token_count=10,
            )
            chunks.append(chunk)

        embeddings = np.random.randn(100, 128).astype(np.float32)

        count = indexer.index_chunks(chunks, embeddings, batch_size=25)

        assert count == 100

        # Cleanup
        indexer.client.delete_collection(indexer.collection_name)
