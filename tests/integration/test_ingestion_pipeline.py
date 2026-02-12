"""
Integration tests for the ingestion pipeline.

Tests end-to-end PDF processing: parsing → chunking → embedding → indexing.
"""

import asyncio
from pathlib import Path

import pytest

from services.ingestion.src.chunking import SemanticChunker
from services.ingestion.src.embedding import Embedder
from services.ingestion.src.indexing import QdrantIndexer
from services.ingestion.src.parsers.pdf_parser import PDFParser
from services.ingestion.src.sync import FileTracker
from shared.models import ProcessingStatus


@pytest.mark.integration
@pytest.mark.requires_api_keys
class TestIngestionPipeline:
    """Integration tests for the full ingestion pipeline."""

    @pytest.fixture(scope="class")
    async def pipeline_components(self, test_settings):
        """Initialize all pipeline components."""
        # Skip if API keys not available
        if not test_settings.llamaparse_api_key or test_settings.llamaparse_api_key == "test_llamaparse_key":
            pytest.skip("LlamaParse API key not configured")

        parser = PDFParser()
        chunker = SemanticChunker()
        embedder = Embedder()
        indexer = QdrantIndexer()
        file_tracker = FileTracker()

        # Ensure collection exists
        indexer.ensure_collection_exists(embedder.get_embedding_dimension())

        return {
            "parser": parser,
            "chunker": chunker,
            "embedder": embedder,
            "indexer": indexer,
            "file_tracker": file_tracker,
        }

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_parse_sample_pdf(self, pipeline_components, temp_document_store):
        """Test parsing a sample PDF file."""
        parser = pipeline_components["parser"]

        # Create a simple test PDF (or use a sample file)
        # For this test, we'll create markdown text
        sample_pdf = temp_document_store / "sample_guideline.pdf"

        # Skip if no sample PDF available
        if not sample_pdf.exists():
            pytest.skip("Sample PDF not available")

        # Parse the PDF
        result = await parser.parse_pdf_with_metadata(sample_pdf)

        assert "text" in result
        assert "metadata" in result
        assert len(result["text"]) > 0
        assert result["metadata"]["page_count"] is not None

    @pytest.mark.asyncio
    async def test_chunk_parsed_text(self, pipeline_components):
        """Test chunking parsed text."""
        chunker = pipeline_components["chunker"]

        sample_text = """
# Clinical Guideline: Hypertension Management

## Introduction
Hypertension affects millions of patients worldwide.

## Treatment Protocol

### First-Line Agents
- ACE Inhibitors
- ARBs
- Thiazide diuretics

### Dosing Guidelines
| Drug | Starting Dose | Maximum Dose |
|------|--------------|--------------|
| Lisinopril | 10mg daily | 40mg daily |
| Losartan | 50mg daily | 100mg daily |
"""

        from uuid import uuid4

        document_id = uuid4()
        chunks = chunker.chunk_text(sample_text, document_id)

        assert len(chunks) > 0
        assert all(chunk.document_id == document_id for chunk in chunks)
        assert any(chunk.section_path is not None for chunk in chunks)

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, pipeline_components):
        """Test embedding generation."""
        embedder = pipeline_components["embedder"]

        texts = [
            "Hypertension is a common condition.",
            "ACE inhibitors are first-line agents.",
            "Monitor blood pressure regularly.",
        ]

        embeddings = embedder.embed_batch(texts)

        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedder.get_embedding_dimension()

    @pytest.mark.asyncio
    async def test_index_to_qdrant(self, pipeline_components):
        """Test indexing chunks to Qdrant."""
        chunker = pipeline_components["chunker"]
        embedder = pipeline_components["embedder"]
        indexer = pipeline_components["indexer"]

        from uuid import uuid4

        # Create test chunks
        document_id = uuid4()
        sample_text = "Test content for indexing. " * 10
        chunks = chunker.chunk_text(sample_text, document_id)

        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_batch(chunk_texts)

        # Set embedding model on chunks
        for chunk in chunks:
            chunk.embedding_model = embedder.model_name

        # Index
        indexed_count = indexer.index_chunks(chunks, embeddings)

        assert indexed_count == len(chunks)

        # Verify we can search
        query_embedding = embedder.embed_text("test content")
        results = indexer.search(query_embedding, top_k=5)

        assert len(results) > 0
        assert results[0]["score"] > 0.5  # Should have decent similarity

@pytest.mark.integration
class TestQdrantConnection:
    """Integration tests for Qdrant connectivity."""

    def test_qdrant_health(self):
        """Test that Qdrant is accessible."""
        from qdrant_client import QdrantClient

        try:
            client = QdrantClient(url="http://qdrant:6333", timeout=5)
            collections = client.get_collections()
            assert collections is not None
        except Exception as e:
            pytest.fail(f"Qdrant connection failed: {e}")

    def test_create_and_delete_collection(self):
        """Test creating and deleting a test collection."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(url="http://qdrant:6333")
        test_collection = "test_collection_temp"

        try:
            # Create collection
            client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=128, distance=Distance.COSINE),
            )

            # Verify it exists
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            assert test_collection in collection_names

            # Delete collection
            client.delete_collection(test_collection)

            # Verify deletion
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            assert test_collection not in collection_names

        except Exception as e:
            # Cleanup on failure
            try:
                client.delete_collection(test_collection)
            except:
                pass
            pytest.fail(f"Collection test failed: {e}")


@pytest.mark.integration
class TestFileTracking:
    """Integration tests for file tracking."""

    def test_track_multiple_files(self, temp_document_store, tmp_path):
        """Test tracking multiple files through processing."""
        from uuid import uuid4

        tracker = FileTracker(tracking_file=tmp_path / "tracking.json")

        # Create test files
        files = []
        for i in range(3):
            file = temp_document_store / f"doc{i}.pdf"
            file.write_bytes(f"Content {i}".encode())
            files.append(file)

        # All should need processing initially
        to_process = tracker.get_files_to_process(temp_document_store)
        assert len(to_process) == 3

        # Mark first two as processed
        for i, file in enumerate(files[:2]):
            tracker.mark_file_processed(
                file,
                document_id=str(uuid4()),
                chunk_count=5 + i,
            )

        # Only one should need processing now
        to_process = tracker.get_files_to_process(temp_document_store)
        assert len(to_process) == 1
        assert to_process[0].name == "doc2.pdf"
