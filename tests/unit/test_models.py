"""
Unit tests for Pydantic data models.

Validates model creation, validation, and serialization.
"""

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from shared.models import (
    ChunkMetadata,
    Citation,
    DocumentMetadata,
    DocumentType,
    ProcessingStatus,
    QueryRequest,
    QueryResponse,
)


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_create_document_metadata(self):
        """Test creating a valid DocumentMetadata instance."""
        doc = DocumentMetadata(
            title="Test Document",
            document_type=DocumentType.CLINICAL_GUIDELINE,
            source_path="/data/test.pdf",
            file_hash="abc123",
            department="Cardiology",
            page_count=25,
        )

        assert doc.title == "Test Document"
        assert doc.document_type == DocumentType.CLINICAL_GUIDELINE
        assert doc.department == "Cardiology"
        assert doc.page_count == 25
        assert isinstance(doc.document_id, UUID)
        assert doc.processing_status == ProcessingStatus.PENDING

    def test_document_metadata_with_dates(self):
        """Test DocumentMetadata with effective and review dates."""
        effective_date = datetime(2026, 1, 1)
        review_date = datetime(2027, 1, 1)

        doc = DocumentMetadata(
            title="Policy Document",
            document_type=DocumentType.POLICY,
            source_path="/data/policy.pdf",
            file_hash="xyz789",
            effective_date=effective_date,
            review_date=review_date,
        )

        assert doc.effective_date == effective_date
        assert doc.review_date == review_date

    def test_document_metadata_invalid_page_count(self):
        """Test that page_count must be >= 1."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentMetadata(
                title="Invalid Doc",
                document_type=DocumentType.FORMULARY,
                source_path="/data/test.pdf",
                file_hash="abc",
                page_count=0,  # Invalid: must be >= 1
            )

        assert "page_count" in str(exc_info.value)


class TestChunkMetadata:
    """Tests for ChunkMetadata model."""

    def test_create_chunk_metadata(self, sample_document_metadata):
        """Test creating a valid ChunkMetadata instance."""
        chunk = ChunkMetadata(
            document_id=sample_document_metadata.document_id,
            chunk_index=0,
            text="This is a test chunk with meaningful content.",
            section_path="Chapter 1 > Introduction",
            heading="Introduction",
            page_numbers=[1, 2],
            token_count=10,
        )

        assert chunk.chunk_index == 0
        assert "test chunk" in chunk.text
        assert chunk.section_path == "Chapter 1 > Introduction"
        assert chunk.page_numbers == [1, 2]
        assert isinstance(chunk.chunk_id, UUID)

    def test_chunk_metadata_empty_text_fails(self, sample_document_metadata):
        """Test that empty or whitespace-only text is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkMetadata(
                document_id=sample_document_metadata.document_id,
                chunk_index=0,
                text="   ",  # Whitespace only
            )

        assert "Chunk text cannot be empty" in str(exc_info.value)

    def test_chunk_metadata_minimum_text_length(self, sample_document_metadata):
        """Test minimum text length validation."""
        with pytest.raises(ValidationError):
            ChunkMetadata(
                document_id=sample_document_metadata.document_id,
                chunk_index=0,
                text="short",  # Less than 10 characters
            )


class TestQueryRequest:
    """Tests for QueryRequest model."""

    def test_create_query_request(self):
        """Test creating a valid QueryRequest."""
        request = QueryRequest(
            query="What is the dosage for Aspirin?",
            top_k=5,
            enable_reranking=True,
        )

        assert request.query == "What is the dosage for Aspirin?"
        assert request.top_k == 5
        assert request.enable_reranking is True

    def test_query_request_with_filters(self):
        """Test QueryRequest with metadata filters."""
        filters = {"department": "Cardiology", "document_type": "formulary"}
        request = QueryRequest(
            query="Beta blocker guidelines",
            filters=filters,
        )

        assert request.filters == filters

    def test_query_request_minimum_length(self):
        """Test that query must be at least 3 characters."""
        with pytest.raises(ValidationError):
            QueryRequest(query="ab")  # Too short

    def test_query_request_top_k_bounds(self):
        """Test top_k must be within valid range."""
        # Should work
        QueryRequest(query="test query", top_k=1)
        QueryRequest(query="test query", top_k=20)

        # Should fail
        with pytest.raises(ValidationError):
            QueryRequest(query="test query", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="test query", top_k=21)


class TestQueryResponse:
    """Tests for QueryResponse model."""

    def test_create_query_response(self):
        """Test creating a valid QueryResponse."""
        citation = Citation(
            document_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            document_title="Test Formulary",
            page_numbers=[12],
            text_snippet="Aspirin 81mg daily",
            relevance_score=0.95,
        )

        response = QueryResponse(
            query="What is the dosage for Aspirin?",
            answer="The standard dosage for Aspirin is 81mg daily for cardioprotection.",
            citations=[citation],
            latency_ms=287.3,
            model_used="llama-3.3-70b-versatile",
        )

        assert response.query == "What is the dosage for Aspirin?"
        assert "81mg daily" in response.answer
        assert len(response.citations) == 1
        assert response.citations[0].relevance_score == 0.95
        assert isinstance(response.timestamp, datetime)


class TestEnums:
    """Tests for enumeration types."""

    def test_document_type_enum(self):
        """Test DocumentType enum values."""
        assert DocumentType.CLINICAL_GUIDELINE == "clinical_guideline"
        assert DocumentType.FORMULARY == "formulary"
        assert DocumentType.PROTOCOL == "protocol"

    def test_processing_status_enum(self):
        """Test ProcessingStatus enum values."""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PARSING == "parsing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
