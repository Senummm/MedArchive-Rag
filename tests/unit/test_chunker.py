"""
Unit tests for SemanticChunker.

Tests text splitting, section extraction, and chunk metadata creation.
"""

from uuid import uuid4

import pytest

from services.ingestion.src.chunking import SemanticChunker
from shared.models import ChunkMetadata


class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a chunker with test settings."""
        return SemanticChunker(
            chunk_size=100,  # Small for testing
            chunk_overlap=20,
            min_chunk_size=10,
        )

    @pytest.fixture
    def sample_markdown(self):
        """Sample markdown text with headers."""
        return """
# Introduction

This is the introduction section with some text about the topic.

## Background

More detailed background information goes here. This section has
multiple paragraphs to test chunking.

# Methods

## Study Design

The study design is described in detail here.

### Participants

Information about the study participants.
"""

    def test_chunker_initialization(self, chunker):
        """Test chunker initializes with correct settings."""
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20
        assert chunker.min_chunk_size == 10

    def test_chunk_simple_text(self, chunker):
        """Test chunking simple text without headers."""
        text = "This is a simple text. " * 10  # ~230 characters
        document_id = uuid4()

        chunks = chunker.chunk_text(text, document_id)

        assert len(chunks) > 0
        assert all(isinstance(chunk, ChunkMetadata) for chunk in chunks)
        assert all(chunk.document_id == document_id for chunk in chunks)

    def test_chunk_with_headers(self, chunker, sample_markdown):
        """Test that headers are properly extracted into section paths."""
        document_id = uuid4()

        chunks = chunker.chunk_text(sample_markdown, document_id)

        assert len(chunks) > 0

        # Check that section paths are populated
        section_paths = [chunk.section_path for chunk in chunks if chunk.section_path]
        assert len(section_paths) > 0

        # Check hierarchical paths exist
        assert any(">" in path for path in section_paths)

    def test_chunk_indices_are_sequential(self, chunker):
        """Test that chunk indices are sequential."""
        text = "Test sentence. " * 20
        document_id = uuid4()

        chunks = chunker.chunk_text(text, document_id)

        indices = [chunk.chunk_index for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_text_returns_no_chunks(self, chunker):
        """Test that empty text returns no chunks."""
        document_id = uuid4()

        chunks = chunker.chunk_text("", document_id)

        assert len(chunks) == 0

    def test_very_short_text_returns_no_chunks(self, chunker):
        """Test that text shorter than min_chunk_size returns no chunks."""
        document_id = uuid4()

        chunks = chunker.chunk_text("short", document_id)  # Less than 10 chars

        assert len(chunks) == 0

    def test_token_count_estimation(self, chunker):
        """Test that token counts are estimated."""
        text = "This is a test sentence with multiple words."
        document_id = uuid4()

        chunks = chunker.chunk_text(text, document_id)

        assert len(chunks) > 0
        assert all(chunk.token_count > 0 for chunk in chunks)

    def test_extract_sections_single_level(self, chunker):
        """Test section extraction with single-level headers."""
        text = """
# Section 1
Content 1

# Section 2
Content 2
"""
        sections = chunker._extract_sections(text)

        assert len(sections) == 2
        assert sections[0]["section_path"] == "Section 1"
        assert sections[1]["section_path"] == "Section 2"

    def test_extract_sections_nested(self, chunker):
        """Test section extraction with nested headers."""
        text = """
# Chapter 1
Content

## Section 1.1
More content

### Subsection 1.1.1
Even more content
"""
        sections = chunker._extract_sections(text)

        # Check hierarchical paths
        paths = [s["section_path"] for s in sections]
        assert any("Chapter 1 > Section 1.1" in path for path in paths)
        assert any("Subsection 1.1.1" in path for path in paths)

    def test_chunk_metadata_has_required_fields(self, chunker):
        """Test that all chunks have required metadata fields."""
        text = "Test content with sufficient length for chunking purposes."
        document_id = uuid4()

        chunks = chunker.chunk_text(text, document_id)

        for chunk in chunks:
            assert chunk.document_id == document_id
            assert isinstance(chunk.chunk_index, int)
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0
            assert chunk.token_count > 0
