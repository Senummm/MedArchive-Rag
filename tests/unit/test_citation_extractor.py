"""
Unit tests for the citation extractor.
"""

from uuid import uuid4

import pytest

from services.api.src.citations.extractor import CitationExtractor
from shared.models import SearchResult, Citation


class TestCitationExtractor:
    """Test cases for the CitationExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a citation extractor."""
        return CitationExtractor()

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        doc_id = uuid4()
        results = []
        for i in range(3):
            result = SearchResult(
                chunk_id=uuid4(),
                document_id=doc_id,
                text=f"Sample medical text about treatment {i}. Important clinical information.",
                score=0.9 - i * 0.1,
                source_file=f"guideline_{i}.pdf",
                page_numbers=[10 + i, 11 + i],
                chunk_index=i,
            )
            results.append(result)
        return results

    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.source_pattern is not None

    def test_extract_citations_from_answer(self, extractor, sample_results):
        """Test extracting citations from answer with [Source N] references."""
        answer = "According to [Source 1], the treatment is effective. [Source 2] confirms this."

        citations = extractor.extract_citations(answer, sample_results)

        assert len(citations) == 2
        assert all(isinstance(c, Citation) for c in citations)

    def test_extract_citations_no_references(self, extractor, sample_results):
        """Test extraction when answer has no source references."""
        answer = "This is an answer without any source citations."

        citations = extractor.extract_citations(answer, sample_results)

        assert len(citations) == 0

    def test_extract_citations_with_titles(self, extractor, sample_results):
        """Test extraction with document titles provided."""
        answer = "See [Source 1] for details."
        doc_titles = {sample_results[0].document_id: "Clinical Guidelines 2026"}

        citations = extractor.extract_citations(answer, sample_results, doc_titles)

        assert len(citations) == 1
        assert citations[0].document_title == "Clinical Guidelines 2026"

    def test_extract_citations_uses_filename_fallback(self, extractor, sample_results):
        """Test that filename is used when title not provided."""
        answer = "See [Source 2]."

        citations = extractor.extract_citations(answer, sample_results)

        assert citations[0].document_title == "guideline_1.pdf"

    def test_format_citations_numeric(self, extractor):
        """Test numeric citation formatting."""
        citations = [
            Citation(
                document_id=uuid4(),
                document_title="Document 1",
                page_numbers=[5, 6],
                text_snippet="Sample text",
                relevance_score=0.95,
            ),
            Citation(
                document_id=uuid4(),
                document_title="Document 2",
                page_numbers=[10],
                text_snippet="More text",
                relevance_score=0.85,
            ),
        ]

        formatted = extractor.format_citations(citations, style="numeric")

        assert "1. Document 1" in formatted
        assert "2. Document 2" in formatted
        assert "Page 5, 6" in formatted

    def test_format_citations_apa(self, extractor):
        """Test APA citation formatting."""
        citations = [
            Citation(
                document_id=uuid4(),
                document_title="Clinical Guidelines",
                page_numbers=[15],
                text_snippet="Text",
                relevance_score=0.9,
            ),
        ]

        formatted = extractor.format_citations(citations, style="apa")

        assert "Clinical Guidelines" in formatted
        assert "(p. 15)" in formatted

    def test_format_citations_footnote(self, extractor):
        """Test footnote citation formatting."""
        citations = [
            Citation(
                document_id=uuid4(),
                document_title="Guidelines",
                page_numbers=[20],
                text_snippet="Important clinical information",
                relevance_score=0.9,
            ),
        ]

        formatted = extractor.format_citations(citations, style="footnote")

        assert "[1]" in formatted
        assert "Guidelines" in formatted
        assert "Important clinical information" in formatted

    def test_create_snippet_short_text(self, extractor):
        """Test snippet creation with short text."""
        text = "Short medical text."

        snippet = extractor._create_snippet(text, max_length=200)

        assert snippet == "Short medical text."

    def test_create_snippet_long_text(self, extractor):
        """Test snippet creation with long text."""
        text = "This is a very long medical text. " * 20

        snippet = extractor._create_snippet(text, max_length=100)

        assert len(snippet) <= 110  # Allow some buffer for word boundaries

    def test_create_snippet_breaks_at_sentence(self, extractor):
        """Test that snippet breaks at sentence boundary."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        snippet = extractor._create_snippet(text, max_length=50)

        assert snippet.endswith(".")

    def test_deduplicate_citations(self, extractor):
        """Test citation deduplication."""
        doc_id = uuid4()
        citations = [
            Citation(
                document_id=doc_id,
                document_title="Doc",
                page_numbers=[5],
                text_snippet="Text",
                relevance_score=0.9,
            ),
            Citation(
                document_id=doc_id,
                document_title="Doc",
                page_numbers=[5],  # Same page
                text_snippet="Different text",
                relevance_score=0.8,
            ),
            Citation(
                document_id=doc_id,
                document_title="Doc",
                page_numbers=[6],  # Different page
                text_snippet="Text",
                relevance_score=0.85,
            ),
        ]

        unique = extractor.deduplicate_citations(citations)

        assert len(unique) == 2  # First and third should remain

    def test_group_by_document(self, extractor):
        """Test grouping citations by document."""
        doc_id1 = uuid4()
        doc_id2 = uuid4()
        citations = [
            Citation(
                document_id=doc_id1,
                document_title="Doc1",
                page_numbers=[5],
                text_snippet="Text",
                relevance_score=0.9,
            ),
            Citation(
                document_id=doc_id2,
                document_title="Doc2",
                page_numbers=[10],
                text_snippet="Text",
                relevance_score=0.85,
            ),
            Citation(
                document_id=doc_id1,
                document_title="Doc1",
                page_numbers=[6],
                text_snippet="Text",
                relevance_score=0.8,
            ),
        ]

        grouped = extractor.group_by_document(citations)

        assert len(grouped) == 2
        assert len(grouped[doc_id1]) == 2
        assert len(grouped[doc_id2]) == 1

    def test_merge_page_ranges(self, extractor):
        """Test merging page numbers into ranges."""
        pages = [1, 2, 3, 5, 7, 8, 9, 12]

        merged = extractor.merge_page_ranges(pages)

        assert merged == "1-3, 5, 7-9, 12"

    def test_merge_page_ranges_single_page(self, extractor):
        """Test merging with single page."""
        pages = [5]

        merged = extractor.merge_page_ranges(pages)

        assert merged == "5"

    def test_merge_page_ranges_empty(self, extractor):
        """Test merging with no pages."""
        merged = extractor.merge_page_ranges([])

        assert merged == "N/A"

    def test_add_inline_citations(self, extractor, sample_results):
        """Test adding inline citations to answer."""
        answer = "This is an answer about treatment."

        with_citations = extractor.add_inline_citations(answer, sample_results)

        assert "[Source 1]" in with_citations
        assert "[Source 2]" in with_citations
        assert "[Source 3]" in with_citations
