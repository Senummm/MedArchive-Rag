"""
Unit tests for the PDF parser module.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from services.ingestion.src.parsers.pdf_parser import PDFParser


class TestPDFParser:
    """Test cases for the PDFParser class."""

    @pytest.fixture
    def parser(self, test_settings):
        """Create a parser instance for testing."""
        return PDFParser()

    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a sample PDF file for testing."""
        pdf_path = tmp_path / "sample.pdf"
        # Create a minimal valid PDF
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Test content) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
trailer<</Size 5/Root 1 0 R>>
startxref
306
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        return pdf_path

    def test_parser_initialization(self, parser, test_settings):
        """Test that the parser initializes correctly."""
        assert parser.api_key == test_settings.llamaparse_api_key
        assert parser.parser is not None

    @pytest.mark.asyncio
    @pytest.mark.requires_api_keys
    async def test_parse_pdf_returns_text(self, parser):
        """Test that parse_pdf returns text content."""
        # Skip if no API key
        if parser.api_key == "test_llamaparse_key":
            pytest.skip("LlamaParse API key not configured")

        # Mock the llamaparse response
        with patch.object(parser.parser, 'aload_data', new_callable=AsyncMock) as mock_load:
            mock_doc = Mock()
            mock_doc.text = "Sample parsed text from PDF"
            mock_load.return_value = [mock_doc]

            result = await parser.parse_pdf(Path("test.pdf"))

            assert result == "Sample parsed text from PDF"
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_pdf_empty_result(self, parser):
        """Test handling of empty parse result."""
        with patch.object(parser.parser, 'aload_data', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = []

            result = await parser.parse_pdf(Path("test.pdf"))

            assert result == ""

    @pytest.mark.asyncio
    async def test_parse_pdf_multiple_pages(self, parser):
        """Test parsing PDF with multiple pages."""
        with patch.object(parser.parser, 'aload_data', new_callable=AsyncMock) as mock_load:
            mock_doc1 = Mock()
            mock_doc1.text = "Page 1 content"
            mock_doc2 = Mock()
            mock_doc2.text = "Page 2 content"
            mock_load.return_value = [mock_doc1, mock_doc2]

            result = await parser.parse_pdf(Path("test.pdf"))

            # Should concatenate pages
            assert "Page 1 content" in result
            assert "Page 2 content" in result

    def test_extract_pdf_metadata(self, parser, sample_pdf_path):
        """Test extracting metadata from PDF."""
        metadata = parser._extract_pdf_metadata(sample_pdf_path)

        assert "page_count" in metadata
        assert "file_size" in metadata
        assert metadata["page_count"] == 1
        assert metadata["file_size"] > 0

    def test_extract_metadata_nonexistent_file(self, parser):
        """Test metadata extraction for non-existent file."""
        metadata = parser._extract_pdf_metadata(Path("nonexistent.pdf"))

        # Should return empty dict on error
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_parse_pdf_with_metadata(self, parser, sample_pdf_path):
        """Test parsing PDF with metadata extraction."""
        with patch.object(parser.parser, 'aload_data', new_callable=AsyncMock) as mock_load:
            mock_doc = Mock()
            mock_doc.text = "Sample text"
            mock_load.return_value = [mock_doc]

            result = await parser.parse_pdf_with_metadata(sample_pdf_path)

            assert "text" in result
            assert "metadata" in result
            assert result["text"] == "Sample text"
            assert result["metadata"]["page_count"] == 1

    @pytest.mark.asyncio
    async def test_parse_pdf_with_metadata_includes_filename(self, parser, sample_pdf_path):
        """Test that metadata includes filename."""
        with patch.object(parser.parser, 'aload_data', new_callable=AsyncMock) as mock_load:
            mock_doc = Mock()
            mock_doc.text = "Test"
            mock_load.return_value = [mock_doc]

            result = await parser.parse_pdf_with_metadata(sample_pdf_path)

            assert result["metadata"]["file_name"] == "sample.pdf"

    @pytest.mark.asyncio
    async def test_parse_pdf_error_handling(self, parser):
        """Test error handling during parsing."""
        with patch.object(parser.parser, 'aload_data', new_callable=AsyncMock) as mock_load:
            mock_load.side_effect = Exception("Parse error")

            with pytest.raises(Exception, match="Parse error"):
                await parser.parse_pdf(Path("test.pdf"))

    def test_extract_metadata_with_pypdf_error(self, parser, tmp_path):
        """Test metadata extraction handles pypdf errors gracefully."""
        # Create an invalid PDF
        invalid_pdf = tmp_path / "invalid.pdf"
        invalid_pdf.write_bytes(b"Not a PDF file")

        metadata = parser._extract_pdf_metadata(invalid_pdf)

        # Should return empty dict, not raise exception
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_parse_pdf_whitespace_handling(self, parser):
        """Test that parser handles whitespace correctly."""
        with patch.object(parser.parser, 'aload_data', new_callable=AsyncMock) as mock_load:
            mock_doc = Mock()
            mock_doc.text = "  \n\n  Text with whitespace  \n\n  "
            mock_load.return_value = [mock_doc]

            result = await parser.parse_pdf(Path("test.pdf"))

            # Original text should be preserved (chunker will handle cleanup)
            assert result == "  \n\n  Text with whitespace  \n\n  "

    @pytest.mark.asyncio
    async def test_parse_pdf_unicode_content(self, parser):
        """Test parsing PDF with unicode characters."""
        with patch.object(parser.parser, 'aload_data', new_callable=AsyncMock) as mock_load:
            mock_doc = Mock()
            mock_doc.text = "β-blocker dosing: 10mg ± 2mg, efficacy ≥ 90%"
            mock_load.return_value = [mock_doc]

            result = await parser.parse_pdf(Path("test.pdf"))

            assert "β-blocker" in result
            assert "≥" in result

    def test_metadata_includes_all_fields(self, parser, sample_pdf_path):
        """Test that metadata includes all expected fields."""
        metadata = parser._extract_pdf_metadata(sample_pdf_path)

        expected_fields = ["page_count", "file_size", "file_name"]
        for field in expected_fields:
            assert field in metadata

    @pytest.mark.asyncio
    async def test_parse_large_pdf(self, parser):
        """Test parsing a large PDF with many pages."""
        with patch.object(parser.parser, 'aload_data', new_callable=AsyncMock) as mock_load:
            # Simulate 100 pages
            mock_docs = [Mock(text=f"Page {i} content") for i in range(100)]
            mock_load.return_value = mock_docs

            result = await parser.parse_pdf(Path("large.pdf"))

            # All pages should be included
            assert "Page 0 content" in result
            assert "Page 99 content" in result
