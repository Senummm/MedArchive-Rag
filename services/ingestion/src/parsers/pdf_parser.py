"""
PDF Parser using LlamaParse.

LlamaParse provides table-aware parsing, converting complex clinical
PDFs with dosage tables into clean markdown that LLMs understand.
"""

from pathlib import Path
from typing import Dict, Optional

from llama_parse import LlamaParse
from pypdf import PdfReader

from shared.utils import get_settings, setup_logging

settings = get_settings()
logger = setup_logging("ingestion.parser", settings.log_level)


class PDFParser:
    """
    Table-aware PDF parser using LlamaParse.

    Converts clinical guidelines with heavy tables into markdown format,
    preserving table structure for accurate information extraction.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LlamaParse client.

        Args:
            api_key: LlamaParse API key (defaults to settings)
        """
        self.api_key = api_key or settings.llamaparse_api_key
        self.result_type = settings.llamaparse_result_type
        self.verbose = settings.llamaparse_verbose

        # Initialize LlamaParse client
        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type=self.result_type,
            verbose=self.verbose,
            language="en",
            num_workers=4,
            invalidate_cache=False,
        )

        logger.info(
            "PDFParser initialized",
            extra={"result_type": self.result_type},
        )

    async def parse_pdf(self, file_path: Path) -> str:
        """
        Parse a PDF file into markdown text.

        Args:
            file_path: Path to the PDF file

        Returns:
            Markdown-formatted text with preserved table structure

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If parsing fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info(
            "Parsing PDF with LlamaParse",
            extra={
                "file": file_path.name,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            },
        )

        try:
            # Use LlamaParse for table-aware parsing
            documents = await self.parser.aload_data(str(file_path))
            markdown_text = "\n\n".join([doc.text for doc in documents])

            logger.info(
                "PDF parsed successfully",
                extra={
                    "file": file_path.name,
                    "text_length": len(markdown_text),
                    "pages": len(documents),
                },
            )

            return markdown_text

        except Exception as e:
            logger.error(
                "PDF parsing failed",
                exc_info=e,
                extra={"file": file_path.name},
            )
            raise ValueError(f"Failed to parse PDF: {str(e)}") from e

    async def parse_pdf_with_metadata(self, file_path: Path) -> Dict[str, any]:
        """
        Parse PDF and extract structured metadata.

        Args:
            file_path: Path to the PDF

        Returns:
            Dictionary with:
                - text: Parsed markdown
                - metadata: Title, author, page count, etc.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        # Parse content with LlamaParse
        text = await self.parse_pdf(file_path)

        # Extract metadata using pypdf
        metadata = self._extract_pdf_metadata(file_path)

        return {
            "text": text,
            "metadata": metadata,
        }

    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, any]:
        """
        Extract metadata from PDF using pypdf.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with title, author, page_count, etc.
        """
        try:
            reader = PdfReader(str(file_path))

            # Extract basic metadata
            info = reader.metadata or {}

            metadata = {
                "title": info.get("/Title", file_path.stem),
                "author": info.get("/Author"),
                "subject": info.get("/Subject"),
                "creator": info.get("/Creator"),
                "producer": info.get("/Producer"),
                "page_count": len(reader.pages),
                "file_size_bytes": file_path.stat().st_size,
            }

            logger.debug(
                "Extracted PDF metadata",
                extra={
                    "file": file_path.name,
                    "page_count": metadata["page_count"],
                },
            )

            return metadata

        except Exception as e:
            logger.warning(
                "Failed to extract PDF metadata",
                exc_info=e,
                extra={"file": file_path.name},
            )
            # Return minimal metadata on failure
            return {
                "title": file_path.stem,
                "author": None,
                "page_count": None,
                "file_size_bytes": file_path.stat().st_size,
            }
