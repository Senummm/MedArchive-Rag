"""
PDF Parser using LlamaParse.

LlamaParse provides table-aware parsing, converting complex clinical
PDFs with dosage tables into clean markdown that LLMs understand.
"""

from pathlib import Path
from typing import Optional

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

        # TODO Phase 2: Initialize LlamaParse client
        # from llama_parse import LlamaParse
        # self.parser = LlamaParse(
        #     api_key=self.api_key,
        #     result_type=self.result_type,
        #     verbose=self.verbose
        # )

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
            "Parsing PDF",
            extra={
                "file": file_path.name,
                "size_mb": file_path.stat().st_size / (1024 * 1024),
            },
        )

        # TODO Phase 2: Implement LlamaParse integration
        # documents = await self.parser.aload_data(str(file_path))
        # markdown_text = "\n\n".join([doc.text for doc in documents])
        # return markdown_text

        raise NotImplementedError("LlamaParse integration pending (Phase 2)")

    async def parse_pdf_with_metadata(self, file_path: Path) -> dict:
        """
        Parse PDF and extract structured metadata.

        Args:
            file_path: Path to the PDF

        Returns:
            Dictionary with:
                - text: Parsed markdown
                - metadata: Title, author, page count, etc.
        """
        # TODO Phase 2: Extract metadata from PDF
        # - Use pypdf for metadata extraction
        # - Combine with LlamaParse output

        raise NotImplementedError("Metadata extraction pending (Phase 2)")
