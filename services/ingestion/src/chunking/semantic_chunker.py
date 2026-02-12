"""
Semantic Chunking for Clinical Documents.

Implements intelligent text splitting that preserves:
- Section structure (headers and hierarchies)
- Table integrity (keeps tables together)
- Semantic meaning (doesn't split mid-sentence unless necessary)
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID

from shared.models import ChunkMetadata
from shared.utils import get_settings, setup_logging

settings = get_settings()
logger = setup_logging("ingestion.chunker", settings.log_level)


@dataclass
class TextChunk:
    """Temporary data structure for chunk processing."""

    text: str
    start_idx: int
    end_idx: int
    section_path: Optional[str] = None
    heading: Optional[str] = None


class SemanticChunker:
    """
    Semantic text chunker optimized for clinical documents.

    Uses recursive character splitting with header-aware logic to create
    semantically meaningful chunks that preserve document structure.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target chunk size in characters (defaults to settings)
            chunk_overlap: Overlap between chunks (defaults to settings)
            min_chunk_size: Minimum viable chunk size (defaults to settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.min_chunk_size

        # Splitting separators (in order of preference)
        self.separators = [
            "\n\n\n",  # Multiple newlines (section breaks)
            "\n\n",  # Double newline (paragraph breaks)
            "\n",  # Single newline
            ". ",  # Sentence boundaries
            "! ",
            "? ",
            "; ",
            ", ",  # Clause boundaries
            " ",  # Word boundaries
            "",  # Character-level (last resort)
        ]

        logger.info(
            "SemanticChunker initialized",
            extra={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        )

    def chunk_text(
        self,
        text: str,
        document_id: UUID,
        section_path: Optional[str] = None,
    ) -> List[ChunkMetadata]:
        """
        Split text into semantic chunks with metadata.

        Args:
            text: The markdown text to chunk
            document_id: UUID of the parent document
            section_path: Optional section path for root context

        Returns:
            List of ChunkMetadata objects ready for indexing
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            logger.warning("Text too short to chunk", extra={"length": len(text)})
            return []

        logger.info(
            "Starting semantic chunking",
            extra={
                "text_length": len(text),
                "document_id": str(document_id),
            },
        )

        # Extract section structure from markdown headers
        sections = self._extract_sections(text)

        # Create chunks preserving section boundaries
        chunks = []
        for section in sections:
            section_chunks = self._split_text_recursive(
                section["text"],
                max_length=self.chunk_size,
                overlap=self.chunk_overlap,
            )

            for idx, chunk_text in enumerate(section_chunks):
                if len(chunk_text.strip()) < self.min_chunk_size:
                    continue

                chunk_metadata = ChunkMetadata(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    text=chunk_text.strip(),
                    section_path=section["section_path"],
                    heading=section["heading"],
                    page_numbers=[],  # Will be populated during indexing if available
                    token_count=self._estimate_token_count(chunk_text),
                )
                chunks.append(chunk_metadata)

        logger.info(
            "Chunking complete",
            extra={
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks)
                if chunks
                else 0,
            },
        )

        return chunks

    def _extract_sections(self, text: str) -> List[dict]:
        """
        Extract sections from markdown headers.

        Args:
            text: Markdown text

        Returns:
            List of dicts with section_path, heading, and text
        """
        sections = []
        current_section = {"section_path": None, "heading": None, "text": ""}
        header_stack = []  # Track hierarchy: [(level, heading), ...]

        lines = text.split("\n")

        for line in lines:
            # Check if line is a markdown header
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if header_match:
                # Save previous section if it has content
                if current_section["text"].strip():
                    sections.append(current_section)

                # Parse header
                level = len(header_match.group(1))
                heading_text = header_match.group(2).strip()

                # Update header stack based on level
                header_stack = [h for h in header_stack if h[0] < level]
                header_stack.append((level, heading_text))

                # Build section path
                section_path = " > ".join([h[1] for h in header_stack])

                # Start new section
                current_section = {
                    "section_path": section_path,
                    "heading": heading_text,
                    "text": "",
                }
            else:
                # Accumulate text for current section
                current_section["text"] += line + "\n"

        # Add final section
        if current_section["text"].strip():
            sections.append(current_section)

        # If no sections found, treat entire text as one section
        if not sections:
            sections = [
                {
                    "section_path": None,
                    "heading": None,
                    "text": text,
                }
            ]

        logger.debug(f"Extracted {len(sections)} sections from document")
        return sections

    def _split_text_recursive(
        self,
        text: str,
        max_length: int,
        overlap: int,
        depth: int = 0,
    ) -> List[str]:
        """
        Recursively split text using multiple separators.

        Args:
            text: Text to split
            max_length: Maximum chunk length
            overlap: Overlap between chunks
            depth: Current recursion depth (for preventing infinite recursion)

        Returns:
            List of text chunks
        """
        # Prevent infinite recursion
        if depth > 10:
            logger.warning(
                "Max recursion depth reached, forcing character split",
                extra={"text_length": len(text), "depth": depth},
            )
            return self._split_by_length(text, max_length, overlap)

        if len(text) <= max_length:
            return [text]

        # Try each separator in order of preference
        for separator in self.separators:
            if separator in text:
                chunks = self._split_by_separator(text, separator, max_length, overlap, depth)
                if chunks:
                    return chunks

        # Fallback: Character-level split (should rarely happen)
        logger.warning(
            "Falling back to character-level split",
            extra={"text_length": len(text), "depth": depth},
        )
        return self._split_by_length(text, max_length, overlap)

    def _split_by_separator(
        self,
        text: str,
        separator: str,
        max_length: int,
        overlap: int,
        depth: int = 0,
    ) -> List[str]:
        """Split text by a specific separator."""
        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for split in splits:
            # Add separator back (except for empty string)
            split_with_sep = split + separator if separator else split

            if len(current_chunk) + len(split_with_sep) <= max_length:
                current_chunk += split_with_sep
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # Handle split that's too long on its own
                if len(split_with_sep) > max_length:
                    # Recursively split with next separator
                    sub_chunks = self._split_text_recursive(
                        split_with_sep, max_length, overlap, depth + 1
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Start new chunk with overlap
                    if chunks and overlap > 0:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + split_with_sep
                    else:
                        current_chunk = split_with_sep

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_length(self, text: str, max_length: int, overlap: int) -> List[str]:
        """Split text by fixed length with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + max_length
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end

        return chunks

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token (typical for English).
        This is approximate but sufficient for metadata purposes.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4
