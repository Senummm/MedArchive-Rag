"""
Citation extraction and formatting service.

Extracts citations from search results and formats them for references.
"""

import logging
import re
from typing import Dict, List, Optional, Set
from uuid import UUID

from shared.models import Citation, SearchResult
from shared.utils import get_logger

logger = get_logger(__name__)


class CitationExtractor:
    """
    Extract and format citations from search results.
    
    Maps source references ([Source N]) in generated text to actual citations
    with document metadata.
    """

    def __init__(self):
        """Initialize the citation extractor."""
        self.source_pattern = re.compile(r"\[Source (\d+)\]")

    def extract_citations(
        self,
        answer: str,
        search_results: List[SearchResult],
        document_titles: Optional[Dict[UUID, str]] = None,
    ) -> List[Citation]:
        """
        Extract citations from generated answer.

        Args:
            answer: Generated answer text with [Source N] references
            search_results: Original search results used for generation
            document_titles: Optional mapping of document_id -> title

        Returns:
            List of Citation objects
        """
        # Find all source references in answer
        referenced_sources = set()
        for match in self.source_pattern.finditer(answer):
            source_num = int(match.group(1))
            referenced_sources.add(source_num)

        logger.info(f"Found {len(referenced_sources)} source references in answer")

        # Build citations for referenced sources
        citations = []
        for source_num in sorted(referenced_sources):
            # Source numbers are 1-indexed
            if source_num <= len(search_results):
                result = search_results[source_num - 1]
                
                # Get document title
                doc_title = "Unknown Document"
                if document_titles and result.document_id in document_titles:
                    doc_title = document_titles[result.document_id]
                elif result.source_file:
                    # Use filename as fallback
                    doc_title = result.source_file

                # Create citation
                citation = Citation(
                    document_id=result.document_id,
                    document_title=doc_title,
                    page_numbers=result.page_numbers,
                    text_snippet=self._create_snippet(result.text),
                    relevance_score=result.score,
                )
                citations.append(citation)

        logger.info(f"Created {len(citations)} citations")
        return citations

    def format_citations(
        self,
        citations: List[Citation],
        style: str = "numeric",
    ) -> str:
        """
        Format citations for display.

        Args:
            citations: List of citations
            style: Citation style ('numeric', 'apa', 'footnote')

        Returns:
            Formatted citation string
        """
        if not citations:
            return ""

        if style == "numeric":
            return self._format_numeric(citations)
        elif style == "apa":
            return self._format_apa(citations)
        elif style == "footnote":
            return self._format_footnote(citations)
        else:
            return self._format_numeric(citations)

    def _format_numeric(self, citations: List[Citation]) -> str:
        """Format citations in numeric style."""
        lines = []
        for i, citation in enumerate(citations, 1):
            pages = ", ".join(str(p) for p in citation.page_numbers) if citation.page_numbers else "N/A"
            lines.append(
                f"{i}. {citation.document_title}, Page {pages}"
            )
        return "\n".join(lines)

    def _format_apa(self, citations: List[Citation]) -> str:
        """Format citations in APA style."""
        lines = []
        for citation in citations:
            pages = ", ".join(str(p) for p in citation.page_numbers) if citation.page_numbers else "N/A"
            lines.append(
                f"{citation.document_title} (p. {pages})"
            )
        return "\n".join(lines)

    def _format_footnote(self, citations: List[Citation]) -> str:
        """Format citations as footnotes."""
        lines = []
        for i, citation in enumerate(citations, 1):
            pages = ", ".join(str(p) for p in citation.page_numbers) if citation.page_numbers else "N/A"
            snippet = citation.text_snippet[:100] + "..." if len(citation.text_snippet) > 100 else citation.text_snippet
            lines.append(
                f"[{i}] {citation.document_title}, p. {pages}: \"{snippet}\""
            )
        return "\n".join(lines)

    def _create_snippet(self, text: str, max_length: int = 200) -> str:
        """
        Create a text snippet for citation.

        Args:
            text: Full chunk text
            max_length: Maximum snippet length

        Returns:
            Truncated snippet with ellipsis if needed
        """
        if len(text) <= max_length:
            return text.strip()

        # Try to break at sentence boundary
        snippet = text[:max_length]
        last_period = snippet.rfind(". ")
        if last_period > max_length // 2:
            snippet = snippet[:last_period + 1]
        else:
            snippet = snippet.rsplit(" ", 1)[0] + "..."

        return snippet.strip()

    def add_inline_citations(
        self,
        answer: str,
        search_results: List[SearchResult],
    ) -> str:
        """
        Add inline source numbers to answer text.
        
        Useful when answer doesn't already have [Source N] references.

        Args:
            answer: Generated answer
            search_results: Search results to cite

        Returns:
            Answer with inline citations added
        """
        # This is a simplified approach - in production, would use more
        # sophisticated methods to match text spans to sources
        
        # For now, just append sources at the end
        if not search_results:
            return answer

        source_refs = ", ".join(
            f"[Source {i+1}]" for i in range(len(search_results))
        )
        
        return f"{answer}\n\nSources: {source_refs}"

    def deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """
        Remove duplicate citations (same document + page).

        Args:
            citations: List potentially containing duplicates

        Returns:
            Deduplicated list
        """
        seen: Set[tuple] = set()
        unique_citations = []

        for citation in citations:
            # Create key from document_id and pages
            pages_tuple = tuple(sorted(citation.page_numbers))
            key = (citation.document_id, pages_tuple)

            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)

        if len(unique_citations) < len(citations):
            logger.info(
                f"Deduplicated citations: {len(citations)} -> {len(unique_citations)}"
            )

        return unique_citations

    def group_by_document(
        self,
        citations: List[Citation],
    ) -> Dict[UUID, List[Citation]]:
        """
        Group citations by document.

        Args:
            citations: List of citations

        Returns:
            Dictionary mapping document_id to citations
        """
        grouped: Dict[UUID, List[Citation]] = {}

        for citation in citations:
            if citation.document_id not in grouped:
                grouped[citation.document_id] = []
            grouped[citation.document_id].append(citation)

        return grouped

    def merge_page_ranges(self, page_numbers: List[int]) -> str:
        """
        Format page numbers with ranges (e.g., "5-7, 10, 12-14").

        Args:
            page_numbers: List of page numbers

        Returns:
            Formatted page range string
        """
        if not page_numbers:
            return "N/A"

        # Sort and deduplicate
        pages = sorted(set(page_numbers))

        if len(pages) == 1:
            return str(pages[0])

        # Build ranges
        ranges = []
        start = pages[0]
        end = pages[0]

        for page in pages[1:]:
            if page == end + 1:
                end = page
            else:
                # Add current range
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = page

        # Add final range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ", ".join(ranges)
