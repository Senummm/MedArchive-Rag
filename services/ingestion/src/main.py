"""
MedArchive RAG Ingestion Service

Background worker for processing clinical PDFs into searchable chunks.

Pipeline:
1. Watch document_store for new/changed PDFs
2. Parse PDFs with LlamaParse (table-aware)
3. Chunk semantically with metadata enrichment
4. Generate embeddings (BAAI/bge-large-en-v1.5)
5. Index into Qdrant with binary quantization
"""

import asyncio
import sys
from pathlib import Path

from shared.utils import get_settings, setup_logging

# Initialize configuration and logging
settings = get_settings()
logger = setup_logging(
    service_name="ingestion",
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_output=settings.log_output,
)


async def initialize_services():
    """
    Initialize ingestion pipeline services.

    Sets up:
    - Qdrant collection (with binary quantization)
    - Embedding model
    - LlamaParse client
    """
    logger.info(
        "Initializing ingestion services",
        extra={
            "document_source": settings.document_source_path,
            "qdrant_url": settings.qdrant_url,
            "embedding_model": settings.embedding_model,
        },
    )

    # TODO Phase 2: Initialize LlamaParse client
    # TODO Phase 3: Initialize Qdrant collection with correct schema
    # TODO Phase 2: Load embedding model

    logger.info("Ingestion services initialized successfully")


async def watch_document_directory():
    """
    Watch document store for new or modified PDFs.

    Uses file hashing (MD5) for incremental sync:
    - New files: Full processing
    - Changed files: Atomic update in Qdrant
    - Unchanged files: Skip
    """
    document_path = Path(settings.document_source_path)

    if not document_path.exists():
        logger.warning(
            f"Document source path does not exist: {document_path}. Creating directory..."
        )
        document_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Watching directory: {document_path}")

    # TODO Phase 2: Implement file watcher with hashing
    # TODO Phase 2: Process new/changed files through pipeline

    # Placeholder: Sleep to keep container running
    while True:
        await asyncio.sleep(60)
        logger.debug("Checking for new documents...")


async def process_document(file_path: Path):
    """
    Process a single PDF document through the ingestion pipeline.

    Args:
        file_path: Path to the PDF file

    Pipeline stages:
    1. Parse PDF with LlamaParse (markdown output for tables)
    2. Extract metadata (title, department, effective date)
    3. Chunk semantically (preserve section structure)
    4. Enrich chunks with metadata (section path, page numbers)
    5. Generate embeddings
    6. Index into Qdrant
    """
    logger.info(f"Processing document: {file_path.name}")

    # TODO Phase 2: Implement full pipeline
    # - Parse with LlamaParse
    # - Chunk with RecursiveCharacterTextSplitter
    # - Embed with sentence-transformers
    # - Index into Qdrant

    pass


async def main():
    """Main entry point for the ingestion service."""
    logger.info("Starting MedArchive Ingestion Service")

    try:
        await initialize_services()
        await watch_document_directory()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error("Fatal error in ingestion service", exc_info=e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
