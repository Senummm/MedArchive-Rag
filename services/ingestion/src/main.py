"""
MedArchive RAG Ingestion Service

Background worker for processing clinical PDFs into searchable chunks.

Pipeline:
1. Watch document_store for new/changed PDFs
2. Parse PDFs with LlamaParse (table-aware)
3. Chunk semantically with metadata enrichment
4. Generate embeddings (BAAI/bge-large-en-v1.5)
5. Index into Qdrant
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from shared.models import DocumentMetadata, DocumentType, ProcessingStatus
from shared.utils import get_settings, setup_logging
from services.ingestion.src.chunking import SemanticChunker
from services.ingestion.src.embedding import Embedder
from services.ingestion.src.indexing import QdrantIndexer
from services.ingestion.src.parsers.pdf_parser import PDFParser
from services.ingestion.src.sync import FileTracker

# Initialize configuration and logging
settings = get_settings()
logger = setup_logging(
    service_name="ingestion",
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_output=settings.log_output,
)

# Global service instances (initialized in initialize_services)
pdf_parser: Optional[PDFParser] = None
chunker: Optional[SemanticChunker] = None
embedder: Optional[Embedder] = None
indexer: Optional[QdrantIndexer] = None
file_tracker: Optional[FileTracker] = None


async def initialize_services():
    """
    Initialize ingestion pipeline services.

    Sets up:
    - PDFParser (LlamaParse client)
    - SemanticChunker
    - Embedder (sentence-transformers)
    - QdrantIndexer (vector database)
    - FileTracker (incremental sync)
    """
    global pdf_parser, chunker, embedder, indexer, file_tracker

    logger.info(
        "Initializing ingestion services",
        extra={
            "document_source": settings.document_source_path,
            "qdrant_url": settings.qdrant_url,
            "embedding_model": settings.embedding_model,
        },
    )

    try:
        # Initialize PDF parser
        pdf_parser = PDFParser()

        # Initialize chunker
        chunker = SemanticChunker()

        # Initialize embedder
        embedder = Embedder()

        # Initialize Qdrant indexer
        indexer = QdrantIndexer()

        # Ensure Qdrant collection exists
        indexer.ensure_collection_exists(vector_size=embedder.get_embedding_dimension())

        # Initialize file tracker
        file_tracker = FileTracker()

        logger.info(
            "Ingestion services initialized successfully",
            extra={
                "embedding_dim": embedder.get_embedding_dimension(),
                "collection": indexer.collection_name,
                "tracked_files": len(file_tracker.file_registry),
            },
        )

    except Exception as e:
        logger.error("Failed to initialize ingestion services", exc_info=e)
        raise


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

    # Initial scan
    await scan_and_process_documents()

    # Continuous monitoring
    while True:
        await asyncio.sleep(60)  # Check every minute
        logger.debug("Checking for new documents...")
        await scan_and_process_documents()


async def scan_and_process_documents():
    """Scan directory and process new/changed documents."""
    document_path = Path(settings.document_source_path)

    if not file_tracker:
        logger.error("FileTracker not initialized")
        return

    # Get files that need processing
    files_to_process = file_tracker.get_files_to_process(document_path)

    if not files_to_process:
        logger.debug("No new or changed files to process")
        return

    logger.info(f"Found {len(files_to_process)} documents to process")

    # Process each file
    for file_path in files_to_process:
        try:
            await process_document(file_path)
        except Exception as e:
            logger.error(
                f"Failed to process document: {file_path.name}",
                exc_info=e,
            )
            file_tracker.mark_file_failed(file_path, str(e))


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
    if not all([pdf_parser, chunker, embedder, indexer, file_tracker]):
        raise RuntimeError("Services not initialized")

    logger.info(
        f"Processing document: {file_path.name}",
        extra={"file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)},
    )

    try:
        # Step 1: Parse PDF with metadata
        parse_result = await pdf_parser.parse_pdf_with_metadata(file_path)
        markdown_text = parse_result["text"]
        pdf_metadata = parse_result["metadata"]

        logger.info(
            f"PDF parsed: {file_path.name}",
            extra={
                "text_length": len(markdown_text),
                "page_count": pdf_metadata.get("page_count"),
            },
        )

        # Step 2: Create DocumentMetadata
        doc_metadata = DocumentMetadata(
            title=pdf_metadata.get("title", file_path.stem),
            document_type=_infer_document_type(file_path.stem),
            source_path=str(file_path),
            file_hash=file_tracker.compute_file_hash(file_path),
            page_count=pdf_metadata.get("page_count"),
            author=pdf_metadata.get("author"),
            processing_status=ProcessingStatus.CHUNKING,
        )

        # Step 3: Chunk semantically
        chunks = chunker.chunk_text(
            text=markdown_text,
            document_id=doc_metadata.document_id,
        )

        logger.info(
            f"Document chunked: {file_path.name}",
            extra={"chunk_count": len(chunks)},
        )

        if not chunks:
            logger.warning(f"No chunks generated for: {file_path.name}")
            return

        # Step 4: Update chunk metadata with embedding model and document info
        for chunk in chunks:
            chunk.embedding_model = embedder.model_name
            # Add document metadata to chunks for citation extraction
            chunk.document_title = doc_metadata.title
            chunk.source_file = file_path.name

        # Step 5: Generate embeddings
        doc_metadata.processing_status = ProcessingStatus.EMBEDDING
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_batch(chunk_texts, show_progress=True)

        logger.info(
            f"Embeddings generated: {file_path.name}",
            extra={"embedding_shape": embeddings.shape},
        )

        # Step 6: Index into Qdrant (with atomic update if document exists)
        doc_metadata.processing_status = ProcessingStatus.INDEXING

        # Check if document was previously indexed
        existing_info = file_tracker.get_file_info(file_path)
        if existing_info and "document_id" in existing_info:
            # Delete old chunks before indexing new ones (atomic update)
            logger.info(f"Deleting old chunks for updated document: {file_path.name}")
            indexer.delete_document_chunks(doc_metadata.document_id)

        # Index new chunks
        indexed_count = indexer.index_chunks(chunks, embeddings)

        # Step 7: Mark as completed
        doc_metadata.processing_status = ProcessingStatus.COMPLETED
        file_tracker.mark_file_processed(
            file_path=file_path,
            document_id=str(doc_metadata.document_id),
            chunk_count=len(chunks),
            metadata={
                "title": doc_metadata.title,
                "page_count": doc_metadata.page_count,
                "document_type": doc_metadata.document_type.value,
            },
        )

        logger.info(
            f"✅ Document processed successfully: {file_path.name}",
            extra={
                "document_id": str(doc_metadata.document_id),
                "chunks": len(chunks),
                "indexed": indexed_count,
            },
        )

    except Exception as e:
        logger.error(
            f"❌ Failed to process document: {file_path.name}",
            exc_info=e,
        )
        file_tracker.mark_file_failed(file_path, str(e))
        raise


def _infer_document_type(filename: str) -> DocumentType:
    """
    Infer document type from filename.

    Args:
        filename: Name of the file (without extension)

    Returns:
        DocumentType enum value
    """
    filename_lower = filename.lower()

    if any(word in filename_lower for word in ["guideline", "guide"]):
        return DocumentType.CLINICAL_GUIDELINE
    elif any(word in filename_lower for word in ["formulary", "drug", "medication"]):
        return DocumentType.FORMULARY
    elif any(word in filename_lower for word in ["protocol", "procedure"]):
        return DocumentType.PROTOCOL
    elif any(word in filename_lower for word in ["policy", "policies"]):
        return DocumentType.POLICY
    elif any(word in filename_lower for word in ["research", "study", "paper"]):
        return DocumentType.RESEARCH_PAPER
    else:
        return DocumentType.OTHER


async def main():
    """Main entry point for the ingestion service."""
    logger.info(
        "🚀 Starting MedArchive Ingestion Service",
        extra={"version": "0.1.0", "phase": "2"},
    )

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
