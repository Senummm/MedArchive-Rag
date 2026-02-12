"""
File Tracker for Incremental Sync.

Tracks PDF files using MD5 hashing to detect changes and avoid
redundant processing. Enables efficient incremental updates.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from shared.utils import get_settings, setup_logging

settings = get_settings()
logger = setup_logging("ingestion.file_tracker", settings.log_level)


class FileTracker:
    """
    Tracks file hashes for incremental synchronization.

    Uses MD5 hashing to detect file changes and maintain a registry
    of processed documents to avoid redundant processing.
    """

    def __init__(self, tracking_file: Optional[Path] = None):
        """
        Initialize the file tracker.

        Args:
            tracking_file: Path to JSON file storing file hashes
                          (defaults to vector_storage_path/file_tracking.json)
        """
        if tracking_file is None:
            storage_path = Path(settings.vector_storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            tracking_file = storage_path / "file_tracking.json"

        self.tracking_file = tracking_file
        self.file_registry: Dict[str, Dict] = self._load_registry()

        logger.info(
            "FileTracker initialized",
            extra={
                "tracking_file": str(self.tracking_file),
                "tracked_files": len(self.file_registry),
            },
        )

    def compute_file_hash(self, file_path: Path) -> str:
        """
        Compute MD5 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash as hexadecimal string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        hash_md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)

        file_hash = hash_md5.hexdigest()

        logger.debug(
            "Computed file hash",
            extra={"file": file_path.name, "hash": file_hash},
        )

        return file_hash

    def has_file_changed(self, file_path: Path) -> bool:
        """
        Check if file has changed since last processing.

        Args:
            file_path: Path to the file

        Returns:
            True if file is new or has changed, False if unchanged
        """
        file_key = str(file_path.absolute())
        current_hash = self.compute_file_hash(file_path)

        if file_key not in self.file_registry:
            logger.info(f"New file detected: {file_path.name}")
            return True

        stored_hash = self.file_registry[file_key].get("hash")
        has_changed = current_hash != stored_hash

        if has_changed:
            logger.info(
                f"File changed: {file_path.name}",
                extra={
                    "old_hash": stored_hash,
                    "new_hash": current_hash,
                },
            )
        else:
            logger.debug(f"File unchanged: {file_path.name}")

        return has_changed

    def mark_file_processed(
        self,
        file_path: Path,
        document_id: str,
        chunk_count: int,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Mark a file as successfully processed.

        Args:
            file_path: Path to the processed file
            document_id: UUID of the document in the system
            chunk_count: Number of chunks created from this document
            metadata: Optional additional metadata to store
        """
        file_key = str(file_path.absolute())
        file_hash = self.compute_file_hash(file_path)

        self.file_registry[file_key] = {
            "hash": file_hash,
            "document_id": document_id,
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "chunk_count": chunk_count,
            "last_processed": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        self._save_registry()

        logger.info(
            "File marked as processed",
            extra={
                "file": file_path.name,
                "document_id": document_id,
                "chunks": chunk_count,
            },
        )

    def mark_file_failed(
        self,
        file_path: Path,
        error_message: str,
    ) -> None:
        """
        Mark a file as failed processing.

        Args:
            file_path: Path to the file that failed
            error_message: Error message describing the failure
        """
        file_key = str(file_path.absolute())

        self.file_registry[file_key] = {
            "hash": self.compute_file_hash(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "status": "failed",
            "error": error_message,
            "last_attempted": datetime.utcnow().isoformat(),
        }

        self._save_registry()

        logger.warning(
            "File marked as failed",
            extra={"file": file_path.name, "error": error_message},
        )

    def get_files_to_process(self, directory: Path) -> List[Path]:
        """
        Get list of PDF files that need processing.

        Args:
            directory: Directory to scan for PDFs

        Returns:
            List of file paths that are new or changed
        """
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return []

        pdf_files = list(directory.glob("*.pdf"))
        files_to_process = []

        for pdf_file in pdf_files:
            if self.has_file_changed(pdf_file):
                files_to_process.append(pdf_file)

        logger.info(
            f"Found {len(files_to_process)} files to process out of {len(pdf_files)} total"
        )

        return files_to_process

    def get_processed_document_ids(self) -> Set[str]:
        """
        Get set of all document IDs that have been processed.

        Returns:
            Set of document ID strings
        """
        return {
            entry["document_id"]
            for entry in self.file_registry.values()
            if "document_id" in entry
        }

    def get_file_info(self, file_path: Path) -> Optional[Dict]:
        """
        Get stored information about a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file info, or None if not tracked
        """
        file_key = str(file_path.absolute())
        return self.file_registry.get(file_key)

    def remove_file(self, file_path: Path) -> None:
        """
        Remove a file from tracking (e.g., when file is deleted).

        Args:
            file_path: Path to the file
        """
        file_key = str(file_path.absolute())

        if file_key in self.file_registry:
            del self.file_registry[file_key]
            self._save_registry()
            logger.info(f"Removed file from tracking: {file_path.name}")

    def _load_registry(self) -> Dict[str, Dict]:
        """Load the file registry from disk."""
        if not self.tracking_file.exists():
            logger.info("No existing file registry found, starting fresh")
            return {}

        try:
            with open(self.tracking_file, "r", encoding="utf-8") as f:
                registry = json.load(f)
                logger.info(f"Loaded {len(registry)} tracked files from registry")
                return registry
        except Exception as e:
            logger.error(
                "Failed to load file registry",
                exc_info=e,
                extra={"file": str(self.tracking_file)},
            )
            return {}

    def _save_registry(self) -> None:
        """Save the file registry to disk."""
        try:
            self.tracking_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.tracking_file, "w", encoding="utf-8") as f:
                json.dump(self.file_registry, f, indent=2, ensure_ascii=False)

            logger.debug(
                "File registry saved",
                extra={"tracked_files": len(self.file_registry)},
            )
        except Exception as e:
            logger.error(
                "Failed to save file registry",
                exc_info=e,
                extra={"file": str(self.tracking_file)},
            )
