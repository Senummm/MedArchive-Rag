"""
Unit tests for FileTracker.

Tests file hashing, change detection, and registry management.
"""

from pathlib import Path
from uuid import uuid4

import pytest

from services.ingestion.src.sync import FileTracker


class TestFileTracker:
    """Tests for FileTracker class."""

    @pytest.fixture
    def temp_tracker(self, tmp_path):
        """Create a file tracker with temporary tracking file."""
        tracking_file = tmp_path / "test_tracking.json"
        return FileTracker(tracking_file=tracking_file)

    def test_tracker_initialization(self, temp_tracker):
        """Test file tracker initializes correctly."""
        assert temp_tracker.file_registry is not None
        assert len(temp_tracker.file_registry) == 0

    def test_compute_file_hash(self, temp_tracker, tmp_path):
        """Test file hash computation."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        file_hash = temp_tracker.compute_file_hash(test_file)

        assert isinstance(file_hash, str)
        assert len(file_hash) == 32  # MD5 hash length

        # Same content should produce same hash
        hash2 = temp_tracker.compute_file_hash(test_file)
        assert file_hash == hash2

    def test_compute_hash_nonexistent_file(self, temp_tracker, tmp_path):
        """Test that hashing nonexistent file raises error."""
        fake_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            temp_tracker.compute_file_hash(fake_file)

    def test_has_file_changed_new_file(self, temp_tracker, tmp_path):
        """Test that new files are detected as changed."""
        test_file = tmp_path / "new_file.txt"
        test_file.write_text("New content")

        assert temp_tracker.has_file_changed(test_file) is True

    def test_has_file_changed_unchanged_file(self, temp_tracker, tmp_path):
        """Test that unchanged files are detected."""
        test_file = tmp_path / "unchanged.txt"
        test_file.write_text("Original content")

        # Mark as processed
        temp_tracker.mark_file_processed(
            test_file,
            document_id=str(uuid4()),
            chunk_count=5,
        )

        # Should not be detected as changed
        assert temp_tracker.has_file_changed(test_file) is False

    def test_has_file_changed_modified_file(self, temp_tracker, tmp_path):
        """Test that modified files are detected."""
        test_file = tmp_path / "modified.txt"
        test_file.write_text("Original content")

        # Mark as processed
        temp_tracker.mark_file_processed(
            test_file,
            document_id=str(uuid4()),
            chunk_count=5,
        )

        # Modify file
        test_file.write_text("Modified content")

        # Should be detected as changed
        assert temp_tracker.has_file_changed(test_file) is True

    def test_mark_file_processed(self, temp_tracker, tmp_path):
        """Test marking file as processed."""
        test_file = tmp_path / "processed.txt"
        test_file.write_text("Content")

        doc_id = str(uuid4())
        temp_tracker.mark_file_processed(
            test_file,
            document_id=doc_id,
            chunk_count=10,
            metadata={"title": "Test Document"},
        )

        # Check registry entry
        file_key = str(test_file.absolute())
        assert file_key in temp_tracker.file_registry
        entry = temp_tracker.file_registry[file_key]

        assert entry["document_id"] == doc_id
        assert entry["chunk_count"] == 10
        assert entry["metadata"]["title"] == "Test Document"

    def test_mark_file_failed(self, temp_tracker, tmp_path):
        """Test marking file as failed."""
        test_file = tmp_path / "failed.txt"
        test_file.write_text("Content")

        temp_tracker.mark_file_failed(test_file, "Parse error occurred")

        # Check registry entry
        file_key = str(test_file.absolute())
        entry = temp_tracker.file_registry[file_key]

        assert entry["status"] == "failed"
        assert "Parse error" in entry["error"]

    def test_get_files_to_process(self, temp_tracker, tmp_path):
        """Test getting list of files to process."""
        # Create test PDF files
        file1 = tmp_path / "doc1.pdf"
        file2 = tmp_path / "doc2.pdf"
        file3 = tmp_path / "doc3.pdf"

        file1.write_bytes(b"PDF content 1")
        file2.write_bytes(b"PDF content 2")
        file3.write_bytes(b"PDF content 3")

        # Mark file2 as processed
        temp_tracker.mark_file_processed(
            file2,
            document_id=str(uuid4()),
            chunk_count=5,
        )

        # Get files to process
        files = temp_tracker.get_files_to_process(tmp_path)

        # Should include file1 and file3 (new), but not file2 (processed)
        file_names = [f.name for f in files]
        assert "doc1.pdf" in file_names
        assert "doc3.pdf" in file_names
        assert "doc2.pdf" not in file_names

    def test_registry_persistence(self, tmp_path):
        """Test that registry persists across instances."""
        tracking_file = tmp_path / "persistent_tracking.json"

        # Create first tracker and add data
        tracker1 = FileTracker(tracking_file=tracking_file)
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"Content")

        doc_id = str(uuid4())
        tracker1.mark_file_processed(test_file, document_id=doc_id, chunk_count=3)

        # Create second tracker instance
        tracker2 = FileTracker(tracking_file=tracking_file)

        # Should load previous data
        assert len(tracker2.file_registry) == 1
        file_key = str(test_file.absolute())
        assert file_key in tracker2.file_registry
        assert tracker2.file_registry[file_key]["document_id"] == doc_id

    def test_get_processed_document_ids(self, temp_tracker, tmp_path):
        """Test getting set of processed document IDs."""
        file1 = tmp_path / "doc1.pdf"
        file2 = tmp_path / "doc2.pdf"
        file1.write_bytes(b"Content 1")
        file2.write_bytes(b"Content 2")

        id1 = str(uuid4())
        id2 = str(uuid4())

        temp_tracker.mark_file_processed(file1, document_id=id1, chunk_count=5)
        temp_tracker.mark_file_processed(file2, document_id=id2, chunk_count=3)

        doc_ids = temp_tracker.get_processed_document_ids()

        assert len(doc_ids) == 2
        assert id1 in doc_ids
        assert id2 in doc_ids

    def test_remove_file(self, temp_tracker, tmp_path):
        """Test removing file from tracking."""
        test_file = tmp_path / "to_remove.pdf"
        test_file.write_bytes(b"Content")

        temp_tracker.mark_file_processed(
            test_file,
            document_id=str(uuid4()),
            chunk_count=5,
        )

        assert len(temp_tracker.file_registry) == 1

        temp_tracker.remove_file(test_file)

        assert len(temp_tracker.file_registry) == 0
