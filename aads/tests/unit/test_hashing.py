"""Tests for aads.tools.filesystem.hashing — file hash computation."""

from pathlib import Path

import pytest

from aads.core.exceptions import DataLoadError
from aads.tools.filesystem.hashing import compute_file_hash


class TestComputeFileHash:

    def test_consistent_hash(self, tmp_path):
        """Same content must produce the same hash every time."""
        f = tmp_path / "test.csv"
        f.write_text("a,b,c\n1,2,3\n")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        """Different content must produce different hashes."""
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("hello")
        f2.write_text("world")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_hash_is_hex_string(self, tmp_path):
        """Hash should be a lowercase hex string."""
        f = tmp_path / "test.csv"
        f.write_text("data")
        h = compute_file_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 = 64 hex chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_nonexistent_file_raises(self):
        with pytest.raises(DataLoadError, match="not found"):
            compute_file_hash("/nonexistent/file.csv")

    def test_custom_algorithm(self, tmp_path):
        """Support alternate hash algorithms."""
        f = tmp_path / "test.csv"
        f.write_text("data")
        h_md5 = compute_file_hash(f, algorithm="md5")
        h_sha = compute_file_hash(f, algorithm="sha256")
        assert h_md5 != h_sha
        assert len(h_md5) == 32  # MD5 = 32 hex chars
