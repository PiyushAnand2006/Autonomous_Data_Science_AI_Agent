"""
File hashing utility for reproducibility tracking.

Provides a deterministic SHA-256 hash of any file so the system can verify
that the same raw dataset was used across runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from aads.core.exceptions import DataLoadError


def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """Compute a hex-digest hash of a file.

    Reads the file in 64 KB chunks to keep memory usage bounded even for
    very large datasets.

    Args:
        path: Path to the file to hash.
        algorithm: Hash algorithm name (default ``sha256``).

    Returns:
        Hex-encoded hash string.

    Raises:
        DataLoadError: If the file does not exist or cannot be read.
    """
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"Cannot hash: file not found at '{path}'")

    h = hashlib.new(algorithm)
    buffer_size = 65_536  # 64 KB

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(buffer_size)
                if not chunk:
                    break
                h.update(chunk)
    except OSError as exc:
        raise DataLoadError(f"Cannot read file for hashing: {exc}") from exc

    return h.hexdigest()
