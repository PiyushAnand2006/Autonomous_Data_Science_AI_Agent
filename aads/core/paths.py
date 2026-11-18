"""Central path-safety helpers for AADS.

Every user-influenceable path (storage directories, uploaded filenames,
run identifiers) must flow through these helpers before touching the
filesystem. They reject ``..`` traversal segments and enforce containment
within an intended root.
"""

from __future__ import annotations

import re
from pathlib import Path

_TRAVERSAL_SEGMENT = re.compile(r"^(?:\.{1,2})$")
_SEGMENT_SPLIT = re.compile(r"[\\/]+")


def reject_traversal(raw: str | Path, *, field: str = "path") -> str:
    """Raise ValueError if *raw* contains ``..`` / ``.`` traversal segments.

    Returns the unchanged string on success so callers can continue using it.
    """
    text = str(raw)
    for segment in _SEGMENT_SPLIT.split(text):
        if segment and _TRAVERSAL_SEGMENT.match(segment):
            raise ValueError(
                f"{field} contains a path traversal segment ({segment!r}): {text!r}"
            )
    return text


def ensure_within(root: Path, path: Path) -> Path:
    """Resolve *path* and enforce that it stays inside *root*.

    Raises ValueError when the resolved path escapes the root (symlinks are
    resolved on both sides before the containment check).
    """
    root_resolved = Path(root).resolve()
    path_resolved = Path(path).resolve()
    if path_resolved != root_resolved and root_resolved not in path_resolved.parents:
        raise ValueError(
            f"path escapes allowed root: {path_resolved} is not within {root_resolved}"
        )
    return path_resolved


def sanitize_filename(name: str) -> str:
    """Strip any directory components from a user-supplied filename."""
    return Path(str(name)).name
