"""
Abstract base class for dataset loaders.

Every file-format loader must inherit from ``DatasetLoader`` and implement
the ``load``, ``supported_extensions``, and ``validate`` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DatasetLoader(ABC):
    """Abstract base class for all AADS dataset loaders."""

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return a list of file extensions this loader handles (e.g. ['.csv']).

        Extensions must be lowercase and include the leading dot.
        """

    def supports(self, path: str | Path) -> bool:
        """Check whether this loader can handle the given file path."""
        ext = Path(path).suffix.lower()
        return ext in self.supported_extensions()

    @abstractmethod
    def load(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Load a file into a Pandas DataFrame.

        Args:
            path: Absolute or relative path to the data file.
            **kwargs: Loader-specific options.

        Returns:
            A Pandas DataFrame with the loaded data.

        Raises:
            DataLoadError: If the file cannot be read or validated.
        """

    def validate(self, path: str | Path) -> None:
        """Validate that the file exists and has a supported extension.

        Args:
            path: Path to validate.

        Raises:
            DataLoadError: If validation fails.
        """
        from aads.core.exceptions import DataLoadError

        p = Path(path)
        if not p.exists():
            raise DataLoadError(f"File not found: {p}")
        if not p.is_file():
            raise DataLoadError(f"Path is not a file: {p}")
        if not self.supports(p):
            raise DataLoadError(
                f"Unsupported extension '{p.suffix}' for {self.__class__.__name__}. "
                f"Supported: {self.supported_extensions()}"
            )
