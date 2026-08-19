"""
Loader Registry — auto-discovers and routes files to the correct loader.

Usage:
    from aads.tools.loaders import LoaderRegistry

    registry = LoaderRegistry()
    df = registry.load("path/to/data.csv")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aads.core.exceptions import DataLoadError
from aads.core.logging import get_logger
from aads.tools.loaders.base import DatasetLoader
from aads.tools.loaders.csv_loader import CSVLoader
from aads.tools.loaders.excel_loader import ExcelLoader
from aads.tools.loaders.parquet_loader import ParquetLoader

logger = get_logger(__name__)


class LoaderRegistry:
    """Registry that maps file extensions to the appropriate DatasetLoader.

    All built-in loaders are registered automatically on construction.
    Custom loaders can be added via ``register()``.
    """

    def __init__(self) -> None:
        self._loaders: list[DatasetLoader] = []
        # Register built-in loaders
        self.register(CSVLoader())
        self.register(ExcelLoader())
        self.register(ParquetLoader())

    def register(self, loader: DatasetLoader) -> None:
        """Add a loader to the registry.

        Args:
            loader: A DatasetLoader instance to register.
        """
        self._loaders.append(loader)
        logger.debug(
            "loader_registered",
            loader=loader.__class__.__name__,
            extensions=loader.supported_extensions(),
        )

    def get_loader(self, path: str | Path) -> DatasetLoader:
        """Find the loader that supports the given file path.

        Args:
            path: Path to the data file.

        Returns:
            The first matching DatasetLoader.

        Raises:
            DataLoadError: If no registered loader supports the file extension.
        """
        for loader in self._loaders:
            if loader.supports(path):
                return loader

        ext = Path(path).suffix.lower()
        supported = []
        for loader in self._loaders:
            supported.extend(loader.supported_extensions())
        raise DataLoadError(
            f"No loader registered for extension '{ext}'. "
            f"Supported extensions: {sorted(set(supported))}"
        )

    def load(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Load a dataset by auto-detecting the correct loader.

        Args:
            path: Path to the data file.
            **kwargs: Passed through to the selected loader.

        Returns:
            A Pandas DataFrame with the loaded data.
        """
        loader = self.get_loader(path)
        logger.info(
            "loading_dataset",
            loader=loader.__class__.__name__,
            path=str(path),
        )
        return loader.load(path, **kwargs)

    @property
    def supported_extensions(self) -> list[str]:
        """Return all supported file extensions across registered loaders."""
        exts: list[str] = []
        for loader in self._loaders:
            exts.extend(loader.supported_extensions())
        return sorted(set(exts))
