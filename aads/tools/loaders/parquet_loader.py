"""
Parquet dataset loader.

Handles ``.parquet`` files via PyArrow.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aads.core.exceptions import DataLoadError
from aads.core.logging import get_logger
from aads.tools.loaders.base import DatasetLoader

logger = get_logger(__name__)


class ParquetLoader(DatasetLoader):
    """Loads Parquet files into Pandas DataFrames."""

    def supported_extensions(self) -> list[str]:
        return [".parquet", ".pq"]

    def load(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Load a Parquet file.

        Args:
            path: Path to the Parquet file.
            **kwargs: Passed through to ``pd.read_parquet``.

        Returns:
            DataFrame with the loaded data.
        """
        self.validate(path)
        path = Path(path)

        try:
            df = pd.read_parquet(path, engine="pyarrow", **kwargs)
            logger.info(
                "parquet_loaded",
                path=str(path),
                rows=len(df),
                cols=len(df.columns),
            )
            return df
        except Exception as exc:
            raise DataLoadError(f"Failed to read Parquet file '{path}': {exc}") from exc
