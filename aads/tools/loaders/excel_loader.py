"""
Excel dataset loader.

Handles ``.xlsx`` and ``.xls`` files using ``openpyxl`` (for xlsx) and
the Pandas built-in xlrd fallback (for xls).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aads.core.exceptions import DataLoadError
from aads.core.logging import get_logger
from aads.tools.loaders.base import DatasetLoader

logger = get_logger(__name__)


class ExcelLoader(DatasetLoader):
    """Loads Excel files into Pandas DataFrames."""

    def supported_extensions(self) -> list[str]:
        return [".xlsx", ".xls"]

    def load(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Load an Excel file.

        Args:
            path: Path to the Excel file.
            **kwargs: Passed through to ``pd.read_excel`` (e.g. ``sheet_name``).

        Returns:
            DataFrame with the loaded data.
        """
        self.validate(path)
        path = Path(path)

        try:
            engine = "openpyxl" if path.suffix.lower() == ".xlsx" else None
            df = pd.read_excel(path, engine=engine, **kwargs)
            logger.info(
                "excel_loaded",
                path=str(path),
                rows=len(df),
                cols=len(df.columns),
            )
            return df
        except Exception as exc:
            raise DataLoadError(f"Failed to read Excel file '{path}': {exc}") from exc
