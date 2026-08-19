"""
CSV dataset loader.

Handles ``.csv`` files with automatic delimiter sniffing and encoding
detection for common cases.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aads.core.exceptions import DataLoadError
from aads.core.logging import get_logger
from aads.tools.loaders.base import DatasetLoader

logger = get_logger(__name__)


class CSVLoader(DatasetLoader):
    """Loads CSV files into Pandas DataFrames."""

    def supported_extensions(self) -> list[str]:
        return [".csv", ".tsv", ".txt"]

    def load(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Load a CSV file.

        Supports common delimiters (comma, tab, semicolon, pipe) and will
        attempt ``utf-8`` then ``latin-1`` encoding if not specified.

        Args:
            path: Path to the CSV file.
            **kwargs: Passed through to ``pd.read_csv``.

        Returns:
            DataFrame with the loaded data.
        """
        self.validate(path)
        path = Path(path)

        # Default delimiter sniffing via Python engine if not provided
        read_kwargs: dict = {
            "sep": kwargs.pop("sep", None),
            "encoding": kwargs.pop("encoding", None),
            **kwargs,
        }

        # Try to load with automatic settings
        encodings = [read_kwargs["encoding"]] if read_kwargs["encoding"] else ["utf-8", "latin-1"]

        for enc in encodings:
            try:
                df = pd.read_csv(
                    path,
                    sep=read_kwargs["sep"],
                    encoding=enc,
                    engine="python" if read_kwargs["sep"] is None else None,
                    **{k: v for k, v in read_kwargs.items() if k not in ("sep", "encoding")},
                )
                logger.info(
                    "csv_loaded",
                    path=str(path),
                    rows=len(df),
                    cols=len(df.columns),
                    encoding=enc,
                )
                return df
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                raise DataLoadError(f"Failed to read CSV '{path}': {exc}") from exc

        raise DataLoadError(f"Could not decode CSV '{path}' with tried encodings: {encodings}")
