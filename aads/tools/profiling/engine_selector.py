"""
AADS Engine Selector — dynamically selects the execution engine
based on dataset size, memory usage, and user configuration.
"""

from __future__ import annotations

from aads.core.config import AADSConfig
from aads.core.schemas import ExecutionEngine


def select_engine(
    n_rows: int,
    memory_mb: float,
    config: AADSConfig | None = None,
) -> ExecutionEngine:
    """Select the appropriate data-processing engine.

    Rules (per MASTER_PLAN §8):
    - Small datasets (< row limit or < 100MB): Pandas (default).
    - Medium / Large datasets (>= row limit or >= 100MB): Polars or DuckDB.

    Args:
        n_rows: Number of rows in the dataset.
        memory_mb: Estimated memory usage in MB.
        config: Optional configuration with custom thresholds.

    Returns:
        ExecutionEngine enum value (PANDAS, POLARS, DUCKDB, DASK).
    """
    row_limit = config.pandas_row_limit if config else 500_000
    preferred_default = config.default_engine if config else ExecutionEngine.PANDAS

    # If the user explicitly configured a non-pandas default, respect it
    if preferred_default != ExecutionEngine.PANDAS:
        return preferred_default

    # Memory > 500MB or rows > threshold -> Polars
    if n_rows >= row_limit or memory_mb >= 500.0:
        return ExecutionEngine.POLARS

    return ExecutionEngine.PANDAS
