"""
AADS Profiler Tool — deterministic statistical profiling of tabular datasets.

Computes comprehensive schema, missing value, cardinality, distribution,
and heuristic target/anomaly indicators on Pandas DataFrames without requiring an LLM.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from aads.core.config import AADSConfig
from aads.core.schemas import ColumnProfile, DatasetProfile, ExecutionEngine
from aads.tools.profiling.engine_selector import select_engine

# Common ID-like column name patterns
_ID_PATTERNS = re.compile(r"(^|_)(id|uuid|guid|key|index|identifier|pk|code)($|_)", re.IGNORECASE)

# Common target column name patterns
_TARGET_PATTERNS = re.compile(
    r"(^|_)(target|label|class|y|response|outcome|price|salary|churn|survived|status|default|sales|revenue)($|_)",
    re.IGNORECASE,
)


def _is_id_column(name: str, unique_count: int, n_rows: int, is_numeric: bool) -> bool:
    """Check whether a column is likely an identifier."""
    if n_rows == 0:
        return False
    # If 100% unique and name matches ID pattern
    if unique_count == n_rows:
        if _ID_PATTERNS.search(name) or not is_numeric:
            return True
    # If very high cardinality (>98%) and matches pattern
    if unique_count / n_rows > 0.98 and _ID_PATTERNS.search(name):
        return True
    return False


def profile_dataset(
    df: pd.DataFrame,
    config: AADSConfig | None = None,
) -> DatasetProfile:
    """Compute deterministic statistical profile of a dataset.

    Args:
        df: Pandas DataFrame to profile.
        config: Optional configuration for engine selection thresholds.

    Returns:
        A complete DatasetProfile model.
    """
    n_rows, n_cols = df.shape
    memory_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)

    # Missing & duplicates
    total_cells = n_rows * n_cols if (n_rows * n_cols) > 0 else 1
    total_missing_cells = int(df.isna().sum().sum())
    total_missing_pct = float((total_missing_cells / total_cells) * 100.0) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum()) if n_rows > 0 else 0
    duplicate_pct = float((duplicate_rows / n_rows) * 100.0) if n_rows > 0 else 0.0

    column_profiles: list[ColumnProfile] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    boolean_cols: list[str] = []

    suspected_ids: list[str] = []
    constant_cols: list[str] = []
    all_null_cols: list[str] = []
    high_card_cols: list[str] = []
    target_candidates: list[str] = []

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)

        # Missing counts
        missing_count = int(series.isna().sum())
        missing_pct = float((missing_count / n_rows) * 100.0) if n_rows > 0 else 0.0

        # Unique & cardinality
        non_null_series = series.dropna()
        unique_count = int(non_null_series.nunique())
        cardinality_pct = float((unique_count / n_rows) * 100.0) if n_rows > 0 else 0.0

        is_num = bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))
        is_bool = bool(pd.api.types.is_bool_dtype(series))
        is_dt = bool(pd.api.types.is_datetime64_any_dtype(series))
        # If object/category or string
        is_cat = not is_num and not is_bool and not is_dt

        # Check flags
        is_all_null = missing_count == n_rows
        is_const = unique_count == 1
        is_id = _is_id_column(str(col), unique_count, n_rows, is_num)

        if is_all_null:
            all_null_cols.append(str(col))
        if is_const:
            constant_cols.append(str(col))
        if is_id:
            suspected_ids.append(str(col))

        # Categorize
        if is_num:
            numeric_cols.append(str(col))
        elif is_bool:
            boolean_cols.append(str(col))
        elif is_dt:
            datetime_cols.append(str(col))
        else:
            categorical_cols.append(str(col))
            if unique_count > 50 and not is_id:
                high_card_cols.append(str(col))

        # Check target candidates
        if _TARGET_PATTERNS.search(str(col)) and not is_id and not is_const and not is_all_null:
            target_candidates.append(str(col))

        # Numeric statistics
        min_val, max_val, mean_val, median_val, std_val = None, None, None, None, None
        skew_val, kurt_val, zeros_count, neg_count = None, None, None, None

        if is_num and len(non_null_series) > 0:
            try:
                min_val = float(non_null_series.min())
                max_val = float(non_null_series.max())
                mean_val = float(non_null_series.mean())
                median_val = float(non_null_series.median())
                std_val = float(non_null_series.std()) if len(non_null_series) > 1 else 0.0
                if len(non_null_series) > 2 and std_val != 0.0:
                    skew_val = float(non_null_series.skew())
                    kurt_val = float(non_null_series.kurtosis())
                zeros_count = int((non_null_series == 0).sum())
                neg_count = int((non_null_series < 0).sum())
            except Exception:
                pass

        # Categorical top values
        top_values: list[dict[str, Any]] | None = None
        if (is_cat or is_bool) and len(non_null_series) > 0:
            val_counts = non_null_series.value_counts().head(5)
            top_values = [{"value": str(k), "count": int(v)} for k, v in val_counts.items()]

        col_prof = ColumnProfile(
            name=str(col),
            dtype=dtype_str,
            is_numeric=is_num,
            is_categorical=is_cat,
            is_datetime=is_dt,
            is_boolean=is_bool,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            cardinality_pct=cardinality_pct,
            min=min_val,
            max=max_val,
            mean=mean_val,
            median=median_val,
            std=std_val,
            skew=skew_val,
            kurtosis=kurt_val,
            zeros_count=zeros_count,
            negative_count=neg_count,
            top_values=top_values,
            is_suspected_id=is_id,
            is_constant=is_const,
            is_all_null=is_all_null,
        )
        column_profiles.append(col_prof)

    # If no target candidate found from names, suggest the last non-id, non-constant column if available
    if not target_candidates and column_profiles:
        candidates = [c.name for c in column_profiles if not c.is_suspected_id and not c.is_constant and not c.is_all_null]
        if candidates:
            target_candidates.append(candidates[-1])

    recommended_engine = select_engine(n_rows, memory_mb, config)

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        memory_mb=round(memory_mb, 4),
        columns=column_profiles,
        total_missing_cells=total_missing_cells,
        total_missing_pct=round(total_missing_pct, 2),
        duplicate_rows=duplicate_rows,
        duplicate_pct=round(duplicate_pct, 2),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        boolean_columns=boolean_cols,
        suspected_id_columns=suspected_ids,
        constant_columns=constant_cols,
        all_null_columns=all_null_cols,
        high_cardinality_columns=high_card_cols,
        target_candidates=target_candidates,
        recommended_engine=recommended_engine,
    )
