"""
AADS Data Cleaner Tool — deterministic data hygiene, missing value imputation,
type sanitization, and outlier treatment.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from aads.core.logging import get_logger

logger = get_logger(__name__)

_HIDDEN_NULL_STRINGS = {"?", "na", "n/a", "null", "none", "", "nan", "inf", "-inf", "missing", "unknown"}


def clean_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    missing_drop_threshold: float = 0.80,
    winsorize_outliers: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a dataset deterministically, returning the cleaned DataFrame and a structured log.

    Args:
        df: Input raw or messy DataFrame.
        target_column: Optional target column (missing target rows are dropped).
        missing_drop_threshold: Drop columns missing more than this fraction of rows.
        winsorize_outliers: Whether to clip severe numeric outliers (outside 3x IQR).

    Returns:
        Tuple of (cleaned_df, cleaning_log_dict).
    """
    cleaned = df.copy()
    n_initial_rows, n_initial_cols = cleaned.shape
    log: dict[str, Any] = {
        "initial_shape": [n_initial_rows, n_initial_cols],
        "dropped_columns": [],
        "dropped_rows": 0,
        "imputations": {},
        "winsorized_columns": [],
        "hidden_nulls_replaced": {},
        "deduplicated_rows": 0,
    }

    # 1. Standardize hidden placeholder null strings in object and string columns
    for col in cleaned.columns:
        if not pd.api.types.is_numeric_dtype(cleaned[col]) and not pd.api.types.is_bool_dtype(cleaned[col]) and not pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            mask = cleaned[col].astype(str).str.strip().str.lower().isin(_HIDDEN_NULL_STRINGS)
            replaced_count = int(mask.sum())
            if replaced_count > 0:
                cleaned.loc[mask, col] = np.nan
                log["hidden_nulls_replaced"][str(col)] = replaced_count

    # 2. Drop exact duplicate rows
    n_dups = int(cleaned.duplicated().sum())
    if n_dups > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        log["deduplicated_rows"] = n_dups

    # 3. Drop rows with missing target values (if target provided)
    if target_column and target_column in cleaned.columns:
        target_nulls = int(cleaned[target_column].isna().sum())
        if target_nulls > 0:
            cleaned = cleaned.dropna(subset=[target_column]).reset_index(drop=True)
            log["dropped_rows"] += target_nulls

    # 4. Handle Date & High-Cardinality ID columns
    cols_to_drop = []
    for col in cleaned.columns:
        if col == target_column:
            continue
        series = cleaned[col]
        missing_pct = series.isna().sum() / len(cleaned) if len(cleaned) > 0 else 0
        unique_cnt = series.dropna().nunique()

        if missing_pct >= missing_drop_threshold:
            cols_to_drop.append((col, f"High missingness ({round(missing_pct*100, 1)}%)"))
            continue
        elif unique_cnt <= 1:
            cols_to_drop.append((col, "Zero variance (constant column)"))
            continue

        # Check if column is a string / object
        if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # Attempt date parsing if string looks like date
            if "date" in str(col).lower() or "time" in str(col).lower():
                try:
                    parsed_dt = pd.to_datetime(series, errors="coerce")
                    if parsed_dt.notna().sum() > 0.5 * len(cleaned):
                        # Extract calendar features if not already present
                        for feat_name, dt_attr in [
                            (f"{col}_year", parsed_dt.dt.year),
                            (f"{col}_month", parsed_dt.dt.month),
                            (f"{col}_day", parsed_dt.dt.day),
                            (f"{col}_dayofweek", parsed_dt.dt.dayofweek),
                            (f"{col}_quarter", parsed_dt.dt.quarter),
                        ]:
                            if feat_name not in cleaned.columns:
                                cleaned[feat_name] = dt_attr.fillna(dt_attr.median())
                        cols_to_drop.append((col, "Parsed into calendar attributes (raw datetime string dropped)"))
                        continue
                except Exception:
                    pass

            # Detect high-cardinality ID columns (unique count > 100 and high ratio)
            if unique_cnt > 100 and (unique_cnt / len(cleaned)) > 0.3:
                cols_to_drop.append((col, f"High-cardinality identifier ({unique_cnt} unique values)"))
                continue

    for col, reason in cols_to_drop:
        cleaned = cleaned.drop(columns=[col])
        log["dropped_columns"].append({"column": str(col), "reason": reason})

    # 5. Missing value imputation
    for col in cleaned.columns:
        if col == target_column:
            continue
        series = cleaned[col]
        missing_count = int(series.isna().sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            fill_val = float(series.median()) if not np.isnan(series.median()) else 0.0
            cleaned[col] = series.fillna(fill_val)
            log["imputations"][str(col)] = {"strategy": "median", "value": fill_val, "count": missing_count}
        else:
            mode_val = series.mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Missing"
            cleaned[col] = series.fillna(fill_val)
            log["imputations"][str(col)] = {"strategy": "mode", "value": str(fill_val), "count": missing_count}

    # 6. Outlier Winsorization / Clipping (3x IQR bounds)
    if winsorize_outliers:
        for col in cleaned.columns:
            if col == target_column:
                continue
            series = cleaned[col]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and len(series) > 10:
                q25, q75 = float(series.quantile(0.25)), float(series.quantile(0.75))
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - 3.0 * iqr
                    upper_bound = q75 + 3.0 * iqr
                    clipped = series.clip(lower=lower_bound, upper=upper_bound)
                    clipped_count = int((series != clipped).sum())
                    if clipped_count > 0:
                        cleaned[col] = clipped
                        log["winsorized_columns"].append({
                            "column": str(col),
                            "lower": lower_bound,
                            "upper": upper_bound,
                            "clipped_count": clipped_count,
                        })

    log["final_shape"] = list(cleaned.shape)
    logger.info("clean_dataset_completed", initial_shape=log["initial_shape"], final_shape=log["final_shape"])
    return cleaned, log
