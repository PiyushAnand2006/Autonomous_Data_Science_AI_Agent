"""
AADS Splitter Tool — leakage-safe train / validation / test partitioning.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aads.core.exceptions import DataLoadError
from aads.core.logging import get_logger
from aads.core.schemas import TaskType

logger = get_logger(__name__)


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.20,
    val_size: float = 0.15,
    task_type: TaskType = TaskType.CLASSIFICATION,
    random_state: int = 42,
    time_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Partition a dataset into X_train, X_val, X_test, y_train, y_val, y_test.

    Args:
        df: Input DataFrame.
        target_column: Target column name.
        test_size: Fraction for test partition (e.g. 0.20).
        val_size: Fraction of the remaining training set for validation (e.g. 0.15).
        task_type: Classification or Regression.
        random_state: Seed for reproducibility.
        time_column: Optional time column for temporal sequential splitting.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    if target_column not in df.columns:
        raise DataLoadError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    n_samples = len(df)
    if n_samples < 5:
        raise DataLoadError(f"Dataset has only {n_samples} rows; cannot perform train/test split.")

    # 1. Temporal splitting (if time_column specified)
    if time_column and time_column in df.columns:
        sorted_indices = df[time_column].sort_values().index
        X_sorted = X.loc[sorted_indices]
        y_sorted = y.loc[sorted_indices]

        n_test = max(1, int(n_samples * test_size))
        n_val = max(1, int(n_samples * val_size)) if val_size > 0 else 0
        n_train = n_samples - n_test - n_val

        X_train, y_train = X_sorted.iloc[:n_train], y_sorted.iloc[:n_train]
        X_val, y_val = X_sorted.iloc[n_train : n_train + n_val], y_sorted.iloc[n_train : n_train + n_val]
        X_test, y_test = X_sorted.iloc[n_train + n_val :], y_sorted.iloc[n_train + n_val :]

        logger.info("temporal_split_completed", train=len(X_train), val=len(X_val), test=len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test

    # 2. Stratification for classification if class distribution permits
    stratify = None
    if task_type == TaskType.CLASSIFICATION:
        class_counts = y.value_counts()
        if (class_counts >= 2).all():
            stratify = y

    # First split: Train+Val vs Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Second split: Train vs Val (from Train+Val)
    if val_size > 0.0:
        val_ratio = val_size / (1.0 - test_size)
        stratify_val = None
        if task_type == TaskType.CLASSIFICATION:
            class_counts_val = y_train_val.value_counts()
            if (class_counts_val >= 2).all():
                stratify_val = y_train_val

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=stratify_val
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val, y_val = pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype)

    logger.info(
        "dataset_split_completed",
        train_rows=len(X_train),
        val_rows=len(X_val),
        test_rows=len(X_test),
        features=len(X.columns),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
