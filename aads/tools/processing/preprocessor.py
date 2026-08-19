"""
AADS Preprocessor Tool — Scikit-Learn based column encoding and scaling pipelines.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from aads.core.logging import get_logger

logger = get_logger(__name__)


def build_and_fit_preprocessor(
    X_train: pd.DataFrame,
    scale_numeric: bool = True,
) -> tuple[ColumnTransformer, pd.DataFrame, list[str]]:
    """Build and fit a leakage-safe preprocessing pipeline on X_train only.

    Args:
        X_train: Training features.
        scale_numeric: Whether to apply scaling to numeric variables.

    Returns:
        Tuple of (fitted_column_transformer, X_train_transformed_df, feature_names).
    """
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c]) and not pd.api.types.is_bool_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    transformers = []

    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))

    if cat_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.fit(X_train)

    # Transform training set to DataFrame
    X_train_trans = preprocessor.transform(X_train)
    feature_names = _extract_feature_names(preprocessor, num_cols, cat_cols)

    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    logger.info("preprocessor_fit_completed", num_features=len(num_cols), cat_features=len(cat_cols), total_out=len(feature_names))
    return preprocessor, X_train_df, feature_names


def transform_with_preprocessor(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Transform validation or test features using an already fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Unseen validation or test features.
        feature_names: Feature names generated during training fit.

    Returns:
        Transformed DataFrame.
    """
    if len(X) == 0:
        return pd.DataFrame(columns=feature_names)
    arr = preprocessor.transform(X)
    return pd.DataFrame(arr, columns=feature_names, index=X.index)


def _extract_feature_names(
    preprocessor: ColumnTransformer,
    num_cols: list[str],
    cat_cols: list[str],
) -> list[str]:
    """Retrieve output feature names from the fitted preprocessor."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = list(num_cols)
        for c in cat_cols:
            names.append(f"cat_{c}")
        return names
