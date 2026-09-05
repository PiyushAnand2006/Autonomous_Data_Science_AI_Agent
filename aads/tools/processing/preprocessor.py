"""
AADS Preprocessor Tool — Scikit-Learn based adaptive column encoding and scaling pipelines.
Intelligently partitions low-cardinality nominals (One-Hot) and high-cardinality features
(Frequency/Ordinal Encoding) to prevent column explosion while maximizing predictive signal.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aads.core.logging import get_logger

logger = get_logger(__name__)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns by normalized category frequency.

    High-cardinality categories are transformed into a single dense continuous feature
    representing category frequency in the training split. Unseen categories default to 0.
    """

    def __init__(self, cols: Optional[list[str]] = None) -> None:
        self.cols = cols or []
        self.freq_maps_: dict[str, dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> FrequencyEncoder:
        X_df = pd.DataFrame(X)
        self.freq_maps_ = {}
        for col in self.cols:
            if col in X_df.columns:
                val_counts = X_df[col].astype(str).value_counts(normalize=True).to_dict()
                self.freq_maps_[col] = val_counts
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col, fmap in self.freq_maps_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].astype(str).map(fmap).fillna(0.0).astype(float)
        return X_df


class AdaptivePreprocessor(BaseEstimator, TransformerMixin):
    """Complete adaptive preprocessing pipeline combining numeric scaling,

    low-cardinality one-hot encoding, and high-cardinality frequency encoding.
    """

    def __init__(
        self,
        num_cols: Optional[list[str]] = None,
        low_card_cols: Optional[list[str]] = None,
        high_card_cols: Optional[list[str]] = None,
        scale_numeric: bool = True,
    ) -> None:
        self.num_cols = num_cols or []
        self.low_card_cols = low_card_cols or []
        self.high_card_cols = high_card_cols or []
        self.scale_numeric = scale_numeric

        self.num_imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.low_card_imputer_: Optional[SimpleImputer] = None
        self.onehot_: Optional[OneHotEncoder] = None
        self.freq_encoder_: Optional[FrequencyEncoder] = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> AdaptivePreprocessor:
        X_df = pd.DataFrame(X).copy().replace([np.inf, -np.inf], np.nan)
        names: list[str] = []

        # 1. Numeric pipeline
        if self.num_cols:
            self.num_imputer_ = SimpleImputer(strategy="median")
            num_data = self.num_imputer_.fit_transform(X_df[self.num_cols])
            if self.scale_numeric:
                self.scaler_ = StandardScaler()
                self.scaler_.fit(num_data)
            names.extend(self.num_cols)

        # 2. High-cardinality frequency pipeline
        if self.high_card_cols:
            self.freq_encoder_ = FrequencyEncoder(cols=self.high_card_cols)
            self.freq_encoder_.fit(X_df[self.high_card_cols])
            for col in self.high_card_cols:
                names.append(f"freq_{col}")

        # 3. Low-cardinality one-hot pipeline
        if self.low_card_cols:
            self.low_card_imputer_ = SimpleImputer(strategy="constant", fill_value="missing")
            cat_data = self.low_card_imputer_.fit_transform(X_df[self.low_card_cols])
            self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)
            self.onehot_.fit(cat_data)
            try:
                oh_names = list(self.onehot_.get_feature_names_out(self.low_card_cols))
            except Exception:
                oh_names = [f"oh_{col}_{idx}" for col in self.low_card_cols for idx in range(10)]
            names.extend(oh_names)

        self.feature_names_ = names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy().replace([np.inf, -np.inf], np.nan)
        parts: list[pd.DataFrame] = []

        # 1. Numeric
        if self.num_cols and self.num_imputer_:
            num_data = self.num_imputer_.transform(X_df[self.num_cols])
            if self.scaler_:
                num_data = self.scaler_.transform(num_data)
            parts.append(pd.DataFrame(num_data, columns=self.num_cols, index=X_df.index))

        # 2. High-cardinality frequency
        if self.high_card_cols and self.freq_encoder_:
            freq_df = self.freq_encoder_.transform(X_df[self.high_card_cols])
            freq_df.columns = [f"freq_{c}" for c in self.high_card_cols]
            parts.append(freq_df)

        # 3. Low-cardinality one-hot
        if self.low_card_cols and self.low_card_imputer_ and self.onehot_:
            cat_data = self.low_card_imputer_.transform(X_df[self.low_card_cols])
            oh_data = self.onehot_.transform(cat_data)
            try:
                oh_names = list(self.onehot_.get_feature_names_out(self.low_card_cols))
            except Exception:
                oh_names = [f"oh_{i}" for i in range(oh_data.shape[1])]
            parts.append(pd.DataFrame(oh_data, columns=oh_names, index=X_df.index))

        if not parts:
            return pd.DataFrame(index=X_df.index)

        res_df = pd.concat(parts, axis=1).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return res_df

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return self.feature_names_


def build_and_fit_preprocessor(
    X_train: pd.DataFrame,
    scale_numeric: bool = True,
    max_low_cardinality: int = 12,
) -> tuple[AdaptivePreprocessor, pd.DataFrame, list[str]]:
    """Build and fit a leakage-safe adaptive preprocessing pipeline on X_train only.

    Args:
        X_train: Training features DataFrame.
        scale_numeric: Whether to apply StandardScaler to numeric columns.
        max_low_cardinality: Maximum distinct values for One-Hot encoding.

    Returns:
        Tuple of (fitted_preprocessor, X_train_transformed_df, feature_names).
    """
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c]) and not pd.api.types.is_bool_dtype(X_train[c])]
    cat_candidates = [c for c in X_train.columns if c not in num_cols]

    low_card_cols: list[str] = []
    high_card_cols: list[str] = []

    for col in cat_candidates:
        nunique = X_train[col].nunique(dropna=False)
        if nunique <= max_low_cardinality:
            low_card_cols.append(col)
        else:
            high_card_cols.append(col)

    preprocessor = AdaptivePreprocessor(
        num_cols=num_cols,
        low_card_cols=low_card_cols,
        high_card_cols=high_card_cols,
        scale_numeric=scale_numeric,
    )
    preprocessor.fit(X_train)
    X_train_df = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    logger.info(
        "adaptive_preprocessor_fit_completed",
        numeric=len(num_cols),
        low_cardinality_onehot=len(low_card_cols),
        high_cardinality_frequency=len(high_card_cols),
        total_ml_ready_features=len(feature_names),
    )
    return preprocessor, X_train_df, feature_names


def transform_with_preprocessor(
    preprocessor: AdaptivePreprocessor,
    X: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Transform validation or test features using an already fitted adaptive preprocessor.

    Args:
        preprocessor: Fitted AdaptivePreprocessor.
        X: Unseen validation or test features.
        feature_names: Feature names generated during training fit.

    Returns:
        Transformed DataFrame matching training schema.
    """
    if len(X) == 0:
        return pd.DataFrame(columns=feature_names)
    transformed_df = preprocessor.transform(X)
    # Ensure columns alignment
    for col in feature_names:
        if col not in transformed_df.columns:
            transformed_df[col] = 0.0
    return transformed_df[feature_names]
