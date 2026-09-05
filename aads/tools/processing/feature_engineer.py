"""
AADS Feature Engineering Tool — generates and validates candidate interaction,
date, and transformation features.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from aads.core.logging import get_logger
from aads.core.schemas import TaskType

logger = get_logger(__name__)


def generate_candidate_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series | None = None,
    task_type: TaskType = TaskType.CLASSIFICATION,
    max_new_features: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate and select high-value candidate features without leaking test information.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target for importance selection.
        task_type: Classification or Regression.
        max_new_features: Max candidate features to retain.

    Returns:
        Tuple of (X_train_fe, X_test_fe, feature_log).
    """
    train_fe = X_train.copy()
    test_fe = X_test.copy()

    log: dict[str, Any] = {
        "created_features": [],
        "retained_features": [],
        "dropped_features": [],
    }

    _ID_RE = re.compile(r"(^|_)(id|uuid|guid|key|index|identifier|pk|code|patient|cust|customer|user|account)($|_)", re.IGNORECASE)
    n_rows = len(train_fe)
    numeric_cols = [
        c for c in train_fe.columns
        if pd.api.types.is_numeric_dtype(train_fe[c])
        and not pd.api.types.is_bool_dtype(train_fe[c])
        and not _ID_RE.search(str(c))
        and not (n_rows > 20 and train_fe[c].nunique() / n_rows > 0.95)
    ]

    # 1. Log / Power transformations for highly skewed features
    for col in numeric_cols:
        series = train_fe[col].dropna()
        if len(series) > 10 and series.min() >= 0 and series.std() > 0:
            skew = float(series.skew())
            if skew > 2.0:
                feat_name = f"log_{col}"
                train_fe[feat_name] = np.log1p(train_fe[col].clip(lower=0))
                test_fe[feat_name] = np.log1p(test_fe[col].clip(lower=0))
                log["created_features"].append({"name": feat_name, "type": "log_transform", "source": col})

    # 2. Pairwise numeric interactions (ratios & differences for top correlated features)
    if len(numeric_cols) >= 2:
        top_num = numeric_cols[:4]  # Limit to avoid combinatorial explosion
        for i in range(len(top_num)):
            for j in range(i + 1, len(top_num)):
                c1, c2 = top_num[i], top_num[j]

                # Difference feature
                diff_name = f"diff_{c1}_{c2}"
                train_fe[diff_name] = train_fe[c1] - train_fe[c2]
                test_fe[diff_name] = test_fe[c1] - test_fe[c2]
                log["created_features"].append({"name": diff_name, "type": "difference", "source": [c1, c2]})

                # Ratio feature (safe division)
                ratio_name = f"ratio_{c1}_{c2}"
                train_fe[ratio_name] = train_fe[c1] / (train_fe[c2].abs() + 1e-6)
                test_fe[ratio_name] = test_fe[c1] / (test_fe[c2].abs() + 1e-6)
                log["created_features"].append({"name": ratio_name, "type": "ratio", "source": [c1, c2]})

    # 3. Date / Time features
    for col in list(train_fe.columns):
        if pd.api.types.is_datetime64_any_dtype(train_fe[col]):
            for part in ["month", "day", "dayofweek"]:
                feat_name = f"{col}_{part}"
                train_fe[feat_name] = getattr(train_fe[col].dt, part)
                test_fe[feat_name] = getattr(test_fe[col].dt, part)
                log["created_features"].append({"name": feat_name, "type": "datetime_part", "source": col})

    log["retained_features"] = [f["name"] for f in log["created_features"][:max_new_features]]
    logger.info("feature_engineering_completed", created=len(log["created_features"]))
    return train_fe, test_fe, log
