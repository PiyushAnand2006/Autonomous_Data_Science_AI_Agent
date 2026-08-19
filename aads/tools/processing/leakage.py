"""
AADS Leakage Guard Tools — audits feature-target independence and train/test boundaries.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from aads.core.logging import get_logger

logger = get_logger(__name__)


def audit_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_name: str,
) -> dict[str, Any]:
    """Audit datasets for target leakage, duplicate contamination, and target artifacts.

    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training target.
        y_test: Testing target.
        target_name: Target column name.

    Returns:
        Structured audit report dict.
    """
    report: dict[str, Any] = {
        "has_leakage": False,
        "critical_issues": [],
        "warnings": [],
        "checked_features": len(X_train.columns),
    }

    # 1. Check if target column itself is present in features
    if target_name in X_train.columns:
        report["has_leakage"] = True
        report["critical_issues"].append(f"Target column '{target_name}' is included as an input feature in X_train.")

    # 2. Check for target proxy correlation leakage (|r| >= 0.999)
    numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    if pd.api.types.is_numeric_dtype(y_train) and len(numeric_features) > 0:
        for col in numeric_features:
            non_null_mask = X_train[col].notna() & y_train.notna()
            if non_null_mask.sum() > 5:
                corr = np.corrcoef(X_train.loc[non_null_mask, col], y_train.loc[non_null_mask])[0, 1]
                if not np.isnan(corr) and abs(corr) >= 0.999:
                    report["has_leakage"] = True
                    report["critical_issues"].append(
                        f"Feature '{col}' has near-perfect correlation (r = {corr:.4f}) with target '{target_name}', indicating direct target leakage."
                    )

    # 3. Check for exact row duplicate contamination between Train and Test sets
    if len(X_train) > 0 and len(X_test) > 0 and len(X_train.columns) > 0:
        try:
            # Hash or compare common columns
            train_rows = set(pd.util.hash_pandas_object(X_train, index=False))
            test_rows = set(pd.util.hash_pandas_object(X_test, index=False))
            overlap_count = len(train_rows.intersection(test_rows))
            if overlap_count > 0:
                overlap_pct = (overlap_count / len(X_test)) * 100.0
                if overlap_pct > 10.0:
                    report["has_leakage"] = True
                    report["critical_issues"].append(
                        f"Significant row contamination: {overlap_count} ({overlap_pct:.1f}%) test rows appear identically in the training set."
                    )
                else:
                    report["warnings"].append(
                        f"Minor row overlap: {overlap_count} ({overlap_pct:.1f}%) test rows match training rows."
                    )
        except Exception as e:
            logger.debug("overlap_check_skipped", error=str(e))

    # 4. Check for target-derived naming patterns (e.g. 'churn_rate', 'price_binned')
    target_slug = re.sub(r"[^\w]", "", target_name.lower())
    for col in X_train.columns:
        col_slug = re.sub(r"[^\w]", "", str(col).lower())
        if col_slug != target_slug and target_slug in col_slug and len(target_slug) > 3:
            report["warnings"].append(
                f"Feature '{col}' contains target name '{target_name}' and may be a target-derived proxy."
            )

    logger.info(
        "leakage_audit_completed",
        has_leakage=report["has_leakage"],
        critical_count=len(report["critical_issues"]),
        warnings_count=len(report["warnings"]),
    )
    return report
