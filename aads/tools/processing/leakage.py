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
        Structured audit report dict containing has_leakage, leaky_columns, critical_issues, and warnings.
    """
    report: dict[str, Any] = {
        "has_leakage": False,
        "leaky_columns": [],
        "critical_issues": [],
        "warnings": [],
        "checked_features": len(X_train.columns),
    }

    leaky_cols: list[str] = []

    # 1. Check if target column itself is present in features
    if target_name in X_train.columns:
        report["has_leakage"] = True
        leaky_cols.append(target_name)
        report["critical_issues"].append(f"Target column '{target_name}' is included as an input feature in X_train.")

    # 2. Check for target-derived naming patterns (e.g. 'Diabetes_Risk_Score', 'churn_rate')
    target_slug = re.sub(r"[^\w]", "", target_name.lower())
    for col in X_train.columns:
        col_slug = re.sub(r"[^\w]", "", str(col).lower())
        if col_slug != target_slug and len(target_slug) > 3:
            if target_slug in col_slug or (col_slug in target_slug and len(col_slug) > 3):
                if any(k in col_slug for k in ["score", "risk", "class", "target", "pred", "outcome", "result", "level", "prob", "index"]):
                    report["has_leakage"] = True
                    if col not in leaky_cols:
                        leaky_cols.append(str(col))
                    report["critical_issues"].append(
                        f"Feature '{col}' is a target-derived score/proxy for target '{target_name}'."
                    )
                else:
                    report["warnings"].append(
                        f"Feature '{col}' contains target name '{target_name}' and may be a target-derived proxy."
                    )

    # 3. Numeric Target Leakage Correlation (|r| >= 0.99)
    numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    is_num_target = pd.api.types.is_numeric_dtype(y_train) and (pd.api.types.is_float_dtype(y_train) or y_train.nunique() >= 5)

    if is_num_target and len(numeric_features) > 0:
        for col in numeric_features:
            non_null_mask = X_train[col].notna() & y_train.notna()
            if non_null_mask.sum() > 5:
                corr = np.corrcoef(X_train.loc[non_null_mask, col], y_train.loc[non_null_mask])[0, 1]
                if not np.isnan(corr) and abs(corr) >= 0.99:
                    report["has_leakage"] = True
                    if col not in leaky_cols:
                        leaky_cols.append(str(col))
                    report["critical_issues"].append(
                        f"Feature '{col}' has near-perfect correlation (r = {corr:.4f}) with target '{target_name}', indicating direct target leakage."
                    )

    # 4. Categorical Target Leakage Detection (Multi-class / Binary Classification)
    if not is_num_target and len(y_train.dropna()) > 20 and y_train.nunique() > 1:
        # Check categorical features for 100% deterministic class partitioning (e.g. AI_Health_Recommendation)
        cat_features = [c for c in X_train.columns if c not in numeric_features]
        for col in cat_features:
            try:
                ct = pd.crosstab(X_train[col], y_train)
                # If each feature value maps to exactly one target class (zero entropy mapping)
                if len(ct) > 1:
                    max_per_row = ct.max(axis=1)
                    row_sums = ct.sum(axis=1)
                    purity = (max_per_row == row_sums).mean()
                    if purity >= 0.98 and len(row_sums[row_sums > 5]) > 2:
                        report["has_leakage"] = True
                        if col not in leaky_cols:
                            leaky_cols.append(str(col))
                        report["critical_issues"].append(
                            f"Feature '{col}' has a deterministic 1-to-1 mapping with target '{target_name}' (purity = {purity*100:.1f}%), indicating post-outcome target leakage."
                        )
            except Exception:
                pass

        # Check numeric features for non-overlapping boundary cuts of the target (e.g. Risk_Score: Low 13-34, Med 35-64, High 65-100)
        for col in numeric_features:
            if col in leaky_cols:
                continue
            try:
                grouped = X_train.groupby(y_train)[col].agg(["min", "max", "count"]).dropna()
                if len(grouped) > 1 and (grouped["count"] > 10).all():
                    # Sort groups by min
                    sorted_g = grouped.sort_values("min")
                    # Check if max of group i <= min of group i+1 (strict non-overlapping ranges)
                    non_overlapping = True
                    for idx in range(len(sorted_g) - 1):
                        if sorted_g.iloc[idx]["max"] >= sorted_g.iloc[idx + 1]["min"]:
                            non_overlapping = False
                            break
                    if non_overlapping:
                        report["has_leakage"] = True
                        if col not in leaky_cols:
                            leaky_cols.append(str(col))
                        report["critical_issues"].append(
                            f"Feature '{col}' has completely non-overlapping numeric ranges across target '{target_name}' classes, indicating it was discretized directly from '{target_name}'."
                        )
            except Exception:
                pass

    # 5. Check for exact row duplicate contamination between Train and Test sets
    if len(X_train) > 0 and len(X_test) > 0 and len(X_train.columns) > 0:
        try:
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

    report["leaky_columns"] = leaky_cols
    report["has_leakage"] = len(report["critical_issues"]) > 0 or len(leaky_cols) > 0

    logger.info(
        "leakage_audit_completed",
        has_leakage=report["has_leakage"],
        leaky_count=len(leaky_cols),
        critical_count=len(report["critical_issues"]),
        warnings_count=len(report["warnings"]),
    )
    return report
