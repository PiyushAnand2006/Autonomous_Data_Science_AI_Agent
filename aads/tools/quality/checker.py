"""
AADS Data Quality Checker — performs multi-point integrity audits on tabular data.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from aads.core.schemas import DataQualityIssue, DataQualityReport

_HIDDEN_NULL_STRINGS = {"?", "na", "n/a", "null", "none", "", "nan", "inf", "-inf", "missing", "unknown"}
_SUSPICIOUS_NON_NEGATIVE = re.compile(r"(^|_)(age|price|salary|revenue|cost|count|quantity|volume|amount|duration|tenure)($|_)", re.IGNORECASE)


def audit_data_quality(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> DataQualityReport:
    """Perform comprehensive data quality checks on a DataFrame.

    Args:
        df: Pandas DataFrame to evaluate.
        target_column: Optional name of the target variable for target-specific audits.

    Returns:
        DataQualityReport containing detected issues, summaries, and an overall score.
    """
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        return DataQualityReport(
            overall_score=0.0,
            issues=[
                DataQualityIssue(
                    issue_type="empty_dataset",
                    column=None,
                    severity="critical",
                    description="The dataset contains zero rows or columns.",
                    affected_count=0,
                    affected_pct=100.0,
                    recommended_action="Provide a non-empty dataset.",
                )
            ],
            has_critical_issues=True,
        )

    issues: list[DataQualityIssue] = []
    missing_summary: dict[str, Any] = {}
    outlier_summary: dict[str, Any] = {}
    constant_columns: list[str] = []
    high_cardinality_columns: list[str] = []

    # 1. Exact Duplicate Rows
    dup_count = int(df.duplicated().sum())
    dup_pct = round((dup_count / n_rows) * 100.0, 2)
    duplicate_summary = {"count": dup_count, "pct": dup_pct}
    if dup_count > 0:
        severity = "high" if dup_pct > 10.0 else "medium" if dup_pct > 1.0 else "low"
        issues.append(
            DataQualityIssue(
                issue_type="duplicate_rows",
                column=None,
                severity=severity,
                description=f"Found {dup_count} ({dup_pct}%) exact duplicate rows.",
                affected_count=dup_count,
                affected_pct=dup_pct,
                recommended_action="Deduplicate rows during cleaning.",
            )
        )

    # 2. Target Column Integrity Check
    if target_column:
        if target_column not in df.columns:
            issues.append(
                DataQualityIssue(
                    issue_type="target_not_found",
                    column=target_column,
                    severity="critical",
                    description=f"Specified target column '{target_column}' is missing from the dataset.",
                    affected_count=n_rows,
                    affected_pct=100.0,
                    recommended_action="Select a valid target column present in the dataset.",
                )
            )
        else:
            target_series = df[target_column]
            target_missing = int(target_series.isna().sum())
            if target_missing > 0:
                t_pct = round((target_missing / n_rows) * 100.0, 2)
                issues.append(
                    DataQualityIssue(
                        issue_type="target_missing_values",
                        column=target_column,
                        severity="critical",
                        description=f"Target column '{target_column}' contains {target_missing} ({t_pct}%) missing values.",
                        affected_count=target_missing,
                        affected_pct=t_pct,
                        recommended_action="Drop rows with missing target values before training.",
                    )
                )

            # Class Imbalance Check (for discrete / categorical target)
            if target_series.nunique() <= 20 and len(target_series.dropna()) > 0:
                val_counts = target_series.value_counts(normalize=True)
                min_class_pct = val_counts.min() * 100.0
                if min_class_pct < 5.0:
                    issues.append(
                        DataQualityIssue(
                            issue_type="severe_class_imbalance",
                            column=target_column,
                            severity="high",
                            description=f"Target '{target_column}' has extreme class imbalance (minority class: {min_class_pct:.2f}%).",
                            affected_count=int(target_series.value_counts().min()),
                            affected_pct=round(min_class_pct, 2),
                            recommended_action="Use stratified splitting, PR-AUC / F1 evaluation, and consider class-weighting or resampling.",
                        )
                    )

    # 3. Column-by-Column Audits
    for col in df.columns:
        series = df[col]
        col_name = str(col)
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / n_rows) * 100.0, 2)
        unique_count = int(series.dropna().nunique())
        is_numeric = bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))

        missing_summary[col_name] = {"missing_count": missing_count, "missing_pct": missing_pct}

        # A. Heavy Missingness
        if missing_pct >= 80.0:
            issues.append(
                DataQualityIssue(
                    issue_type="extreme_missing_values",
                    column=col_name,
                    severity="critical" if col_name != target_column else "critical",
                    description=f"Column '{col_name}' is {missing_pct}% missing.",
                    affected_count=missing_count,
                    affected_pct=missing_pct,
                    recommended_action="Drop column due to excessive missingness (>80%).",
                )
            )
        elif missing_pct >= 40.0:
            issues.append(
                DataQualityIssue(
                    issue_type="high_missing_values",
                    column=col_name,
                    severity="high",
                    description=f"Column '{col_name}' has {missing_pct}% missing values.",
                    affected_count=missing_count,
                    affected_pct=missing_pct,
                    recommended_action="Consider missingness indicator + imputation or evaluation during feature selection.",
                )
            )
        elif missing_pct > 0.0:
            issues.append(
                DataQualityIssue(
                    issue_type="missing_values",
                    column=col_name,
                    severity="low",
                    description=f"Column '{col_name}' has {missing_count} ({missing_pct}%) missing values.",
                    affected_count=missing_count,
                    affected_pct=missing_pct,
                    recommended_action="Impute with median/mode or model-based imputation.",
                )
            )

        # B. Hidden String Nulls in Object Columns
        if not is_numeric and len(series.dropna()) > 0:
            non_null_strs = series.dropna().astype(str).str.strip().str.lower()
            hidden_null_count = int(non_null_strs.isin(_HIDDEN_NULL_STRINGS).sum())
            if hidden_null_count > 0:
                h_pct = round((hidden_null_count / n_rows) * 100.0, 2)
                issues.append(
                    DataQualityIssue(
                        issue_type="hidden_null_strings",
                        column=col_name,
                        severity="medium",
                        description=f"Column '{col_name}' contains {hidden_null_count} ({h_pct}%) placeholder null strings (e.g. '?', 'NA', 'None').",
                        affected_count=hidden_null_count,
                        affected_pct=h_pct,
                        recommended_action="Standardize placeholder strings to true NaN during data cleaning.",
                    )
                )

        # C. Constant Columns
        if unique_count <= 1:
            constant_columns.append(col_name)
            issues.append(
                DataQualityIssue(
                    issue_type="constant_column",
                    column=col_name,
                    severity="high",
                    description=f"Column '{col_name}' has only {unique_count} distinct value(s) and provides zero variance.",
                    affected_count=n_rows,
                    affected_pct=100.0,
                    recommended_action="Drop constant column to simplify model and avoid colinearity.",
                )
            )

        # D. High Cardinality Categoricals / ID Leakage
        if not is_numeric and unique_count > 50 and (unique_count / n_rows) > 0.7:
            high_cardinality_columns.append(col_name)
            issues.append(
                DataQualityIssue(
                    issue_type="high_cardinality_identifier",
                    column=col_name,
                    severity="medium",
                    description=f"Column '{col_name}' has {unique_count} distinct categories ({round((unique_count/n_rows)*100, 1)}% cardinality).",
                    affected_count=unique_count,
                    affected_pct=round((unique_count / n_rows) * 100.0, 2),
                    recommended_action="Exclude identifier columns from features to avoid overfitting.",
                )
            )

        # E. Numeric Outliers & Range Anomalies
        if is_numeric and len(series.dropna()) > 10:
            non_null = series.dropna().astype(float)
            q25, q75 = float(non_null.quantile(0.25)), float(non_null.quantile(0.75))
            iqr = q75 - q25
            severe_outliers = 0
            severe_lower, severe_upper = None, None

            if iqr > 0:
                severe_lower = q25 - 3.0 * iqr
                severe_upper = q75 + 3.0 * iqr
                severe_outliers = int(((non_null < severe_lower) | (non_null > severe_upper)).sum())
            elif non_null.std() > 0:
                # Fallback to 3 standard deviations when IQR is 0 (e.g. dominant single value)
                mean_val = float(non_null.mean())
                std_val = float(non_null.std())
                severe_lower = mean_val - 3.0 * std_val
                severe_upper = mean_val + 3.0 * std_val
                severe_outliers = int(((non_null < severe_lower) | (non_null > severe_upper)).sum())

            if severe_outliers > 0:
                out_pct = round((severe_outliers / len(non_null)) * 100.0, 2)
                outlier_summary[col_name] = {
                    "count": severe_outliers,
                    "pct": out_pct,
                    "lower_bound": severe_lower,
                    "upper_bound": severe_upper,
                }
                if out_pct > 0.5:
                    issues.append(
                        DataQualityIssue(
                            issue_type="extreme_outliers",
                            column=col_name,
                            severity="medium",
                            description=f"Column '{col_name}' has {severe_outliers} ({out_pct}%) extreme outliers outside normal distribution/IQR.",
                            affected_count=severe_outliers,
                            affected_pct=out_pct,
                            recommended_action="Consider robust scaling, winsorization, or log transformation.",
                        )
                    )

            # Check for suspicious negative values
            if _SUSPICIOUS_NON_NEGATIVE.search(col_name):
                neg_count = int((non_null < 0).sum())
                if neg_count > 0:
                    neg_pct = round((neg_count / len(non_null)) * 100.0, 2)
                    issues.append(
                        DataQualityIssue(
                            issue_type="suspicious_negative_values",
                            column=col_name,
                            severity="high",
                            description=f"Column '{col_name}' represents a strictly positive concept but has {neg_count} negative values.",
                            affected_count=neg_count,
                            affected_pct=neg_pct,
                            recommended_action="Investigate sensor/logging errors and clip or impute invalid negative values.",
                        )
                    )

    # 4. Compute Health Score
    score = 100.0
    for issue in issues:
        if issue.severity == "critical":
            score -= 25.0
        elif issue.severity == "high":
            score -= 10.0
        elif issue.severity == "medium":
            score -= 5.0
        elif issue.severity == "low":
            score -= 2.0
    score = max(0.0, min(100.0, score))

    has_critical = any(i.severity == "critical" for i in issues)

    return DataQualityReport(
        overall_score=round(score, 1),
        issues=issues,
        missing_summary=missing_summary,
        duplicate_summary=duplicate_summary,
        outlier_summary=outlier_summary,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        has_critical_issues=has_critical,
    )
