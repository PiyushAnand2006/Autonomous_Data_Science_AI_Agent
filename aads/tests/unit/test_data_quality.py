"""
Tests for AADS Data Quality Tools and Checker.
"""

import numpy as np
import pandas as pd
import pytest

from aads.core.schemas import DataQualityReport
from aads.tools.quality.checker import audit_data_quality


@pytest.fixture
def messy_dataset() -> pd.DataFrame:
    """Create a dataset with intentional quality flaws."""
    return pd.DataFrame({
        "id": range(100),
        "age": [25, 30, -5, 40] + [35] * 96,  # negative age flaw
        "income": [50000.0] * 90 + [10_000_000.0] * 10,  # extreme outlier
        "feedback": ["Great", "?", "N/A", "none"] + ["Good"] * 96,  # hidden null strings
        "constant_feature": [42] * 100,  # zero variance constant
        "almost_empty": [1.0] * 5 + [np.nan] * 95,  # >80% missing
        "churn": [0] * 98 + [1] * 2,  # severe class imbalance (2%)
    })


class TestDataQualityAudit:
    """Verify detection of various data quality issues."""

    def test_audit_identifies_injected_issues(self, messy_dataset):
        report = audit_data_quality(messy_dataset, target_column="churn")

        assert isinstance(report, DataQualityReport)
        assert report.overall_score < 80.0  # Quality score should be reduced due to issues

        issue_types = {issue.issue_type for issue in report.issues}

        # Verify specific detections
        assert "suspicious_negative_values" in issue_types  # age < 0
        assert "extreme_outliers" in issue_types  # 10M income
        assert "hidden_null_strings" in issue_types  # '?', 'N/A'
        assert "constant_column" in issue_types  # constant_feature
        assert "extreme_missing_values" in issue_types  # almost_empty
        assert "severe_class_imbalance" in issue_types  # churn (2%)

    def test_audit_handles_clean_dataset(self):
        clean_df = pd.DataFrame({
            "a": np.random.normal(50, 5, 100),
            "b": np.random.choice(["X", "Y"], 100),
            "target": np.random.choice([0, 1], 100),
        })
        report = audit_data_quality(clean_df, target_column="target")
        assert report.overall_score >= 90.0
        assert not report.has_critical_issues

    def test_audit_handles_missing_target(self, messy_dataset):
        report = audit_data_quality(messy_dataset, target_column="nonexistent_target")
        assert report.has_critical_issues
        assert any(i.issue_type == "target_not_found" for i in report.issues)

    def test_audit_handles_empty_dataframe(self):
        empty_df = pd.DataFrame()
        report = audit_data_quality(empty_df)
        assert report.has_critical_issues
        assert report.overall_score == 0.0
