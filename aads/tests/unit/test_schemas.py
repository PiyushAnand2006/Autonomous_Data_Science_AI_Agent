"""Tests for aads.core.schemas — enums and Pydantic models."""

from datetime import datetime

import pytest

from aads.core.schemas import (
    ArtifactRecord,
    ArtifactType,
    AutonomyMode,
    DatasetMeta,
    DecisionRecord,
    ExecutionEngine,
    ExperimentRecord,
    TaskType,
)


class TestEnums:
    """Verify enum members and string values."""

    def test_task_type_values(self):
        assert TaskType.REGRESSION.value == "regression"
        assert TaskType.CLASSIFICATION.value == "classification"
        assert TaskType.CLUSTERING.value == "clustering"
        assert TaskType.DESCRIPTIVE.value == "descriptive"
        assert TaskType.ANOMALY.value == "anomaly"
        assert TaskType.FORECASTING.value == "forecasting"

    def test_autonomy_mode_values(self):
        assert AutonomyMode.FULLY_AUTONOMOUS.value == "fully_autonomous"
        assert AutonomyMode.SEMI_AUTONOMOUS.value == "semi_autonomous"
        assert AutonomyMode.MANUAL_APPROVAL.value == "manual_approval"

    def test_artifact_type_has_expected_members(self):
        expected = {
            "RAW_DATA", "CLEANED_DATA", "FEATURE_ENGINEERED_DATA",
            "ML_READY_DATA", "TRAIN_SPLIT", "VALIDATION_SPLIT", "TEST_SPLIT",
            "NOTEBOOK", "MODEL", "PREPROCESSING_PIPELINE",
            "VISUALIZATION", "REPORT", "EXPERIMENT_LOG",
            "METADATA", "DECISION_LOG",
        }
        actual = {m.name for m in ArtifactType}
        assert expected.issubset(actual)

    def test_execution_engine_values(self):
        assert ExecutionEngine.PANDAS.value == "pandas"
        assert ExecutionEngine.POLARS.value == "polars"

    def test_task_type_from_string(self):
        assert TaskType("regression") == TaskType.REGRESSION


class TestDatasetMeta:
    """Verify DatasetMeta creation and defaults."""

    def test_minimal_creation(self):
        meta = DatasetMeta(file_path="/data/test.csv", file_hash="abc123")
        assert meta.file_path == "/data/test.csv"
        assert meta.file_hash == "abc123"
        assert meta.n_rows == 0
        assert meta.n_cols == 0
        assert meta.memory_mb == 0.0

    def test_full_creation(self):
        meta = DatasetMeta(
            file_path="/data/test.csv",
            file_hash="abc123",
            file_size_bytes=1024,
            file_format="csv",
            n_rows=100,
            n_cols=10,
            column_names=["a", "b"],
            dtypes={"a": "int64", "b": "float64"},
            memory_mb=0.5,
            missing_cells=3,
            duplicate_rows=1,
        )
        assert meta.n_rows == 100
        assert meta.column_names == ["a", "b"]

    def test_serialization_roundtrip(self):
        meta = DatasetMeta(file_path="/data/test.csv", file_hash="abc123", n_rows=50)
        data = meta.model_dump()
        restored = DatasetMeta.model_validate(data)
        assert restored.n_rows == 50
        assert restored.file_hash == "abc123"


class TestArtifactRecord:
    """Verify ArtifactRecord."""

    def test_creation(self):
        record = ArtifactRecord(
            artifact_type=ArtifactType.RAW_DATA,
            path="01_Raw_Data/data.csv",
            description="Raw upload",
        )
        assert record.artifact_type == ArtifactType.RAW_DATA
        assert record.path == "01_Raw_Data/data.csv"
        assert isinstance(record.created_at, datetime)


class TestDecisionRecord:
    """Verify DecisionRecord."""

    def test_creation_defaults(self):
        record = DecisionRecord(
            agent="cleaning",
            action="remove_duplicates",
            reason="42 exact duplicates found",
        )
        assert record.approved is True
        assert record.approval_mode == AutonomyMode.FULLY_AUTONOMOUS

    def test_creation_with_details(self):
        record = DecisionRecord(
            agent="cleaning",
            action="drop_rows",
            reason="Outliers in price column",
            approved=False,
            details={"rows_affected": 150},
        )
        assert record.approved is False
        assert record.details["rows_affected"] == 150


class TestExperimentRecord:
    """Verify ExperimentRecord."""

    def test_creation(self):
        record = ExperimentRecord(
            experiment_id="exp_001",
            model_name="RandomForestRegressor",
            metrics={"rmse": 1.5, "r2": 0.89},
            is_baseline=True,
        )
        assert record.experiment_id == "exp_001"
        assert record.metrics["r2"] == 0.89
        assert record.is_baseline is True
        assert record.is_best is False
