"""
AADS Schemas — Pydantic models and enums shared across the system.

These are the canonical data contracts. Agents and tools communicate
through these types to keep interfaces explicit and validated.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """Supported ML / analysis task types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    DESCRIPTIVE = "descriptive"
    ANOMALY = "anomaly"
    FORECASTING = "forecasting"


class AutonomyMode(str, Enum):
    """Level of human oversight during a run."""

    FULLY_AUTONOMOUS = "fully_autonomous"
    SEMI_AUTONOMOUS = "semi_autonomous"
    MANUAL_APPROVAL = "manual_approval"


class ArtifactType(str, Enum):
    """Categories of artifacts produced during a run."""

    RAW_DATA = "raw_data"
    CLEANED_DATA = "cleaned_data"
    FEATURE_ENGINEERED_DATA = "feature_engineered_data"
    ML_READY_DATA = "ml_ready_data"
    TRAIN_SPLIT = "train_split"
    VALIDATION_SPLIT = "validation_split"
    TEST_SPLIT = "test_split"
    NOTEBOOK = "notebook"
    MODEL = "model"
    PREPROCESSING_PIPELINE = "preprocessing_pipeline"
    VISUALIZATION = "visualization"
    REPORT = "report"
    EXPERIMENT_LOG = "experiment_log"
    METADATA = "metadata"
    DECISION_LOG = "decision_log"


class ExecutionEngine(str, Enum):
    """Data-processing engine selected based on dataset size."""

    PANDAS = "pandas"
    POLARS = "polars"
    DUCKDB = "duckdb"
    DASK = "dask"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class DatasetMeta(BaseModel):
    """Metadata captured during dataset profiling."""

    file_path: str = Field(description="Path to the original uploaded file")
    file_hash: str = Field(description="SHA-256 hash of the raw file")
    file_size_bytes: int = Field(default=0, description="Size of the raw file in bytes")
    file_format: str = Field(default="csv", description="Detected file format (csv, xlsx, parquet)")
    n_rows: int = Field(default=0, description="Number of rows")
    n_cols: int = Field(default=0, description="Number of columns")
    column_names: list[str] = Field(default_factory=list, description="Column names")
    dtypes: dict[str, str] = Field(default_factory=dict, description="Column name → dtype string")
    memory_mb: float = Field(default=0.0, description="Estimated in-memory size in MB")
    missing_cells: int = Field(default=0, description="Total missing cells")
    duplicate_rows: int = Field(default=0, description="Number of duplicate rows")
    execution_engine: ExecutionEngine = Field(
        default=ExecutionEngine.PANDAS,
        description="Processing engine selected for this dataset",
    )


class ArtifactRecord(BaseModel):
    """Registry entry for a generated artifact."""

    artifact_type: ArtifactType
    path: str = Field(description="Relative path from the run root")
    description: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionRecord(BaseModel):
    """Log entry for an autonomous or approved decision."""

    agent: str = Field(description="Name of the agent that made the decision")
    action: str = Field(description="Short description of what was done")
    reason: str = Field(description="Why this action was taken")
    approved: bool = Field(default=True, description="Whether the action was approved")
    approval_mode: AutonomyMode = Field(default=AutonomyMode.FULLY_AUTONOMOUS)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (affected rows, metrics, etc.)",
    )


class ExperimentRecord(BaseModel):
    """Summary of a single ML experiment run."""

    experiment_id: str = Field(description="Unique experiment identifier")
    model_name: str = Field(description="Name/family of the model")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    is_baseline: bool = Field(default=False)
    is_best: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    notes: str = Field(default="")


# ---------------------------------------------------------------------------
# Phase 2 — Profiling and Planning models
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    """Per-column profiling statistics."""

    name: str
    dtype: str = Field(description="Pandas dtype string")
    is_numeric: bool = False
    is_categorical: bool = False
    is_datetime: bool = False
    is_boolean: bool = False

    # Missing values
    missing_count: int = 0
    missing_pct: float = Field(default=0.0, description="Percentage of missing values")

    # Cardinality
    unique_count: int = 0
    cardinality_pct: float = Field(default=0.0, description="unique / total rows %")

    # Numeric stats (None when not numeric)
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    zeros_count: int | None = None
    negative_count: int | None = None

    # Categorical stats (None when not categorical)
    top_values: list[dict[str, Any]] | None = Field(
        default=None, description="Top-N value:count pairs"
    )

    # Flags
    is_suspected_id: bool = Field(default=False, description="Looks like an ID column")
    is_constant: bool = Field(default=False, description="Only one unique value")
    is_all_null: bool = Field(default=False, description="Entirely missing")


class DatasetProfile(BaseModel):
    """Complete profiling result for a dataset."""

    # Basic shape
    n_rows: int
    n_cols: int
    memory_mb: float

    # Column-level
    columns: list[ColumnProfile]

    # Dataset-level summaries
    total_missing_cells: int = 0
    total_missing_pct: float = 0.0
    duplicate_rows: int = 0
    duplicate_pct: float = 0.0

    # Column classification lists
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    boolean_columns: list[str] = Field(default_factory=list)

    # Suspicious columns
    suspected_id_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)
    all_null_columns: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(
        default_factory=list, description="Categorical cols with cardinality > 50"
    )

    # Target detection
    target_candidates: list[str] = Field(
        default_factory=list, description="Columns that look like potential targets"
    )

    # Execution engine recommendation
    recommended_engine: ExecutionEngine = ExecutionEngine.PANDAS

    def summary_text(self) -> str:
        """Return a concise text summary for LLM consumption."""
        lines = [
            f"Dataset: {self.n_rows} rows × {self.n_cols} columns ({self.memory_mb:.1f} MB)",
            f"Missing: {self.total_missing_cells} cells ({self.total_missing_pct:.1f}%)",
            f"Duplicates: {self.duplicate_rows} rows ({self.duplicate_pct:.1f}%)",
            f"Numeric columns ({len(self.numeric_columns)}): {self.numeric_columns[:10]}",
            f"Categorical columns ({len(self.categorical_columns)}): {self.categorical_columns[:10]}",
            f"Datetime columns: {self.datetime_columns}",
            f"Suspected ID columns: {self.suspected_id_columns}",
            f"Constant columns: {self.constant_columns}",
            f"High cardinality categoricals: {self.high_cardinality_columns}",
            f"Target candidates: {self.target_candidates}",
            f"Recommended engine: {self.recommended_engine.value}",
        ]
        return "\n".join(lines)


class PlanStep(BaseModel):
    """A single step in the workflow plan."""

    step_id: str = Field(description="Step identifier from allowed vocabulary")
    description: str = Field(default="", description="What this step will do")
    depends_on: list[str] = Field(
        default_factory=list, description="Step IDs that must complete first"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Step-specific configuration hints"
    )


# Allowed step vocabulary (adapted from AI Data Science Team's pattern)
ALLOWED_PLAN_STEPS = [
    "profiling",
    "data_quality",
    "eda",
    "cleaning",
    "split",
    "leakage_check",
    "feature_engineering",
    "preprocessing",
    "ml_experiment",
    "evaluation",
    "replanning",
    "notebook_generation",
    "report_generation",
]


class TaskPlan(BaseModel):
    """Structured plan produced by the Goal/Planning Agent."""

    task_type: TaskType = Field(description="Detected task type")
    target_column: str | None = Field(
        default=None, description="Target column for supervised tasks"
    )
    steps: list[PlanStep] = Field(description="Ordered workflow steps")
    reasoning: str = Field(
        default="", description="LLM's explanation for the chosen plan"
    )
    questions: list[str] = Field(
        default_factory=list,
        description="Clarifying questions if info is missing",
    )
    metric: str | None = Field(
        default=None, description="Primary evaluation metric"
    )
    notes: list[str] = Field(
        default_factory=list, description="Additional observations"
    )


# ---------------------------------------------------------------------------
# Phase 3 — Data Quality and EDA models
# ---------------------------------------------------------------------------

class DataQualityIssue(BaseModel):
    """An individual data quality issue detected in the dataset."""

    issue_type: str = Field(description="Category of issue (e.g. missing_values, extreme_outliers, constant_column)")
    column: str | None = Field(default=None, description="Affected column name, or None if row/dataset-level")
    severity: str = Field(default="medium", description="Severity level: low, medium, high, critical")
    description: str = Field(description="Human-readable description of the issue")
    affected_count: int = Field(default=0, description="Number of affected rows/cells")
    affected_pct: float = Field(default=0.0, description="Percentage of dataset affected")
    recommended_action: str = Field(description="Suggested remediation strategy")


class DataQualityReport(BaseModel):
    """Comprehensive data quality audit report."""

    overall_score: float = Field(default=100.0, description="Data health score (0-100)")
    issues: list[DataQualityIssue] = Field(default_factory=list)
    missing_summary: dict[str, Any] = Field(default_factory=dict)
    duplicate_summary: dict[str, Any] = Field(default_factory=dict)
    outlier_summary: dict[str, Any] = Field(default_factory=dict)
    constant_columns: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(default_factory=list)
    has_critical_issues: bool = Field(default=False)

    def summary_text(self) -> str:
        """Textual summary for logging and agent reasoning."""
        critical_count = sum(1 for i in self.issues if i.severity == "critical")
        high_count = sum(1 for i in self.issues if i.severity == "high")
        lines = [
            f"Data Quality Score: {self.overall_score:.1f}/100",
            f"Total Issues Detected: {len(self.issues)} (Critical: {critical_count}, High: {high_count})",
        ]
        for issue in self.issues[:10]:
            col_str = f" in '{issue.column}'" if issue.column else ""
            lines.append(f"- [{issue.severity.upper()}] {issue.issue_type}{col_str}: {issue.description}")
        return "\n".join(lines)


class EDAFindings(BaseModel):
    """Structured insights generated by Exploratory Data Analysis."""

    summary: str = Field(default="", description="High-level narrative of dataset characteristics")
    target_column: str | None = Field(default=None)
    univariate_insights: list[str] = Field(default_factory=list)
    bivariate_insights: list[str] = Field(default_factory=list)
    correlation_insights: list[str] = Field(default_factory=list)
    generated_visualizations: list[str] = Field(default_factory=list, description="Relative paths to saved charts")


# ---------------------------------------------------------------------------
# Phase 6 & 7 — Model Export & Notebook Validation Models
# ---------------------------------------------------------------------------

class TopModelRecord(BaseModel):
    """Metadata for a selected top-performing model."""

    rank: int = Field(description="1-based rank (1 is best)")
    model: str = Field(description="Name/algorithm of the model")
    filename: str = Field(description="Exported model filename (e.g. model_01_xgboost.pkl)")
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    test_metrics: dict[str, float] = Field(default_factory=dict)
    training_time: float = Field(default=0.0, description="Training time in seconds")
    feature_set: str = Field(default="feature_engineered", description="Feature set used")
    selection_reason: str = Field(default="", description="Why this model was selected for its rank")


class ModelMetadata(BaseModel):
    """Comprehensive export metadata for all top selected models."""

    top_models: list[TopModelRecord] = Field(default_factory=list)
    total_models_evaluated: int = 0
    selection_metric: str = "primary_metric"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class NotebookValidationResult(BaseModel):
    """Validation report from programmatic top-to-bottom notebook execution."""

    success: bool = True
    executed_cells: int = 0
    total_cells: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)

