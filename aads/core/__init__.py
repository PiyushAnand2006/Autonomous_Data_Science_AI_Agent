"""
AADS Core — configuration, state, schemas, logging, and exceptions.
"""

from aads.core.config import AADSConfig
from aads.core.exceptions import (
    AADSError,
    ArtifactError,
    ConfigError,
    DataLoadError,
    LeakageError,
)
from aads.core.schemas import (
    ArtifactRecord,
    ArtifactType,
    AutonomyMode,
    ColumnProfile,
    DataQualityIssue,
    DataQualityReport,
    DatasetMeta,
    DatasetProfile,
    DecisionRecord,
    EDAFindings,
    ExecutionEngine,
    ModelMetadata,
    NotebookValidationResult,
    PlanStep,
    TaskPlan,
    TaskType,
    TopModelRecord,
)
from aads.core.state import RunState

__all__ = [
    "AADSConfig",
    "RunState",
    "DatasetMeta",
    "DatasetProfile",
    "ColumnProfile",
    "TaskType",
    "TaskPlan",
    "PlanStep",
    "AutonomyMode",
    "ArtifactType",
    "ArtifactRecord",
    "DecisionRecord",
    "DataQualityIssue",
    "DataQualityReport",
    "EDAFindings",
    "ExecutionEngine",
    "TopModelRecord",
    "ModelMetadata",
    "NotebookValidationResult",
    "AADSError",
    "DataLoadError",
    "LeakageError",
    "ArtifactError",
    "ConfigError",
]
