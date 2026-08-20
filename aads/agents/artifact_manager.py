"""
Artifact Manager — owns all generated file paths and metadata.

Creates the run-scoped output directory following the MASTER_PLAN §10
folder contract. No other agent should invent arbitrary paths; everything
goes through the ArtifactManager.

Usage:
    from aads.agents.artifact_manager import ArtifactManager
    am = ArtifactManager(storage_root="aads/storage/runs")
    run_dir = am.initialize_run(run_id="abc123")
    am.copy_raw_data("/path/to/data.csv")
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from aads.core.exceptions import ArtifactError
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactRecord, ArtifactType

logger = get_logger(__name__)


# Subdirectory contract from MASTER_PLAN §10
_SUBDIRS = {
    "raw_data": "01_Raw_Data",
    "cleaned_data": "02_Cleaned_Data",
    "feature_engineered_data": "03_Feature_Engineered_Data",
    "ml_ready_data": "04_ML_Ready_Data",
    "notebook": "05_Notebook",
    "notebooks": "05_Notebook",
    "models": "06_Models",
    "visualizations": "07_Visualizations",
    "reports": "08_Reports",
    "experiments": "09_Experiments",
    "metadata": "10_Metadata",
}

# Visualization sub-folders
_VIZ_SUBDIRS = [
    "distributions",
    "categorical",
    "correlations",
    "outliers",
    "model_evaluation",
]


class ArtifactManager:
    """Manages the run output directory and artifact registration.

    Attributes:
        storage_root: Base directory under which all run folders live.
        run_id: Unique identifier for the current run.
        run_dir: Absolute path to the current run's output folder.
    """

    def __init__(self, storage_root: str | Path = "aads/storage/runs") -> None:
        self.storage_root = Path(storage_root).resolve()
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self._artifacts: list[ArtifactRecord] = []

    @property
    def current_run_dir(self) -> Optional[Path]:
        """Alias for run_dir."""
        return self.run_dir

    def initialize_run(self, run_id: str) -> Path:
        """Create the full run output directory tree.

        Args:
            run_id: Unique run identifier.

        Returns:
            Absolute path to the run root directory.

        Raises:
            ArtifactError: If the directory already exists.
        """
        self.run_id = run_id
        folder_name = f"Generated_Project_{run_id}"
        self.run_dir = self.storage_root / folder_name

        if self.run_dir.exists():
            raise ArtifactError(
                f"Run directory already exists: {self.run_dir}. "
                "Use a unique run_id or clean up old runs."
            )

        # Create main subdirectories
        for key, subdir in _SUBDIRS.items():
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Create visualization sub-folders
        viz_root = self.run_dir / _SUBDIRS["visualizations"]
        for sub in _VIZ_SUBDIRS:
            (viz_root / sub).mkdir(parents=True, exist_ok=True)

        logger.info("run_directory_created", run_id=run_id, path=str(self.run_dir))
        return self.run_dir

    def _ensure_initialized(self) -> Path:
        """Guard that ensures initialize_run has been called."""
        if self.run_dir is None:
            raise ArtifactError("ArtifactManager not initialized. Call initialize_run() first.")
        return self.run_dir

    def get_path(self, key: str) -> Path:
        """Get the absolute path for a named subdirectory.

        Args:
            key: One of the keys in the subdirectory contract
                 (e.g. 'raw_data', 'models', 'notebook').

        Returns:
            Absolute path to the subdirectory.
        """
        run_dir = self._ensure_initialized()
        if key not in _SUBDIRS:
            raise ArtifactError(
                f"Unknown artifact key '{key}'. Valid keys: {list(_SUBDIRS.keys())}"
            )
        return run_dir / _SUBDIRS[key]

    def copy_raw_data(self, source_path: str | Path) -> Path:
        """Copy the original dataset to 01_Raw_Data/ without modification.

        Args:
            source_path: Path to the raw dataset file.

        Returns:
            Path to the copied file in the run directory.
        """
        run_dir = self._ensure_initialized()
        src = Path(source_path)
        if not src.is_file():
            raise ArtifactError(f"Source file not found: {src}")

        dest = self.get_path("raw_data") / src.name
        shutil.copy2(str(src), str(dest))

        canonical_name = f"original_dataset{src.suffix}"
        canonical_dest = self.get_path("raw_data") / canonical_name
        if canonical_dest != dest:
            shutil.copy2(str(src), str(canonical_dest))

        record = ArtifactRecord(
            artifact_type=ArtifactType.RAW_DATA,
            path=str(canonical_dest.relative_to(run_dir)),
            description=f"Immutable copy of original dataset: {canonical_name}",
        )
        self._artifacts.append(record)

        logger.info("raw_data_copied", source=str(src), dest=str(dest), canonical=str(canonical_dest))
        return dest

    def register_artifact(
        self,
        artifact_type: ArtifactType,
        path: str | Path,
        description: str = "",
    ) -> ArtifactRecord:
        """Register a generated artifact in the internal registry.

        Args:
            artifact_type: Type of the artifact.
            path: Absolute path to the artifact file.
            description: Human-readable description.

        Returns:
            The created ArtifactRecord.
        """
        run_dir = self._ensure_initialized()
        abs_path = Path(path).resolve()

        try:
            rel_path = str(abs_path.relative_to(run_dir))
        except ValueError:
            rel_path = str(abs_path)

        record = ArtifactRecord(
            artifact_type=artifact_type,
            path=rel_path,
            description=description,
        )
        self._artifacts.append(record)
        logger.info(
            "artifact_registered",
            type=artifact_type.value,
            path=rel_path,
        )
        return record

    @property
    def artifacts(self) -> list[ArtifactRecord]:
        """Return a copy of all registered artifacts."""
        return list(self._artifacts)

    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> list[ArtifactRecord]:
        """Filter registered artifacts by type."""
        return [a for a in self._artifacts if a.artifact_type == artifact_type]
