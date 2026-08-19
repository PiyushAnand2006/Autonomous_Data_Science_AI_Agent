"""
AADS Dataset Profiler Agent — wraps deterministic profiling tools,
updates RunState with dataset metadata, and exports profiling artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import (
    ArtifactType,
    DatasetMeta,
    DatasetProfile,
    DecisionRecord,
)
from aads.core.state import RunState
from aads.tools.profiling.profiler import profile_dataset

logger = get_logger(__name__)


class ProfilerAgent:
    """Agent responsible for inspecting and profiling raw datasets."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager

    def run(
        self,
        df: pd.DataFrame,
        state: RunState,
        file_path: str = "",
        file_hash: str = "",
        file_format: str = "csv",
    ) -> DatasetProfile:
        """Profile the dataset, update state, and persist metadata artifact.

        Args:
            df: The loaded pandas DataFrame.
            state: The current RunState.
            file_path: Path to the original raw file.
            file_hash: SHA-256 hash of the raw file.
            file_format: Format string (csv, xlsx, parquet, etc.).

        Returns:
            Computed DatasetProfile.
        """
        logger.info("profiler_agent_start", run_id=state.run_id, shape=df.shape)

        profile = profile_dataset(df, config=self.config)

        # Build DatasetMeta for RunState
        meta = DatasetMeta(
            file_path=file_path,
            file_hash=file_hash,
            file_size_bytes=int(Path(file_path).stat().st_size) if file_path and Path(file_path).exists() else 0,
            file_format=file_format,
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
            column_names=[col.name for col in profile.columns],
            dtypes={col.name: col.dtype for col in profile.columns},
            memory_mb=profile.memory_mb,
            missing_cells=profile.total_missing_cells,
            duplicate_rows=profile.duplicate_rows,
            execution_engine=profile.recommended_engine,
        )

        state.dataset_meta = meta
        state.mark_phase_complete("profiling")

        # Save metadata artifact if artifact_manager is present
        if self.artifact_manager:
            try:
                meta_dir = self.artifact_manager.get_path("metadata")
                profile_path = meta_dir / "dataset_profile.json"
                profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=profile_path,
                    description="Comprehensive statistical profile of the dataset",
                )
            except Exception as e:
                logger.warning("profiler_artifact_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="profiler",
                action="profile_dataset",
                reason=f"Profiled {profile.n_rows} rows and {profile.n_cols} cols; selected {profile.recommended_engine.value} engine.",
                approval_mode=state.autonomy_mode,
                details={
                    "shape": [profile.n_rows, profile.n_cols],
                    "missing_pct": profile.total_missing_pct,
                    "target_candidates": profile.target_candidates,
                },
            )
        )

        logger.info(
            "profiler_agent_completed",
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
            engine=profile.recommended_engine.value,
        )

        return profile
