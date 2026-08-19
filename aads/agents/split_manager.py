"""
AADS Split Manager Agent — handles train / val / test splitting,
guarantees data integrity, and exports split partitions to 04_ML_Ready_Data/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord, TaskType
from aads.core.state import RunState
from aads.tools.processing.splitter import split_dataset

logger = get_logger(__name__)


class SplitManager:
    """Agent responsible for dataset partitioning and persisting split artifacts."""

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
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        time_column: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Perform train/val/test split and export to 04_ML_Ready_Data/.

        Args:
            df: Cleaned DataFrame.
            state: The current RunState.
            test_size: Fraction for test set.
            val_size: Fraction for validation set.
            time_column: Optional time column.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        target = state.target_column or df.columns[-1]
        t_size = test_size if test_size is not None else self.config.test_size
        v_size = val_size if val_size is not None else self.config.validation_size
        task_type = state.task_type or TaskType.CLASSIFICATION

        logger.info(
            "split_manager_start",
            run_id=state.run_id,
            target=target,
            test_size=t_size,
            val_size=v_size,
        )

        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            df=df,
            target_column=target,
            test_size=t_size,
            val_size=v_size,
            task_type=task_type,
            random_state=state.random_seed,
            time_column=time_column,
        )

        state.mark_phase_complete("split")

        # Save artifacts into 04_ML_Ready_Data/
        if self.artifact_manager:
            try:
                ml_dir = self.artifact_manager.get_path("ml_ready_data")

                X_train.to_parquet(ml_dir / "X_train.parquet", index=False)
                self.artifact_manager.register_artifact(ArtifactType.TRAIN_SPLIT, ml_dir / "X_train.parquet", "Training features")

                X_test.to_parquet(ml_dir / "X_test.parquet", index=False)
                self.artifact_manager.register_artifact(ArtifactType.TEST_SPLIT, ml_dir / "X_test.parquet", "Test features")

                pd.DataFrame({target: y_train}).to_parquet(ml_dir / "y_train.parquet", index=False)
                pd.DataFrame({target: y_test}).to_parquet(ml_dir / "y_test.parquet", index=False)

                if len(X_val) > 0:
                    X_val.to_parquet(ml_dir / "X_val.parquet", index=False)
                    self.artifact_manager.register_artifact(ArtifactType.VALIDATION_SPLIT, ml_dir / "X_val.parquet", "Validation features")
                    pd.DataFrame({target: y_val}).to_parquet(ml_dir / "y_val.parquet", index=False)

            except Exception as e:
                logger.warning("split_artifact_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="split_manager",
                action="split_dataset",
                reason=f"Partitioned data: Train={len(X_train)} rows, Val={len(X_val)} rows, Test={len(X_test)} rows.",
                approval_mode=state.autonomy_mode,
                details={
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test),
                    "features": list(X_train.columns),
                },
            )
        )

        logger.info("split_manager_completed", train=len(X_train), val=len(X_val), test=len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test
