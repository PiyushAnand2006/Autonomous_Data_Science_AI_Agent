"""
AADS Feature Engineering Agent — discovers, constructs, and selects candidate features,
exporting transformed datasets to 03_Feature_Engineered_Data/.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord, TaskType
from aads.core.state import RunState
from aads.tools.processing.feature_engineer import generate_candidate_features

logger = get_logger(__name__)


class FeatureEngineeringAgent:
    """Agent that creates interaction, transformation, and domain features."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        state: RunState,
        X_val: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict[str, Any]]:
        """Run feature engineering on training features and project to test/val sets.

        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training target.
            state: The current RunState.
            X_val: Optional validation features.

        Returns:
            Tuple of (X_train_fe, X_test_fe, X_val_fe, feature_log).
        """
        logger.info("feature_engineering_agent_start", run_id=state.run_id)
        task_type = state.task_type or TaskType.CLASSIFICATION

        X_train_fe, X_test_fe, log = generate_candidate_features(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            task_type=task_type,
            max_new_features=10,
        )

        X_val_fe = None
        if X_val is not None and len(X_val) > 0:
            _, X_val_fe, _ = generate_candidate_features(
                X_train=X_train,
                X_test=X_val,
                y_train=y_train,
                task_type=task_type,
                max_new_features=10,
            )

        state.mark_phase_complete("feature_engineering")

        # Save artifacts to 03_Feature_Engineered_Data/
        if self.artifact_manager:
            try:
                fe_dir = self.artifact_manager.get_path("feature_engineered_data")

                X_train_fe.to_parquet(fe_dir / "X_train_fe.parquet", index=False)
                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.FEATURE_ENGINEERED_DATA,
                    path=fe_dir / "X_train_fe.parquet",
                    description=f"Feature-engineered training data ({len(X_train_fe.columns)} features)",
                )

                log_path = fe_dir / "feature_engineering_log.json"
                log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=log_path,
                    description="Detailed log of generated feature candidates",
                )
            except Exception as e:
                logger.warning("feature_engineering_artifact_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="feature_engineering",
                action="generate_features",
                reason=f"Engineered {len(log['created_features'])} candidate features (retained {len(log['retained_features'])}).",
                approval_mode=state.autonomy_mode,
                details=log,
            )
        )

        logger.info(
            "feature_engineering_agent_completed",
            new_features=len(log["created_features"]),
            total_features=len(X_train_fe.columns),
        )
        return X_train_fe, X_test_fe, X_val_fe, log
