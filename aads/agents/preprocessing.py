"""
AADS Preprocessing Agent — fits encoders and scalers on training data,
transforms all splits to ML-ready format, and saves the reusable preprocessing pipeline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord
from aads.core.state import RunState
from aads.tools.processing.preprocessor import (
    build_and_fit_preprocessor,
    transform_with_preprocessor,
)

logger = get_logger(__name__)


class PreprocessingAgent:
    """Agent that builds, fits, and serializes the complete data preprocessing pipeline."""

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
        state: RunState,
        X_val: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Any]:
        """Fit preprocessor on X_train and transform all partitions.

        Args:
            X_train: Training features.
            X_test: Test features.
            state: The current RunState.
            X_val: Optional validation features.

        Returns:
            Tuple of (X_train_enc, X_test_enc, X_val_enc, fitted_pipeline).
        """
        logger.info("preprocessing_agent_start", run_id=state.run_id, n_features=len(X_train.columns))

        pipeline, X_train_enc, feature_names = build_and_fit_preprocessor(X_train, scale_numeric=True)
        X_test_enc = transform_with_preprocessor(pipeline, X_test, feature_names)

        X_val_enc = None
        if X_val is not None and len(X_val) > 0:
            X_val_enc = transform_with_preprocessor(pipeline, X_val, feature_names)

        state.mark_phase_complete("preprocessing")

        # Save artifacts
        if self.artifact_manager:
            try:
                models_dir = self.artifact_manager.get_path("models")
                pipe_path = models_dir / "preprocessing_pipeline.pkl"
                with open(pipe_path, "wb") as f:
                    pickle.dump(pipeline, f)

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.PREPROCESSING_PIPELINE,
                    path=pipe_path,
                    description=f"Fitted Scikit-Learn preprocessing pipeline ({len(feature_names)} encoded features)",
                )

                ml_dir = self.artifact_manager.get_path("ml_ready_data")
                X_train_enc.to_parquet(ml_dir / "X_train_encoded.parquet", index=False)
                X_test_enc.to_parquet(ml_dir / "X_test_encoded.parquet", index=False)
                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.ML_READY_DATA,
                    path=ml_dir / "X_train_encoded.parquet",
                    description="One-hot encoded and scaled training dataset ready for ML models",
                )
            except Exception as e:
                logger.warning("preprocessing_artifact_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="preprocessing",
                action="fit_preprocessing_pipeline",
                reason=f"Fitted imputer, standard scaler, and one-hot encoder on train set; yielded {len(feature_names)} ML-ready features.",
                approval_mode=state.autonomy_mode,
                details={"feature_names": feature_names[:20], "total_features": len(feature_names)},
            )
        )

        logger.info("preprocessing_agent_completed", total_features=len(feature_names))
        return X_train_enc, X_test_enc, X_val_enc, pipeline
