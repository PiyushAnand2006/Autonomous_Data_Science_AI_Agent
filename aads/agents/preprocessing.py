"""
AADS Preprocessing Agent — fits adaptive encoders and scalers on training data,
transforms all splits to ML-ready format, and saves the reusable preprocessing pipeline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord
from aads.core.state import RunState
from aads.tools.processing.preprocessor import (
    AdaptivePreprocessor,
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
        llm: Any = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager
        self.llm = llm

    def _consult_ai_encoding_strategy(
        self,
        X_train: pd.DataFrame,
        state: RunState,
    ) -> Optional[str]:
        """Query LLM in AI Mode for domain-aware encoding strategy and column reasoning."""
        if getattr(self.config, "execution_mode", "local") != "ai":
            return None

        llm_client = self.llm
        if llm_client is None:
            try:
                from aads.core.llm import get_llm
                llm_client = get_llm(self.config)
            except Exception:
                return None

        try:
            col_summary = []
            for col in X_train.columns[:25]:
                nunique = X_train[col].nunique(dropna=False)
                dtype = str(X_train[col].dtype)
                col_summary.append(f"- {col} (dtype={dtype}, distinct={nunique})")

            prompt = (
                f"You are a Principal Machine Learning Engineer. Analyze these feature columns for preprocessing:\n"
                + "\n".join(col_summary)
                + f"\n\nObjective: {state.user_objective}\n"
                + "Recommend the optimal encoding strategy (One-Hot for low-cardinality nominals, Frequency/Target encoding for high-cardinality identifiers, and Standard scaling for numericals) in 2-3 sentences."
            )
            response = llm_client.invoke(prompt)
            return str(getattr(response, "content", "") or response).strip()
        except Exception as e:
            logger.debug("ai_encoding_consultation_skipped", error=str(e))
            return None

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        state: RunState,
        X_val: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], AdaptivePreprocessor]:
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

        # Consult AI if in AI Mode
        ai_rationale = self._consult_ai_encoding_strategy(X_train, state)

        # Fit adaptive pipeline
        pipeline, X_train_enc, feature_names = build_and_fit_preprocessor(X_train, scale_numeric=True)
        X_test_enc = transform_with_preprocessor(pipeline, X_test, feature_names)

        X_val_enc = None
        if X_val is not None and len(X_val) > 0:
            X_val_enc = transform_with_preprocessor(pipeline, X_val, feature_names)

        state.mark_phase_complete("preprocessing")

        state.add_decision(
            DecisionRecord(
                agent="PreprocessingAgent",
                action="adaptive_feature_encoding",
                reason=ai_rationale or f"Partitioned {len(pipeline.low_card_cols)} low-cardinality features to One-Hot, {len(pipeline.high_card_cols)} high-cardinality features to Frequency Encoding, and scaled {len(pipeline.num_cols)} numericals.",
                details={
                    "total_features_out": len(feature_names),
                    "numeric_count": len(pipeline.num_cols),
                    "low_cardinality_count": len(pipeline.low_card_cols),
                    "high_cardinality_count": len(pipeline.high_card_cols),
                },
            )
        )

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
                    description=f"Fitted adaptive preprocessing pipeline ({len(feature_names)} encoded features)",
                )

                ml_dir = self.artifact_manager.get_path("ml_ready_data")

                # Export full ML-ready dataset as CSV and Parquet
                full_ml_df = pd.concat([X_train_enc, X_test_enc], axis=0).reset_index(drop=True)
                ml_csv_path = ml_dir / "ml_ready_dataset.csv"
                full_ml_df.to_csv(ml_csv_path, index=False)

                ml_parquet_path = ml_dir / "ml_ready_dataset.parquet"
                full_ml_df.to_parquet(ml_parquet_path, index=False)

                X_train_enc.to_parquet(ml_dir / "X_train_encoded.parquet", index=False)
                X_test_enc.to_parquet(ml_dir / "X_test_encoded.parquet", index=False)

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.ML_READY_DATA,
                    path=ml_csv_path,
                    description=f"Final ML-ready encoded dataset CSV ({len(full_ml_df)} rows × {len(full_ml_df.columns)} features)",
                )
                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.ML_READY_DATA,
                    path=ml_dir / "X_train_encoded.parquet",
                    description=f"ML-ready training partition ({len(feature_names)} features)",
                )
            except Exception as e:
                logger.warning("preprocessing_artifact_registration_failed", error=str(e))

        logger.info("preprocessing_agent_completed", total_features=len(feature_names))
        return X_train_enc, X_test_enc, X_val_enc, pipeline
