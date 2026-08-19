"""
AADS ML Experiment Agent — trains multiple candidate models, evaluates performance,
and selects the best performing model artifact.
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord, ExperimentRecord, TaskType
from aads.core.state import RunState
from aads.tools.ml.trainer import get_candidate_models, train_and_evaluate_model

logger = get_logger(__name__)


class MLExperimentAgent:
    """Agent that conducts reproducible ML experiments, tunes models, and tracks best models."""

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
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        state: RunState,
        candidate_models: Optional[list[str]] = None,
    ) -> tuple[Any, str, dict[str, float], list[ExperimentRecord]]:
        """Run ML training experiments across candidate models and select the best model.

        Args:
            X_train: Encoded train features.
            y_train: Train target.
            X_eval: Encoded validation or test features.
            y_eval: Validation or test target.
            state: The current RunState.
            candidate_models: Optional model names to train.

        Returns:
            Tuple of (best_model, best_model_name, best_metrics, all_experiments).
        """
        task_type = state.task_type or TaskType.CLASSIFICATION
        models_to_run = candidate_models or get_candidate_models(task_type)
        primary_metric = "rmse" if task_type == TaskType.REGRESSION else "f1"

        logger.info(
            "ml_experiment_agent_start",
            run_id=state.run_id,
            models=models_to_run,
            primary_metric=primary_metric,
        )

        all_experiments: list[ExperimentRecord] = []
        best_model: Any = None
        best_model_name: str = ""
        best_score: float = float("inf") if task_type == TaskType.REGRESSION else -float("inf")
        best_metrics: dict[str, float] = {}

        for idx, model_name in enumerate(models_to_run):
            exp_id = f"exp_{state.run_id}_{idx+1:02d}_{model_name}"
            is_baseline = (idx == 0)

            fitted_model, metrics, _ = train_and_evaluate_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_eval,
                y_val=y_eval,
                task_type=task_type,
                random_state=state.random_seed,
            )

            current_score = metrics.get(primary_metric, 0.0)
            is_better = (current_score < best_score) if task_type == TaskType.REGRESSION else (current_score > best_score)

            if is_better or best_model is None:
                best_score = current_score
                best_model = fitted_model
                best_model_name = model_name
                best_metrics = metrics

            exp_rec = ExperimentRecord(
                experiment_id=exp_id,
                model_name=model_name,
                hyperparameters={},
                metrics=metrics,
                is_baseline=is_baseline,
                is_best=False,  # Updated after loop
                notes=f"Trained {model_name} with default parameters",
            )
            all_experiments.append(exp_rec)
            state.add_experiment(exp_rec)

        # Mark best experiment
        for exp in all_experiments:
            if exp.model_name == best_model_name:
                exp.is_best = True

        state.mark_phase_complete("ml_experiment")

        # Save artifacts: best_model.pkl and experiment_results.csv
        if self.artifact_manager:
            try:
                models_dir = self.artifact_manager.get_path("models")
                best_model_path = models_dir / "best_model.pkl"
                with open(best_model_path, "wb") as f:
                    pickle.dump(best_model, f)

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.MODEL,
                    path=best_model_path,
                    description=f"Best ML model: {best_model_name} ({primary_metric}: {best_score})",
                )

                exp_dir = self.artifact_manager.get_path("experiments")
                exp_rows = [
                    {"experiment_id": e.experiment_id, "model": e.model_name, "is_best": e.is_best, **e.metrics}
                    for e in all_experiments
                ]
                exp_df = pd.DataFrame(exp_rows)
                exp_csv_path = exp_dir / "experiment_results.csv"
                exp_df.to_csv(exp_csv_path, index=False)

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.EXPERIMENT_LOG,
                    path=exp_csv_path,
                    description=f"Summary of {len(all_experiments)} model experiments",
                )
            except Exception as e:
                logger.warning("ml_experiment_artifact_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="ml_experiment",
                action="train_candidate_models",
                reason=f"Evaluated {len(models_to_run)} model(s). Selected {best_model_name} as best model ({primary_metric} = {best_score}).",
                approval_mode=state.autonomy_mode,
                details={"best_model": best_model_name, "best_metrics": best_metrics, "total_runs": len(all_experiments)},
            )
        )

        logger.info(
            "ml_experiment_agent_completed",
            best_model=best_model_name,
            best_score=best_score,
            metric=primary_metric,
        )
        return best_model, best_model_name, best_metrics, all_experiments
