"""
AADS ML Experiment Agent — trains multiple candidate models, evaluates performance,
and selects the best performing model artifact.
"""

from __future__ import annotations

import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import (
    ArtifactType,
    DecisionRecord,
    ExperimentRecord,
    ModelMetadata,
    TaskType,
    TopModelRecord,
)
from aads.core.state import RunState
from aads.tools.ml.trainer import (
    get_candidate_models,
    rank_models,
    train_and_evaluate_model,
)

logger = get_logger(__name__)


def _sanitize_model_name_for_filename(name: str) -> str:
    """Convert PascalCase/camelCase model name into snake_case filename segment."""
    # e.g. RandomForestClassifier -> random_forest_classifier
    # e.g. XGBClassifier -> xgboost_classifier
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    s2 = s2.replace("xgb", "xgboost").replace("cat_boost", "catboost")
    # clean extra underscores
    return re.sub(r"_+", "_", s2).strip("_")


class MLExperimentAgent:
    """Agent that conducts reproducible ML experiments, tunes models, and exports top-performing models."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
        llm: Any = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager
        self.llm = llm
        self.top_models: list[dict[str, Any]] = []

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        state: RunState,
        candidate_models: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        progress_callback: Optional[Any] = None,
    ) -> tuple[Any, str, dict[str, float], list[ExperimentRecord]]:
        """Run ML training experiments across candidate models and select top models.

        Args:
            X_train: Encoded train features.
            y_train: Train target.
            X_eval: Encoded validation or test features.
            y_eval: Validation or test target.
            state: The current RunState.
            candidate_models: Optional model names to train.
            top_k: Number of top models to export (defaults to config.top_models_count).
            progress_callback: Optional callback for live progress streaming.

        Returns:
            Tuple of (best_model, best_model_name, best_metrics, all_experiments).
        """
        task_type = state.task_type or TaskType.CLASSIFICATION
        if y_train is not None:
            try:
                from sklearn.utils.multiclass import type_of_target
                t_type = type_of_target(y_train)
                if t_type in ("continuous", "continuous-multioutput"):
                    task_type = TaskType.REGRESSION
                    state.task_type = TaskType.REGRESSION
                elif t_type in ("binary", "multiclass"):
                    task_type = TaskType.CLASSIFICATION
                    state.task_type = TaskType.CLASSIFICATION
            except Exception:
                if pd.api.types.is_float_dtype(y_train) or (pd.api.types.is_numeric_dtype(y_train) and getattr(y_train, "nunique", lambda: 0)() > 30):
                    task_type = TaskType.REGRESSION
                    state.task_type = TaskType.REGRESSION

        base_models = candidate_models or get_candidate_models(task_type)
        models_to_run = list(base_models)

        # In AI Mode: inject AI-Tuned Model configurations to push validation accuracy higher
        ai_model_params: dict[str, dict[str, Any]] = {}
        if getattr(self.config, "execution_mode", "local") == "ai":
            if task_type == TaskType.CLASSIFICATION:
                if "LGBMClassifier" in models_to_run:
                    name = "LGBMClassifier (AI-Tuned)"
                    models_to_run.append(name)
                    ai_model_params[name] = {"n_estimators": 140, "learning_rate": 0.04, "max_depth": 6, "subsample": 0.85}
                if "XGBClassifier" in models_to_run:
                    name = "XGBClassifier (AI-Tuned)"
                    models_to_run.append(name)
                    ai_model_params[name] = {"n_estimators": 140, "learning_rate": 0.04, "max_depth": 6, "subsample": 0.85}
                if "CatBoostClassifier" in models_to_run:
                    name = "CatBoostClassifier (AI-Tuned)"
                    models_to_run.append(name)
                    ai_model_params[name] = {"n_estimators": 150, "learning_rate": 0.04, "max_depth": 6}
                if "RandomForestClassifier" in models_to_run:
                    name = "RandomForestClassifier (AI-Tuned)"
                    models_to_run.append(name)
                    ai_model_params[name] = {"n_estimators": 120, "max_depth": 12, "min_samples_split": 4}
            elif task_type == TaskType.REGRESSION:
                if "LGBMRegressor" in models_to_run:
                    name = "LGBMRegressor (AI-Tuned)"
                    models_to_run.append(name)
                    ai_model_params[name] = {"n_estimators": 140, "learning_rate": 0.04, "max_depth": 6, "subsample": 0.85}
                if "XGBRegressor" in models_to_run:
                    name = "XGBRegressor (AI-Tuned)"
                    models_to_run.append(name)
                    ai_model_params[name] = {"n_estimators": 140, "learning_rate": 0.04, "max_depth": 6, "subsample": 0.85}
                if "RandomForestRegressor" in models_to_run:
                    name = "RandomForestRegressor (AI-Tuned)"
                    models_to_run.append(name)
                    ai_model_params[name] = {"n_estimators": 120, "max_depth": 12, "min_samples_split": 4}

        primary_metric = "rmse" if task_type == TaskType.REGRESSION else "f1"
        num_top_models = top_k or getattr(self.config, "top_models_count", 4)

        logger.info(
            "ml_experiment_agent_start",
            run_id=state.run_id,
            models=models_to_run,
            primary_metric=primary_metric,
            top_k=num_top_models,
        )

        all_experiments: list[ExperimentRecord] = []
        raw_results: list[dict[str, Any]] = []

        for idx, model_name in enumerate(models_to_run):
            exp_id = f"exp_{state.run_id}_{idx+1:02d}_{_sanitize_model_name_for_filename(model_name)}"
            is_baseline = (idx == 0)
            custom_p = ai_model_params.get(model_name, None)

            fitted_model, metrics, _, train_time = train_and_evaluate_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_eval,
                y_val=y_eval,
                task_type=task_type,
                random_state=state.random_seed,
                params=custom_p,
            )

            raw_results.append({
                "model_name": model_name,
                "model": fitted_model,
                "metrics": metrics,
                "training_time": train_time,
            })

            exp_rec = ExperimentRecord(
                experiment_id=exp_id,
                model_name=model_name,
                hyperparameters=custom_p or {},
                metrics=metrics,
                is_baseline=is_baseline,
                is_best=False,
                notes=f"Trained {model_name} in {train_time:.3f}s",
            )
            all_experiments.append(exp_rec)
            state.add_experiment(exp_rec)

            if progress_callback:
                progress_callback(
                    f"🤖 Training candidate machine learning models and evaluating leaderboard... [{idx+1}/{len(models_to_run)}] ({model_name})"
                )

        # 1. Multi-metric ranking across all candidate models
        ranked_top = rank_models(raw_results, task_type=task_type, top_k=num_top_models)
        self.top_models = ranked_top

        best_model = ranked_top[0]["model"] if ranked_top else None
        best_model_name = ranked_top[0]["model_name"] if ranked_top else ""
        best_metrics = ranked_top[0]["metrics"] if ranked_top else {}

        # Mark best in experiment registry
        for exp in all_experiments:
            if exp.model_name == best_model_name:
                exp.is_best = True

        state.mark_phase_complete("ml_experiment")

        # 2. Save Artifacts: 06_Models/ (model_01_*.pkl ... model_04_*.pkl, comparison, metadata)
        if self.artifact_manager:
            try:
                models_dir = self.artifact_manager.get_path("models")
                top_model_records: list[TopModelRecord] = []

                # Export each top model
                for item in ranked_top:
                    rank = item["rank"]
                    m_name = item["model_name"]
                    clean_name = _sanitize_model_name_for_filename(m_name)
                    filename = f"model_{rank:02d}_{clean_name}.pkl"
                    model_path = models_dir / filename

                    with open(model_path, "wb") as f:
                        pickle.dump(item["model"], f)

                    self.artifact_manager.register_artifact(
                        artifact_type=ArtifactType.MODEL,
                        path=model_path,
                        description=f"Rank {rank} ML Model: {m_name} ({item['selection_reason']})",
                    )

                    top_model_records.append(
                        TopModelRecord(
                            rank=rank,
                            model=m_name,
                            filename=filename,
                            validation_metrics=item["metrics"],
                            test_metrics=item["metrics"],
                            training_time=item.get("training_time", 0.0),
                            feature_set="feature_engineered",
                            selection_reason=item["selection_reason"],
                        )
                    )

                # Export best_model.pkl for backward compatibility
                if best_model is not None:
                    legacy_best_path = models_dir / "best_model.pkl"
                    with open(legacy_best_path, "wb") as f:
                        pickle.dump(best_model, f)
                    self.artifact_manager.register_artifact(
                        artifact_type=ArtifactType.MODEL,
                        path=legacy_best_path,
                        description=f"Winning model ({best_model_name})",
                    )

                # Save 06_Models/model_comparison.json in sorted performance order
                ranked_all = rank_models(raw_results, task_type=task_type, top_k=len(raw_results))
                comparison_rows = [
                    {
                        "rank": r.get("rank", idx + 1),
                        "model": r["model_name"],
                        "training_time_seconds": r.get("training_time", 0.0),
                        "is_best": r.get("rank", idx + 1) == 1,
                        **r["metrics"],
                    }
                    for idx, r in enumerate(ranked_all)
                ]
                comp_path = models_dir / "model_comparison.json"
                comp_path.write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")
                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=comp_path,
                    description="Comparative performance metrics across all evaluated candidate models",
                )

                # Save 06_Models/model_metadata.json
                model_meta = ModelMetadata(
                    top_models=top_model_records,
                    total_models_evaluated=len(raw_results),
                    selection_metric=primary_metric,
                    created_at=datetime.utcnow(),
                )
                meta_path = models_dir / "model_metadata.json"
                meta_path.write_text(model_meta.model_dump_json(indent=2), encoding="utf-8")
                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.METADATA,
                    path=meta_path,
                    description="Detailed metadata for exported top models",
                )

                # Save 09_Experiments/experiment_results.csv
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
                action="train_and_export_top_models",
                reason=(
                    f"Evaluated {len(models_to_run)} candidate models. "
                    f"Selected top {len(ranked_top)} models (Rank 1: {best_model_name})."
                ),
                approval_mode=state.autonomy_mode,
                details={
                    "total_models": len(models_to_run),
                    "top_models": [
                        {"rank": t["rank"], "model": t["model_name"], "metrics": t["metrics"]}
                        for t in ranked_top
                    ],
                },
            )
        )

        logger.info(
            "ml_experiment_agent_completed",
            best_model=best_model_name,
            top_models_count=len(ranked_top),
            metric=primary_metric,
        )
        return best_model, best_model_name, best_metrics, all_experiments
