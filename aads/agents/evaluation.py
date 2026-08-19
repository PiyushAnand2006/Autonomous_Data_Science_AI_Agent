"""
AADS Evaluation Agent — generates deep diagnostic metrics, confusion matrices,
residual distributions, and exports the final model evaluation report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import ArtifactType, DecisionRecord, TaskType
from aads.core.state import RunState

logger = get_logger(__name__)


class EvaluationAgent:
    """Agent that performs holdout evaluation, generates diagnostic plots, and compiles reports."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager

    def run(
        self,
        model: Any,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        state: RunState,
    ) -> dict[str, Any]:
        """Evaluate the model on holdout test data and save diagnostic visualizations.

        Args:
            model: Fitted model object.
            model_name: Name of the model.
            X_test: Encoded test features.
            y_test: Test target.
            state: The current RunState.

        Returns:
            Dictionary containing evaluation metrics and diagnostics.
        """
        task_type = state.task_type or TaskType.CLASSIFICATION
        logger.info("evaluation_agent_start", run_id=state.run_id, model=model_name)

        y_pred = model.predict(X_test)
        report: dict[str, Any] = {
            "model_name": model_name,
            "task_type": task_type.value,
            "test_samples": len(y_test),
            "metrics": {},
            "diagnostics": {},
            "generated_charts": [],
        }

        eval_dir: Optional[Path] = None
        if self.artifact_manager:
            eval_dir = self.artifact_manager.get_path("visualizations") / "model_evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)

        # 1. Regression Diagnostics
        if task_type == TaskType.REGRESSION:
            residuals = y_test.values - y_pred
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals**2)))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))

            report["metrics"] = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
            report["diagnostics"]["mean_residual"] = round(float(np.mean(residuals)), 4)
            report["diagnostics"]["residual_std"] = round(float(np.std(residuals)), 4)

            # Generate Residuals Plot
            if eval_dir:
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].scatter(y_pred, residuals, alpha=0.6, color="#2563eb", edgecolors="none")
                ax[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
                ax[0].set_title("Residuals vs Predicted")
                ax[0].set_xlabel("Predicted Value")
                ax[0].set_ylabel("Residual (Actual - Pred)")
                ax[0].grid(True, linestyle=":", alpha=0.5)

                ax[1].hist(residuals, bins=25, color="#3b82f6", edgecolor="white", alpha=0.85)
                ax[1].set_title("Residual Distribution")
                ax[1].set_xlabel("Residual")
                ax[1].grid(axis="y", linestyle=":", alpha=0.5)
                plt.tight_layout()

                res_path = eval_dir / "residuals_plot.png"
                fig.savefig(res_path, dpi=150)
                plt.close(fig)

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.VISUALIZATION,
                    path=res_path,
                    description="Residuals vs Predicted and Residual Distribution",
                )
                report["generated_charts"].append(str(res_path))

        # 2. Classification Diagnostics
        else:
            acc = float(np.mean(y_pred == y_test.values))
            report["metrics"] = {"accuracy": round(acc, 4)}

            # Generate Confusion Matrix
            labels = np.unique(y_test)
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            report["diagnostics"]["confusion_matrix"] = cm.tolist()

            if eval_dir:
                fig, ax = plt.subplots(figsize=(5, 4))
                cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
                fig.colorbar(cax)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels([str(l) for l in labels])
                ax.set_yticklabels([str(l) for l in labels])

                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=11)

                ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold", pad=15)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                plt.tight_layout()

                cm_path = eval_dir / "confusion_matrix.png"
                fig.savefig(cm_path, dpi=150)
                plt.close(fig)

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.VISUALIZATION,
                    path=cm_path,
                    description="Confusion matrix heatmap",
                )
                report["generated_charts"].append(str(cm_path))

        state.mark_phase_complete("evaluation")

        # Save model_report.json in 08_Reports/
        if self.artifact_manager:
            try:
                rep_dir = self.artifact_manager.get_path("reports")
                report_path = rep_dir / "model_report.json"
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

                self.artifact_manager.register_artifact(
                    artifact_type=ArtifactType.REPORT,
                    path=report_path,
                    description=f"Model evaluation report for {model_name}",
                )
            except Exception as e:
                logger.warning("evaluation_report_save_failed", error=str(e))

        # Log decision
        state.add_decision(
            DecisionRecord(
                agent="evaluation",
                action="evaluate_model",
                reason=f"Completed holdout evaluation for {model_name}. Test metrics: {report['metrics']}.",
                approval_mode=state.autonomy_mode,
                details=report,
            )
        )

        logger.info("evaluation_agent_completed", metrics=report["metrics"])
        return report
