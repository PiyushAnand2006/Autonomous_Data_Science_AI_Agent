"""
AADS Replanning Agent — analyzes model performance and diagnostics to decide
whether to terminate the workflow or launch follow-up experiments.
"""

from __future__ import annotations

from typing import Any, Optional

from aads.agents.artifact_manager import ArtifactManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import DecisionRecord, TaskType
from aads.core.state import RunState

logger = get_logger(__name__)


class ReplanningAgent:
    """Agent that analyzes experimental results, diagnoses weaknesses, and determines the next action."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        artifact_manager: Optional[ArtifactManager] = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.artifact_manager = artifact_manager

    def run(
        self,
        evaluation_report: dict[str, Any],
        state: RunState,
        current_iteration: int = 1,
    ) -> dict[str, Any]:
        """Analyze evaluation results and decide next action.

        Args:
            evaluation_report: Dict from EvaluationAgent.
            state: The current RunState.
            current_iteration: Iteration counter in the replanning loop.

        Returns:
            Dict containing decision (should_stop: bool, next_action: str, reasoning: str).
        """
        task_type = state.task_type or TaskType.CLASSIFICATION
        metrics = evaluation_report.get("metrics", {})
        max_iters = self.config.max_experiment_iterations

        logger.info("replanning_agent_start", run_id=state.run_id, iteration=current_iteration)

        should_stop = True
        next_action = "conclude_experiments"
        reasoning = ""

        if current_iteration >= max_iters:
            should_stop = True
            next_action = "conclude_budget_exhausted"
            reasoning = f"Reached maximum configured iteration budget ({max_iters}). Concluding experimentation."
        elif task_type == TaskType.REGRESSION:
            r2 = metrics.get("r2", 0.0)
            if r2 >= 0.85:
                should_stop = True
                reasoning = f"Excellent regression performance achieved (R2 = {r2:.3f}). Concluding experiments."
            elif r2 < 0.30 and current_iteration < 2:
                should_stop = False
                next_action = "tune_hyperparameters"
                reasoning = f"Moderate regression score (R2 = {r2:.3f}). Recommending hyperparameter search on gradient boosting."
            else:
                should_stop = True
                reasoning = f"Acceptable regression score (R2 = {r2:.3f}) within iteration budget."
        else:
            acc = metrics.get("accuracy", 0.0)
            if acc >= 0.90:
                should_stop = True
                reasoning = f"High classification accuracy achieved ({acc*100:.1f}%). Concluding experiments."
            elif acc < 0.65 and current_iteration < 2:
                should_stop = False
                next_action = "tune_hyperparameters"
                reasoning = f"Classification accuracy is {acc*100:.1f}%. Suggesting parameter tuning."
            else:
                should_stop = True
                reasoning = f"Classification accuracy ({acc*100:.1f}%) satisfies baseline objective."

        state.mark_phase_complete("replanning")

        decision_result = {
            "should_stop": should_stop,
            "next_action": next_action,
            "reasoning": reasoning,
            "current_iteration": current_iteration,
        }

        state.add_decision(
            DecisionRecord(
                agent="replanning",
                action="evaluate_replan",
                reason=reasoning,
                approval_mode=state.autonomy_mode,
                details=decision_result,
            )
        )

        logger.info("replanning_agent_completed", should_stop=should_stop, next_action=next_action)
        return decision_result
