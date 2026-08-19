"""
AADS Goal & Planning Agent — interprets user intent and dataset profile
to formulate structured, executable task plans.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from aads.core.config import AADSConfig
from aads.core.exceptions import AgentError
from aads.core.llm import get_llm
from aads.core.logging import get_logger
from aads.core.schemas import (
    ALLOWED_PLAN_STEPS,
    DatasetProfile,
    DecisionRecord,
    PlanStep,
    TaskPlan,
    TaskType,
)
from aads.core.state import RunState
from aads.prompts.agents.planner_prompt import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT_TEMPLATE,
)

logger = get_logger(__name__)


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON dictionary from raw model text."""
    text = text.strip()
    if not text:
        return {}
    # Direct JSON parse attempt
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Markdown block parse attempt ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Greedily match outermost { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {}


class GoalPlannerAgent:
    """Agent that translates natural-language objectives and data profiles into structured plans."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        llm: Any = None,
    ) -> None:
        self.config = config or AADSConfig()
        self.llm = llm

    def _get_llm_instance(self) -> Any:
        if self.llm is not None:
            return self.llm
        try:
            return get_llm(self.config)
        except Exception as e:
            logger.debug("llm_init_skipped_or_failed", error=str(e))
            return None

    def plan_heuristically(
        self,
        profile: DatasetProfile,
        user_objective: str,
        target_column: Optional[str] = None,
    ) -> TaskPlan:
        """Deterministic fallback planner when LLM is unavailable."""
        obj_lower = user_objective.lower()

        # 1. Infer task type
        if any(w in obj_lower for w in ["cluster", "grouping", "segmentation"]):
            task_type = TaskType.CLUSTERING
        elif any(w in obj_lower for w in ["anomaly", "outlier detection"]):
            task_type = TaskType.ANOMALY
        elif any(w in obj_lower for w in ["forecast", "time series", "future trends"]):
            task_type = TaskType.FORECASTING
        elif any(w in obj_lower for w in ["classif", "churn", "predict category", "spam", "survived", "fraud", "binary", "default"]):
            task_type = TaskType.CLASSIFICATION
        elif any(w in obj_lower for w in ["regress", "price", "cost", "salary", "continuous", "amount", "sales", "revenue"]):
            task_type = TaskType.REGRESSION
        elif any(w in obj_lower for w in ["explore", "eda", "summary", "profile", "describe"]) and not any(w in obj_lower for w in ["predict", "train", "model"]):
            task_type = TaskType.DESCRIPTIVE
        else:
            # Default heuristic based on target candidate type
            task_type = TaskType.CLASSIFICATION

        # 2. Determine target column if applicable
        target = target_column
        if not target and task_type in (TaskType.REGRESSION, TaskType.CLASSIFICATION):
            if profile.target_candidates:
                target = profile.target_candidates[0]

        # 3. Determine steps based on task type
        if task_type == TaskType.DESCRIPTIVE:
            steps_list = [
                PlanStep(step_id="profiling", description="Inspect dataset schema, missingness, and distributions"),
                PlanStep(step_id="data_quality", description="Detect missing values, anomalies, and inconsistencies", depends_on=["profiling"]),
                PlanStep(step_id="eda", description="Generate distribution plots, correlations, and exploratory summaries", depends_on=["data_quality"]),
                PlanStep(step_id="notebook_generation", description="Generate reproducible Jupyter notebook for analysis", depends_on=["eda"]),
                PlanStep(step_id="report_generation", description="Generate EDA summary report", depends_on=["notebook_generation"]),
            ]
            metric = None
        else:
            steps_list = [
                PlanStep(step_id="profiling", description="Inspect dataset schema, missingness, and distributions"),
                PlanStep(step_id="data_quality", description="Detect missing values and anomalies", depends_on=["profiling"]),
                PlanStep(step_id="eda", description="Generate distribution plots and feature-target relationships", depends_on=["data_quality"]),
                PlanStep(step_id="cleaning", description="Apply justified missing-value imputation and outlier treatment", depends_on=["eda"]),
                PlanStep(step_id="split", description="Create leakage-safe train/test partitions", depends_on=["cleaning"]),
                PlanStep(step_id="leakage_check", description="Verify no train/test contamination or target leakage", depends_on=["split"]),
                PlanStep(step_id="feature_engineering", description="Generate candidate interaction and domain features", depends_on=["leakage_check"]),
                PlanStep(step_id="preprocessing", description="Fit encoders and scalers strictly on training data", depends_on=["feature_engineering"]),
                PlanStep(step_id="ml_experiment", description="Train baseline and candidate machine learning models", depends_on=["preprocessing"]),
                PlanStep(step_id="evaluation", description="Compute diagnostics, metrics, and error analyses", depends_on=["ml_experiment"]),
                PlanStep(step_id="replanning", description="Review model performance and evaluate follow-up runs", depends_on=["evaluation"]),
                PlanStep(step_id="notebook_generation", description="Generate reproducible end-to-end Jupyter notebook", depends_on=["evaluation"]),
                PlanStep(step_id="report_generation", description="Produce final model report, metrics, and artifacts", depends_on=["notebook_generation"]),
            ]
            metric = "rmse" if task_type == TaskType.REGRESSION else "f1"

        return TaskPlan(
            task_type=task_type,
            target_column=target,
            metric=metric,
            steps=steps_list,
            reasoning=f"Heuristic plan formulated for objective: '{user_objective}'. Task inferred as {task_type.value}.",
            questions=["Please confirm the target column is correct."] if (not target_column and target) else [],
            notes=[f"Dataset has {profile.n_rows} rows and {profile.n_cols} columns."],
        )

    def plan(
        self,
        profile: DatasetProfile,
        state: RunState,
    ) -> TaskPlan:
        """Formulate a task plan using LLM or heuristic fallback.

        Args:
            profile: DatasetProfile from the ProfilerAgent.
            state: The current RunState.

        Returns:
            Validated TaskPlan.
        """
        user_objective = state.user_objective or "Analyze the dataset and build a predictive model."
        target_column_hint = state.target_column or (profile.target_candidates[0] if profile.target_candidates else "None")

        llm = self._get_llm_instance()
        plan_obj: TaskPlan

        if llm is None:
            logger.info("planner_running_heuristically", reason="No LLM instance available")
            plan_obj = self.plan_heuristically(profile, user_objective, state.target_column)
        else:
            try:
                user_prompt = PLANNER_USER_PROMPT_TEMPLATE.format(
                    user_objective=user_objective,
                    target_column_hint=target_column_hint,
                    dataset_profile_summary=profile.summary_text(),
                )

                try:
                    from langchain_core.messages import HumanMessage, SystemMessage
                    messages = [
                        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                        HumanMessage(content=user_prompt),
                    ]
                except ImportError:
                    messages = [
                        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ]

                response = llm.invoke(messages)
                content = getattr(response, "content", "") or str(response)

                parsed_json = _extract_json_from_text(content)
                if not parsed_json or "steps" not in parsed_json:
                    logger.warning("planner_llm_output_invalid_json", raw_content=content[:200])
                    plan_obj = self.plan_heuristically(profile, user_objective, state.target_column)
                else:
                    # Sanitize and validate steps against allowed vocabulary
                    raw_steps = parsed_json.get("steps", [])
                    validated_steps: list[PlanStep] = []
                    for s in raw_steps:
                        if isinstance(s, dict):
                            s_id = str(s.get("step_id", "")).strip()
                            if s_id in ALLOWED_PLAN_STEPS:
                                validated_steps.append(
                                    PlanStep(
                                        step_id=s_id,
                                        description=str(s.get("description", "")),
                                        depends_on=[d for d in s.get("depends_on", []) if d in ALLOWED_PLAN_STEPS],
                                        config=s.get("config", {}) if isinstance(s.get("config"), dict) else {},
                                    )
                                )
                        elif isinstance(s, str) and s.strip() in ALLOWED_PLAN_STEPS:
                            validated_steps.append(PlanStep(step_id=s.strip()))

                    if not validated_steps:
                        plan_obj = self.plan_heuristically(profile, user_objective, state.target_column)
                    else:
                        task_type_str = str(parsed_json.get("task_type", "classification")).lower().strip()
                        try:
                            task_type = TaskType(task_type_str)
                        except ValueError:
                            task_type = TaskType.CLASSIFICATION

                        plan_obj = TaskPlan(
                            task_type=task_type,
                            target_column=parsed_json.get("target_column") or state.target_column,
                            metric=parsed_json.get("metric"),
                            steps=validated_steps,
                            reasoning=str(parsed_json.get("reasoning", "")),
                            questions=parsed_json.get("questions", []) if isinstance(parsed_json.get("questions"), list) else [],
                            notes=parsed_json.get("notes", []) if isinstance(parsed_json.get("notes"), list) else [],
                        )
            except Exception as e:
                logger.warning("planner_llm_execution_failed", error=str(e))
                plan_obj = self.plan_heuristically(profile, user_objective, state.target_column)

        # Update state with plan
        state.task_type = plan_obj.task_type
        if plan_obj.target_column:
            state.target_column = plan_obj.target_column
        state.task_plan = [s.step_id for s in plan_obj.steps]
        state.mark_phase_complete("planning")

        state.add_decision(
            DecisionRecord(
                agent="planner",
                action="formulate_plan",
                reason=f"Formulated {plan_obj.task_type.value} plan with {len(plan_obj.steps)} steps. Target: {plan_obj.target_column}.",
                approval_mode=state.autonomy_mode,
                details={
                    "task_type": plan_obj.task_type.value,
                    "target_column": plan_obj.target_column,
                    "metric": plan_obj.metric,
                    "steps": state.task_plan,
                },
            )
        )

        logger.info(
            "planner_agent_completed",
            task_type=plan_obj.task_type.value,
            target=plan_obj.target_column,
            step_count=len(plan_obj.steps),
        )

        return plan_obj
