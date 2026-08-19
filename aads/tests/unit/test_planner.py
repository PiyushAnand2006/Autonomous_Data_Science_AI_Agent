"""
Tests for AADS Goal & Planning Agent.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from aads.agents.planner import GoalPlannerAgent, _extract_json_from_text
from aads.core.schemas import DatasetProfile, TaskPlan, TaskType
from aads.core.state import RunState
from aads.tools.profiling.profiler import profile_dataset


@pytest.fixture
def sample_profile() -> DatasetProfile:
    df = pd.DataFrame({
        "id": range(10),
        "feature_1": [1.0 * i for i in range(10)],
        "target_price": [100.0 + 10.0 * i for i in range(10)],
    })
    return profile_dataset(df)


class TestJsonExtraction:
    """Verify JSON extraction from model outputs."""

    def test_clean_json(self):
        text = '{"task_type": "regression", "steps": []}'
        data = _extract_json_from_text(text)
        assert data.get("task_type") == "regression"

    def test_markdown_fenced_json(self):
        text = 'Here is the plan:\n```json\n{"task_type": "classification", "steps": []}\n```'
        data = _extract_json_from_text(text)
        assert data.get("task_type") == "classification"

    def test_invalid_text_returns_empty(self):
        text = "Sorry, I cannot help with that."
        data = _extract_json_from_text(text)
        assert data == {}


class TestHeuristicPlanning:
    """Verify deterministic fallback planning."""

    def test_regression_objective(self, sample_profile):
        agent = GoalPlannerAgent()
        plan = agent.plan_heuristically(
            sample_profile,
            user_objective="Predict house prices based on features",
        )
        assert plan.task_type == TaskType.REGRESSION
        assert plan.target_column == "target_price"
        assert len(plan.steps) > 5

    def test_classification_objective(self, sample_profile):
        agent = GoalPlannerAgent()
        plan = agent.plan_heuristically(
            sample_profile,
            user_objective="Predict customer churn category",
        )
        assert plan.task_type == TaskType.CLASSIFICATION

    def test_descriptive_objective(self, sample_profile):
        agent = GoalPlannerAgent()
        plan = agent.plan_heuristically(
            sample_profile,
            user_objective="Perform exploratory data analysis and summary visualizations",
        )
        assert plan.task_type == TaskType.DESCRIPTIVE
        # Descriptive workflow should omit ML training steps
        step_ids = [s.step_id for s in plan.steps]
        assert "ml_experiment" not in step_ids
        assert "eda" in step_ids


class TestLLMPlanning:
    """Verify LLM-driven planning with mock responses."""

    def test_mock_llm_valid_plan(self, sample_profile):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """```json
{
  "task_type": "regression",
  "target_column": "target_price",
  "metric": "rmse",
  "steps": [
    {"step_id": "profiling", "description": "Profile data"},
    {"step_id": "eda", "description": "Explore distributions"},
    {"step_id": "split", "description": "Train test split"},
    {"step_id": "ml_experiment", "description": "Train models"},
    {"step_id": "evaluation", "description": "Evaluate RMSE"}
  ],
  "reasoning": "Standard regression flow",
  "questions": [],
  "notes": []
}
```"""
        mock_llm.invoke.return_value = mock_response

        agent = GoalPlannerAgent(llm=mock_llm)
        state = RunState.create(user_objective="Predict house price")

        plan = agent.plan(sample_profile, state)

        assert plan.task_type == TaskType.REGRESSION
        assert plan.target_column == "target_price"
        assert len(plan.steps) == 5
        assert state.task_type == TaskType.REGRESSION
        assert "planning" in state.completed_phases
        assert len(state.decisions) == 1

    def test_mock_llm_filters_invalid_steps(self, sample_profile):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """{
  "task_type": "classification",
  "target_column": "target_price",
  "steps": [
    {"step_id": "profiling"},
    {"step_id": "hack_the_system"},
    {"step_id": "eda"}
  ]
}"""
        mock_llm.invoke.return_value = mock_response

        agent = GoalPlannerAgent(llm=mock_llm)
        state = RunState.create(user_objective="Classify")

        plan = agent.plan(sample_profile, state)

        step_ids = [s.step_id for s in plan.steps]
        assert "hack_the_system" not in step_ids
        assert "profiling" in step_ids
        assert "eda" in step_ids
