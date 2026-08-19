"""Tests for aads.core.state — RunState lifecycle."""

import json
from pathlib import Path

import pytest

from aads.core.schemas import (
    ArtifactRecord,
    ArtifactType,
    AutonomyMode,
    DatasetMeta,
    DecisionRecord,
    ExperimentRecord,
    TaskType,
)
from aads.core.state import RunState


class TestRunStateCreation:
    """Verify RunState factory and defaults."""

    def test_create_minimal(self):
        state = RunState.create(user_objective="Predict house prices")
        assert state.user_objective == "Predict house prices"
        assert state.run_id  # should not be empty
        assert len(state.run_id) == 12
        assert state.current_phase == "initialized"
        assert state.autonomy_mode == AutonomyMode.SEMI_AUTONOMOUS

    def test_create_with_options(self):
        state = RunState.create(
            user_objective="Classify spam",
            autonomy_mode=AutonomyMode.FULLY_AUTONOMOUS,
            target_column="is_spam",
            random_seed=123,
        )
        assert state.target_column == "is_spam"
        assert state.autonomy_mode == AutonomyMode.FULLY_AUTONOMOUS
        assert state.random_seed == 123

    def test_unique_run_ids(self):
        s1 = RunState.create(user_objective="A")
        s2 = RunState.create(user_objective="B")
        assert s1.run_id != s2.run_id


class TestRunStateRegistries:
    """Verify artifact, decision, and experiment helpers."""

    def test_add_artifact(self):
        state = RunState.create(user_objective="Test")
        record = ArtifactRecord(
            artifact_type=ArtifactType.RAW_DATA,
            path="01_Raw_Data/data.csv",
        )
        state.add_artifact(record)
        assert len(state.artifacts) == 1
        assert state.artifacts[0].artifact_type == ArtifactType.RAW_DATA

    def test_add_decision(self):
        state = RunState.create(user_objective="Test")
        decision = DecisionRecord(
            agent="cleaning",
            action="remove_nulls",
            reason="Column had 80% nulls",
        )
        state.add_decision(decision)
        assert len(state.decisions) == 1

    def test_add_experiment(self):
        state = RunState.create(user_objective="Test")
        exp = ExperimentRecord(
            experiment_id="exp_001",
            model_name="LinearRegression",
            metrics={"rmse": 2.0},
            is_baseline=True,
        )
        state.add_experiment(exp)
        assert len(state.experiments) == 1

    def test_get_best_experiment_none(self):
        state = RunState.create(user_objective="Test")
        assert state.get_best_experiment() is None

    def test_get_best_experiment(self):
        state = RunState.create(user_objective="Test")
        state.add_experiment(ExperimentRecord(
            experiment_id="exp_001", model_name="LR", metrics={"rmse": 2.0},
        ))
        state.add_experiment(ExperimentRecord(
            experiment_id="exp_002", model_name="RF", metrics={"rmse": 1.5}, is_best=True,
        ))
        best = state.get_best_experiment()
        assert best is not None
        assert best.experiment_id == "exp_002"


class TestRunStatePhases:
    """Verify phase tracking."""

    def test_mark_phase_complete(self):
        state = RunState.create(user_objective="Test")
        state.mark_phase_complete("profiling")
        assert "profiling" in state.completed_phases

    def test_no_duplicate_phases(self):
        state = RunState.create(user_objective="Test")
        state.mark_phase_complete("profiling")
        state.mark_phase_complete("profiling")
        assert state.completed_phases.count("profiling") == 1


class TestRunStatePersistence:
    """Verify JSON serialization/deserialization."""

    def test_save_and_load(self, tmp_path):
        state = RunState.create(
            user_objective="Predict prices",
            target_column="price",
        )
        state.mark_phase_complete("profiling")
        state.add_decision(DecisionRecord(
            agent="test", action="test_action", reason="testing",
        ))

        save_path = tmp_path / "state.json"
        state.save(save_path)

        assert save_path.exists()

        loaded = RunState.load(save_path)
        assert loaded.run_id == state.run_id
        assert loaded.user_objective == "Predict prices"
        assert loaded.target_column == "price"
        assert "profiling" in loaded.completed_phases
        assert len(loaded.decisions) == 1

    def test_save_creates_parent_dirs(self, tmp_path):
        state = RunState.create(user_objective="Test")
        save_path = tmp_path / "deep" / "nested" / "state.json"
        state.save(save_path)
        assert save_path.exists()
