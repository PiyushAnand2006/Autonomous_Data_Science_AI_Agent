"""
Tests for Cleaning, Splitting, and Leakage Guard tools & agents.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aads.agents.artifact_manager import ArtifactManager
from aads.agents.cleaning import CleaningAgent
from aads.agents.leakage_guard import LeakageGuard
from aads.agents.split_manager import SplitManager
from aads.core.exceptions import LeakageError
from aads.core.schemas import ArtifactType, TaskType
from aads.core.state import RunState
from aads.tools.processing.cleaner import clean_dataset
from aads.tools.processing.leakage import audit_leakage
from aads.tools.processing.splitter import split_dataset


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(100),
        "feature_1": [1.0, 2.0, np.nan, 4.0] + [3.0] * 96,
        "feature_2": ["A", "B", "?", "A"] + ["B"] * 96,
        "constant_col": ["Fixed"] * 100,
        "target": [0, 1, 0, 1] + [0] * 96,
    })


class TestCleaningTool:

    def test_clean_dataset_imputes_and_drops(self, messy_df):
        cleaned, log = clean_dataset(messy_df, target_column="target")

        assert "constant_col" not in cleaned.columns
        assert cleaned["feature_1"].isna().sum() == 0
        assert cleaned["feature_2"].isna().sum() == 0
        assert "?" not in cleaned["feature_2"].values
        assert len(log["dropped_columns"]) == 1


class TestSplitterTool:

    def test_split_dataset_stratification(self):
        df = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "target": [0] * 70 + [1] * 30,
        })
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            df, target_column="target", test_size=0.2, val_size=0.1, task_type=TaskType.CLASSIFICATION
        )

        assert len(X_train) + len(X_val) + len(X_test) == 100
        assert "target" not in X_train.columns
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestLeakageTool:

    def test_detects_target_inclusion(self):
        X_train = pd.DataFrame({"f1": [1, 2, 3], "target": [10, 20, 30]})
        X_test = pd.DataFrame({"f1": [4, 5, 6], "target": [40, 50, 60]})
        y_train = pd.Series([10, 20, 30])
        y_test = pd.Series([40, 50, 60])

        report = audit_leakage(X_train, X_test, y_train, y_test, target_name="target")
        assert report["has_leakage"] is True
        assert any("included as an input feature" in err for err in report["critical_issues"])

    def test_detects_target_proxy_correlation(self):
        y_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_test = pd.Series([11.0, 12.0])
        # Perfect correlation proxy
        X_train = pd.DataFrame({"proxy": y_train * 2.0 + 5.0, "clean_feat": np.random.randn(10)})
        X_test = pd.DataFrame({"proxy": y_test * 2.0 + 5.0, "clean_feat": np.random.randn(2)})

        report = audit_leakage(X_train, X_test, y_train, y_test, target_name="price")
        assert report["has_leakage"] is True
        assert any("near-perfect correlation" in err for err in report["critical_issues"])


class TestPhase4Agents:

    def test_full_pipeline_agents(self, tmp_path, messy_df):
        mgr = ArtifactManager(storage_root=tmp_path)
        mgr.initialize_run("phase4_run")
        state = RunState.create(user_objective="Predict target", target_column="target")

        # 1. Cleaning Agent
        clean_agent = CleaningAgent(artifact_manager=mgr)
        cleaned_df, _ = clean_agent.run(messy_df, state)
        assert "cleaning" in state.completed_phases

        # Check artifact
        clean_dir = mgr.get_path("cleaned_data")
        assert (clean_dir / "cleaned_dataset.parquet").exists()
        assert (clean_dir / "cleaned_dataset.csv").exists()

        # 2. Split Manager
        split_agent = SplitManager(artifact_manager=mgr)
        X_train, X_val, X_test, y_train, y_val, y_test = split_agent.run(cleaned_df, state)
        assert "split" in state.completed_phases

        # Check artifact
        ml_dir = mgr.get_path("ml_ready_data")
        assert (ml_dir / "X_train.parquet").exists()

        # 3. Leakage Guard
        guard = LeakageGuard(artifact_manager=mgr)
        leakage_rep = guard.run(X_train, X_test, y_train, y_test, state)
        assert "leakage_check" in state.completed_phases
        assert (mgr.get_path("metadata") / "leakage_audit.json").exists()
