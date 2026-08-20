"""
Tests for Feature Engineering and Preprocessing agents and tools.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aads.agents.artifact_manager import ArtifactManager
from aads.agents.feature_engineering import FeatureEngineeringAgent
from aads.agents.preprocessing import PreprocessingAgent
from aads.core.schemas import ArtifactType, TaskType
from aads.core.state import RunState
from aads.tools.processing.feature_engineer import generate_candidate_features
from aads.tools.processing.preprocessor import (
    build_and_fit_preprocessor,
    transform_with_preprocessor,
)


@pytest.fixture
def sample_fe_df() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X_train = pd.DataFrame({
        "num1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "num2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "cat1": ["Low", "High", "Low", "Medium", "High"],
        "skewed": [1.0, 2.0, 1.0, 1.0, 1000.0],
    })
    X_test = pd.DataFrame({
        "num1": [15.0, 25.0],
        "num2": [3.0, 5.0],
        "cat1": ["Low", "High"],
        "skewed": [1.5, 500.0],
    })
    y_train = pd.Series([0, 1, 0, 1, 1])
    return X_train, X_test, y_train


class TestFeatureEngineering:

    def test_generate_candidate_features(self, sample_fe_df):
        X_train, X_test, y_train = sample_fe_df
        X_tr_fe, X_te_fe, log = generate_candidate_features(X_train, X_test, y_train)

        assert len(X_tr_fe.columns) > len(X_train.columns)
        assert len(X_te_fe.columns) == len(X_tr_fe.columns)
        assert any("diff_" in c or "ratio_" in c for c in X_tr_fe.columns)


class TestPreprocessing:

    def test_build_and_fit_preprocessor(self, sample_fe_df):
        X_train, X_test, _ = sample_fe_df
        pipeline, X_tr_enc, names = build_and_fit_preprocessor(X_train)

        assert len(names) > 0
        assert X_tr_enc.shape[1] == len(names)
        assert not X_tr_enc.isna().any().any()

        # Transform test
        X_te_enc = transform_with_preprocessor(pipeline, X_test, names)
        assert X_te_enc.shape[1] == len(names)
        assert not X_te_enc.isna().any().any()


class TestPhase5Agents:

    def test_agents_lifecycle(self, tmp_path, sample_fe_df):
        mgr = ArtifactManager(storage_root=tmp_path)
        mgr.initialize_run("phase5_run")
        state = RunState.create(user_objective="Predict category", target_column="target")

        X_train, X_test, y_train = sample_fe_df

        # 1. Feature Engineering Agent
        fe_agent = FeatureEngineeringAgent(artifact_manager=mgr)
        X_tr_fe, X_te_fe, _, _ = fe_agent.run(X_train, X_test, y_train, state)
        assert "feature_engineering" in state.completed_phases
        assert (mgr.get_path("feature_engineered_data") / "feature_engineered_dataset.csv").exists()
        assert (mgr.get_path("feature_engineered_data") / "X_train_fe.parquet").exists()

        # 2. Preprocessing Agent
        prep_agent = PreprocessingAgent(artifact_manager=mgr)
        X_tr_enc, X_te_enc, _, pipeline = prep_agent.run(X_tr_fe, X_te_fe, state)
        assert "preprocessing" in state.completed_phases
        assert (mgr.get_path("models") / "preprocessing_pipeline.pkl").exists()
        assert (mgr.get_path("ml_ready_data") / "ml_ready_dataset.csv").exists()
        assert (mgr.get_path("ml_ready_data") / "X_train_encoded.parquet").exists()
