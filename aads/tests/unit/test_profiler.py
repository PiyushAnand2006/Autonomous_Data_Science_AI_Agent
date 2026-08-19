"""
Tests for AADS Dataset Profiler and Engine Selector.
"""

import numpy as np
import pandas as pd
import pytest

from aads.agents.artifact_manager import ArtifactManager
from aads.agents.profiler import ProfilerAgent
from aads.core.config import AADSConfig
from aads.core.schemas import ExecutionEngine
from aads.core.state import RunState
from aads.tools.profiling.engine_selector import select_engine
from aads.tools.profiling.profiler import profile_dataset


@pytest.fixture
def sample_mixed_df() -> pd.DataFrame:
    """Create a DataFrame with numeric, categorical, boolean, and missing values."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": [f"ID_{i}" for i in range(100)],
        "age": [20 + (i % 50) if i % 10 != 0 else np.nan for i in range(100)],
        "income": [30000.0 + i * 500.0 for i in range(100)],
        "category": ["A" if i % 2 == 0 else "B" for i in range(100)],
        "is_active": [True if i % 3 == 0 else False for i in range(100)],
        "constant_col": [1 for _ in range(100)],
        "all_null": [np.nan for _ in range(100)],
        "churn": [1 if i % 4 == 0 else 0 for i in range(100)],
    })


class TestProfilerTool:
    """Verify profile_dataset tool calculations."""

    def test_shape_and_counts(self, sample_mixed_df):
        profile = profile_dataset(sample_mixed_df)
        assert profile.n_rows == 100
        assert profile.n_cols == 8
        assert profile.total_missing_cells == 100 + 10  # all_null (100) + age (10)

    def test_column_classifications(self, sample_mixed_df):
        profile = profile_dataset(sample_mixed_df)
        assert "income" in profile.numeric_columns
        assert "category" in profile.categorical_columns
        assert "is_active" in profile.boolean_columns
        assert "customer_id" in profile.suspected_id_columns
        assert "constant_col" in profile.constant_columns
        assert "all_null" in profile.all_null_columns

    def test_target_candidate_detection(self, sample_mixed_df):
        profile = profile_dataset(sample_mixed_df)
        assert "churn" in profile.target_candidates

    def test_numeric_statistics(self, sample_mixed_df):
        profile = profile_dataset(sample_mixed_df)
        income_col = next(c for c in profile.columns if c.name == "income")
        assert income_col.min == 30000.0
        assert income_col.max == 30000.0 + 99 * 500.0
        assert income_col.mean is not None
        assert income_col.std is not None

    def test_categorical_top_values(self, sample_mixed_df):
        profile = profile_dataset(sample_mixed_df)
        cat_col = next(c for c in profile.columns if c.name == "category")
        assert cat_col.top_values is not None
        assert len(cat_col.top_values) == 2


class TestEngineSelector:
    """Verify engine selection rules."""

    def test_small_data_selects_pandas(self):
        engine = select_engine(n_rows=10_000, memory_mb=5.0)
        assert engine == ExecutionEngine.PANDAS

    def test_large_rows_selects_polars(self):
        engine = select_engine(n_rows=600_000, memory_mb=50.0)
        assert engine == ExecutionEngine.POLARS

    def test_large_memory_selects_polars(self):
        engine = select_engine(n_rows=100_000, memory_mb=600.0)
        assert engine == ExecutionEngine.POLARS

    def test_config_override_respected(self):
        cfg = AADSConfig(default_engine=ExecutionEngine.DUCKDB)
        engine = select_engine(n_rows=100, memory_mb=1.0, config=cfg)
        assert engine == ExecutionEngine.DUCKDB


class TestProfilerAgent:
    """Verify ProfilerAgent lifecycle, state updates, and artifact creation."""

    def test_agent_run(self, tmp_path, sample_mixed_df):
        storage_root = tmp_path / "runs"
        mgr = ArtifactManager(storage_root=storage_root)
        mgr.initialize_run("profiler_test_run")

        state = RunState.create(user_objective="Predict churn")
        agent = ProfilerAgent(artifact_manager=mgr)

        profile = agent.run(
            df=sample_mixed_df,
            state=state,
            file_path="test.csv",
            file_hash="dummy_hash",
        )

        # Check state updates
        assert state.dataset_meta is not None
        assert state.dataset_meta.n_rows == 100
        assert state.dataset_meta.n_cols == 8
        assert "profiling" in state.completed_phases
        assert len(state.decisions) == 1

        # Check artifact written
        meta_dir = mgr.get_path("metadata")
        assert (meta_dir / "dataset_profile.json").exists()
        assert len(mgr.artifacts) == 1
