"""
Tests for DataQualityAgent and EDAAgent workflows.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aads.agents.artifact_manager import ArtifactManager
from aads.agents.data_quality import DataQualityAgent
from aads.agents.eda import EDAAgent
from aads.core.schemas import ArtifactType, DataQualityReport, EDAFindings
from aads.core.state import RunState


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(35, 10, 80),
        "income": np.random.normal(60000, 15000, 80),
        "department": np.random.choice(["Sales", "Engineering", "Marketing"], 80),
        "performance": np.random.choice([1, 2, 3, 4, 5], 80),
    })


class TestDataQualityAgent:
    """Verify DataQualityAgent execution, state updates, and artifact writing."""

    def test_run_data_quality_agent(self, tmp_path, sample_dataset):
        mgr = ArtifactManager(storage_root=tmp_path)
        mgr.initialize_run("dq_test_run")
        state = RunState.create(user_objective="Analyze employee performance", target_column="performance")

        agent = DataQualityAgent(artifact_manager=mgr)
        report = agent.run(sample_dataset, state)

        assert isinstance(report, DataQualityReport)
        assert "data_quality" in state.completed_phases
        assert len(state.decisions) == 1

        # Check artifact written to 10_Metadata/
        meta_dir = mgr.get_path("metadata")
        assert (meta_dir / "data_quality_report.json").exists()


class TestEDAAgent:
    """Verify EDAAgent execution, chart generation, and findings extraction."""

    def test_run_eda_agent(self, tmp_path, sample_dataset):
        mgr = ArtifactManager(storage_root=tmp_path)
        mgr.initialize_run("eda_test_run")
        state = RunState.create(user_objective="Analyze employee performance", target_column="performance")

        agent = EDAAgent(artifact_manager=mgr)
        findings = agent.run(sample_dataset, state)

        assert isinstance(findings, EDAFindings)
        assert "eda" in state.completed_phases
        assert len(findings.generated_visualizations) > 0
        assert len(state.decisions) == 1

        # Check artifact written to 10_Metadata/
        meta_dir = mgr.get_path("metadata")
        assert (meta_dir / "eda_findings.json").exists()

        # Check charts were created in 07_Visualizations/
        viz_artifacts = mgr.get_artifacts_by_type(ArtifactType.VISUALIZATION)
        assert len(viz_artifacts) >= 4
