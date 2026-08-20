"""
Tests for Jupyter Notebook generation, multi-cell structure, and programmatic execution validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from aads.agents.artifact_manager import ArtifactManager
from aads.agents.notebook_generator import NotebookGeneratorAgent
from aads.core.schemas import TaskType
from aads.core.state import RunState
from aads.tools.notebook.generator import build_project_notebook
from aads.tools.notebook.validator import validate_and_execute_notebook


class TestNotebookGenerationAndValidation:

    def test_notebook_structure_has_multiple_cells(self):
        state = RunState.create(user_objective="Predict continuous values", target_column="target")
        state.task_type = TaskType.REGRESSION

        top_models = [
            {"rank": 1, "model_name": "RandomForestRegressor", "metrics": {"rmse": 1.23, "r2": 0.88}, "selection_reason": "Best"},
            {"rank": 2, "model_name": "GradientBoostingRegressor", "metrics": {"rmse": 1.45, "r2": 0.85}, "selection_reason": "Runner up"},
        ]

        nb_dict = build_project_notebook(
            state=state,
            raw_data_filename="data.csv",
            best_model_name="RandomForestRegressor",
            top_models=top_models,
        )

        assert "cells" in nb_dict
        cells = nb_dict["cells"]
        # Must have multiple separate markdown and code cells, never just 1 monolithic cell
        assert len(cells) >= 15

        markdown_cells = [c for c in cells if c["cell_type"] == "markdown"]
        code_cells = [c for c in cells if c["cell_type"] == "code"]

        assert len(markdown_cells) >= 6
        assert len(code_cells) >= 6

    def test_notebook_agent_and_validator_run(self, tmp_path):
        mgr = ArtifactManager(storage_root=tmp_path)
        mgr.initialize_run("nb_test_run")
        state = RunState.create(user_objective="Predict binary target", target_column="target")
        state.task_type = TaskType.CLASSIFICATION

        # Create raw csv in 01_Raw_Data
        raw_dir = mgr.get_path("raw_data")
        dummy_df = pd.DataFrame({
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
            "target": [0, 1] * 25,
        })
        dummy_csv = raw_dir / "original_dataset.csv"
        dummy_df.to_csv(dummy_csv, index=False)

        top_models = [
            {"rank": 1, "model_name": "RandomForestClassifier", "metrics": {"f1": 0.95, "accuracy": 0.94}, "selection_reason": "Best F1"},
            {"rank": 2, "model_name": "LogisticRegression", "metrics": {"f1": 0.90, "accuracy": 0.89}, "selection_reason": "Baseline"},
        ]

        agent = NotebookGeneratorAgent(artifact_manager=mgr)
        nb_dict = agent.run(
            state=state,
            best_model_name="RandomForestClassifier",
            raw_data_filename="original_dataset.csv",
            top_models=top_models,
        )

        nb_dir = mgr.get_path("notebook")
        assert (nb_dir / "autonomous_analysis.ipynb").exists()
        assert (nb_dir / "notebook_validation.json").exists()

        # Check validation execution result
        val_result = validate_and_execute_notebook(nb_dir / "autonomous_analysis.ipynb", working_dir=nb_dir)
        assert val_result.total_cells >= 6
        assert val_result.success is True
        assert len(val_result.errors) == 0
