"""
End-to-End Integration Tests — tests AADSOrchestrator running the complete autonomous lifecycle.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aads.agents.orchestrator import AADSOrchestrator
from aads.core.config import AADSConfig
from aads.core.schemas import TaskType


@pytest.fixture
def synthetic_classification_csv(tmp_path) -> Path:
    """Create a raw synthetic classification dataset with slight noise."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": [20 + (i % 45) for i in range(n)],
        "monthly_spend": [100.0 + (i * 12.5) for i in range(n)],
        "contract_type": ["Month-to-Month" if i % 2 == 0 else "Two-Year" for i in range(n)],
        "support_calls": [i % 5 for i in range(n)],
        "churn": [1 if (i % 3 == 0 or i % 7 == 0) else 0 for i in range(n)],
    })
    # Inject a few missing values and a duplicate row
    df.loc[5, "monthly_spend"] = np.nan
    df.loc[10, "contract_type"] = "?"
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    csv_path = tmp_path / "raw_churn_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_regression_csv(tmp_path) -> Path:
    """Create a raw synthetic regression dataset."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "house_id": range(n),
        "sqft": [800.0 + i * 25.0 for i in range(n)],
        "bedrooms": [2 + (i % 4) for i in range(n)],
        "neighborhood": ["Urban" if i % 2 == 0 else "Suburban" for i in range(n)],
        "price": [150_000.0 + i * 3500.0 for i in range(n)],
    })
    csv_path = tmp_path / "raw_housing_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestEndToEndPipeline:
    """Verify complete autonomous pipeline execution from raw file to final models & reports."""

    def test_end_to_end_classification_run(self, tmp_path, synthetic_classification_csv):
        storage_root = tmp_path / "runs"
        config = AADSConfig(storage_root=storage_root, random_seed=42)
        orchestrator = AADSOrchestrator(config=config, storage_root=storage_root)

        result = orchestrator.run_pipeline(
            data_path=synthetic_classification_csv,
            user_objective="Predict whether customer churn occurs based on customer spending and contract type.",
            target_column="churn",
        )

        assert result["run_id"] is not None
        assert result["best_model_name"] is not None
        assert "accuracy" in result["best_metrics"] or "f1" in result["best_metrics"]
        assert result["total_artifacts"] >= 10

        run_dir = Path(result["run_dir"])

        # Check all contract directories contain their expected artifacts
        assert (run_dir / "01_Raw_Data" / "raw_churn_data.csv").exists()
        assert (run_dir / "02_Cleaned_Data" / "cleaned_dataset.csv").exists()
        assert (run_dir / "03_Feature_Engineered_Data" / "feature_engineered_dataset.csv").exists()
        assert (run_dir / "04_ML_Ready_Data" / "ml_ready_dataset.csv").exists()
        assert (run_dir / "05_Notebook" / "autonomous_analysis.ipynb").exists()
        assert (run_dir / "05_Notebook" / "notebook_validation.json").exists()
        assert (run_dir / "06_Models" / "preprocessing_pipeline.pkl").exists()
        assert (run_dir / "06_Models" / "model_comparison.json").exists()
        assert (run_dir / "06_Models" / "model_metadata.json").exists()
        assert len(list((run_dir / "06_Models").glob("model_*.pkl"))) >= 3
        assert (run_dir / "07_Visualizations" / "correlations" / "correlation_matrix.png").exists()
        assert (run_dir / "08_Reports" / "executive_summary.md").exists()
        assert (run_dir / "09_Experiments" / "experiment_results.csv").exists()
        assert (run_dir / "10_Metadata" / "run_state.json").exists()
        assert (run_dir / "README.md").exists()

        # Check state transitions
        state = result["state"]
        expected_phases = {
            "profiling",
            "planning",
            "data_quality",
            "eda",
            "cleaning",
            "split",
            "leakage_check",
            "feature_engineering",
            "preprocessing",
            "ml_experiment",
            "evaluation",
            "replanning",
            "notebook_generation",
            "report_generation",
        }
        for phase in expected_phases:
            assert phase in state.completed_phases

    def test_end_to_end_regression_run(self, tmp_path, synthetic_regression_csv):
        storage_root = tmp_path / "runs_reg"
        config = AADSConfig(storage_root=storage_root, random_seed=42)
        orchestrator = AADSOrchestrator(config=config, storage_root=storage_root)

        result = orchestrator.run_pipeline(
            data_path=synthetic_regression_csv,
            user_objective="Predict continuous house price from characteristics.",
            target_column="price",
        )

        assert result["state"].task_type == TaskType.REGRESSION
        assert "rmse" in result["best_metrics"]
        assert "r2" in result["best_metrics"]
        run_dir = Path(result["run_dir"])
        assert (run_dir / "02_Cleaned_Data" / "cleaned_dataset.csv").exists()
        assert (run_dir / "03_Feature_Engineered_Data" / "feature_engineered_dataset.csv").exists()
        assert (run_dir / "04_ML_Ready_Data" / "ml_ready_dataset.csv").exists()
        assert (run_dir / "05_Notebook" / "autonomous_analysis.ipynb").exists()
        assert (run_dir / "06_Models" / "model_comparison.json").exists()
        assert len(list((run_dir / "06_Models").glob("model_*.pkl"))) >= 3
