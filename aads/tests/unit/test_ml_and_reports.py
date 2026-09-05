"""
Tests for ML Experimentation, Evaluation, Replanning, Notebook, and Report agents.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aads.agents.artifact_manager import ArtifactManager
from aads.agents.evaluation import EvaluationAgent
from aads.agents.ml_experiment import MLExperimentAgent
from aads.agents.notebook_generator import NotebookGeneratorAgent
from aads.agents.replanning import ReplanningAgent
from aads.agents.report_generator import ReportGeneratorAgent
from aads.core.schemas import TaskType
from aads.core.state import RunState
from aads.tools.ml.trainer import train_and_evaluate_model


@pytest.fixture
def sample_ml_data():
    np.random.seed(42)
    X_train = pd.DataFrame({
        "feature_1": np.random.randn(80),
        "feature_2": np.random.randn(80),
    })
    y_train = pd.Series(np.random.choice([0, 1], 80))

    X_test = pd.DataFrame({
        "feature_1": np.random.randn(20),
        "feature_2": np.random.randn(20),
    })
    y_test = pd.Series(np.random.choice([0, 1], 20))

    return X_train, y_train, X_test, y_test


class TestMLExperimentation:

    def test_train_and_evaluate_tool(self, sample_ml_data):
        X_train, y_train, X_test, y_test = sample_ml_data
        model, metrics, y_pred, training_time = train_and_evaluate_model(
            "LogisticRegression", X_train, y_train, X_test, y_test, TaskType.CLASSIFICATION
        )
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert len(y_pred) == len(y_test)
        assert training_time >= 0

    def test_extended_regression_models(self):
        np.random.seed(42)
        X_tr = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        y_tr = pd.Series(np.random.randn(50))
        X_te = pd.DataFrame({"a": np.random.randn(10), "b": np.random.randn(10)})
        y_te = pd.Series(np.random.randn(10))

        reg_models = [
            "LinearRegression", "Ridge", "Lasso", "ElasticNet",
            "KNeighborsRegressor", "DecisionTreeRegressor",
            "ExtraTreesRegressor", "AdaBoostRegressor",
            "HistGradientBoostingRegressor", "LinearSVR"
        ]
        for m in reg_models:
            fitted, metrics, preds, duration = train_and_evaluate_model(
                m, X_tr, y_tr, X_te, y_te, task_type=TaskType.REGRESSION
            )
            assert "rmse" in metrics, f"Missing rmse for {m}"
            assert "r2" in metrics, f"Missing r2 for {m}"
            assert len(preds) == len(y_te)

    def test_extended_classification_models(self, sample_ml_data):
        X_train, y_train, X_test, y_test = sample_ml_data
        clf_models = [
            "KNeighborsClassifier", "DecisionTreeClassifier",
            "ExtraTreesClassifier", "AdaBoostClassifier",
            "HistGradientBoostingClassifier", "GaussianNB", "LinearSVC"
        ]
        for m in clf_models:
            fitted, metrics, preds, duration = train_and_evaluate_model(
                m, X_train, y_train, X_test, y_test, task_type=TaskType.CLASSIFICATION
            )
            assert "accuracy" in metrics, f"Missing accuracy for {m}"
            assert "f1" in metrics, f"Missing f1 for {m}"
            assert len(preds) == len(y_test)

    def test_clustering_models(self):
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(60), "f2": np.random.randn(60)})
        for m in ["KMeans", "GaussianMixture", "DBSCAN"]:
            fitted, metrics, preds, duration = train_and_evaluate_model(
                m, X, None, X, None, task_type=TaskType.CLUSTERING
            )
            assert "silhouette" in metrics, f"Missing silhouette for {m}"
            assert "n_clusters" in metrics, f"Missing n_clusters for {m}"


    def test_ml_experiment_agent_and_reports(self, tmp_path, sample_ml_data):
        mgr = ArtifactManager(storage_root=tmp_path)
        mgr.initialize_run("ml_test_run")
        state = RunState.create(user_objective="Classify records", target_column="target")

        X_train, y_train, X_test, y_test = sample_ml_data

        # 1. ML Experiment Agent
        ml_agent = MLExperimentAgent(artifact_manager=mgr)
        best_model, best_name, best_metrics, experiments = ml_agent.run(
            X_train, y_train, X_test, y_test, state, candidate_models=["LogisticRegression", "RandomForestClassifier"]
        )
        assert "ml_experiment" in state.completed_phases
        assert (mgr.get_path("models") / "best_model.pkl").exists()
        assert (mgr.get_path("models") / "model_comparison.json").exists()
        assert (mgr.get_path("models") / "model_metadata.json").exists()
        assert len(list(mgr.get_path("models").glob("model_01_*.pkl"))) >= 1
        assert (mgr.get_path("experiments") / "experiment_results.csv").exists()

        # 2. Evaluation Agent
        eval_agent = EvaluationAgent(artifact_manager=mgr)
        eval_report = eval_agent.run(best_model, best_name, X_test, y_test, state)
        assert "evaluation" in state.completed_phases
        assert (mgr.get_path("reports") / "model_report.json").exists()

        # 3. Replanning Agent
        replan_agent = ReplanningAgent(artifact_manager=mgr)
        replan_res = replan_agent.run(eval_report, state)
        assert "replanning" in state.completed_phases

        # 4. Notebook Generator Agent
        nb_agent = NotebookGeneratorAgent(artifact_manager=mgr)
        nb_dict = nb_agent.run(state, best_model_name=best_name, top_models=ml_agent.top_models)
        assert "notebook_generation" in state.completed_phases
        assert (mgr.get_path("notebook") / "autonomous_analysis.ipynb").exists()
        assert (mgr.get_path("notebook") / "notebook_validation.json").exists()

        # 5. Report Generator Agent
        rep_agent = ReportGeneratorAgent(artifact_manager=mgr)
        summary_md = rep_agent.run(state, best_model_name=best_name, best_metrics=best_metrics, eval_report=eval_report, top_models=ml_agent.top_models)
        assert "report_generation" in state.completed_phases
        assert (mgr.get_path("reports") / "executive_summary.md").exists()
        assert (mgr.current_run_dir / "README.md").exists()

    def test_feature_engineering_biomarker_safeguard(self):
        from aads.agents.feature_engineering import FeatureEngineeringAgent

        state = RunState(user_objective="Predict diabetes risk")
        agent = FeatureEngineeringAgent()

        X_tr = pd.DataFrame({"Blood_Glucose": [100, 120, 150], "HbA1c": [5.4, 6.1, 7.2], "Constant_Col": [1, 1, 1]})
        X_te = X_tr.copy()

        # Mock AI returning attempt to drop biomarkers and constant column
        agent._consult_ai_feature_engineering = lambda *args, **kwargs: (["Blood_Glucose", "HbA1c", "Constant_Col"], [])

        X_tr_fe, X_te_fe, _, log = agent.run(X_tr, X_te, None, state)
        assert "Blood_Glucose" in X_tr_fe.columns, "Blood_Glucose must NOT be dropped!"
        assert "HbA1c" in X_tr_fe.columns, "HbA1c must NOT be dropped!"
        assert "Constant_Col" not in X_tr_fe.columns, "Constant_Col should be dropped because nunique() <= 1"
        assert "Constant_Col" in log["dropped_features"]

