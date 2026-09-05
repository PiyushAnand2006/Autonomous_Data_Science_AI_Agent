"""
AADS Master Orchestrator — coordinates all specialist agents through the full
autonomous end-to-end data-science lifecycle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from aads.agents.artifact_manager import ArtifactManager
from aads.agents.cleaning import CleaningAgent
from aads.agents.data_quality import DataQualityAgent
from aads.agents.eda import EDAAgent
from aads.agents.evaluation import EvaluationAgent
from aads.agents.feature_engineering import FeatureEngineeringAgent
from aads.agents.leakage_guard import LeakageGuard
from aads.agents.ml_experiment import MLExperimentAgent
from aads.agents.notebook_generator import NotebookGeneratorAgent
from aads.agents.planner import GoalPlannerAgent
from aads.agents.preprocessing import PreprocessingAgent
from aads.agents.profiler import ProfilerAgent
from aads.agents.replanning import ReplanningAgent
from aads.agents.report_generator import ReportGeneratorAgent
from aads.agents.split_manager import SplitManager
from aads.core.config import AADSConfig
from aads.core.logging import get_logger
from aads.core.schemas import AutonomyMode
from aads.core.state import RunState
from aads.tools.filesystem.hashing import compute_file_hash
from aads.tools.loaders.registry import LoaderRegistry

logger = get_logger(__name__)


class AADSOrchestrator:
    """The master supervisor agent coordinating the entire AADS lifecycle."""

    def __init__(
        self,
        config: Optional[AADSConfig] = None,
        storage_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config or AADSConfig()
        if storage_root and str(storage_root).strip():
            p = Path(storage_root)
            self.storage_root = p.resolve() if p.is_absolute() else (self.config.project_root / p).resolve()
        else:
            self.storage_root = self.config.storage_root
        self.loader_registry = LoaderRegistry()

    def run_pipeline(
        self,
        data_path: Union[str, Path],
        user_objective: str,
        target_column: Optional[str] = None,
        run_id: Optional[str] = None,
        autonomy_mode: AutonomyMode = AutonomyMode.FULLY_AUTONOMOUS,
        progress_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Execute the complete end-to-end autonomous data-science workflow.

        Args:
            data_path: Path to the raw dataset file (CSV, XLSX, Parquet).
            user_objective: Natural language task or goal description.
            target_column: Optional explicit target column name.
            run_id: Optional unique run identifier (auto-generated if None).
            autonomy_mode: Fully autonomous, semi-autonomous, or manual.
            progress_callback: Optional callable receiving real-time phase updates.

        Returns:
            Dictionary summarizing all generated artifacts, model metrics, and final RunState.
        """
        import os
        # Resolve source path across direct and standard upload paths
        candidate_paths = [
            Path(data_path),
            Path(data_path).resolve(),
            self.config.project_root / data_path,
            self.config.project_root / "storage" / "temp_uploads" / Path(data_path).name,
            self.config.project_root / "data" / Path(data_path).name,
            Path("storage/temp_uploads") / Path(data_path).name,
            Path("data") / Path(data_path).name,
        ]
        source_path = None
        for p in candidate_paths:
            if p.exists() and p.is_file():
                source_path = p.resolve()
                break

        if not source_path:
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        def _notify(msg: str) -> None:
            if progress_callback:
                try:
                    progress_callback(msg)
                    import time
                    time.sleep(0.45)
                except Exception:
                    pass

        _notify("◈ [INIT] Initializing run state and copying raw data...")

        # 1. Initialize State & Artifact Manager
        state = RunState.create(
            user_objective=user_objective,
            target_column=target_column,
            autonomy_mode=autonomy_mode,
            random_seed=self.config.random_seed,
        )
        if run_id:
            state.run_id = run_id

        artifact_mgr = ArtifactManager(storage_root=self.storage_root)
        run_dir = artifact_mgr.initialize_run(state.run_id)

        logger.info("orchestrator_pipeline_started", run_id=state.run_id, data_path=str(source_path))

        # 2. Immutable Raw Data Copy
        raw_copy_path = artifact_mgr.copy_raw_data(source_path)
        file_hash = compute_file_hash(raw_copy_path)

        # 3. Load Dataset
        raw_df = self.loader_registry.load(raw_copy_path)

        # Initialize LLM if in AI Mode
        llm = None
        if self.config.execution_mode == "ai":
            try:
                from aads.core.llm import get_llm
                llm = get_llm(self.config)
                _notify(f"✦ [AI SETUP: {self.config.llm_provider.upper()} ({self.config.llm_model})] Initialized provider reasoning client.")
            except Exception as e:
                _notify(f"⚠ [AI WARNING] Failed to initialize provider ({e}). Falling back to local offline reasoning.")

        # 4. Agent: Profiler
        _notify("⬡ [PROFILER] Profiling dataset dimensions, statistics, and column semantics...")
        profiler = ProfilerAgent(config=self.config, artifact_manager=artifact_mgr)
        profile = profiler.run(
            df=raw_df,
            state=state,
            file_path=str(raw_copy_path),
            file_hash=file_hash,
            file_format=source_path.suffix.lstrip(".").lower(),
        )

        # 5. Agent: Goal & Planner
        if self.config.execution_mode == "ai":
            _notify(f"✦ [AI PLANNER] Querying {self.config.llm_provider.upper()} ({self.config.llm_model}) for strategy & task formulation...")
        else:
            _notify("◈ [PLANNER] Formulating task strategy and deterministic execution plan...")
        planner = GoalPlannerAgent(config=self.config, llm=llm)
        plan = planner.plan(profile=profile, state=state)

        # 6. Agent: Data Quality
        _notify("⬡ [DATA QUALITY] Auditing data quality, missing values, and anomalies...")
        dq_agent = DataQualityAgent(config=self.config, artifact_manager=artifact_mgr)
        dq_report = dq_agent.run(df=raw_df, state=state)

        # 7. Agent: EDA & Visualization
        _notify("◈ [EDA] Generating exploratory data analysis charts and correlation plots...")
        eda_agent = EDAAgent(config=self.config, artifact_manager=artifact_mgr)
        eda_findings = eda_agent.run(df=raw_df, state=state)
        import gc; gc.collect()

        # 8. Agent: Data Cleaning
        _notify("⬡ [CLEANER] Sanitizing missing values, date columns, and deduplicating...")
        cleaning_agent = CleaningAgent(config=self.config, artifact_manager=artifact_mgr)
        cleaned_df, cleaning_log = cleaning_agent.run(df=raw_df, state=state)
        gc.collect()

        # 9. Agent: Split Manager
        _notify("◈ [SPLITTER] Partitioning data into train/val/test holdouts...")
        split_mgr = SplitManager(config=self.config, artifact_manager=artifact_mgr)
        X_train, X_val, X_test, y_train, y_val, y_test = split_mgr.run(df=cleaned_df, state=state)

        # 10. Agent: Leakage Guard
        _notify("⬡ [LEAKAGE GUARD] Verifying strict data leakage guards and pruning leaky proxies...")
        leakage_guard = LeakageGuard(config=self.config, artifact_manager=artifact_mgr)
        X_train, X_test, X_val, leakage_report = leakage_guard.run(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, state=state, X_val=X_val
        )

        # 11. Agent: Feature Engineering
        if self.config.execution_mode == "ai":
            _notify(f"✦ [AI FEATURE ENG] Selecting high-value features, pruning noise & synthesizing domain interactions with {self.config.llm_provider.upper()}...")
        else:
            _notify("◈ [FEATURE ENG] Constructing domain interactions and feature transformations...")
        fe_agent = FeatureEngineeringAgent(config=self.config, artifact_manager=artifact_mgr, llm=llm)
        X_train_fe, X_test_fe, X_val_fe, fe_log = fe_agent.run(
            X_train=X_train, X_test=X_test, y_train=y_train, state=state, X_val=X_val
        )
        gc.collect()

        # 12. Agent: Preprocessing Pipeline
        if self.config.execution_mode == "ai":
            _notify(f"✦ [AI PREPROCESS] Analyzing feature cardinalities & fitting adaptive encoder pipeline with {self.config.llm_provider.upper()}...")
        else:
            _notify("◈ [PREPROCESS] Fitting adaptive encoder and scaling pipeline...")
        prep_agent = PreprocessingAgent(config=self.config, artifact_manager=artifact_mgr, llm=llm)
        X_train_enc, X_test_enc, X_val_enc, preprocessor = prep_agent.run(
            X_train=X_train_fe, X_test=X_test_fe, state=state, X_val=X_val_fe
        )
        gc.collect()

        # 13. Agent: ML Experimentation
        _notify("⚡ [ML EXPERIMENT] Training candidate machine learning models and evaluating leaderboard... [0/12]")
        X_eval_enc = X_val_enc if (X_val_enc is not None and len(X_val_enc) > 0) else X_test_enc
        y_eval = y_val if (y_val is not None and len(y_val) > 0) else y_test

        ml_agent = MLExperimentAgent(config=self.config, artifact_manager=artifact_mgr, llm=llm)
        best_model, best_model_name, best_metrics, experiments = ml_agent.run(
            X_train=X_train_enc,
            y_train=y_train,
            X_eval=X_eval_enc,
            y_eval=y_eval,
            state=state,
            progress_callback=_notify,
        )
        gc.collect()

        # 14. Agent: Evaluation & Diagnostics
        _notify("◈ [EVALUATION] Computing residual diagnostics and holdout test metrics...")
        eval_agent = EvaluationAgent(config=self.config, artifact_manager=artifact_mgr)
        eval_report = eval_agent.run(
            model=best_model,
            model_name=best_model_name,
            X_test=X_test_enc,
            y_test=y_test,
            state=state,
        )

        # 15. Agent: Replanning
        replanning_agent = ReplanningAgent(config=self.config, artifact_manager=artifact_mgr)
        replanning_agent.run(evaluation_report=eval_report, state=state, current_iteration=1)

        # 16. Agent: Notebook Generation & Validation
        _notify("⬡ [NOTEBOOK] Synthesizing and programmatically validating Jupyter Notebook...")
        nb_agent = NotebookGeneratorAgent(config=self.config, artifact_manager=artifact_mgr)
        notebook_dict = nb_agent.run(
            state=state,
            best_model_name=best_model_name,
            raw_data_filename=raw_copy_path.name,
            top_models=ml_agent.top_models,
        )

        # 17. Agent: Report Generation
        if self.config.execution_mode == "ai":
            _notify(f"✦ [AI REPORTING] Querying {self.config.llm_provider.upper()} ({self.config.llm_model}) for Chief AI Scientist analytical narrative...")
        else:
            _notify("◈ [EXECUTIVE REPORT] Synthesizing comprehensive in-depth executive summary...")
        report_agent = ReportGeneratorAgent(config=self.config, artifact_manager=artifact_mgr, llm=llm)
        exec_summary = report_agent.run(
            state=state,
            best_model_name=best_model_name,
            best_metrics=best_metrics,
            eval_report=eval_report,
            top_models=ml_agent.top_models,
        )

        # 18. Persist final RunState JSON to 10_Metadata/run_state.json
        state_save_path = run_dir / "10_Metadata" / "run_state.json"
        state.save(state_save_path)

        logger.info(
            "orchestrator_pipeline_finished_successfully",
            run_id=state.run_id,
            completed_phases=state.completed_phases,
            best_model=best_model_name,
            top_models_count=len(ml_agent.top_models),
            artifacts_count=len(artifact_mgr.artifacts),
        )

        return {
            "run_id": state.run_id,
            "run_dir": str(run_dir),
            "state": state,
            "best_model_name": best_model_name,
            "best_metrics": best_metrics,
            "top_models": ml_agent.top_models,
            "task_plan": plan,
            "data_quality_report": dq_report,
            "eda_findings": eda_findings,
            "total_artifacts": len(artifact_mgr.artifacts),
            "executive_summary": exec_summary,
        }
