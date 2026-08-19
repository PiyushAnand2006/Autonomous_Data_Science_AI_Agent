"""
AADS Report Builder Tool — synthesizes executive summary and project documentation.
"""

from __future__ import annotations

from typing import Any

from aads.core.state import RunState


def build_executive_summary_md(
    state: RunState,
    best_model_name: str,
    best_metrics: dict[str, float],
    eval_report: dict[str, Any] | None = None,
) -> str:
    """Construct a comprehensive Executive Summary in Markdown."""
    target = state.target_column or "N/A"
    task_type = state.task_type.value if state.task_type else "N/A"
    n_rows = state.dataset_meta.n_rows if state.dataset_meta else "Unknown"
    n_cols = state.dataset_meta.n_cols if state.dataset_meta else "Unknown"

    metrics_table_rows = "\n".join([f"| **{k.upper()}** | `{v}` |" for k, v in best_metrics.items()])

    experiments_table = ""
    if state.experiments:
        exp_rows = []
        for exp in state.experiments:
            is_best_str = "⭐ **BEST**" if exp.is_best else "Baseline" if exp.is_baseline else "Candidate"
            metrics_str = ", ".join([f"{k}={v}" for k, v in exp.metrics.items()])
            exp_rows.append(f"| `{exp.model_name}` | {is_best_str} | {metrics_str} |")
        experiments_table = "\n".join(exp_rows)

    decisions_list = "\n".join([
        f"- **[{d.agent.upper()}]** {d.action}: {d.reason}" for d in state.decisions[:10]
    ])

    content = f"""# AADS Executive Project Report
**Run ID:** `{state.run_id}`  
**Objective:** {state.user_objective or 'Autonomous Data Science Workflow'}  
**Status:** Completed  

---

## 1. Problem Formulation & Dataset
- **Task Type:** `{task_type}`
- **Target Variable:** `{target}`
- **Dataset Dimensions:** {n_rows} rows × {n_cols} columns
- **Execution Engine:** {state.dataset_meta.execution_engine.value if state.dataset_meta else 'pandas'}

---

## 2. Winning Model Performance
The best performing model identified across all experiments is **`{best_model_name}`**.

| Metric | Score |
|---|---|
{metrics_table_rows}

---

## 3. Experimentation Log
| Model Architecture | Role | Holdout Validation Metrics |
|---|---|---|
{experiments_table if experiments_table else '| None | N/A | N/A |'}

---

## 4. Autonomous Audit Trail & Decisions
{decisions_list if decisions_list else '- No decisions recorded.'}

---
*Generated automatically by Autonomous AI Data Scientist (AADS).*
"""
    return content


def build_project_readme_md(
    state: RunState,
    best_model_name: str,
    best_metrics: dict[str, float],
) -> str:
    """Construct the run directory README.md."""
    return f"""# AADS Project Run — `{state.run_id}`

## Overview
- **Objective:** {state.user_objective}
- **Target:** `{state.target_column}`
- **Winning Model:** `{best_model_name}`
- **Primary Metrics:** {best_metrics}

## Directory Structure
- `01_Raw_Data/` — Immutable original uploaded dataset
- `02_Cleaned_Data/` — Sanitized and imputed data (`cleaned_dataset.parquet`)
- `03_Feature_Engineered_Data/` — Feature interactions & transforms
- `04_ML_Ready_Data/` — Encoded Train/Val/Test partitions
- `05_Notebooks/` — Reproducible end-to-end Jupyter Notebook (`pipeline_notebook.ipynb`)
- `06_Models/` — Serialized best model (`best_model.pkl`) & pipeline (`preprocessing_pipeline.pkl`)
- `07_Visualizations/` — EDA charts, distributions, correlations & diagnostic plots
- `08_Reports/` — Executive summary & model evaluation reports
- `09_Experiments/` — Experiment logs (`experiment_results.csv`)
- `10_Metadata/` — Run state (`run_state.json`), quality audits & data profiles
"""
