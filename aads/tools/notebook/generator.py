"""
AADS Notebook Generator Tool — creates structured, standalone, executable Jupyter Notebooks.
"""

from __future__ import annotations

import json
from typing import Any

from aads.core.schemas import TaskType
from aads.core.state import RunState


def _make_code_cell(source_code: str) -> dict[str, Any]:
    """Create a Jupyter v4 format code cell dictionary."""
    lines = [line + "\n" for line in source_code.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def _make_markdown_cell(markdown_text: str) -> dict[str, Any]:
    """Create a Jupyter v4 format markdown cell dictionary."""
    lines = [line + "\n" for line in markdown_text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def build_project_notebook(
    state: RunState,
    raw_data_filename: str = "dataset.csv",
    best_model_name: str = "RandomForestClassifier",
) -> dict[str, Any]:
    """Construct a complete, self-contained, reproducible Jupyter Notebook.

    Args:
        state: RunState containing execution details.
        raw_data_filename: Name of the raw data file.
        best_model_name: Name of the selected best model.

    Returns:
        Dictionary conforming to Jupyter Notebook v4 JSON schema.
    """
    target_col = state.target_column or "target"
    task_type = state.task_type or TaskType.CLASSIFICATION
    objective = state.user_objective or "End-to-End Data Science Pipeline"

    cells: list[dict[str, Any]] = []

    # 1. Title & Executive Summary
    cells.append(_make_markdown_cell(f"""# Autonomous AI Data Science (AADS) — Reproducible Pipeline
**Run ID:** `{state.run_id}`  
**Objective:** {objective}  
**Target Column:** `{target_col}`  
**Task Type:** `{task_type.value}`  

---
This notebook contains the complete, reproducible end-to-end data processing, feature engineering, and model training pipeline.
"""))

    # 2. Imports & Seed
    cells.append(_make_markdown_cell("## 1. Environment Setup & Imports"))
    cells.append(_make_code_cell(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import {best_model_name}
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Set seed for exact reproducibility
RANDOM_SEED = {state.random_seed}
np.random.seed(RANDOM_SEED)
print("Environment setup complete.")
"""))

    # 3. Data Loading
    cells.append(_make_markdown_cell("## 2. Load Raw Dataset"))
    cells.append(_make_code_cell(f"""# Load raw data from 01_Raw_Data
data_path = Path("../01_Raw_Data/{raw_data_filename}")
if not data_path.exists():
    # Fallback to local search
    data_path = Path("{raw_data_filename}")

df = pd.read_csv(data_path) if data_path.suffix.lower() == '.csv' else pd.read_parquet(data_path)
print(f"Loaded dataset: {{df.shape[0]}} rows, {{df.shape[1]}} columns")
df.head()
"""))

    # 4. Cleaning & Imputation
    cells.append(_make_markdown_cell("## 3. Data Cleaning & Hygiene"))
    cells.append(_make_code_cell(f"""# Standardize placeholder null strings
hidden_nulls = ["?", "na", "n/a", "null", "none", "", "nan"]
for col in df.select_dtypes(include='object').columns:
    mask = df[col].astype(str).str.strip().str.lower().isin(hidden_nulls)
    df.loc[mask, col] = np.nan

# Drop exact duplicates and missing target rows
df = df.drop_duplicates()
if "{target_col}" in df.columns:
    df = df.dropna(subset=["{target_col}"]).reset_index(drop=True)

# Drop constant columns
non_constant_cols = [c for c in df.columns if df[c].dropna().nunique() > 1 or c == "{target_col}"]
df = df[non_constant_cols]

print(f"Shape after cleaning: {{df.shape}}")
"""))

    # 5. Train / Test Split
    cells.append(_make_markdown_cell("## 4. Leakage-Safe Splitting"))
    stratify_code = f"stratify=y" if task_type == TaskType.CLASSIFICATION else "stratify=None"
    cells.append(_make_code_cell(f"""X = df.drop(columns=["{target_col}"])
y = df["{target_col}"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, {stratify_code}
)
print(f"Train set: {{X_train.shape}}, Test set: {{X_test.shape}}")
"""))

    # 6. Preprocessing & Model Pipeline
    cells.append(_make_markdown_cell("## 5. Model Training & Pipeline"))
    cells.append(_make_code_cell(f"""numeric_cols = list(X_train.select_dtypes(include=['int64', 'float64']).columns)
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_cols)
])

# Complete end-to-end model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', {best_model_name}(random_state=RANDOM_SEED))
])

print("Training best model pipeline...")
pipeline.fit(X_train, y_train)
print("Model training complete.")
"""))

    # 7. Model Evaluation
    cells.append(_make_markdown_cell("## 6. Holdout Test Evaluation"))
    if task_type == TaskType.REGRESSION:
        cells.append(_make_code_cell("""y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2 Score: {r2:.4f}")

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.6, color="#2563eb")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Actual vs Predicted")
plt.show()
"""))
    else:
        cells.append(_make_code_cell("""y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Weighted F1: {f1:.4f}")
"""))

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook_dict
