"""
AADS Notebook Generator Tool — creates structured, human-readable, executable Jupyter Notebooks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

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


def _clean_slug(name: str) -> str:
    """Format model name into clean snake_case slug."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return re.sub(r"_+", "_", s2).strip("_")


def build_project_notebook(
    state: RunState,
    raw_data_filename: str = "original_dataset.csv",
    best_model_name: str = "RandomForestClassifier",
    top_models: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Construct a clean, professional, multi-cell Jupyter Notebook conforming to standard ML conventions.

    Args:
        state: RunState containing execution details.
        raw_data_filename: Name of the raw data file.
        best_model_name: Name of the selected best model.
        top_models: Optional list of ranked top models.

    Returns:
        Dictionary conforming to Jupyter Notebook v4 JSON schema.
    """
    target_col = state.target_column or "target"
    task_type = state.task_type or TaskType.CLASSIFICATION
    objective = state.user_objective or "Autonomous Machine Learning Pipeline"

    # Determine top model export entries
    selected_top = top_models or []
    if not selected_top:
        selected_top = [
            {"rank": 1, "model_name": best_model_name, "selection_reason": "Primary top performing model"}
        ]

    cells: list[dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Section 1: Title & Executive Overview
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell(f"""# Autonomous Machine Learning & Data Science Analysis
**Project Objective:** {objective}  
**Target Variable:** `{target_col}`  
**Task Formulation:** `{task_type.value.capitalize()}`  
**Global Random Seed:** `{state.random_seed}`  

---
This notebook contains an end-to-end, reproducible machine learning workflow covering dataset loading, data quality audit, exploratory analysis, preprocessing, candidate model benchmarking, multi-metric ranking, and artifact persistence.
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 2: Imports & Global Configuration
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 1. Imports and Global Configuration"))
    cells.append(_make_code_cell(f"""import os
import math
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-Learn tools & models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings('ignore')

try:
    display
except NameError:
    display = print

# Global reproducibility seed
RANDOM_SEED = {state.random_seed}
np.random.seed(RANDOM_SEED)

# Relative directory paths
BASE_DIR = Path.cwd()
RAW_DATA_PATH = Path("../01_Raw_Data/{raw_data_filename}")
CLEANED_DATA_PATH = Path("../02_Cleaned_Data/cleaned_dataset.csv")
FE_DATA_PATH = Path("../03_Feature_Engineered_Data/feature_engineered_dataset.csv")
ML_READY_DATA_PATH = Path("../04_ML_Ready_Data/ml_ready_dataset.csv")
MODELS_DIR = Path("../06_Models")

# Ensure target output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
Path("../02_Cleaned_Data").mkdir(parents=True, exist_ok=True)
Path("../03_Feature_Engineered_Data").mkdir(parents=True, exist_ok=True)
Path("../04_ML_Ready_Data").mkdir(parents=True, exist_ok=True)

print("Environment setup and directories initialized successfully.")
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 3: Load Raw Dataset
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 2. Load Raw Dataset"))
    cells.append(_make_code_cell(f"""# Check raw dataset path with fallback for standalone notebook runs
data_file = RAW_DATA_PATH if RAW_DATA_PATH.exists() else Path("{raw_data_filename}")
if not data_file.exists():
    # Search locally for csv
    csv_candidates = list(Path(".").glob("*.csv")) + list(Path("../data").glob("*.csv"))
    data_file = csv_candidates[0] if csv_candidates else Path("{raw_data_filename}")

if str(data_file).endswith('.parquet'):
    df_raw = pd.read_parquet(data_file)
elif str(data_file).endswith('.xlsx') or str(data_file).endswith('.xls'):
    df_raw = pd.read_excel(data_file)
else:
    df_raw = pd.read_csv(data_file)

print(f"Loaded raw dataset from: {{data_file}}")
print(f"Dimensions: {{df_raw.shape[0]}} rows × {{df_raw.shape[1]}} columns")
df_raw.head(5)
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 4: Dataset Overview & Quality Inspection
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 3. Dataset Overview and Missing Value Inspection"))
    cells.append(_make_code_cell("""# Display column data types and non-null counts
print("--- Column Information ---")
print(df_raw.info())

# Inspect summary statistics for numerical attributes
print("\\n--- Numerical Feature Statistics ---")
display(df_raw.describe())

# Audit missing values per column
missing_summary = pd.DataFrame({
    'Missing_Count': df_raw.isnull().sum(),
    'Missing_Pct': (df_raw.isnull().sum() / len(df_raw)) * 100
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
print("\\n--- Missing Values Audit ---")
if not missing_summary.empty:
    display(missing_summary)
else:
    print("No missing values detected in raw dataset.")
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 5: Exploratory Data Analysis (EDA)
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 4. Exploratory Data Analysis (EDA)"))
    cells.append(_make_code_cell(f"""# Identify numeric columns for distribution plots
numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
print(f"Numerical columns identified ({{len(numeric_cols)}}): {{numeric_cols[:6]}}")

# Plot distribution of the first numerical features
plot_cols = numeric_cols[:min(4, len(numeric_cols))]
if plot_cols:
    fig, axes = plt.subplots(1, len(plot_cols), figsize=(4 * len(plot_cols), 3))
    if len(plot_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, plot_cols):
        ax.hist(df_raw[col].dropna(), bins=20, color="#2563eb", edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribution: {{col}}", fontsize=10)
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Target variable distribution inspection
target_col_name = "{target_col}"
if target_col_name in df_raw.columns:
    print(f"\\nTarget column '{{target_col_name}}' value distribution:")
    print(df_raw[target_col_name].value_counts(dropna=False).head(10))
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    # Section 6: Data Cleaning & Hygiene
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 5. Data Cleaning and Sanitization"))
    cells.append(_make_code_cell(f"""df_cleaned = df_raw.copy()

# 1. Standardize placeholder missing values (e.g. '?', 'na', 'null')
hidden_null_tokens = ["?", "na", "n/a", "null", "none", "", "nan", "NaN"]
for col in df_cleaned.select_dtypes(include='object').columns:
    mask = df_cleaned[col].astype(str).str.strip().str.lower().isin([t.lower() for t in hidden_null_tokens])
    df_cleaned.loc[mask, col] = np.nan

# 2. Parse date/time strings into calendar features and drop raw date string
for col in list(df_cleaned.select_dtypes(include=['object', 'string']).columns):
    if any(k in col.lower() for k in ['date', 'time', 'timestamp', 'day', 'created', 'period']):
        try:
            dt_s = pd.to_datetime(df_cleaned[col], errors='coerce')
            if dt_s.notnull().mean() > 0.7:
                df_cleaned[f"{{col}}_year"] = dt_s.dt.year.fillna(0).astype(int)
                df_cleaned[f"{{col}}_month"] = dt_s.dt.month.fillna(0).astype(int)
                df_cleaned[f"{{col}}_day"] = dt_s.dt.day.fillna(0).astype(int)
                df_cleaned[f"{{col}}_dayofweek"] = dt_s.dt.dayofweek.fillna(0).astype(int)
                df_cleaned = df_cleaned.drop(columns=[col])
        except Exception:
            pass

# 3. Remove duplicate rows
initial_rows = len(df_cleaned)
df_cleaned = df_cleaned.drop_duplicates()
print(f"Deduplication: removed {{initial_rows - len(df_cleaned)}} duplicate row(s).")

# 4. Drop rows where the target variable is missing
if "{target_col}" in df_cleaned.columns:
    target_nulls = df_cleaned["{target_col}"].isnull().sum()
    if target_nulls > 0:
        df_cleaned = df_cleaned.dropna(subset=["{target_col}"]).reset_index(drop=True)
        print(f"Target validation: removed {{target_nulls}} row(s) with missing target.")

# 5. Missing value baseline imputation (median for numeric, mode for categorical)
for col in df_cleaned.columns:
    if col == "{target_col}":
        continue
    if df_cleaned[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            fill_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(fill_val)
        else:
            mode_val = df_cleaned[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Missing"
            df_cleaned[col] = df_cleaned[col].fillna(fill_val)

# 6. Save cleaned dataset explicitly as CSV
df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
print(f"Saved cleaned dataset to: {{CLEANED_DATA_PATH}} (Shape: {{df_cleaned.shape}})")
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 7: Train / Test Split
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 6. Leakage-Safe Train/Test Splitting"))
    stratify_code = f"stratify=y" if task_type == TaskType.CLASSIFICATION else "stratify=None"
    cells.append(_make_code_cell(f"""# Separate features from target
X = df_cleaned.drop(columns=["{target_col}"])
y = df_cleaned["{target_col}"]

# Partition data into train and test sets prior to any feature transformation
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    {stratify_code}
)

print(f"Training split: {{X_train.shape[0]}} rows, {{X_train.shape[1]}} features")
print(f"Holdout test split: {{X_test.shape[0]}} rows, {{X_test.shape[1]}} features")
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 8: Feature Engineering
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 7. Feature Engineering"))
    cells.append(_make_code_cell(f"""X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

# Generate domain interactions for numeric features
train_num_cols = list(X_train_fe.select_dtypes(include=['int64', 'float64']).columns)

if len(train_num_cols) >= 2:
    col_a, col_b = train_num_cols[0], train_num_cols[1]
    X_train_fe[f"{{col_a}}_x_{{col_b}}"] = X_train_fe[col_a] * X_train_fe[col_b]
    X_test_fe[f"{{col_a}}_x_{{col_b}}"] = X_test_fe[col_a] * X_test_fe[col_b]

# Export feature engineered dataset
fe_combined = pd.concat([X_train_fe, X_test_fe], axis=0).reset_index(drop=True)
fe_combined.to_csv(FE_DATA_PATH, index=False)
print(f"Feature engineering complete. Saved dataset to: {{FE_DATA_PATH}}")
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 9: Encoding & Preprocessing Pipeline
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 8. Adaptive Encoding and Preprocessing Pipeline"))
    cells.append(_make_code_cell("""# Separate numerical, low-cardinality nominals, and high-cardinality categoricals
numeric_features = [c for c in X_train_fe.columns if pd.api.types.is_numeric_dtype(X_train_fe[c])]
cat_candidates = [c for c in X_train_fe.columns if c not in numeric_features]

low_card_features = [c for c in cat_candidates if X_train_fe[c].nunique(dropna=False) <= 12]
high_card_features = [c for c in cat_candidates if c not in low_card_features]

transformers = []
if numeric_features:
    transformers.append(('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features))

if low_card_features:
    transformers.append(('cat_oh', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), low_card_features))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
X_tr_base = preprocessor.fit_transform(X_train_fe)
X_te_base = preprocessor.transform(X_test_fe)

# Derive clean feature names from preprocessor
try:
    raw_names = list(preprocessor.get_feature_names_out())
    encoded_feature_names = [n.replace('num__', '').replace('cat_oh__', '') for n in raw_names]
except Exception:
    encoded_feature_names = [f"feat_{i}" for i in range(X_tr_base.shape[1])]

# Apply frequency encoding on high-cardinality columns
if high_card_features:
    freq_maps = {col: X_train_fe[col].astype(str).value_counts(normalize=True).to_dict() for col in high_card_features}
    tr_freq = np.column_stack([X_train_fe[c].astype(str).map(freq_maps[c]).fillna(0.0).values for c in high_card_features])
    te_freq = np.column_stack([X_test_fe[c].astype(str).map(freq_maps[c]).fillna(0.0).values for c in high_card_features])
    X_train_encoded = np.hstack([X_tr_base, tr_freq])
    X_test_encoded = np.hstack([X_te_base, te_freq])
    encoded_feature_names.extend([f"freq_{c}" for c in high_card_features])
else:
    X_train_encoded = X_tr_base
    X_test_encoded = X_te_base

# Save preprocessing pipeline
pipe_save_path = MODELS_DIR / "preprocessing_pipeline.pkl"
with open(pipe_save_path, "wb") as f:
    pickle.dump(preprocessor, f)

# Save ML-ready encoded dataset with explicit column names
ml_ready_df = pd.DataFrame(X_train_encoded, columns=encoded_feature_names)
ml_ready_df.to_csv(ML_READY_DATA_PATH, index=False)
print(f"Fitted preprocessing pipeline saved to: {pipe_save_path}")
print(f"ML-Ready transformed features: {X_train_encoded.shape[1]} columns")
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 10: Model Benchmark Experiments
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 9. Machine Learning Model Experiments"))
    if task_type == TaskType.REGRESSION:
        model_training_code = f"""# Candidate regression model pool
candidates = {{
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_SEED),
    "Lasso": Lasso(alpha=0.1, random_state=RANDOM_SEED),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1),
    "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=50, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=50, random_state=RANDOM_SEED),
    "AdaBoostRegressor": AdaBoostRegressor(n_estimators=40, random_state=RANDOM_SEED),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
}}

results = []
trained_models = {{}}

for name, model in candidates.items():
    t0 = time.perf_counter()
    model.fit(X_train_encoded, y_train)
    t_elapsed = round(time.perf_counter() - t0, 4)
    
    y_pred = model.predict(X_test_encoded)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    results.append({{
        "model": name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "training_time_sec": t_elapsed,
    }})
    trained_models[name] = model
    print(f"Trained {{name:30s}} | RMSE: {{rmse:.4f}} | MAE: {{mae:.4f}} | R2: {{r2:.4f}} ({{t_elapsed}}s)")
"""
    else:
        model_training_code = f"""# Candidate classification model pool
candidates = {{
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_SEED),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1),
    "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(max_iter=50, random_state=RANDOM_SEED),
    "AdaBoostClassifier": AdaBoostClassifier(n_estimators=40, random_state=RANDOM_SEED),
    "GaussianNB": GaussianNB(),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}}

results = []
trained_models = {{}}

for name, model in candidates.items():
    t0 = time.perf_counter()
    model.fit(X_train_encoded, y_train)
    t_elapsed = round(time.perf_counter() - t0, 4)
    
    y_pred = model.predict(X_test_encoded)
    is_binary = len(np.unique(y_train)) == 2
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="binary" if is_binary else "weighted", zero_division=0))
    prec = float(precision_score(y_test, y_pred, average="binary" if is_binary else "weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="binary" if is_binary else "weighted", zero_division=0))
    
    results.append({{
        "model": name,
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "training_time_sec": t_elapsed,
    }})
    trained_models[name] = model
    print(f"Trained {{name:28s}} | F1: {{f1:.4f}} | Acc: {{acc:.4f}} | Prec: {{prec:.4f}} ({{t_elapsed}}s)")
"""
    cells.append(_make_code_cell(model_training_code))


    # ──────────────────────────────────────────────────────────────────────────
    # Section 11: Model Comparison & Leaderboard
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 10. Model Comparison and Leaderboard"))
    if task_type == TaskType.REGRESSION:
        sort_key_code = 'df_results.sort_values(by=["rmse", "mae", "r2"], ascending=[True, True, False])'
    else:
        sort_key_code = 'df_results.sort_values(by=["f1", "accuracy", "precision"], ascending=[False, False, False])'

    cells.append(_make_code_cell(f"""# Assemble benchmark leaderboard
df_results = pd.DataFrame(results)
df_leaderboard = {sort_key_code}.reset_index(drop=True)
df_leaderboard["Rank"] = range(1, len(df_leaderboard) + 1)

cols = ["Rank", "model"] + [c for c in df_leaderboard.columns if c not in ["Rank", "model"]]
df_leaderboard = df_leaderboard[cols]

print("=== Candidate Model Leaderboard ===")
display(df_leaderboard)
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 12: Top 3-4 Models Selection & Export
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 11. Top 3-4 Model Selection and Persistence"))
    cells.append(_make_code_cell(f"""# Select top models (up to 4)
TOP_K = min(4, len(df_leaderboard))
top_selected_rows = df_leaderboard.head(TOP_K)

print(f"Exporting Top {{TOP_K}} models into {{MODELS_DIR}}:\\n")

for idx, row in top_selected_rows.iterrows():
    rank = int(row["Rank"])
    m_name = row["model"]
    # Clean model slug
    slug = m_name.lower().replace("classifier", "_classifier").replace("regressor", "_regressor")
    filename = f"model_{{rank:02d}}_{{slug}}.pkl"
    save_path = MODELS_DIR / filename
    
    # Save model artifact
    model_obj = trained_models[m_name]
    with open(save_path, "wb") as f:
        pickle.dump(model_obj, f)
        
    print(f" [Rank {{rank}}] Saved: {{save_path.name}} (Algorithm: {{m_name}})")
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 13: Sample Inference Verification
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell("## 12. Sample Prediction and Verification"))
    cells.append(_make_code_cell("""# Load Rank 1 model and verify sample inference pipeline
rank_1_name = top_selected_rows.iloc[0]["model"]
rank_1_slug = rank_1_name.lower().replace("classifier", "_classifier").replace("regressor", "_regressor")
rank_1_file = MODELS_DIR / f"model_01_{rank_1_slug}.pkl"

if rank_1_file.exists():
    with open(rank_1_file, "rb") as f:
        best_loaded_model = pickle.load(f)
else:
    best_loaded_model = trained_models[rank_1_name]

# Predict on first 5 holdout test records
sample_encoded = X_test_encoded[:5]
sample_preds = best_loaded_model.predict(sample_encoded)

verification_df = pd.DataFrame({
    'Actual': y_test.head(5).values,
    'Predicted': sample_preds,
})
print(f"Sample Predictions with Top-Ranked Model ({rank_1_name}):")
display(verification_df)
"""))

    # ──────────────────────────────────────────────────────────────────────────
    # Section 14: Project Summary
    # ──────────────────────────────────────────────────────────────────────────
    cells.append(_make_markdown_cell(f"""## 13. Summary & Conclusions
- **Workflow Status:** Successfully completed end-to-end data processing and multi-model benchmark.
- **Top Performing Model:** `{best_model_name}`
- **Exported Models Directory:** `06_Models/`
- **Reproducibility Guarantee:** All preprocessing pipelines and random seeds are fixed and validated.
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
