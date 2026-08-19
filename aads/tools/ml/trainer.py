"""
AADS Model Trainer Tool — trains and evaluates baseline, tree-based, and boosting models.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from aads.core.logging import get_logger
from aads.core.schemas import TaskType

logger = get_logger(__name__)


def get_candidate_models(task_type: TaskType) -> list[str]:
    """Get the standard candidate model list for a given task type."""
    if task_type == TaskType.REGRESSION:
        return ["LinearRegression", "Ridge", "RandomForestRegressor", "GradientBoostingRegressor"]
    else:
        return ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]


def _instantiate_model(model_name: str, task_type: TaskType, random_state: int = 42, params: Optional[dict] = None) -> Any:
    """Instantiate model by name with random seed and optional hyperparameters."""
    p = params or {}

    if model_name == "LinearRegression":
        return LinearRegression(**p)
    elif model_name == "Ridge":
        return Ridge(random_state=random_state, **p)
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 8), random_state=random_state, n_jobs=-1)
    elif model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressor(n_estimators=p.get("n_estimators", 100), learning_rate=p.get("learning_rate", 0.1), max_depth=p.get("max_depth", 4), random_state=random_state)
    elif model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=random_state, **p)
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 8), random_state=random_state, n_jobs=-1)
    elif model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(n_estimators=p.get("n_estimators", 100), learning_rate=p.get("learning_rate", 0.1), max_depth=p.get("max_depth", 4), random_state=random_state)
    else:
        # Default fallback
        if task_type == TaskType.REGRESSION:
            return RandomForestRegressor(random_state=random_state, n_jobs=-1)
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)


def train_and_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: TaskType = TaskType.CLASSIFICATION,
    random_state: int = 42,
    params: Optional[dict] = None,
) -> tuple[Any, dict[str, float], np.ndarray]:
    """Train a model and compute holdout validation metrics.

    Args:
        model_name: Name of model to train.
        X_train: Encoded train features.
        y_train: Train target.
        X_val: Encoded validation features (or test features if no val set).
        y_val: Validation target.
        task_type: Classification or Regression.
        random_state: Reproducibility seed.
        params: Custom hyperparameter overrides.

    Returns:
        Tuple of (fitted_model, metrics_dict, y_pred_array).
    """
    model = _instantiate_model(model_name, task_type, random_state=random_state, params=params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    metrics: dict[str, float] = {}

    if task_type == TaskType.REGRESSION:
        mse = float(mean_squared_error(y_val, y_pred))
        rmse = float(math.sqrt(mse))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))
        metrics = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "mse": round(mse, 4),
        }
    else:
        acc = float(accuracy_score(y_val, y_pred))
        is_binary = len(np.unique(y_val)) == 2
        f1 = float(f1_score(y_val, y_pred, average="binary" if is_binary else "weighted", zero_division=0))
        prec = float(precision_score(y_val, y_pred, average="binary" if is_binary else "weighted", zero_division=0))
        rec = float(recall_score(y_val, y_pred, average="binary" if is_binary else "weighted", zero_division=0))

        metrics = {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        }

        if is_binary and hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_val)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_val, y_prob)), 4)
            except Exception:
                pass

    logger.info("model_evaluation_completed", model=model_name, metrics=metrics)
    return model, metrics, y_pred
