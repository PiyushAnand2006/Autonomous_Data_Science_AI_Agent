"""
AADS Model Trainer Tool — trains and evaluates baseline, tree-based, and boosting models.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from aads.core.logging import get_logger
from aads.core.schemas import TaskType

logger = get_logger(__name__)

# Optional XGBoost support
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Optional CatBoost support
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Optional LightGBM support
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def get_candidate_models(task_type: TaskType) -> list[str]:
    """Get the standard candidate model list for a given task type."""
    if task_type == TaskType.REGRESSION:
        models = [
            "LinearRegression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "KNeighborsRegressor",
            "DecisionTreeRegressor",
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "AdaBoostRegressor",
            "GradientBoostingRegressor",
            "HistGradientBoostingRegressor",
            "LinearSVR",
        ]
        if HAS_XGBOOST:
            models.append("XGBRegressor")
        if HAS_LIGHTGBM:
            models.append("LGBMRegressor")
        if HAS_CATBOOST:
            models.append("CatBoostRegressor")
        return models
    elif task_type == TaskType.CLUSTERING:
        return ["KMeans", "GaussianMixture", "DBSCAN"]
    else:
        models = [
            "LogisticRegression",
            "KNeighborsClassifier",
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
            "GaussianNB",
            "LinearSVC",
        ]
        if HAS_XGBOOST:
            models.append("XGBClassifier")
        if HAS_LIGHTGBM:
            models.append("LGBMClassifier")
        if HAS_CATBOOST:
            models.append("CatBoostClassifier")
        return models


def _instantiate_model(
    model_name: str,
    task_type: TaskType,
    random_state: int = 42,
    params: Optional[dict] = None,
) -> Any:
    """Instantiate model by name with random seed and optional hyperparameters."""
    p = params or {}
    model_name = model_name.replace(" (AI-Tuned)", "").replace(" (Tuned)", "").strip()

    # ── Regression Models ─────────────────────────────────────────────────────
    if model_name == "LinearRegression":
        return LinearRegression(**p)
    elif model_name == "Ridge":
        return Ridge(random_state=random_state, **p)
    elif model_name == "Lasso":
        return Lasso(alpha=p.get("alpha", 0.1), max_iter=1000, random_state=random_state)
    elif model_name == "ElasticNet":
        return ElasticNet(alpha=p.get("alpha", 0.1), l1_ratio=p.get("l1_ratio", 0.5), max_iter=1000, random_state=random_state)
    elif model_name == "KNeighborsRegressor":
        return KNeighborsRegressor(
            n_neighbors=p.get("n_neighbors", 5),
            weights=p.get("weights", "distance"),
            n_jobs=-1,
        )
    elif model_name == "DecisionTreeRegressor":
        return DecisionTreeRegressor(
            max_depth=p.get("max_depth", 8),
            random_state=random_state,
        )
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 80),
            max_depth=p.get("max_depth", 8),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "ExtraTreesRegressor":
        return ExtraTreesRegressor(
            n_estimators=p.get("n_estimators", 80),
            max_depth=p.get("max_depth", 8),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "AdaBoostRegressor":
        return AdaBoostRegressor(
            n_estimators=p.get("n_estimators", 60),
            random_state=random_state,
        )
    elif model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            n_estimators=p.get("n_estimators", 80),
            learning_rate=p.get("learning_rate", 0.1),
            max_depth=p.get("max_depth", 4),
            subsample=p.get("subsample", 0.8),
            random_state=random_state,
        )
    elif model_name == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(
            max_iter=p.get("n_estimators", 80),
            random_state=random_state,
        )
    elif model_name == "LinearSVR":
        return LinearSVR(
            random_state=random_state,
            max_iter=2000,
            dual="auto",
        )
    elif model_name == "XGBRegressor" and HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=p.get("n_estimators", 80),
            learning_rate=p.get("learning_rate", 0.1),
            max_depth=p.get("max_depth", 4),
            random_state=random_state,
            verbosity=0,
        )
    elif model_name == "LGBMRegressor" and HAS_LIGHTGBM:
        return LGBMRegressor(
            n_estimators=p.get("n_estimators", 80),
            learning_rate=p.get("learning_rate", 0.1),
            max_depth=p.get("max_depth", 4),
            random_state=random_state,
            verbosity=-1,
        )
    elif model_name == "CatBoostRegressor" and HAS_CATBOOST:
        return CatBoostRegressor(
            iterations=p.get("n_estimators", 80),
            learning_rate=p.get("learning_rate", 0.1),
            depth=p.get("max_depth", 4),
            random_seed=random_state,
            verbose=False,
        )

    # ── Classification Models ─────────────────────────────────────────────────
    elif model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=random_state, **p)
    elif model_name == "KNeighborsClassifier":
        return KNeighborsClassifier(
            n_neighbors=p.get("n_neighbors", 5),
            weights=p.get("weights", "distance"),
            n_jobs=-1,
        )
    elif model_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(
            max_depth=p.get("max_depth", 8),
            random_state=random_state,
        )
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 80),
            max_depth=p.get("max_depth", 8),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "ExtraTreesClassifier":
        return ExtraTreesClassifier(
            n_estimators=p.get("n_estimators", 80),
            max_depth=p.get("max_depth", 8),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "AdaBoostClassifier":
        return AdaBoostClassifier(
            n_estimators=p.get("n_estimators", 60),
            random_state=random_state,
        )
    elif model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(
            n_estimators=p.get("n_estimators", 80),
            learning_rate=p.get("learning_rate", 0.1),
            max_depth=p.get("max_depth", 4),
            subsample=p.get("subsample", 0.8),
            random_state=random_state,
        )
    elif model_name == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            max_iter=p.get("n_estimators", 80),
            random_state=random_state,
        )
    elif model_name == "GaussianNB":
        return GaussianNB()
    elif model_name == "LinearSVC":
        return LinearSVC(
            random_state=random_state,
            max_iter=2000,
            dual="auto",
        )
    elif model_name == "XGBClassifier" and HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=p.get("n_estimators", 80),
            learning_rate=p.get("learning_rate", 0.1),
            max_depth=p.get("max_depth", 4),
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
    elif model_name == "LGBMClassifier" and HAS_LIGHTGBM:
        return LGBMClassifier(
            n_estimators=p.get("n_estimators", 80),
            learning_rate=p.get("learning_rate", 0.1),
            max_depth=p.get("max_depth", 4),
            random_state=random_state,
            verbosity=-1,
        )
    elif model_name == "CatBoostClassifier" and HAS_CATBOOST:
        return CatBoostClassifier(
            iterations=p.get("n_estimators", 100),
            learning_rate=p.get("learning_rate", 0.1),
            depth=p.get("max_depth", 4),
            random_seed=random_state,
            verbose=False,
        )

    # ── Clustering Models ─────────────────────────────────────────────────────
    elif model_name == "KMeans":
        return KMeans(
            n_clusters=p.get("n_clusters", 4),
            random_state=random_state,
            n_init="auto",
        )
    elif model_name == "GaussianMixture":
        return GaussianMixture(
            n_components=p.get("n_components", 4),
            random_state=random_state,
        )
    elif model_name == "DBSCAN":
        return DBSCAN(
            eps=p.get("eps", 0.5),
            min_samples=p.get("min_samples", 5),
            n_jobs=-1,
        )
    else:
        # Default fallback
        if task_type == TaskType.REGRESSION:
            return RandomForestRegressor(random_state=random_state, n_jobs=-1)
        elif task_type == TaskType.CLUSTERING:
            return KMeans(n_clusters=4, random_state=random_state, n_init="auto")
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)


def get_memory_budget_samples() -> int:
    """Determine safe training sample size dynamically based on available system RAM.

    - Low Memory / Free Tier Containers (<= 1.2 GB RAM): 25,000 samples (prevents Render 512MB OOM)
    - Medium Memory Instances (1.2 GB - 4 GB RAM): 100,000 samples
    - High Memory Local / Cloud Workstations (> 4 GB RAM): 1,000,000 samples (train on 100% of rows)
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_ram_mb = vm.total / (1024 * 1024)
        avail_ram_mb = vm.available / (1024 * 1024)
        if total_ram_mb <= 1200 or avail_ram_mb < 600:
            return 25000
        elif total_ram_mb <= 4096 or avail_ram_mb < 2000:
            return 100000
        else:
            return 1000000
    except Exception:
        return 100000


def train_and_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: TaskType = TaskType.CLASSIFICATION,
    random_state: int = 42,
    params: Optional[dict] = None,
) -> tuple[Any, dict[str, float], np.ndarray, float]:
    """Train a model and compute holdout validation metrics and training duration."""
    model = _instantiate_model(model_name, task_type, random_state=random_state, params=params)

    # Dynamic RAM Auto-Scaling: Keep Render 512MB safe while unlocking full dataset on local RAM
    max_samples = get_memory_budget_samples()
    if len(X_train) > max_samples:
        if isinstance(X_train, pd.DataFrame):
            if y_train is not None and task_type == TaskType.CLASSIFICATION and getattr(y_train, "nunique", lambda: 0)() > 1:
                try:
                    from sklearn.model_selection import train_test_split
                    _, X_tr, _, y_tr = train_test_split(
                        X_train, y_train, test_size=max_samples, stratify=y_train, random_state=random_state
                    )
                except Exception:
                    sample_idx = X_train.sample(n=max_samples, random_state=random_state).index
                    X_tr = X_train.loc[sample_idx]
                    y_tr = y_train.loc[sample_idx] if y_train is not None else None
            else:
                sample_idx = X_train.sample(n=max_samples, random_state=random_state).index
                X_tr = X_train.loc[sample_idx]
                y_tr = y_train.loc[sample_idx] if y_train is not None else None
        else:
            indices = np.random.RandomState(random_state).choice(len(X_train), max_samples, replace=False)
            X_tr = X_train[indices]
            y_tr = y_train[indices] if y_train is not None else None
    else:
        X_tr = X_train
        y_tr = y_train

    start_time = time.perf_counter()
    if task_type == TaskType.CLUSTERING:
        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X_tr)
        else:
            model.fit(X_tr)
            labels = model.predict(X_tr)
        training_time = round(time.perf_counter() - start_time, 4)

        if hasattr(model, "predict"):
            y_pred = model.predict(X_val)
        else:
            y_pred = labels[:len(X_val)]

        unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]
        if len(unique_labels) > 1 and len(unique_labels) < len(labels):
            sample_size = min(2000, len(X_tr))
            idx_sample = np.random.RandomState(random_state).choice(len(X_tr), sample_size, replace=False)
            X_sub = X_tr.iloc[idx_sample] if hasattr(X_tr, "iloc") else X_tr[idx_sample]
            lbl_sub = labels[idx_sample]
            valid_mask = lbl_sub != -1
            if np.sum(valid_mask) > 10 and len(np.unique(lbl_sub[valid_mask])) > 1:
                sil = float(silhouette_score(X_sub[valid_mask], lbl_sub[valid_mask]))
                db = float(davies_bouldin_score(X_sub[valid_mask], lbl_sub[valid_mask]))
                ch = float(calinski_harabasz_score(X_sub[valid_mask], lbl_sub[valid_mask]))
            else:
                sil, db, ch = 0.0, 999.0, 0.0
            metrics = {
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4),
                "calinski_harabasz": round(ch, 2),
                "n_clusters": int(len(unique_labels)),
            }
        else:
            metrics = {"silhouette": 0.0, "davies_bouldin": 999.0, "n_clusters": 1}

    else:
        # If classification with string/categorical target, encode target to 0..K-1 integers
        le = None
        if task_type == TaskType.CLASSIFICATION and y_tr is not None:
            is_str_target = (
                pd.api.types.is_object_dtype(y_tr)
                or pd.api.types.is_string_dtype(y_tr)
                or pd.api.types.is_categorical_dtype(y_tr)
                or model_name in ["XGBClassifier", "LGBMClassifier"]
            )
            if is_str_target:
                le = LabelEncoder()
                y_tr_fit = le.fit_transform(y_tr)
                if y_val is not None and len(y_val) > 0:
                    try:
                        y_val_fit = le.transform(y_val)
                    except ValueError:
                        # Fallback for rare unseen classes in holdout set
                        all_classes = list(pd.Series(list(y_tr) + list(y_val)).dropna().unique())
                        le.fit(all_classes)
                        y_tr_fit = le.transform(y_tr)
                        y_val_fit = le.transform(y_val)
                else:
                    y_val_fit = y_val
            else:
                y_tr_fit = y_tr
                y_val_fit = y_val
        else:
            y_tr_fit = y_tr
            y_val_fit = y_val

        # Guard against n_neighbors > n_samples_fit in KNeighbors models
        if hasattr(model, "n_neighbors") and len(X_tr) > 0:
            model.n_neighbors = min(model.n_neighbors, max(1, len(X_tr)))

        model.fit(X_tr, y_tr_fit)
        training_time = round(time.perf_counter() - start_time, 4)
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
            # Evaluate using y_val_fit for numerical consistency
            eval_target = y_val_fit if le is not None else y_val
            eval_pred = y_pred

            acc = float(accuracy_score(eval_target, eval_pred))
            is_binary = len(np.unique(eval_target)) == 2
            f1 = float(
                f1_score(
                    eval_target,
                    eval_pred,
                    average="binary" if is_binary else "weighted",
                    zero_division=0,
                )
            )
            prec = float(
                precision_score(
                    eval_target,
                    eval_pred,
                    average="binary" if is_binary else "weighted",
                    zero_division=0,
                )
            )
            rec = float(
                recall_score(
                    eval_target,
                    eval_pred,
                    average="binary" if is_binary else "weighted",
                    zero_division=0,
                )
            )

            metrics = {
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }

            if is_binary and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_val)[:, 1]
                    metrics["roc_auc"] = round(float(roc_auc_score(eval_target, y_prob)), 4)
                except Exception:
                    pass

        # If LabelEncoder was used, inverse transform predictions back to original class strings
        if le is not None and hasattr(le, "inverse_transform"):
            try:
                y_pred = le.inverse_transform(y_pred)
            except Exception:
                pass

    logger.info(
        "model_evaluation_completed",
        model=model_name,
        metrics=metrics,
        training_time=training_time,
    )
    return model, metrics, y_pred, training_time


def rank_models(
    results: list[dict[str, Any]],
    task_type: TaskType = TaskType.CLASSIFICATION,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Multi-metric ranking for candidate models."""
    if not results:
        return []

    if task_type == TaskType.REGRESSION:
        # Multi-metric sort for regression: RMSE (asc), MAE (asc), -R2 (desc)
        sorted_results = sorted(
            results,
            key=lambda r: (
                r["metrics"].get("rmse", float("inf")),
                r["metrics"].get("mae", float("inf")),
                -r["metrics"].get("r2", -float("inf")),
                r.get("training_time", 0.0),
            ),
        )
    elif task_type == TaskType.CLUSTERING:
        # Multi-metric sort for clustering: Silhouette (desc), Davies-Bouldin (asc)
        sorted_results = sorted(
            results,
            key=lambda r: (
                -r["metrics"].get("silhouette", -float("inf")),
                r["metrics"].get("davies_bouldin", float("inf")),
                r.get("training_time", 0.0),
            ),
        )
    else:
        # Multi-metric sort for classification: F1 (desc), ROC-AUC (desc), Precision (desc), Accuracy (desc)
        sorted_results = sorted(
            results,
            key=lambda r: (
                -r["metrics"].get("f1", -float("inf")),
                -r["metrics"].get("roc_auc", -float("inf")),
                -r["metrics"].get("precision", -float("inf")),
                -r["metrics"].get("accuracy", -float("inf")),
                r.get("training_time", 0.0),
            ),
        )

    top_n = min(len(sorted_results), max(1, top_k))
    ranked_top = []

    for rank_idx in range(top_n):
        item = dict(sorted_results[rank_idx])
        rank = rank_idx + 1
        model_name = item["model_name"]
        metrics = item["metrics"]

        if task_type == TaskType.REGRESSION:
            primary = f"RMSE={metrics.get('rmse', 'N/A')}, R²={metrics.get('r2', 'N/A')}"
        elif task_type == TaskType.CLUSTERING:
            primary = f"Silhouette={metrics.get('silhouette', 'N/A')}, Clusters={metrics.get('n_clusters', 'N/A')}"
        else:
            primary = f"F1={metrics.get('f1', 'N/A')}, Accuracy={metrics.get('accuracy', 'N/A')}"

        if rank == 1:
            reason = f"Rank 1: Best overall validation performance ({primary}) with stable generalization."
        elif rank == 2:
            reason = f"Rank 2: Strong runner-up candidate ({primary}) with fast training convergence."
        elif rank == 3:
            reason = f"Rank 3: Competitive baseline ensemble ({primary}) offering architectural diversity."
        else:
            reason = f"Rank {rank}: Solid alternative model architecture ({primary})."

        item["rank"] = rank
        item["selection_reason"] = reason
        ranked_top.append(item)

    return ranked_top

