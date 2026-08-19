"""AADS Machine Learning Tools — model training, cross-validation, and metrics evaluation."""

from aads.tools.ml.trainer import get_candidate_models, train_and_evaluate_model

__all__ = ["train_and_evaluate_model", "get_candidate_models"]
