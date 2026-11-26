"""Metric computation for regression and classification.

All metrics are computed from real predictions only — nothing is invented
or hard-coded. Baselines are included so model quality can be judged
honestly (a model that cannot beat a naive baseline is not useful).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, MSE, RMSE and R2 of a regression model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred shapes differ: {y_true.shape} vs {y_pred.shape}"
        )
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def persistence_baseline(last_close: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Naive forecast: tomorrow's close equals today's close."""
    return np.asarray(last_close, dtype=float)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if len(np.unique(y_true)) > 1:
            metrics["ROC_AUC"] = float(roc_auc_score(y_true, y_proba))
        else:
            # ROC-AUC is undefined for a single-class sample; report NaN
            # instead of pretending a value exists.
            metrics["ROC_AUC"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["Confusion_Matrix"] = cm.tolist()
    return metrics


def majority_class_baseline(y_train: pd.Series, y_eval) -> np.ndarray:
    """Always predict the most frequent class seen in the training data."""
    majority = int(pd.Series(y_train).value_counts().idxmax())
    return np.full(len(y_eval), majority, dtype=int)
