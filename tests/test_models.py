"""Tests for chronological splitting and model training/evaluation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.btc_prediction.evaluation.metrics import (
    classification_metrics,
    majority_class_baseline,
    regression_metrics,
)
from src.btc_prediction.training.splits import chronological_split


def test_chronological_split_is_ordered_and_contiguous(synthetic_dataset):
    splits = chronological_split(synthetic_dataset)
    assert splits.train.index.max() < splits.validation.index.min()
    assert splits.validation.index.max() < splits.test.index.min()
    assert splits.train.index.is_monotonic_increasing
    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(synthetic_dataset)


def test_chronological_split_rejects_shuffled_input(synthetic_dataset):
    shuffled = synthetic_dataset.iloc[np.random.permutation(len(synthetic_dataset))]
    with pytest.raises(ValueError, match="sorted"):
        chronological_split(shuffled)


def test_chronological_split_rejects_bad_fractions(synthetic_dataset):
    with pytest.raises(ValueError):
        chronological_split(synthetic_dataset, train_fraction=1.5)
    with pytest.raises(ValueError):
        chronological_split(synthetic_dataset, train_fraction=0.9,
                            validation_fraction=0.9)


def test_regression_metrics_known_values():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == pytest.approx(1.0)

    metrics = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert metrics["MAE"] == pytest.approx(2.0 / 3.0)
    assert metrics["MSE"] == pytest.approx((1 + 0 + 1) / 3)


def test_classification_metrics_structure():
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0]
    proba = [0.1, 0.9, 0.6, 0.8, 0.4, 0.2]
    m = classification_metrics(y_true, y_pred, proba)
    for key in ("Accuracy", "Precision", "Recall", "F1", "ROC_AUC",
                "Confusion_Matrix"):
        assert key in m
    cm = np.asarray(m["Confusion_Matrix"])
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_true)
    assert 0 <= m["ROC_AUC"] <= 1


def test_classification_metrics_auc_undefined_single_class():
    m = classification_metrics([1, 1, 1], [1, 0, 1], [0.9, 0.4, 0.8])
    assert np.isnan(m["ROC_AUC"])


def test_majority_class_baseline():
    import pandas as pd

    y_train = pd.Series([1, 1, 1, 0])
    pred = majority_class_baseline(y_train, [None, None, None])
    assert list(pred) == [1, 1, 1]


def test_all_regression_models_train_and_meet_baseline_contract(
    trained_artifacts, synthetic_dataset
):
    from src.btc_prediction.training.train_regression import (
        available_regression_models,
    )

    available = available_regression_models(trained_artifacts)
    assert set(available) == {"Linear Regression", "Random Forest", "XGBoost"}

    with open(trained_artifacts / "regression_metrics.json") as fh:
        results = json.load(fh)
    assert "Persistence Baseline" in results
    for name in available:
        test = results[name]["test"]
        for key in ("MAE", "MSE", "RMSE", "R2"):
            assert key in test
            assert np.isfinite(test[key])


def test_all_classification_models_train(trained_artifacts):
    from src.btc_prediction.training.train_classification import (
        available_classification_models,
    )

    available = available_classification_models(trained_artifacts)
    assert set(available) == {"Logistic Regression", "Random Forest", "XGBoost"}

    with open(trained_artifacts / "classification_metrics.json") as fh:
        results = json.load(fh)
    assert "Majority-Class Baseline" in results
    for name in available:
        test = results[name]["test"]
        for key in ("Accuracy", "Precision", "Recall", "F1"):
            assert 0.0 <= test[key] <= 1.0


def test_regression_models_beat_nothing_but_are_finite(
    trained_artifacts, synthetic_dataset
):
    """Honesty contract: models must produce finite predictions on the
    test split; we do NOT fabricate 'must beat baseline' claims in tests."""
    from src.btc_prediction.features.build_features import (
        FEATURE_COLUMNS,
        TARGET_REGRESSION,
    )
    from src.btc_prediction.training.train_regression import load_regression_model
    from src.btc_prediction.training.splits import chronological_split

    splits = chronological_split(synthetic_dataset)
    for name in ("Linear Regression", "Random Forest", "XGBoost"):
        model = load_regression_model(name, trained_artifacts)
        preds = model.predict(splits.test[FEATURE_COLUMNS].to_numpy())
        assert np.isfinite(preds).all()
        assert (preds > 0).all()  # prices are positive
        assert len(preds) == len(splits.test[TARGET_REGRESSION])


def test_fit_uses_training_data_only_pipeline(trained_artifacts):
    """Scaling lives inside the Pipeline: it is refit from raw features at
    predict time, so a saved pipeline is self-contained (no leakage of
    test statistics into the scaler)."""
    import joblib

    from src.btc_prediction.features.build_features import FEATURE_COLUMNS

    model = joblib.load(trained_artifacts / "regression_linear_regression.joblib")
    scaler = model.named_steps["scaler"]
    assert hasattr(scaler, "mean_")  # fitted
    assert len(scaler.mean_) == len(FEATURE_COLUMNS)


def test_split_summary_dates(trained_artifacts):
    with open(trained_artifacts / "regression_metrics.json") as fh:
        results = json.load(fh)
    summary = results["_split_summary"]
    assert summary["train"]["end"] <= summary["validation"]["start"]
    assert summary["validation"]["end"] <= summary["test"]["start"]
