"""Train and evaluate the regression models (next-day close prediction).

For each model the chronological pipeline is:
    fit on train -> select by validation RMSE -> report honest test metrics.
A persistence baseline ("tomorrow = today") is always reported alongside.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    ARTIFACTS_DIR,
    MODEL_NAME_SLUGS,
    RANDOM_SEED,
    REGRESSION_ARTIFACT_TEMPLATE,
    REGRESSION_METRICS_PATH,
    ensure_directories,
    get_logger,
)
from src.btc_prediction.evaluation.metrics import regression_metrics
from src.btc_prediction.features.build_features import (
    FEATURE_COLUMNS,
    TARGET_REGRESSION,
)
from src.btc_prediction.models.regression import get_regression_models
from src.btc_prediction.training.splits import chronological_split

logger = get_logger(__name__)


def train_regression_models(
    dataset: pd.DataFrame,
    artifacts_dir: Path = ARTIFACTS_DIR,
    seed: int = RANDOM_SEED,
) -> dict:
    """Train every regression model; save artifacts and metrics.

    Returns
    -------
    dict
        ``{model_name: {"validation": metrics, "test": metrics}}`` plus a
        ``"Persistence Baseline"`` entry computed without training.
    """
    ensure_directories()
    artifacts_dir = Path(artifacts_dir)

    splits = chronological_split(dataset)
    X_train, y_train, X_val, y_val, X_test, y_test = (
        splits.train[FEATURE_COLUMNS], splits.train[TARGET_REGRESSION],
        splits.validation[FEATURE_COLUMNS], splits.validation[TARGET_REGRESSION],
        splits.test[FEATURE_COLUMNS], splits.test[TARGET_REGRESSION],
    )

    results: dict = {}

    # --- persistence baseline: predict tomorrow = today -----------------
    baseline_val = splits.validation["Close"].to_numpy()
    baseline_test = splits.test["Close"].to_numpy()
    results["Persistence Baseline"] = {
        "validation": regression_metrics(y_val.to_numpy(), baseline_val),
        "test": regression_metrics(y_test.to_numpy(), baseline_test),
    }

    # --- fitted models ---------------------------------------------------
    models = get_regression_models(seed)
    val_rmse = {}
    for name, model in models.items():
        logger.info("Training regression model: %s", name)
        try:
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)
        except Exception as exc:
            logger.error("Regression model %s failed to train: %s", name, exc)
            continue
        results[name] = {
            "validation": regression_metrics(y_val.to_numpy(), pred_val),
            "test": regression_metrics(y_test.to_numpy(), pred_test),
        }
        val_rmse[name] = results[name]["validation"]["RMSE"]

        slug = MODEL_NAME_SLUGS["regression"][name]
        joblib.dump(model, artifacts_dir / REGRESSION_ARTIFACT_TEMPLATE.format(slug=slug), compress=3)
        logger.info("Saved %s artifact.", name)

    if not val_rmse:
        raise RuntimeError(
            "No regression model could be trained. Check the dataset and logs."
        )

    best = min(val_rmse, key=val_rmse.get)
    results["_best_model"] = best
    results["_split_summary"] = splits.summary

    with open(artifacts_dir / "regression_metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Regression training done. Best on validation: %s", best)
    return results


def load_regression_model(name: str, artifacts_dir: Path = ARTIFACTS_DIR):
    """Load a fitted regression model by display name (e.g. 'XGBoost')."""
    slug = MODEL_NAME_SLUGS["regression"][name]
    path = Path(artifacts_dir) / REGRESSION_ARTIFACT_TEMPLATE.format(slug=slug)
    if not path.exists():
        raise FileNotFoundError(
            f"Regression artifact '{path.name}' not found in {artifacts_dir}. "
            "Train the models first: python scripts/train_models.py"
        )
    return joblib.load(path)


def available_regression_models(artifacts_dir: Path = ARTIFACTS_DIR):
    """Names of regression models that have a saved artifact."""
    artifacts_dir = Path(artifacts_dir)
    available = []
    for name, slug in MODEL_NAME_SLUGS["regression"].items():
        if (artifacts_dir / REGRESSION_ARTIFACT_TEMPLATE.format(slug=slug)).exists():
            available.append(name)
    return available
