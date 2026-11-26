"""Train and evaluate the UP/DOWN classification models.

Same disciplined pipeline as regression: chronological splits, fit on
train, model selection on validation, honest metrics on test. A
majority-class baseline is always reported for honest comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    ARTIFACTS_DIR,
    CLASSIFICATION_ARTIFACT_TEMPLATE,
    CLASSIFICATION_METRICS_PATH,
    MODEL_NAME_SLUGS,
    RANDOM_SEED,
    ensure_directories,
    get_logger,
)
from src.btc_prediction.evaluation.metrics import (
    classification_metrics,
    majority_class_baseline,
)
from src.btc_prediction.features.build_features import (
    FEATURE_COLUMNS,
    TARGET_CLASSIFICATION,
)
from src.btc_prediction.models.classification import get_classification_models
from src.btc_prediction.training.splits import chronological_split

logger = get_logger(__name__)


def train_classification_models(
    dataset: pd.DataFrame,
    artifacts_dir: Path = ARTIFACTS_DIR,
    seed: int = RANDOM_SEED,
) -> dict:
    """Train every classifier; save artifacts and metrics."""
    ensure_directories()
    artifacts_dir = Path(artifacts_dir)

    splits = chronological_split(dataset)
    X_train, y_train, X_val, y_val, X_test, y_test = (
        splits.train[FEATURE_COLUMNS], splits.train[TARGET_CLASSIFICATION],
        splits.validation[FEATURE_COLUMNS], splits.validation[TARGET_CLASSIFICATION],
        splits.test[FEATURE_COLUMNS], splits.test[TARGET_CLASSIFICATION],
    )

    results: dict = {}

    # --- majority-class baseline ----------------------------------------
    majority_pred_val = majority_class_baseline(y_train, y_val)
    majority_pred_test = majority_class_baseline(y_train, y_test)
    results["Majority-Class Baseline"] = {
        "validation": classification_metrics(y_val, majority_pred_val),
        "test": classification_metrics(y_test, majority_pred_test),
    }

    # --- fitted models ---------------------------------------------------
    models = get_classification_models(seed)
    val_f1 = {}
    for name, model in models.items():
        logger.info("Training classification model: %s", name)
        try:
            model.fit(X_train, y_train)
            proba_val = model.predict_proba(X_val)[:, 1]
            proba_test = model.predict_proba(X_test)[:, 1]
            pred_val = (proba_val >= 0.5).astype(int)
            pred_test = (proba_test >= 0.5).astype(int)
        except Exception as exc:
            logger.error("Classification model %s failed to train: %s", name, exc)
            continue
        results[name] = {
            "validation": classification_metrics(y_val, pred_val, proba_val),
            "test": classification_metrics(y_test, pred_test, proba_test),
        }
        val_f1[name] = results[name]["validation"]["F1"]

        slug = MODEL_NAME_SLUGS["classification"][name]
        joblib.dump(model, artifacts_dir / CLASSIFICATION_ARTIFACT_TEMPLATE.format(slug=slug), compress=3)
        logger.info("Saved %s artifact.", name)

    if not val_f1:
        raise RuntimeError(
            "No classification model could be trained. Check the dataset and logs."
        )

    best = max(val_f1, key=val_f1.get)
    results["_best_model"] = best
    results["_split_summary"] = splits.summary
    results["_train_class_balance"] = {
        "UP_share": float(np.mean(y_train)),
        "n_train": int(len(y_train)),
    }

    with open(artifacts_dir / "classification_metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Classification training done. Best on validation: %s", best)
    return results


def load_classification_model(name: str, artifacts_dir: Path = ARTIFACTS_DIR):
    """Load a fitted classification model by display name."""
    slug = MODEL_NAME_SLUGS["classification"][name]
    path = Path(artifacts_dir) / CLASSIFICATION_ARTIFACT_TEMPLATE.format(slug=slug)
    if not path.exists():
        raise FileNotFoundError(
            f"Classification artifact '{path.name}' not found in {artifacts_dir}. "
            "Train the models first: python scripts/train_models.py"
        )
    return joblib.load(path)


def available_classification_models(artifacts_dir: Path = ARTIFACTS_DIR):
    """Names of classification models that have a saved artifact."""
    artifacts_dir = Path(artifacts_dir)
    return [
        name
        for name, slug in MODEL_NAME_SLUGS["classification"].items()
        if (artifacts_dir / CLASSIFICATION_ARTIFACT_TEMPLATE.format(slug=slug)).exists()
    ]
