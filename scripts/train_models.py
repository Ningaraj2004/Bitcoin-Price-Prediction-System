#!/usr/bin/env python3
"""Step 3 — Train all models (regression, classification, LSTM) and save
artifacts + genuine evaluation reports.

Usage:
    python scripts/train_models.py                 # everything
    python scripts/train_models.py --skip-lstm     # classical ML only (fast)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

from config.settings import (  # noqa: E402
    FEATURE_COLUMNS_PATH,
    TRAINING_METADATA_PATH,
    ensure_directories,
    get_logger,
)
from src.btc_prediction.data import collector  # noqa: E402
from src.btc_prediction.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    regression_metrics,
)
from src.btc_prediction.evaluation.reporting import generate_all_reports  # noqa: E402
from src.btc_prediction.features.build_features import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    create_dataset,
)
from src.btc_prediction.models.lstm import LstmUnavailableError  # noqa: E402
from src.btc_prediction.training.splits import chronological_split  # noqa: E402
from src.btc_prediction.training.train_classification import (  # noqa: E402
    train_classification_models,
)
from src.btc_prediction.training.train_lstm import train_and_save_lstm  # noqa: E402
from src.btc_prediction.training.train_regression import (  # noqa: E402
    train_regression_models,
)

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all BTC prediction models")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="skip the LSTM (faster; app still works without it)")
    args = parser.parse_args()

    data = collector.load_or_download()
    dataset = create_dataset(data)
    ensure_directories()

    # Persist the canonical feature column order for the prediction module.
    with open(FEATURE_COLUMNS_PATH, "w") as fh:
        json.dump(FEATURE_COLUMNS, fh, indent=2)

    # ------------------------------------------------------------------
    # Classical ML (also collects test predictions for the report figures)
    # ------------------------------------------------------------------
    reg_results = train_regression_models(dataset)
    clf_results = train_classification_models(dataset)

    splits = chronological_split(dataset)
    reg_predictions = {}
    for name in [k for k in reg_results if not k.startswith("_")]:
        if name == "Persistence Baseline":
            continue
        from src.btc_prediction.training.train_regression import load_regression_model

        model = load_regression_model(name)
        reg_predictions[name] = {
            "y_true": splits.test[TARGET_REGRESSION].to_numpy(),
            "y_pred": model.predict(splits.test[FEATURE_COLUMNS].to_numpy()),
        }

    clf_predictions = {}
    from src.btc_prediction.training.train_classification import load_classification_model

    for name in [k for k in clf_results if not k.startswith("_")]:
        if name == "Majority-Class Baseline":
            continue
        model = load_classification_model(name)
        proba = model.predict_proba(splits.test[FEATURE_COLUMNS].to_numpy())[:, 1]
        clf_predictions[name] = {
            "y_true": splits.test[TARGET_CLASSIFICATION].to_numpy().astype(int),
            "y_proba": proba,
            "confusion_matrix": classification_metrics(
                splits.test[TARGET_CLASSIFICATION].to_numpy().astype(int),
                (proba >= 0.5).astype(int), proba,
            )["Confusion_Matrix"],
        }

    # ------------------------------------------------------------------
    # LSTM (optional; the app degrades gracefully without it)
    # ------------------------------------------------------------------
    lstm_payload = None
    if not args.skip_lstm:
        try:
            lstm_payload = train_and_save_lstm(dataset)
        except (LstmUnavailableError, RuntimeError, OSError) as exc:
            logger.warning("LSTM training skipped: %s", exc)
            print(f"⚠️  LSTM skipped: {exc}")
    else:
        print("⏭️  LSTM skipped by --skip-lstm.")

    # ------------------------------------------------------------------
    # Reports, figures and training metadata
    # ------------------------------------------------------------------
    generate_all_reports(
        dataset=dataset,
        predictions={"regression": reg_predictions, "classification": clf_predictions},
    )

    metadata = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_rows": len(dataset),
        "data_range": {
            "start": str(dataset.index.min().date()),
            "end": str(dataset.index.max().date()),
        },
        "splits": splits.summary,
        "feature_columns": FEATURE_COLUMNS,
        "best_regression": reg_results.get("_best_model"),
        "best_classification": clf_results.get("_best_model"),
        "lstm_trained": lstm_payload is not None,
        "regression_test_metrics": {
            name: reg_results[name]["test"]
            for name in reg_results if not name.startswith("_")
        },
        "classification_test_metrics": {
            name: clf_results[name]["test"]
            for name in clf_results if not name.startswith("_")
        },
        "lstm_test_metrics": (lstm_payload or {}).get("test"),
    }
    with open(TRAINING_METADATA_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    print("\n✅ Training complete. Genuine test-set results:")
    for name, m in metadata["regression_test_metrics"].items():
        print(f"   [reg] {name:<20} MAE={m['MAE']:>10,.2f}  RMSE={m['RMSE']:>10,.2f}  R2={m['R2']:>7.4f}")
    if metadata["lstm_test_metrics"]:
        m = metadata["lstm_test_metrics"]
        print(f"   [reg] {'LSTM':<20} MAE={m['MAE']:>10,.2f}  RMSE={m['RMSE']:>10,.2f}  R2={m['R2']:>7.4f}")
    for name, m in metadata["classification_test_metrics"].items():
        auc = m.get("ROC_AUC")
        auc_str = "n/a" if auc is None or auc != auc else f"{auc:.3f}"
        print(f"   [clf] {name:<22} Acc={m['Accuracy']:.3f}  F1={m['F1']:.3f}  AUC={auc_str}")
    print(f"\nArtifacts: {TRAINING_METADATA_PATH.parent}")
    print("Reports:   reports/ (tables + figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
