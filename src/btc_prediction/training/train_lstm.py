"""Train the LSTM model and persist its artifacts (model, scalers, metrics).

Scaling discipline: the feature MinMaxScaler and the target scaler are fitted
on the training block only, then applied to validation/test — the same
anti-leakage rule as the classical ML models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.settings import (
    ARTIFACTS_DIR,
    LSTM_FEATURE_SCALER_PATH,
    LSTM_LOOKBACK,
    LSTM_METRICS_PATH,
    LSTM_MODEL_PATH,
    LSTM_TARGET_SCALER_PATH,
    RANDOM_SEED,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    ensure_directories,
    get_logger,
)
from src.btc_prediction.evaluation.metrics import regression_metrics
from src.btc_prediction.features.build_features import FEATURE_COLUMNS
from src.btc_prediction.models.lstm import (
    LstmUnavailableError,
    make_sequences,
    train_lstm,
)

logger = get_logger(__name__)


def train_and_save_lstm(
    dataset: pd.DataFrame,
    artifacts_dir: Path = ARTIFACTS_DIR,
    seed: int = RANDOM_SEED,
    lookback: int = LSTM_LOOKBACK,
    epochs: int = None,
    patience: int = None,
) -> dict:
    """End-to-end LSTM training: scale, build sequences, fit, evaluate, save.

    Returns a dict with test metrics (MAE/MSE/RMSE/R2) and training info.

    Raises
    ------
    LstmUnavailableError
        If TensorFlow is missing or training fails — callers should catch
        this and continue; the app runs fine without the LSTM.
    """
    ensure_directories()
    artifacts_dir = Path(artifacts_dir)

    n = len(dataset)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
    if n <= lookback + 10:
        raise LstmUnavailableError(
            f"Insufficient data for LSTM training: {n} rows is too few for a "
            f"lookback of {lookback}. Skipping the LSTM gracefully."
        )

    feature_values = dataset[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    close_values = dataset["Close"].to_numpy(dtype=np.float64).reshape(-1, 1)
    # Target = next-day close; the final row has none, so shift target column.
    next_close = np.roll(close_values, -1, axis=0)
    next_close[-1] = np.nan
    valid = ~np.isnan(next_close[:, 0])
    feature_values, next_close = feature_values[valid], next_close[valid]

    # Indices after dropping the final target-less row.
    train_end_idx = int(len(next_close) * TRAIN_FRACTION)
    val_end_idx = int(len(next_close) * (TRAIN_FRACTION + VALIDATION_FRACTION))

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_values[:train_end_idx])
    features_scaled = feature_scaler.transform(feature_values)

    target_scaler = MinMaxScaler()
    target_scaler.fit(next_close[:train_end_idx])
    target_scaled = target_scaler.transform(next_close).ravel()

    train_end_idx = min(train_end_idx, len(target_scaled))
    val_end_idx = min(val_end_idx, len(target_scaled))

    model, history = train_lstm(
        features_scaled[:train_end_idx],
        target_scaled[:train_end_idx],
        features_scaled[:val_end_idx],
        target_scaled[:val_end_idx],
        lookback=lookback,
        seed=seed,
        **({"epochs": epochs} if epochs is not None else {}),
        **({"patience": patience} if patience is not None else {}),
    )
    logger.info("LSTM trained for %d epochs (early stopping).", history["epochs_run"])

    # --- honest test evaluation ------------------------------------------
    X_test, y_test_scaled = make_sequences(
        features_scaled[: len(target_scaled)],
        target_scaled,
        lookback,
    )
    # Keep only sequences whose END row belongs to the test block.
    test_start = val_end_idx  # first test index in the original row space
    seq_end_indices = np.arange(lookback, len(target_scaled))
    test_mask = seq_end_indices >= test_start
    X_test = X_test[test_mask]
    y_test_scaled = y_test_scaled[test_mask]

    pred_scaled = model.predict(X_test, verbose=0).ravel()
    y_test = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    y_pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    metrics = regression_metrics(y_test, y_pred)
    metrics["n_test_sequences"] = int(len(y_test))

    model_path = Path(artifacts_dir) / "lstm_model.keras"
    model.save(model_path)
    joblib.dump(feature_scaler, Path(artifacts_dir) / "lstm_feature_scaler.joblib")
    joblib.dump(target_scaler, Path(artifacts_dir) / "lstm_target_scaler.joblib")

    payload = {
        "test": metrics,
        "training": {
            "epochs_run": history["epochs_run"],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "lookback": lookback,
            "feature_columns": FEATURE_COLUMNS,
            "n_rows_total": int(len(dataset)),
            "train_rows": train_end_idx,
            "val_rows": val_end_idx - train_end_idx,
            "test_rows": len(dataset) - val_end_idx,
        },
        "history": history,
    }
    with open(artifacts_dir / "lstm_metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("LSTM test metrics: %s", metrics)
    return payload


def load_lstm(artifacts_dir: Path = ARTIFACTS_DIR):
    """Load the saved LSTM model plus its scalers.

    Returns ``(model, feature_scaler, target_scaler)``.
    Raises FileNotFoundError with an actionable message when absent.
    """
    artifacts_dir = Path(artifacts_dir)
    if not (artifacts_dir / "lstm_model.keras").exists():
        raise FileNotFoundError(
            "LSTM artifact 'lstm_model.keras' not found. Train it with "
            "`python scripts/train_models.py` (requires TensorFlow). The "
            "dashboard works without the LSTM: pick a classical model instead."
        )
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover
        raise LstmUnavailableError(
            "TensorFlow is required to run the saved LSTM model. Install it "
            "with `pip install tensorflow`, or use a classical model."
        ) from exc

    model = keras.models.load_model(artifacts_dir / "lstm_model.keras")
    feature_scaler = joblib.load(artifacts_dir / "lstm_feature_scaler.joblib")
    target_scaler = joblib.load(artifacts_dir / "lstm_target_scaler.joblib")
    return model, feature_scaler, target_scaler


def lstm_artifacts_available() -> bool:
    """True when the LSTM model file exists on disk."""
    return Path(LSTM_MODEL_PATH).exists()
