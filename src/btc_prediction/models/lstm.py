"""LSTM deep-learning model for next-day Bitcoin close prediction (TensorFlow/Keras)."""

from __future__ import annotations

import numpy as np

from config.settings import (
    LSTM_BATCH_SIZE,
    LSTM_DENSE_UNITS,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_LOOKBACK,
    LSTM_PATIENCE,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    get_logger,
)

logger = get_logger(__name__)


class LstmUnavailableError(RuntimeError):
    """Raised when TensorFlow is missing or the LSTM cannot be trained."""


def _import_tf():
    """Import TensorFlow lazily so the rest of the app works without it."""
    try:
        import tensorflow as tf
        from tensorflow import keras

        return tf, keras
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LstmUnavailableError(
            "TensorFlow is not installed, so the LSTM model is unavailable. "
            "Install it with `pip install tensorflow` (or `tensorflow-cpu`). "
            "All other models keep working without it."
        ) from exc


def build_lstm(n_features: int, lookback: int = LSTM_LOOKBACK):
    """Build the sequential LSTM regressor.

    Architecture: LSTM(64, return_sequences=True) -> Dropout -> LSTM(32)
    -> Dropout -> Dense(16, relu) -> Dense(1) predicting the scaled
    next-day close.
    """
    _, keras = _import_tf()
    tf, _ = _import_tf()

    tf.random.set_seed(42)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(lookback, n_features)),
            keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True),
            keras.layers.Dropout(LSTM_DROPOUT),
            keras.layers.LSTM(LSTM_UNITS_2),
            keras.layers.Dropout(LSTM_DROPOUT),
            keras.layers.Dense(LSTM_DENSE_UNITS, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def make_sequences(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    lookback: int = LSTM_LOOKBACK,
):
    """Turn a scaled feature matrix + target column into sliding sequences.

    ``X[i] = features_scaled[i - lookback : i]`` and ``y[i] =
    target_scaled[i]`` — i.e. the target of row ``i`` is predicted from the
    ``lookback`` days strictly before it, so no future information enters a
    training sample.
    """
    if len(features_scaled) <= lookback:
        raise ValueError(
            f"Need more than {lookback} rows to build LSTM sequences; got "
            f"{len(features_scaled)}."
        )
    X, y = [], []
    for i in range(lookback, len(features_scaled)):
        X.append(features_scaled[i - lookback : i])
        y.append(target_scaled[i])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train_lstm(
    features_scaled_train, target_scaled_train,
    features_scaled_val, target_scaled_val,
    lookback: int = LSTM_LOOKBACK,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH_SIZE,
    patience: int = LSTM_PATIENCE,
    seed: int = 42,
):
    """Train the LSTM with early stopping on a chronological validation set.

    Returns ``(model, history_dict)``.
    """
    tf, _ = _import_tf()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train = make_sequences(features_scaled_train, target_scaled_train, lookback)
    X_val, y_val = make_sequences(features_scaled_val, target_scaled_val, lookback)
    logger.info("LSTM sequences: train=%s, val=%s", X_train.shape, X_val.shape)

    model = build_lstm(X_train.shape[2], lookback)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    return model, {
        "loss": [float(v) for v in history.history.get("loss", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "epochs_run": len(history.history.get("loss", [])),
    }
