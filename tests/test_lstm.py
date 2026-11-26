"""LSTM tests: sequence construction, training, persistence.

Skipped automatically when TensorFlow is unavailable — the project is
designed to run without the LSTM, so its absence must not fail the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import tensorflow  # noqa: F401

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")


def test_make_sequences_shapes():
    from src.btc_prediction.models.lstm import make_sequences

    features = np.random.default_rng(0).random((100, 5))
    target = np.random.default_rng(1).random(100)
    X, y = make_sequences(features, target, lookback=10)
    assert X.shape == (90, 10, 5)
    assert y.shape == (90,)
    # sequence i ends at row i+lookback-1; target row i is row lookback
    np.testing.assert_allclose(X[0, -1], features[9])
    np.testing.assert_allclose(y[0], target[10])


def test_make_sequences_requires_enough_rows():
    from src.btc_prediction.models.lstm import make_sequences

    with pytest.raises(ValueError, match="more than"):
        make_sequences(np.zeros((5, 3)), np.zeros(5), lookback=10)


def test_lstm_sequence_has_no_future_information():
    """X[i] may only contain rows strictly before the target row i."""
    from src.btc_prediction.models.lstm import make_sequences

    features = np.arange(30, dtype=float).reshape(30, 1)
    target = np.arange(30, dtype=float)
    X, y = make_sequences(features, target, lookback=5)
    # last input of sequence for target row 5 must be row 4 (not row 5)
    assert X[0, -1, 0] == 4.0
    assert y[0] == 5.0


def test_lstm_trains_and_saves(synthetic_dataset, tmp_path):
    from sklearn.preprocessing import MinMaxScaler

    from src.btc_prediction.features.build_features import FEATURE_COLUMNS
    from src.btc_prediction.training.train_lstm import train_and_save_lstm

    small = synthetic_dataset.head(200)  # keep the test fast
    payload = train_and_save_lstm(small, artifacts_dir=tmp_path, epochs=3, patience=3)

    assert payload["test"]["n_test_sequences"] > 0
    assert np.isfinite(payload["test"]["MAE"])
    for name in ("lstm_model.keras", "lstm_feature_scaler.joblib",
                 "lstm_target_scaler.joblib", "lstm_metrics.json"):
        assert (tmp_path / name).exists(), f"missing artifact {name}"

    # Reload round-trip
    from src.btc_prediction.training.train_lstm import load_lstm

    model, feat_scaler, target_scaler = load_lstm(tmp_path)
    features = small[FEATURE_COLUMNS].tail(60)
    scaled = feat_scaler.transform(features.to_numpy())
    pred_scaled = model.predict(np.asarray([scaled]), verbose=0).ravel()
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
    assert np.isfinite(pred)
    assert pred > 0


def test_lstm_graceful_insufficient_data(synthetic_dataset, tmp_path):
    from src.btc_prediction.models.lstm import LstmUnavailableError
    from src.btc_prediction.training.train_lstm import train_and_save_lstm

    tiny = synthetic_dataset.head(40)
    with pytest.raises((LstmUnavailableError, ValueError)):
        train_and_save_lstm(tiny, artifacts_dir=tmp_path, lookback=60)
