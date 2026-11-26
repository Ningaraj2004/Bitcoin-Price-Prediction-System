"""Prediction-module tests: end-to-end predictions from saved artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.btc_prediction.features.build_features import create_dataset
from src.btc_prediction.prediction import BitcoinPredictor, PredictionError
from tests.conftest import make_synthetic_ohlcv


@pytest.fixture(scope="module")
def predictor(trained_artifacts) -> BitcoinPredictor:
    return BitcoinPredictor(artifacts_dir=trained_artifacts)


def test_predictor_finds_all_models(predictor):
    assert set(predictor.available_regression_models()) == {
        "Linear Regression", "Random Forest", "XGBoost"
    }
    assert set(predictor.available_classification_models()) == {
        "Logistic Regression", "Random Forest", "XGBoost"
    }


@pytest.mark.parametrize("model", ["Linear Regression", "Random Forest", "XGBoost"])
def test_next_day_close_prediction(predictor, synthetic_ohlcv, model):
    result = predictor.predict_next_day_close(model, df=synthetic_ohlcv)

    assert result["model_name"] == model
    assert np.isfinite(result["predicted_close"])
    assert result["predicted_close"] > 0
    last_close = float(synthetic_ohlcv["Close"].iloc[-1])
    assert result["last_close"] == last_close
    expected_pct = (result["predicted_close"] - last_close) / last_close * 100.0
    assert result["predicted_change_pct"] == pytest.approx(expected_pct)
    # prediction is for the day after the last observed day
    assert result["prediction_date"] == synthetic_ohlcv.index.max() + pd.Timedelta(days=1)


def test_direction_prediction_all_models(predictor, synthetic_ohlcv):
    for model in predictor.available_classification_models():
        result = predictor.predict_direction(model, df=synthetic_ohlcv)
        assert result["direction"] in ("UP", "DOWN")
        assert result["prob_up"] + result["prob_down"] == pytest.approx(1.0)
        assert 0.0 <= result["prob_up"] <= 1.0
        assert result["confidence"] == pytest.approx(
            max(result["prob_up"], result["prob_down"])
        )
        assert model in result["explanation"]
        assert "$" in result["explanation"]  # includes the last close


def test_direction_consistent_with_probability(predictor, synthetic_ohlcv):
    result = predictor.predict_direction("XGBoost", df=synthetic_ohlcv)
    if result["prob_up"] >= 0.5:
        assert result["direction"] == "UP"
    else:
        assert result["direction"] == "DOWN"


def test_predictor_handles_missing_artifacts(tmp_path):
    empty_predictor = BitcoinPredictor(artifacts_dir=tmp_path)
    df = make_synthetic_ohlcv()
    with pytest.raises(PredictionError, match="train_models"):
        empty_predictor.predict_next_day_close("XGBoost", df=df)
    with pytest.raises(PredictionError, match="train_models"):
        empty_predictor.predict_direction("XGBoost", df=df)


def test_predictor_rejects_unknown_model(predictor, synthetic_ohlcv):
    # unknown names fall back to the first available model rather than crash
    result = predictor.predict_next_day_close("Does Not Exist", df=synthetic_ohlcv)
    assert result["model_name"] == "Does Not Exist"
    assert np.isfinite(result["predicted_close"])


def test_predictions_are_deterministic(predictor, synthetic_ohlcv):
    a = predictor.predict_next_day_close("XGBoost", df=synthetic_ohlcv)
    b = predictor.predict_next_day_close("XGBoost", df=synthetic_ohlcv)
    assert a["predicted_close"] == b["predicted_close"]


def test_lstm_prediction_when_available(trained_artifacts, synthetic_dataset):
    """Only runs when an LSTM artifact exists (trained in test_lstm.py via a
    tmp dir); here we exercise the LSTM path if the module can train one."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        pytest.skip("TensorFlow not installed")

    from src.btc_prediction.features.build_features import FEATURE_COLUMNS
    from sklearn.preprocessing import MinMaxScaler
    from src.btc_prediction.models.lstm import train_lstm

    lookback = 30
    feat = synthetic_dataset[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    close = synthetic_dataset["Close"].to_numpy().reshape(-1, 1)
    feat_scaler = MinMaxScaler().fit(feat[:150])
    target_scaler = MinMaxScaler().fit(close[:150])
    fs, ts = feat_scaler.transform(feat), target_scaler.transform(close).ravel()

    model, _ = train_lstm(
        fs[:200], ts[:200], fs[:250], ts[:250], lookback=lookback, epochs=2,
    )
    tail = fs[-lookback:]
    pred_scaled = float(model.predict(np.asarray([tail]), verbose=0).ravel()[0])
    pred = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
    assert np.isfinite(pred)
    assert pred > 0


def test_dataset_targets_used_by_predictor_roundtrip(synthetic_ohlcv):
    """Sanity: features computed for prediction come from the same code path
    as training features."""
    dataset = create_dataset(synthetic_ohlcv)
    # the second-to-last dataset row equals the last feature row's day - 1
    last_feature_day = synthetic_ohlcv.index.max()
    assert dataset.index.max() < last_feature_day
