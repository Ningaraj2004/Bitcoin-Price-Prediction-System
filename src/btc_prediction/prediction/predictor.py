"""Next-day price and market-direction prediction from saved artifacts.

The predictor recomputes the same feature engineering used in training on
the latest processed data, then applies the chosen fitted model. It never
retrains and never invents numbers: predictions come strictly from the
saved, evaluated artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    ARTIFACTS_DIR,
    CLASSIFICATION_METRICS_PATH,
    LSTM_LOOKBACK,
    LSTM_METRICS_PATH,
    REGRESSION_METRICS_PATH,
    get_logger,
)
from src.btc_prediction.data import collector
from src.btc_prediction.features.build_features import (
    FEATURE_COLUMNS,
    features_for_last_row,
)
from src.btc_prediction.training.train_classification import (
    available_classification_models,
    load_classification_model,
)
from src.btc_prediction.training.train_lstm import load_lstm
from src.btc_prediction.training.train_regression import (
    available_regression_models,
    load_regression_model,
)

logger = get_logger(__name__)


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be produced (missing artifact, bad data)."""


class BitcoinPredictor:
    """Facade used by the dashboard: load artifacts lazily, predict next day."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = Path(artifacts_dir)
        self._regression_cache: dict = {}
        self._classification_cache: dict = {}
        self._lstm_cache: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Data / features
    # ------------------------------------------------------------------
    def load_market_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Processed OHLCV data (cached CSV; optional forced re-download)."""
        return collector.load_or_download(force_refresh=force_refresh)

    def _feature_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """Latest single-row feature frame, columns in canonical order."""
        return features_for_last_row(df)

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------
    def available_regression_models(self) -> list:
        return available_regression_models(self.artifacts_dir)

    def available_classification_models(self) -> list:
        return available_classification_models(self.artifacts_dir)

    def lstm_available(self) -> bool:
        return (self.artifacts_dir / "lstm_model.keras").exists()

    def _get_regression(self, name: str):
        if name not in self._regression_cache:
            self._regression_cache[name] = load_regression_model(name, self.artifacts_dir)
        return self._regression_cache[name]

    def _get_classification(self, name: str):
        if name not in self._classification_cache:
            self._classification_cache[name] = load_classification_model(
                name, self.artifacts_dir
            )
        return self._classification_cache[name]

    def _get_lstm(self):
        if self._lstm_cache is None:
            self._lstm_cache = load_lstm(self.artifacts_dir)
        return self._lstm_cache

    def _require_any_regression(self, name: Optional[str]) -> str:
        available = self.available_regression_models()
        if not available:
            raise PredictionError(
                "No regression artifacts found. Train the models first: "
                "`python scripts/train_models.py`."
            )
        if name is None or name not in available:
            return available[0]
        return name

    def _require_any_classification(self, name: Optional[str]) -> str:
        available = self.available_classification_models()
        if not available:
            raise PredictionError(
                "No classification artifacts found. Train the models first: "
                "`python scripts/train_models.py`."
            )
        if name is None or name not in available:
            return available[0]
        return name

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def predict_next_day_close(
        self, model_name: str, df: Optional[pd.DataFrame] = None
    ) -> dict:
        """Predict the next daily close with a classical ML or the LSTM model.

        Returns a dict with the last close, the prediction, the implied
        percentage change and the date the prediction applies to.
        """
        df = self.load_market_data() if df is None else df
        x = self._feature_row(df)

        if model_name.lower().startswith("lstm"):
            prediction = self._predict_close_lstm(df)
        else:
            name = self._require_any_regression(model_name)
            model = self._get_regression(name)
            try:
                prediction = float(model.predict(x)[0])
            except Exception as exc:
                raise PredictionError(
                    f"Model '{name}' failed on the latest features: {exc}"
                ) from exc

        last_close = float(df["Close"].iloc[-1])
        last_date = df.index.max()
        change_pct = (prediction - last_close) / last_close * 100.0
        return {
            "model_name": model_name,
            "last_close": last_close,
            "last_date": last_date,
            "prediction_date": last_date + pd.Timedelta(days=1),
            "predicted_close": prediction,
            "predicted_change_pct": change_pct,
            "as_of": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC"),
        }

    def _predict_close_lstm(self, df: pd.DataFrame) -> float:
        """LSTM prediction from the last `lookback` scaled feature rows."""
        model, feature_scaler, target_scaler = self._get_lstm()

        from src.btc_prediction.features.build_features import build_features

        features = build_features(df)[FEATURE_COLUMNS].tail(LSTM_LOOKBACK)
        if len(features) < LSTM_LOOKBACK:
            raise PredictionError(
                f"LSTM needs at least {LSTM_LOOKBACK} rows of features; got "
                f"{len(features)}."
            )
        scaled = feature_scaler.transform(features.to_numpy(dtype=np.float64))
        x = np.asarray([scaled], dtype=np.float32)
        pred_scaled = float(model.predict(x, verbose=0).ravel()[0])
        return float(target_scaler.inverse_transform([[pred_scaled]])[0, 0])

    def predict_direction(
        self, model_name: str, df: Optional[pd.DataFrame] = None
    ) -> dict:
        """Predict UP/DOWN for the next day with probability scores."""
        df = self.load_market_data() if df is None else df
        name = self._require_any_classification(model_name)
        x = self._feature_row(df)
        model = self._get_classification(name)

        try:
            proba = model.predict_proba(x)[0]
        except Exception as exc:
            raise PredictionError(
                f"Model '{name}' failed on the latest features: {exc}"
            ) from exc

        prob_down, prob_up = float(proba[0]), float(proba[1])
        direction = "UP" if prob_up >= 0.5 else "DOWN"
        last_close = float(df["Close"].iloc[-1])
        last_date = df.index.max()

        # Human-readable explanation grounded in the model's own features.
        feat = x.iloc[0]
        rsi = feat["RSI_14"]
        macd_signal = "above" if feat["MACD"] > feat["MACD_Signal"] else "below"
        sma7_sma30 = "above" if feat["SMA_7"] > feat["SMA_30"] else "below"
        explanation = (
            f"The {name} model assigns a {prob_up:.1%} probability that "
            f"tomorrow's close will be higher than today's (${last_close:,.2f}) "
            f"and a {prob_down:.1%} probability that it will be lower, and "
            f"therefore predicts **{direction}**. Current context: RSI(14) is "
            f"{rsi:.1f} ({'overbought' if rsi >= 70 else 'oversold' if rsi <= 30 else 'neutral'}), "
            f"the MACD line is {macd_signal} its signal line, and the 7-day SMA "
            f"is {sma7_sma30} the 30-day SMA. Directional predictions for "
            "financial markets are highly uncertain — see the disclaimer."
        )
        return {
            "model_name": name,
            "direction": direction,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "confidence": max(prob_up, prob_down),
            "last_close": last_close,
            "last_date": last_date,
            "prediction_date": last_date + pd.Timedelta(days=1),
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    # Evaluation summaries for the dashboard
    # ------------------------------------------------------------------
    def load_regression_metrics(self) -> Optional[dict]:
        path = self.artifacts_dir / "regression_metrics.json"
        return _read_json(path)

    def load_classification_metrics(self) -> Optional[dict]:
        path = self.artifacts_dir / "classification_metrics.json"
        return _read_json(path)

    def load_lstm_metrics(self) -> Optional[dict]:
        return _read_json(LSTM_METRICS_PATH)


def _read_json(path) -> Optional[dict]:
    import json

    if not Path(path).exists():
        return None
    with open(path) as fh:
        return json.load(fh)
