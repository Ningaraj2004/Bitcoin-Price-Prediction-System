"""Prediction façade for the dashboard (loads artifacts, never retrains)."""

from src.btc_prediction.prediction.predictor import (  # noqa: F401
    BitcoinPredictor,
    PredictionError,
)
