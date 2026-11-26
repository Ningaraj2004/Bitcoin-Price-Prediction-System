"""Central configuration for the Bitcoin Price Prediction project.

All paths are computed relative to the project root so the code works
identically on a laptop, in CI and on Streamlit Community Cloud.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (always relative, never machine-specific absolute paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "btc_usd_raw.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "btc_usd_daily.csv"
FEATURES_DATA_PATH = PROCESSED_DATA_DIR / "btc_usd_features.csv"

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
TICKER = "BTC-USD"
# BTC-USD trading history on Yahoo Finance starts 2014-09-17.
DATA_START_DATE = "2014-09-17"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Chronological splitting (never shuffle time-series data)
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15  # remaining 15% is the held-out test set

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
SMA_WINDOWS = (7, 14, 30)
EMA_SPANS = (12, 26)
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLATILITY_WINDOWS = (7, 30)
CLOSE_LAGS = (1, 2, 3, 5)
VOLUME_LAGS = (1, 2)

# ---------------------------------------------------------------------------
# LSTM hyperparameters
# ---------------------------------------------------------------------------
LSTM_LOOKBACK = 60
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
LSTM_DENSE_UNITS = 16
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE = 10
LSTM_VALIDATION_SPLIT_IS_CHRONOLOGICAL = True

# ---------------------------------------------------------------------------
# Model artifact names
# ---------------------------------------------------------------------------
MODEL_NAME_SLUGS = {
    "regression": {
        "Linear Regression": "linear_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    },
    "classification": {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    },
}
LSTM_MODEL_PATH = ARTIFACTS_DIR / "lstm_model.keras"
LSTM_FEATURE_SCALER_PATH = ARTIFACTS_DIR / "lstm_feature_scaler.joblib"
LSTM_TARGET_SCALER_PATH = ARTIFACTS_DIR / "lstm_target_scaler.joblib"
LSTM_METADATA_PATH = ARTIFACTS_DIR / "lstm_metadata.json"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
REGRESSION_METRICS_PATH = ARTIFACTS_DIR / "regression_metrics.json"
CLASSIFICATION_METRICS_PATH = ARTIFACTS_DIR / "classification_metrics.json"
LSTM_METRICS_PATH = ARTIFACTS_DIR / "lstm_metrics.json"
TRAINING_METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"

REGRESSION_ARTIFACT_TEMPLATE = "regression_{slug}.joblib"
CLASSIFICATION_ARTIFACT_TEMPLATE = "classification_{slug}.joblib"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> "logging.Logger":
    """Return a configured logger (console only; no files to keep repo clean)."""
    import logging

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_directories() -> None:
    """Create every directory the pipeline writes to (idempotent)."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
