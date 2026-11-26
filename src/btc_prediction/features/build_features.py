"""Build the model-ready feature dataset and leak-free targets.

Targets
-------
* Regression: ``Next_Day_Close = Close.shift(-1)``.
* Classification: ``Direction = 1`` ("UP") when the next day's close is
  greater than the current day's close, otherwise ``0`` ("DOWN").

Leakage policy
--------------
Feature rows only ever contain information available at the end of that
day (indicators of the current day plus shifted lags). The target is the
only column that looks into the future, and it is never part of
``FEATURE_COLUMNS``. The final row has no next-day close yet, so it is
dropped here for training — the prediction module recomputes features for
that last row separately when predicting.
"""

from __future__ import annotations

import pandas as pd

from config.settings import get_logger
from src.btc_prediction.features.indicators import add_all_indicators

logger = get_logger(__name__)

TARGET_REGRESSION = "Next_Day_Close"
TARGET_CLASSIFICATION = "Direction"

FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "SMA_7",
    "SMA_14",
    "SMA_30",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Daily_Return_Pct",
    "Rolling_Volatility_7",
    "Rolling_Volatility_30",
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_5",
    "Volume_Lag_1",
    "Volume_Lag_2",
    "High_Low_Range",
    "High_Low_Range_Pct",
    "Open_Close_Pct",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append all technical indicators and lag features to an OHLCV frame."""
    if df is None or df.empty:
        raise ValueError("Cannot build features from an empty frame.")
    return add_all_indicators(df)


def add_targets(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add the next-day close (regression) and UP/DOWN direction targets."""
    out = feature_df.copy()
    out[TARGET_REGRESSION] = out["Close"].shift(-1)
    out[TARGET_CLASSIFICATION] = (
        out[TARGET_REGRESSION] > out["Close"]
    ).astype(int)
    return out


def create_dataset(df: pd.DataFrame, keep_helper_columns: bool = False) -> pd.DataFrame:
    """Full pipeline: OHLCV -> features + targets, NaN rows dropped.

    Parameters
    ----------
    df:
        Cleaned daily OHLCV frame.
    keep_helper_columns:
        Keep the raw OHLC columns next to the features (useful for charts).

    Returns
    -------
    Frame with ``FEATURE_COLUMNS`` + both targets, rows with NaN features or
    a missing next-day close removed (this drops the final row, which has no
    target yet).
    """
    features = build_features(df)
    dataset = add_targets(features)

    keep = list(FEATURE_COLUMNS) + [TARGET_REGRESSION, TARGET_CLASSIFICATION]
    if keep_helper_columns:
        keep = sorted(set(keep) | {"Open", "High", "Low", "Close", "Volume"})
    dataset = dataset[keep]

    before = len(dataset)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + [TARGET_REGRESSION])
    dropped = before - len(dataset)
    logger.info(
        "Dataset built: %d rows (dropped %d rows with NaN features/target).",
        len(dataset),
        dropped,
    )
    return dataset


def features_for_last_row(df: pd.DataFrame) -> pd.DataFrame:
    """Feature vector for the most recent day (used at prediction time).

    The last row of the raw frame has no target yet, so ``create_dataset``
    drops it; this helper instead computes indicators over the whole frame
    and returns the single most recent feature row, ready for models.
    """
    features = build_features(df)
    last_row = features.iloc[[-1]][FEATURE_COLUMNS]
    if last_row.isna().any().any():
        raise ValueError(
            "Not enough history to compute all features for the latest day. "
            f"Need at least ~30 days of data; got {len(df)} rows."
        )
    return last_row
