"""Tests for technical indicators and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.btc_prediction.features.build_features import (
    FEATURE_COLUMNS,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    create_dataset,
    features_for_last_row,
)
from src.btc_prediction.features.indicators import (
    add_ema,
    add_macd,
    add_rsi,
    add_sma,
)


def test_sma_matches_rolling_mean(synthetic_ohlcv):
    result = add_sma(synthetic_ohlcv, windows=(7,))
    expected = synthetic_ohlcv["Close"].rolling(7).mean()
    pd.testing.assert_series_equal(result["SMA_7"], expected, check_names=False)


def test_ema_matches_pandas_ewm(synthetic_ohlcv):
    result = add_ema(synthetic_ohlcv, spans=(12,))
    expected = synthetic_ohlcv["Close"].ewm(span=12, adjust=False).mean()
    pd.testing.assert_series_equal(result["EMA_12"], expected, check_names=False)


def test_rsi_is_bounded_and_sane(synthetic_ohlcv):
    result = add_rsi(synthetic_ohlcv, period=14)
    rsi = result["RSI_14"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()
    assert len(rsi) > len(synthetic_ohlcv) - 20  # defined from ~period onward


def test_macd_relationship(synthetic_ohlcv):
    result = add_macd(synthetic_ohlcv)
    # MACD = EMA12 - EMA26 by construction
    ema_fast = synthetic_ohlcv["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = synthetic_ohlcv["Close"].ewm(span=26, adjust=False).mean()
    np.testing.assert_allclose(result["MACD"], ema_fast - ema_slow, rtol=1e-9)
    np.testing.assert_allclose(
        result["MACD_Hist"], result["MACD"] - result["MACD_Signal"], rtol=1e-9
    )


def test_dataset_has_all_features_and_targets(synthetic_dataset):
    for column in FEATURE_COLUMNS:
        assert column in synthetic_dataset.columns, f"missing feature {column}"
    assert TARGET_REGRESSION in synthetic_dataset.columns
    assert TARGET_CLASSIFICATION in synthetic_dataset.columns


def test_dataset_has_no_nan(synthetic_dataset):
    assert not synthetic_dataset[FEATURE_COLUMNS].isna().any().any()
    assert not synthetic_dataset[TARGET_REGRESSION].isna().any()
    assert set(synthetic_dataset[TARGET_CLASSIFICATION].unique()) <= {0, 1}


def test_dataset_drops_final_row_without_target(synthetic_ohlcv):
    dataset = create_dataset(synthetic_ohlcv)
    # last raw row has no next-day close, so it must be dropped
    assert dataset.index.max() < synthetic_ohlcv.index.max()


def test_features_for_last_row_matches_dataset_schema(synthetic_ohlcv):
    row = features_for_last_row(synthetic_ohlcv)
    assert list(row.columns) == FEATURE_COLUMNS
    assert len(row) == 1
    assert not row.isna().any().any()
    # its index is the most recent raw day (the one create_dataset drops)
    assert row.index[0] == synthetic_ohlcv.index.max()


def test_daily_return_matches_pct_change(synthetic_ohlcv):
    from src.btc_prediction.features.indicators import add_daily_returns

    result = add_daily_returns(synthetic_ohlcv)
    expected = synthetic_ohlcv["Close"].pct_change() * 100.0
    pd.testing.assert_series_equal(result["Daily_Return_Pct"], expected,
                                   check_names=False)


def test_lags_are_shifted(synthetic_ohlcv):
    from src.btc_prediction.features.indicators import add_lags

    result = add_lags(synthetic_ohlcv)
    np.testing.assert_allclose(
        result["Close_Lag_1"].iloc[1:], synthetic_ohlcv["Close"].iloc[:-1], rtol=1e-12
    )
    np.testing.assert_allclose(
        result["Volume_Lag_1"].iloc[1:], synthetic_ohlcv["Volume"].iloc[:-1], rtol=1e-12
    )
    assert result["Close_Lag_1"].iloc[0] != result["Close_Lag_1"].iloc[0]  # NaN
