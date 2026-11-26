"""Tests for data collection: validation, cleaning, persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_synthetic_ohlcv


def test_validate_passes_clean_data(synthetic_ohlcv):
    from src.btc_prediction.data.collector import validate_ohlcv

    report = validate_ohlcv(synthetic_ohlcv)
    assert report["n_rows"] == len(synthetic_ohlcv)
    assert report["duplicate_dates"] == 0
    assert report["non_positive_prices"] == 0
    assert report["high_below_low"] == 0


def test_validate_rejects_empty_frame():
    from src.btc_prediction.data.collector import DataValidationError, validate_ohlcv

    with pytest.raises(DataValidationError):
        validate_ohlcv(pd.DataFrame())
    with pytest.raises(DataValidationError):
        validate_ohlcv(None)


def test_validate_rejects_missing_columns(synthetic_ohlcv):
    from src.btc_prediction.data.collector import DataValidationError, validate_ohlcv

    broken = synthetic_ohlcv.drop(columns=["Volume"])
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_ohlcv(broken)


def test_clean_sorts_and_deduplicates(synthetic_ohlcv):
    from src.btc_prediction.data.collector import clean_ohlcv

    shuffled = synthetic_ohlcv.iloc[::-1]
    doubled = pd.concat([shuffled, shuffled.loc[shuffled.index[:5]]])
    cleaned = clean_ohlcv(doubled, drop_incomplete_current_day=False)

    assert cleaned.index.is_monotonic_increasing
    assert cleaned.index.is_unique
    assert len(cleaned) == len(synthetic_ohlcv)
    # kept the LAST occurrence of duplicated dates
    assert cleaned.loc[cleaned.index[0], "Close"] == synthetic_ohlcv.iloc[0]["Close"]


def test_clean_fills_and_drops_missing(synthetic_ohlcv):
    from src.btc_prediction.data.collector import clean_ohlcv

    gappy = synthetic_ohlcv.copy()
    gappy.iloc[10, gappy.columns.get_loc("Close")] = np.nan   # isolated gap -> ffill
    gappy.iloc[0, gappy.columns.get_loc("Open")] = np.nan     # leading gap -> dropped

    cleaned = clean_ohlcv(gappy, drop_incomplete_current_day=False)
    assert cleaned.notna().all().all()
    assert cleaned.index[0] != gappy.index[0]          # leading NaN row removed
    # after dropping the leading row, the ffilled value sits one position up:
    # original row 10 (Close=NaN) now equals original row 9's Close
    assert cleaned.iloc[9]["Close"] == cleaned.iloc[8]["Close"]


def test_incomplete_current_day_is_dropped():
    from src.btc_prediction.data.collector import clean_ohlcv

    df = make_synthetic_ohlcv(n=60, end_today=True)
    assert df.index.max() >= pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

    cleaned = clean_ohlcv(df, drop_incomplete_current_day=True)
    today_utc = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    assert cleaned.index.max() < today_utc


def test_past_days_are_not_dropped(synthetic_ohlcv):
    from src.btc_prediction.data.collector import clean_ohlcv

    cleaned = clean_ohlcv(synthetic_ohlcv, drop_incomplete_current_day=True)
    assert len(cleaned) == len(synthetic_ohlcv)


def test_save_and_roundtrip(tmp_path, synthetic_ohlcv):
    from src.btc_prediction.data.collector import clean_ohlcv, load_processed, save_processed

    cleaned = clean_ohlcv(synthetic_ohlcv, drop_incomplete_current_day=False)
    path = tmp_path / "btc_test.csv"
    save_processed(cleaned, path)
    loaded = load_processed(path)

    expected = cleaned.copy()
    expected.index.name = "Date"  # save_processed stamps the index name
    assert len(loaded) == len(expected)
    assert list(loaded.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(loaded, expected, check_freq=False)


def test_load_processed_missing_file(tmp_path):
    from src.btc_prediction.data.collector import load_processed

    with pytest.raises(FileNotFoundError, match="download_data"):
        load_processed(tmp_path / "does_not_exist.csv")
