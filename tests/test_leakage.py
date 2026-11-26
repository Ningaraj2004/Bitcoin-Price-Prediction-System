"""Data-leakage and target-alignment tests.

The critical property: feature values for day t must depend ONLY on data
from days <= t. If modifying future prices changes past features, there
is leakage. Targets must align exactly with the definitions.
"""

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
from tests.conftest import make_synthetic_ohlcv

LEAK_FORBIDDEN_SUBSTRINGS = ("Next_Day", "Direction")


def test_targets_are_not_features(synthetic_dataset):
    assert TARGET_REGRESSION not in FEATURE_COLUMNS
    assert TARGET_CLASSIFICATION not in FEATURE_COLUMNS


def test_no_lookahead_in_feature_names():
    for name in FEATURE_COLUMNS:
        for forbidden in LEAK_FORBIDDEN_SUBSTRINGS:
            assert forbidden.lower() not in name.lower()


def test_modifying_future_data_does_not_change_past_features():
    """Core leakage check: perturb everything AFTER day k; features at
    rows < k must be bit-for-bit identical."""
    df = make_synthetic_ohlcv(n=250, seed=7)
    k = 150

    perturbed = df.copy()
    perturbed.iloc[k:] = perturbed.iloc[k:] * 3.0  # triple all future prices

    from src.btc_prediction.features.build_features import build_features

    orig_all = build_features(df)[FEATURE_COLUMNS]
    pert_all = build_features(perturbed)[FEATURE_COLUMNS]
    unchanged = orig_all.iloc[:k].equals(pert_all.iloc[:k])
    assert unchanged, "Features before day k changed when future data changed"


def test_modifying_past_data_does_not_change_future_features_past_the_window():
    """A one-day perturbation may only affect features whose lookback window
    still covers that day. Beyond day k + 30 (the widest window), every
    window-based feature must be identical. Exponentially-weighted indicators
    (EMA, RSI, MACD) legitimately decay forever, so they are excluded here."""
    df = make_synthetic_ohlcv(n=250, seed=11)
    k = 100
    perturbed = df.copy()
    perturbed.iloc[k] = perturbed.iloc[k] * 2.0  # double a single day

    from src.btc_prediction.features.build_features import build_features

    window_based = [
        "SMA_7", "SMA_14", "SMA_30",
        "Rolling_Volatility_7", "Rolling_Volatility_30",
        "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
        "Volume_Lag_1", "Volume_Lag_2",
        "High_Low_Range", "High_Low_Range_Pct", "Open_Close_Pct",
        "Daily_Return_Pct",
    ]
    orig = build_features(df)[window_based]
    pert = build_features(perturbed)[window_based]
    max_window = 30
    pd.testing.assert_frame_equal(
        orig.iloc[k + max_window + 1:], pert.iloc[k + max_window + 1:]
    )
    # daily return is only affected on the day right after the perturbation
    assert not np.isclose(
        orig["Daily_Return_Pct"].iloc[k + 1], pert["Daily_Return_Pct"].iloc[k + 1]
    )


def test_next_day_close_target_alignment(synthetic_ohlcv):
    dataset = create_dataset(synthetic_ohlcv)
    # Next_Day_Close[t] must equal Close[t+1] for every dataset row
    closes = synthetic_ohlcv["Close"]
    for i in range(len(dataset) - 1):
        t = dataset.index[i]
        next_date = dataset.index[i + 1]
        expected = closes.loc[next_date]
        assert dataset[TARGET_REGRESSION].iloc[i] == expected
        assert t < next_date  # chronological order preserved


def test_direction_target_alignment(synthetic_ohlcv):
    dataset = create_dataset(synthetic_ohlcv)
    closes = synthetic_ohlcv["Close"]
    for i in range(len(dataset) - 1):
        next_close = closes.loc[dataset.index[i + 1]]
        current_close = dataset["Close"].iloc[i]
        expected = 1 if next_close > current_close else 0
        assert dataset[TARGET_CLASSIFICATION].iloc[i] == expected


def test_next_day_close_equals_shift_minus_one(synthetic_ohlcv):
    """Direct definition check on a fresh frame."""
    from src.btc_prediction.features.build_features import add_targets, build_features

    features = build_features(synthetic_ohlcv)
    targets = add_targets(features)
    np.testing.assert_allclose(
        targets[TARGET_REGRESSION].iloc[:-1],
        synthetic_ohlcv["Close"].shift(-1).iloc[:-1],
        rtol=1e-12,
    )
    assert np.isnan(targets[TARGET_REGRESSION].iloc[-1])


def test_dataset_index_is_chronological_and_unique(synthetic_dataset):
    assert synthetic_dataset.index.is_monotonic_increasing
    assert synthetic_dataset.index.is_unique


def test_prediction_uses_only_information_available_today(synthetic_ohlcv):
    """The feature row used for 'tomorrow's' prediction must be computable
    from data up to today only — truncate the frame at day k and the
    feature row of day k must equal the full-sample row at day k (the
    prefix is identical, so even the EMA warm-up matches exactly)."""
    k = 200
    df = make_synthetic_ohlcv(n=250, seed=3)
    truncated = df.iloc[: k + 1]

    from src.btc_prediction.features.build_features import build_features

    full_row = build_features(df)[FEATURE_COLUMNS].loc[df.index[k]]
    trunc_row = build_features(truncated)[FEATURE_COLUMNS].iloc[-1]
    pd.testing.assert_series_equal(full_row, trunc_row, check_names=False)
