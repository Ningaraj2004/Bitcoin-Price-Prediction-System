"""Shared pytest fixtures: import paths and synthetic OHLCV data.

Tests never touch the network: they use a deterministic synthetic market
(grown with a seeded geometric random walk) so results are reproducible
and the suite is fast and offline-friendly.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(n: int = 300, seed: int = 42,
                         end_today: bool = False) -> pd.DataFrame:
    """Deterministic synthetic BTC-like daily OHLCV frame."""
    rng = np.random.default_rng(seed)
    if end_today:
        dates = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize().tz_localize(None),
                              periods=n, freq="D")
    else:
        dates = pd.bdate_range("2023-01-02", periods=n)
    log_ret = rng.normal(0.0005, 0.02, n)
    close = 20000.0 * np.exp(np.cumsum(log_ret))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.uniform(2e9, 6e9, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    return make_synthetic_ohlcv()


@pytest.fixture(scope="session")
def synthetic_dataset(synthetic_ohlcv: pd.DataFrame) -> pd.DataFrame:
    from src.btc_prediction.features.build_features import create_dataset

    return create_dataset(synthetic_ohlcv)


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory, synthetic_dataset: pd.DataFrame) -> Path:
    """Train every classical model once on the synthetic dataset (session scope)."""
    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    from src.btc_prediction.training.train_classification import (
        train_classification_models,
    )
    from src.btc_prediction.training.train_regression import train_regression_models

    train_regression_models(synthetic_dataset, artifacts_dir=artifacts_dir)
    train_classification_models(synthetic_dataset, artifacts_dir=artifacts_dir)
    return artifacts_dir
