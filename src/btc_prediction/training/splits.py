"""Chronological train/validation/test splitting for time-series data.

Random shuffling is never allowed: with shuffled splits the model could
"see the future" because adjacent days are highly autocorrelated. Splits
here are contiguous, ordered blocks of time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from config.settings import TRAIN_FRACTION, VALIDATION_FRACTION, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChronoSplits:
    """Contiguous chronological splits of a date-indexed frame."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def summary(self) -> dict:
        return {
            "train": {
                "rows": len(self.train),
                "start": str(self.train.index.min().date()),
                "end": str(self.train.index.max().date()),
            },
            "validation": {
                "rows": len(self.validation),
                "start": str(self.validation.index.min().date()),
                "end": str(self.validation.index.max().date()),
            },
            "test": {
                "rows": len(self.test),
                "start": str(self.test.index.min().date()),
                "end": str(self.test.index.max().date()),
            },
        }


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> ChronoSplits:
    """Split a date-indexed frame into ordered train/val/test blocks.

    Parameters
    ----------
    df:
        Frame with a sorted, ascending DatetimeIndex.
    train_fraction, validation_fraction:
        Fractions of rows for train and validation; the remainder is test.

    Returns
    -------
    ChronoSplits
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("chronological_split requires a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("chronological_split requires sorted, ascending dates.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1.")

    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    if any(len(part) == 0 for part in (train, validation, test)):
        raise ValueError(
            f"Not enough rows ({n}) for a chronological "
            "train/validation/test split. Need more historical data."
        )

    logger.info(
        "Chronological split: train=%d (%s..%s), val=%d (%s..%s), test=%d (%s..%s)",
        len(train), train.index.min().date(), train.index.max().date(),
        len(validation), validation.index.min().date(), validation.index.max().date(),
        len(test), test.index.min().date(), test.index.max().date(),
    )
    return ChronoSplits(train=train, validation=validation, test=test)


def split_xy(
    splits: ChronoSplits, feature_columns, target: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Convenience: split into (X_train, y_train, X_val, y_val, X_test, y_test)."""
    return (
        splits.train[feature_columns], splits.train[target],
        splits.validation[feature_columns], splits.validation[target],
        splits.test[feature_columns], splits.test[target],
    )
