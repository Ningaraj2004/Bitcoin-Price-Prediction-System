#!/usr/bin/env python3
"""Step 2 — Build the leak-free feature dataset with targets.

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

from config.settings import (  # noqa: E402
    FEATURES_DATA_PATH,
    ensure_directories,
    get_logger,
)
from src.btc_prediction.data import collector  # noqa: E402
from src.btc_prediction.features.build_features import (  # noqa: E402
    create_dataset,
)

logger = get_logger(__name__)


def main() -> int:
    try:
        data = collector.load_processed()
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1

    dataset = create_dataset(data, keep_helper_columns=True)
    ensure_directories()
    dataset.to_csv(FEATURES_DATA_PATH, index=True, index_label="Date")
    print(f"✅ Feature dataset saved: {FEATURES_DATA_PATH}")
    print(f"   Rows: {len(dataset):,} · Columns: {len(dataset.columns)}")
    print(f"   Range: {dataset.index.min().date()} → {dataset.index.max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
