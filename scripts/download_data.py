#!/usr/bin/env python3
"""Step 1 — Download BTC-USD data: validation, cleaning, raw + processed CSVs.

Usage:
    python scripts/download_data.py            # download if no cache yet
    python scripts/download_data.py --refresh  # force re-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

from src.btc_prediction.data import collector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Download BTC-USD daily data")
    parser.add_argument("--refresh", action="store_true",
                        help="force a fresh download even if a cached file exists")
    parser.add_argument("--start", default=None,
                        help="start date YYYY-MM-DD (default: 2014-09-17)")
    parser.add_argument("--ticker", default=None, help="Yahoo Finance ticker")
    args = parser.parse_args()

    try:
        if args.refresh or not collector.PROCESSED_DATA_PATH.exists():
            raw, clean = collector.download_and_save(
                ticker=args.ticker or collector.TICKER,
                start=args.start or collector.DATA_START_DATE,
            )
            print(f"✅ Saved {len(raw)} raw rows and {len(clean)} cleaned rows.")
            print(f"   Raw:       {collector.RAW_DATA_PATH}")
            print(f"   Processed: {collector.PROCESSED_DATA_PATH}")
        else:
            df = collector.load_processed()
            print(f"✅ Using cached data ({len(df)} rows, "
                  f"{df.index.min().date()} → {df.index.max().date()}).")
            print("   Use --refresh to force a fresh download.")
        return 0
    except collector.DataDownloadError as exc:
        print(f"❌ {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
