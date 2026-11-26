#!/usr/bin/env python3
"""End-to-end pipeline: download → features → train → reports → sample prediction.

Usage:
    python scripts/run_pipeline.py            # full pipeline
    python scripts/run_pipeline.py --skip-lstm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

from src.btc_prediction.data import collector  # noqa: E402


def run(*args) -> int:
    import subprocess

    cmd = [sys.executable, str(Path(__file__).parent / args[0]), *args[1:]]
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full BTC pipeline")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--refresh", action="store_true",
                        help="force fresh data download")
    args = parser.parse_args()

    steps = [
        (["download_data.py", "--refresh"] if args.refresh else ["download_data.py"], "Data collection"),
        (["build_features.py"], "Feature engineering"),
        (["train_models.py", "--skip-lstm"] if args.skip_lstm else ["train_models.py"], "Model training"),
    ]
    for script, label in steps:
        print(f"\n{'=' * 60}\n▶ {label} ({script[0]})\n{'=' * 60}")
        code = run(*script)
        if code != 0:
            print(f"❌ Pipeline failed at step: {label}")
            return code

    # Final smoke test: real prediction from saved artifacts.
    print(f"\n{'=' * 60}\n▶ Smoke test: next-day predictions\n{'=' * 60}")
    from src.btc_prediction.prediction import BitcoinPredictor, PredictionError

    predictor = BitcoinPredictor()
    data = collector.load_processed()
    try:
        for name in predictor.available_regression_models():
            result = predictor.predict_next_day_close(name, df=data)
            print(f"  {name:<20} predicted close ${result['predicted_close']:>12,.2f} "
                  f"({result['predicted_change_pct']:+.2f}%) for "
                  f"{result['prediction_date'].date()}")
        for name in predictor.available_classification_models():
            result = predictor.predict_direction(name, df=data)
            print(f"  {name:<22} {result['direction']} "
                  f"(P(UP)={result['prob_up']:.1%})")
        if predictor.lstm_available():
            result = predictor.predict_next_day_close("LSTM (Deep Learning)", df=data)
            print(f"  {'LSTM':<20} predicted close ${result['predicted_close']:>12,.2f} "
                  f"({result['predicted_change_pct']:+.2f}%)")
        else:
            print("  LSTM not available (skipped).")
    except PredictionError as exc:
        print(f"  ⚠️  Prediction smoke test: {exc}")

    print("\n✅ Pipeline finished. Launch the dashboard with:")
    print("   streamlit run app/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
