"""Cached data and artifact loading for the Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable no matter where Streamlit launches from
# (repo root on Streamlit Community Cloud, project dir locally).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from src.btc_prediction.data import collector
from src.btc_prediction.prediction import BitcoinPredictor, PredictionError

PAGES = {
    "Market Overview": "📈",
    "Historical Price Analysis": "🕯️",
    "Technical Indicators": "📐",
    "Bitcoin Price Prediction": "💰",
    "Market Direction Prediction": "🎯",
    "Model Comparison": "⚖️",
    "Model Evaluation": "🔬",
}


@st.cache_data(ttl=3600, show_spinner="Loading BTC-USD market data ...")
def load_market_data(force_refresh: bool = False) -> pd.DataFrame:
    """Processed daily OHLCV data (CSV cache; optional forced re-download)."""
    return collector.load_or_download(force_refresh=force_refresh)


@st.cache_resource(show_spinner="Loading model artifacts ...")
def get_predictor() -> BitcoinPredictor:
    """One predictor instance per Streamlit session (artifacts cached)."""
    return BitcoinPredictor()


@st.cache_data(ttl=600, show_spinner=False)
def get_test_predictions(model_name: str, kind: str):
    """Recompute genuine test-set predictions of a saved model.

    Returns ``(dates, y_true, y_pred, y_proba_or_None)`` computed by applying
    the fitted model to the chronological test split — never invented data.
    """
    predictor = get_predictor()
    dataset = load_dataset_with_targets()
    from src.btc_prediction.features.build_features import (
        FEATURE_COLUMNS,
        TARGET_CLASSIFICATION,
        TARGET_REGRESSION,
    )
    from src.btc_prediction.training.splits import chronological_split

    splits = chronological_split(dataset)
    target = TARGET_REGRESSION if kind == "regression" else TARGET_CLASSIFICATION
    X_test = splits.test[FEATURE_COLUMNS]
    y_test = splits.test[target]

    if kind == "regression":
        model = predictor._get_regression(model_name)
        y_pred = model.predict(X_test.to_numpy())
        return splits.test.index, y_test.to_numpy(), y_pred, None

    model = predictor._get_classification(model_name)
    y_proba = model.predict_proba(X_test.to_numpy())[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return splits.test.index, y_test.to_numpy(), y_pred, y_proba


@st.cache_data(ttl=600, show_spinner=False)
def load_dataset_with_targets() -> pd.DataFrame:
    """Feature dataset with targets (used for evaluation pages)."""
    data = load_market_data()
    from src.btc_prediction.features.build_features import create_dataset

    return create_dataset(data)


def sidebar(state: dict) -> None:
    """Render the sidebar: navigation, refresh, model pickers, project info."""
    st.sidebar.markdown("## ₿ Bitcoin Prediction")
    st.sidebar.caption("Advanced BTC-USD price prediction & market analysis")

    page = st.sidebar.radio("Navigation", list(PAGES.keys()),
                            format_func=lambda p: f"{PAGES[p]} {p}")
    state["page"] = page

    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh market data", use_container_width=True):
        try:
            collector.download_and_save()
            load_market_data.clear()
            st.sidebar.success("Market data refreshed.")
        except Exception as exc:
            st.sidebar.error(f"Refresh failed: {exc}")

    st.sidebar.divider()
    with st.sidebar.expander("ℹ️ Project information", expanded=False):
        st.markdown(
            """
**Advanced Bitcoin Price Prediction System**

- **Data**: BTC-USD daily OHLCV from Yahoo Finance (yfinance)
- **ML models**: Linear Regression, Random Forest, XGBoost
- **Deep learning**: LSTM (TensorFlow/Keras)
- **Targets**: next-day close (regression) and UP/DOWN direction (classification)
- **Discipline**: chronological splits, leak-free features, honest baselines

⚠️ *Educational project — not financial advice. Crypto markets are highly
volatile and past performance never guarantees future results.*
            """
        )


def require_models() -> BitcoinPredictor | None:
    """Return the predictor, or show a friendly setup notice and return None."""
    predictor = get_predictor()
    try:
        if not predictor.available_regression_models():
            raise PredictionError("no regression artifacts")
    except PredictionError:
        st.warning(
            "**No trained models found.** Run the training pipeline first:\n\n"
            "```bash\npython scripts/download_data.py\n"
            "python scripts/train_models.py\n```\n\n"
            "The market-overview pages work without models; prediction and "
            "evaluation pages need the artifacts."
        )
        return None
    return predictor
