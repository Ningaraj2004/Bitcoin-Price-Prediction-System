"""Advanced Bitcoin Price Prediction & Market Analysis — Streamlit dashboard.

Run from the project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable regardless of where `streamlit run` is invoked from
# (repo root locally or on Streamlit Community Cloud).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.loaders import sidebar  # noqa: E402
from components.sections_market import (  # noqa: E402
    render_historical_analysis,
    render_market_overview,
    render_technical_indicators,
)
from components.sections_models import (  # noqa: E402
    render_direction_prediction,
    render_model_comparison,
    render_model_evaluation,
    render_price_prediction,
)

state: dict = {}
sidebar(state)
page = state.get("page", "Market Overview")

PAGES = {
    "Market Overview": render_market_overview,
    "Historical Price Analysis": render_historical_analysis,
    "Technical Indicators": render_technical_indicators,
    "Bitcoin Price Prediction": render_price_prediction,
    "Market Direction Prediction": render_direction_prediction,
    "Model Comparison": render_model_comparison,
    "Model Evaluation": render_model_evaluation,
}

try:
    PAGES[page]()
except Exception as exc:  # last-resort guard: show the error, keep the app alive
    st.error(f"Something went wrong on this page: {exc}")
    st.caption("Try refreshing the market data from the sidebar, or re-run the "
               "training pipeline (`python scripts/train_models.py`).")
