"""Dashboard import/startup checks (no live Streamlit server needed)."""

from __future__ import annotations

import py_compile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_app_file_compiles():
    py_compile.compile(str(PROJECT_ROOT / "app" / "app.py"), doraise=True)


def test_component_files_compile():
    for component in (PROJECT_ROOT / "app" / "components").glob("*.py"):
        py_compile.compile(str(component), doraise=True)


def test_streamlit_theme_config_parses():
    import tomllib

    config = PROJECT_ROOT / ".streamlit" / "config.toml"
    assert config.exists()
    with open(config, "rb") as fh:
        data = tomllib.load(fh)
    assert "theme" in data
    assert "primaryColor" in data["theme"]


def test_streamlit_entry_point_is_correct():
    """The documented entry point must exist and reference Streamlit."""
    app_file = PROJECT_ROOT / "app" / "app.py"
    assert app_file.exists()
    content = app_file.read_text()
    assert "import streamlit as st" in content
    assert "set_page_config" in content


def test_all_seven_pages_are_registered():
    from components.loaders import PAGES

    expected = {
        "Market Overview",
        "Historical Price Analysis",
        "Technical Indicators",
        "Bitcoin Price Prediction",
        "Market Direction Prediction",
        "Model Comparison",
        "Model Evaluation",
    }
    assert set(PAGES.keys()) == expected


def test_chart_builders_run_on_synthetic_data(synthetic_ohlcv):
    """Every chart builder must produce a plotly Figure without errors."""
    import plotly.graph_objects as go

    from components import charts

    figures = [
        charts.close_price_chart(synthetic_ohlcv),
        charts.volume_chart(synthetic_ohlcv),
        charts.candlestick_chart(synthetic_ohlcv),
        charts.sma_ema_chart(synthetic_ohlcv),
        charts.rsi_chart(synthetic_ohlcv),
        charts.macd_chart(synthetic_ohlcv),
        charts.volatility_chart(synthetic_ohlcv),
        charts.probability_gauge(0.62),
    ]
    assert all(isinstance(f, go.Figure) for f in figures)


def test_evaluation_charts_run_on_real_predictions():
    from components import charts

    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    y_proba = np.array([0.8, 0.4, 0.3, 0.2, 0.7, 0.6, 0.9, 0.8, 0.1, 0.7])

    cm_fig = charts.confusion_matrix_chart([[2, 1], [2, 5]], "Test")
    roc_fig = charts.roc_curve_chart(y_true, y_proba, "Test")
    dates = list(range(10))
    reg_fig = charts.actual_vs_predicted_chart(dates, y_true * 1000.0,
                                               y_pred * 1000.0, "Test")
    err_fig = charts.error_chart(y_true * 1000.0, y_pred * 1000.0)
    assert cm_fig is not None and roc_fig is not None
    assert reg_fig is not None and err_fig is not None


def test_no_absolute_paths_in_source():
    """Deployment rule: no machine-specific absolute paths in the codebase."""
    # Assembled to avoid matching this file itself.
    forbidden = ["/" + "home" + "/" + "z" + "/", "C:" + chr(92) + "Users" + chr(92)]
    offenders = []
    for folder in ("src", "app", "scripts", "config", "tests"):
        for py in (PROJECT_ROOT / folder).rglob("*.py"):
            text = py.read_text()
            for token in forbidden:
                if token in text:
                    offenders.append(str(py.relative_to(PROJECT_ROOT)))
    assert offenders == [], f"Hard-coded absolute paths found: {offenders}"
