"""Exercise the real Streamlit app with the official AppTest framework.

Boots app/app.py and clicks through all seven sidebar pages, asserting no
uncaught exceptions occur. Requires the project artifacts (skipped with a
clear message when they are absent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_PRESENT = (PROJECT_ROOT / "models" / "artifacts" / "training_metadata.json").exists()

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="model artifacts not trained yet (run scripts/train_models.py)",
)


def run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=120)
    at.run()
    return at


def assert_no_exception(at):
    assert not at.exception, [str(e.value) for e in at.exception]


def test_default_page_market_overview():
    at = run_app()
    assert_no_exception(at)
    assert any("Market Overview" in m.value for m in at.title)


@pytest.mark.parametrize(
    "page",
    [
        "Market Overview",
        "Historical Price Analysis",
        "Technical Indicators",
        "Bitcoin Price Prediction",
        "Market Direction Prediction",
        "Model Comparison",
        "Model Evaluation",
    ],
)
def test_every_page_renders_without_exceptions(page):
    at = run_app()
    at.sidebar.radio[0].set_value(page).run()
    assert_no_exception(at)
    assert any(page.split()[0] in t.value for t in at.title), (
        f"page '{page}' did not render its title"
    )


def test_sidebar_refresh_button_exists():
    at = run_app()
    labels = [b.label for b in at.sidebar.button]
    assert any("Refresh market data" in lbl for lbl in labels)
