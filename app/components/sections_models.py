"""Dashboard sections: Price Prediction, Direction Prediction, Model Comparison, Model Evaluation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from components import charts
from components.loaders import (
    get_predictor,
    get_test_predictions,
    load_dataset_with_targets,
)

DISCLAIMER = (
    "⚠️ **Disclaimer:** predictions are statistical estimates from an "
    "educational model, not financial advice. Cryptocurrency markets are "
    "extremely volatile and often unpredictable."
)


def render_price_prediction() -> None:
    st.title("💰 Bitcoin Price Prediction")
    predictor = get_predictor()
    available = predictor.available_regression_models()
    if not available:
        st.warning(
            "No regression artifacts found. Train the models first with "
            "`python scripts/train_models.py`."
        )
        return

    model_names = available + (["LSTM (Deep Learning)"] if predictor.lstm_available() else [])
    col1, _ = st.columns([1, 2])
    model_name = col1.selectbox("Regression model", model_names)

    if st.button("🔮 Predict next-day close", use_container_width=True):
        try:
            result = predictor.predict_next_day_close(model_name)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.subheader("Prediction result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted next-day close", f"${result['predicted_close']:,.2f}")
        c2.metric(
            "Implied change",
            f"{result['predicted_change_pct']:+.2f}%",
            f"${result['predicted_close'] - result['last_close']:+,.2f} "
            f"vs last close ${result['last_close']:,.2f}",
            delta_color="normal",
        )
        c3.metric("Applies to", str(result["prediction_date"].date()))
        st.caption(
            f"Model: **{result['model_name']}** · Last close "
            f"(${result['last_close']:,.2f}) on {result['last_date'].date()} · "
            f"Computed {result['as_of']}"
        )
        st.info(DISCLAIMER, icon="⚠️")

    st.divider()
    st.subheader("Historical performance: actual vs predicted (test set)")
    st.caption(
        "Genuine out-of-sample predictions on the held-out chronological test "
        "split — the model never saw these days during training."
    )
    try:
        dates, y_true, y_pred, _ = get_test_predictions(model_name, "regression")
    except Exception as exc:
        st.warning(f"Could not compute test predictions: {exc}")
        return

    st.plotly_chart(
        charts.actual_vs_predicted_chart(dates, y_true, y_pred, model_name),
        use_container_width=True,
    )
    from src.btc_prediction.evaluation.metrics import regression_metrics

    metrics = regression_metrics(y_true, y_pred)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"${metrics['MAE']:,.2f}")
    m2.metric("RMSE", f"${metrics['RMSE']:,.2f}")
    m3.metric("R²", f"{metrics['R2']:.4f}")
    last_close = float(y_true[-1])
    direction_hits = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    m4.metric("Directional hit rate", f"{direction_hits:.1%}")


def render_direction_prediction() -> None:
    st.title("🎯 Market Direction Prediction")
    predictor = get_predictor()
    available = predictor.available_classification_models()
    if not available:
        st.warning(
            "No classification artifacts found. Train the models first with "
            "`python scripts/train_models.py`."
        )
        return

    model_name = st.selectbox("Classification model", available)
    if st.button("🎲 Predict market direction", use_container_width=True):
        try:
            result = predictor.predict_direction(model_name)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.subheader(f"Prediction: **{result['direction']}** 🚀" if result["direction"] == "UP"
                     else f"Prediction: **{result['direction']}** 📉")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(charts.probability_gauge(result["prob_up"]),
                            use_container_width=True)
        with col_b:
            st.metric("Probability UP", f"{result['prob_up']:.1%}")
            st.metric("Probability DOWN", f"{result['prob_down']:.1%}")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.caption(
                f"Applies to {result['prediction_date'].date()} · last close "
                f"${result['last_close']:,.2f} on {result['last_date'].date()}"
            )

        st.info(result["explanation"], icon="🧠")
        st.info(DISCLAIMER, icon="⚠️")

    st.divider()
    st.subheader("Model performance on the test set")
    try:
        dates, y_true, y_pred, y_proba = get_test_predictions(model_name, "classification")
    except Exception as exc:
        st.warning(f"Could not compute test predictions: {exc}")
        return
    from src.btc_prediction.evaluation.metrics import classification_metrics

    metrics = classification_metrics(y_true, y_pred, y_proba)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    c2.metric("Precision", f"{metrics['Precision']:.1%}")
    c3.metric("Recall", f"{metrics['Recall']:.1%}")
    c4.metric("F1", f"{metrics['F1']:.1%}")
    auc = metrics.get("ROC_AUC")
    c5.metric("ROC-AUC", f"{auc:.3f}" if auc == auc else "n/a")
    st.caption("Note: for daily BTC direction, ~50% accuracy is the random-guess "
               "baseline; the majority-class baseline is shown on the Model "
               "Comparison page for honest context.")


def render_model_comparison() -> None:
    st.title("⚖️ Model Comparison")
    from config.settings import ARTIFACTS_DIR
    from src.btc_prediction.evaluation.reporting import (
        build_classification_comparison,
        build_regression_comparison,
    )

    reg = build_regression_comparison()
    clf = build_classification_comparison()

    if reg is None and clf is None:
        st.warning("No evaluation metrics found — train the models first with "
                   "`python scripts/train_models.py`.")
        return

    st.subheader("Regression: next-day close prediction (test set)")
    if reg is not None:
        st.dataframe(
            reg.style.format({c: "{:.2f}" for c in ("MAE", "MSE", "RMSE", "R2")}),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(
            charts.metric_bar_chart(reg, "MAE",
                                    "Test-set MAE — lower is better (grey = baseline)",
                                    higher_is_better=False),
            use_container_width=True,
        )
        st.plotly_chart(
            charts.metric_bar_chart(reg, "R2",
                                    "Test-set R² — higher is better (grey = baseline, blue = LSTM)",
                                    higher_is_better=True),
            use_container_width=True,
        )
        st.caption(
            "The **Persistence Baseline** predicts tomorrow's close = today's "
            "close. Any useful model must beat it; LSTM (blue) is the deep-"
            "learning entry."
        )

    st.divider()
    st.subheader("Classification: UP/DOWN direction (test set)")
    if clf is not None:
        st.dataframe(
            clf.style.format({c: "{:.3f}" for c in ("Accuracy", "Precision", "Recall", "F1", "ROC_AUC")}),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(
            charts.metric_bar_chart(clf, "F1",
                                    "Test-set F1 — higher is better (grey = majority-class baseline)",
                                    higher_is_better=True),
            use_container_width=True,
        )
        st.caption(
            "The **Majority-Class Baseline** always predicts the most frequent "
            "training class. Accuracy near that baseline means the model adds "
            "little directional signal — a realistic outcome for daily crypto "
            "direction."
        )


def render_model_evaluation() -> None:
    st.title("🔬 Model Evaluation")
    predictor = get_predictor()
    clf_models = predictor.available_classification_models()
    reg_models = predictor.available_regression_models()
    if not clf_models and not reg_models:
        st.warning("No artifacts found — run `python scripts/train_models.py` first.")
        return

    tab_reg, tab_clf = st.tabs(["Regression: actual vs predicted & errors",
                                "Classification: confusion matrix & ROC"])

    with tab_reg:
        if not reg_models:
            st.warning("No regression artifacts.")
            return
        model_name = st.selectbox("Regression model", reg_models)
        try:
            dates, y_true, y_pred, _ = get_test_predictions(model_name, "regression")
        except Exception as exc:
            st.error(f"Could not compute predictions: {exc}")
            return
        st.plotly_chart(
            charts.actual_vs_predicted_chart(dates, y_true, y_pred, model_name),
            use_container_width=True,
        )
        st.plotly_chart(charts.error_chart(y_true, y_pred), use_container_width=True)
        from src.btc_prediction.evaluation.metrics import regression_metrics

        metrics = regression_metrics(y_true, y_pred)
        st.markdown(
            f"**MAE** ${metrics['MAE']:,.2f} · **RMSE** ${metrics['RMSE']:,.2f} · "
            f"**R²** {metrics['R2']:.4f} · **MSE** {metrics['MSE']:,.0f}"
        )
        st.caption(
            "Error analysis: residuals should center on zero with no obvious "
            "trend; clustering of large errors typically happens during "
            "high-volatility regimes."
        )

    with tab_clf:
        if not clf_models:
            st.warning("No classification artifacts.")
            return
        model_name = st.selectbox("Classification model", clf_models)
        try:
            dates, y_true, y_pred, y_proba = get_test_predictions(model_name, "classification")
        except Exception as exc:
            st.error(f"Could not compute predictions: {exc}")
            return
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        col1, col2 = st.columns(2)
        col1.plotly_chart(charts.confusion_matrix_chart(cm, model_name),
                          use_container_width=True)
        if y_proba is not None and len(np.unique(y_true)) > 1:
            col2.plotly_chart(charts.roc_curve_chart(y_true, y_proba, model_name),
                              use_container_width=True)
        st.caption(
            "Confusion matrix on the held-out test set: rows are actual "
            "directions, columns are predictions. ROC-AUC > 0.5 means better "
            "than random ranking of UP probabilities."
        )
