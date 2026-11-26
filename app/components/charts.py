"""Reusable Plotly chart builders for the dashboard (consistent styling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BTC_ORANGE = "#F7931A"
BLUE = "#3B82F6"
GREEN = "#22C55E"
RED = "#EF4444"
GREY = "#9CA3AF"

_AXIS_FONT = dict(size=12)


def _style(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(title_font=_AXIS_FONT)
    fig.update_yaxes(title_font=_AXIS_FONT)
    return fig


def candlestick_chart(df: pd.DataFrame, last_n: int = 180) -> go.Figure:
    """Interactive candlestick chart of the most recent days."""
    view = df.tail(last_n)
    fig = go.Figure(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            name="BTC-USD",
        )
    )
    fig.update_layout(title=f"BTC-USD candlestick (last {len(view)} days)")
    fig.update_xaxes(rangeslider_visible=False)
    return _style(fig, height=520)


def close_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(color=BLUE, width=1.6),
        )
    )
    fig.update_layout(title="BTC-USD closing price (full history)")
    fig.update_yaxes(tickformat="$,.0f")
    return _style(fig)


def volume_chart(df: pd.DataFrame) -> go.Figure:
    colors = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
    fig = go.Figure(
        go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume")
    )
    fig.update_layout(title="BTC-USD daily trading volume")
    fig.update_yaxes(tickformat=".2s", title="Volume (BTC)")
    return _style(fig, height=380)


def sma_ema_chart(df: pd.DataFrame, last_n: int = 365) -> go.Figure:
    from src.btc_prediction.features.indicators import add_ema, add_sma

    view = add_sma(add_ema(df)).tail(last_n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Close",
                             line=dict(color=GREY, width=1.2)))
    for col, color in (("SMA_7", BLUE), ("SMA_14", GREEN),
                       ("SMA_30", BTC_ORANGE), ("EMA_12", "#A855F7"),
                       ("EMA_26", "#EC4899")):
        fig.add_trace(go.Scatter(x=view.index, y=view[col], name=col,
                                 line=dict(width=1.3)))
    fig.update_layout(title="Moving averages (SMA 7/14/30, EMA 12/26)")
    fig.update_yaxes(tickformat="$,.0f")
    return _style(fig)


def rsi_chart(df: pd.DataFrame, last_n: int = 365) -> go.Figure:
    from src.btc_prediction.features.indicators import add_rsi

    view = add_rsi(df).tail(last_n)
    rsi = view["RSI_14"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=rsi, name="RSI(14)",
                             line=dict(color=BLUE, width=1.5)))
    fig.add_hline(y=70, line_dash="dash", line_color=RED,
                  annotation_text="Overbought 70")
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN,
                  annotation_text="Oversold 30")
    fig.update_layout(title="Relative Strength Index (RSI 14)")
    fig.update_yaxes(range=[0, 100])
    return _style(fig, height=380)


def macd_chart(df: pd.DataFrame, last_n: int = 365) -> go.Figure:
    from src.btc_prediction.features.indicators import add_macd

    view = add_macd(df).tail(last_n)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=view.index, y=view["MACD_Hist"], name="MACD histogram",
                         marker_color=[GREEN if v >= 0 else RED for v in view["MACD_Hist"]],
                         opacity=0.5))
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD"], name="MACD",
                             line=dict(color=BLUE, width=1.5)))
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD_Signal"], name="Signal",
                             line=dict(color=BTC_ORANGE, width=1.3)))
    fig.update_layout(title="MACD (12, 26, 9)")
    return _style(fig, height=400)


def volatility_chart(df: pd.DataFrame, last_n: int = 365) -> go.Figure:
    from src.btc_prediction.features.indicators import add_rolling_volatility

    view = add_rolling_volatility(df).tail(last_n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["Rolling_Volatility_7"],
                             name="Volatility 7d", line=dict(color=BLUE, width=1.4)))
    fig.add_trace(go.Scatter(x=view.index, y=view["Rolling_Volatility_30"],
                             name="Volatility 30d", line=dict(color=BTC_ORANGE, width=1.4)))
    fig.update_layout(title="Rolling volatility of daily returns (%)")
    fig.update_yaxes(title="Std-dev of daily return (%)")
    return _style(fig, height=380)


def actual_vs_predicted_chart(dates, y_true, y_pred, model_name: str) -> go.Figure:
    """Test-set actual vs predicted next-day close."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, name="Actual", mode="lines",
                             line=dict(color=BLUE, width=1.8)))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, name="Predicted", mode="lines",
                             line=dict(color=BTC_ORANGE, width=1.5)))
    fig.update_layout(title=f"{model_name}: actual vs predicted next-day close (test set)")
    fig.update_yaxes(tickformat="$,.0f")
    return _style(fig, height=420)


def error_chart(y_true, y_pred) -> go.Figure:
    """Residual scatter + histogram in one figure."""
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residual distribution", "Residuals vs actual"))
    fig.add_trace(go.Histogram(x=residuals, nbinsx=40, marker_color=BTC_ORANGE,
                               name="Errors"), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_true, y=residuals, mode="markers", name="Days",
                             marker=dict(size=5, color=BLUE, opacity=0.5)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
    fig.update_xaxes(title_text="Prediction error (USD)", row=1, col=1)
    fig.update_xaxes(title_text="Actual next-day close (USD)", row=1, col=2)
    fig.update_yaxes(title_text="Days", row=1, col=1)
    fig.update_yaxes(title_text="Error (USD)", row=1, col=2)
    return _style(fig, height=380)


def confusion_matrix_chart(cm, model_name: str) -> go.Figure:
    cm = np.asarray(cm)
    labels = ["DOWN (0)", "UP (1)"]
    fig = go.Figure(
        go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Oranges",
            text=cm, texttemplate="%{text}", textfont=dict(size=18, color="black"),
            showscale=False,
        )
    )
    fig.update_layout(title=f"{model_name}: confusion matrix (test set)")
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height=400)


def roc_curve_chart(y_true, y_proba, model_name: str) -> go.Figure:
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_proba))
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC = {roc_auc:.3f})",
                             line=dict(color=BTC_ORANGE, width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random guessing",
                             line=dict(color=GREY, dash="dash", width=1.5)))
    fig.update_layout(title=f"{model_name}: ROC curve (test set)")
    fig.update_xaxes(title_text="False positive rate")
    fig.update_yaxes(title_text="True positive rate")
    return _style(fig, height=420)


def metric_bar_chart(comparison_df: pd.DataFrame, metric: str, title: str,
                     higher_is_better: bool = True) -> go.Figure:
    """Horizontal bars comparing one metric across models (baselines greyed)."""
    data = comparison_df.dropna(subset=[metric]).copy()
    data = data.sort_values(metric, ascending=not higher_is_better)
    colors = [
        GREY if "Baseline" in m else BLUE if "LSTM" in m else BTC_ORANGE
        for m in data["Model"]
    ]
    fig = go.Figure(
        go.Bar(
            x=data[metric], y=data["Model"], orientation="h",
            marker_color=colors, text=[f"{v:.4g}" for v in data[metric]],
            textposition="outside",
        )
    )
    fig.update_layout(title=title)
    fig.update_xaxes(title_text=metric)
    fig.update_layout(height=max(260, 46 * len(data) + 90))
    return _style(fig, height=max(280, 46 * len(data) + 90))


def probability_gauge(prob_up: float) -> go.Figure:
    """Semicircle gauge showing the probability of an UP day."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_up * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Probability of UP day", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": BTC_ORANGE},
                "steps": [
                    {"range": [0, 50], "color": "#7F1D1D"},
                    {"range": [50, 100], "color": "#14532D"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "value": 50},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=30, b=20))
    return fig
