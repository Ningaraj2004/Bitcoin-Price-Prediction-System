"""Dashboard sections: Market Overview, Historical Price Analysis, Technical Indicators."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import charts
from components.loaders import load_market_data


def render_market_overview() -> None:
    st.title("📈 Market Overview")
    data = load_market_data()
    if data.empty:
        st.error("No market data available. Use the sidebar 'Refresh market data'.")
        return

    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    day_change = latest["Close"] - prev["Close"]
    day_change_pct = day_change / prev["Close"] * 100.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Latest BTC-USD Close",
        f"${latest['Close']:,.2f}",
        f"{day_change_pct:+.2f}% ({day_change:+,.2f})",
        delta_color="normal",
    )
    col2.metric("24h Open", f"${latest['Open']:,.2f}")
    col3.metric("24h High / Low",
                f"${latest['High']:,.2f} / ${latest['Low']:,.2f}")
    col4.metric("Volume", f"{latest['Volume']:,.0f} BTC")

    delta = "▲" if day_change_pct >= 0 else "▼"
    st.caption(
        f"{delta} Data as of **{data.index.max().date()}** — daily candles, "
        f"incomplete current day excluded ({len(data):,} days of history)."
    )

    st.plotly_chart(charts.close_price_chart(data), use_container_width=True)
    st.plotly_chart(charts.volume_chart(data.tail(180)), use_container_width=True)

    st.subheader("Key market statistics")
    stats = pd.DataFrame(
        {
            "Value": {
                "Days of history": f"{len(data):,}",
                "History start": str(data.index.min().date()),
                "All-time high (in data)": f"${data['High'].max():,.2f}",
                "All-time low (in data)": f"${data['Low'].min():,.2f}",
                "Latest close": f"${latest['Close']:,.2f}",
                "30-day return": f"{(latest['Close'] / data['Close'].iloc[-31] - 1) * 100:+.2f}%" if len(data) > 30 else "n/a",
                "365-day return": f"{(latest['Close'] / data['Close'].iloc[-366] - 1) * 100:+.2f}%" if len(data) > 366 else "n/a",
                "Avg daily volume (30d)": f"{data['Volume'].tail(30).mean():,.0f} BTC",
                "Annualized volatility (30d)": f"{data['Close'].pct_change().tail(30).std() * (365 ** 0.5) * 100:.1f}%",
            }
        }
    )
    st.dataframe(stats, use_container_width=True, height=360)


def render_historical_analysis() -> None:
    st.title("🕯️ Historical Price Analysis")
    data = load_market_data()

    col1, col2 = st.columns([1, 1])
    default_start = data.index.max() - pd.Timedelta(days=365)
    start_date, end_date = col1.date_input(
        "Date range",
        value=(default_start.date(), data.index.max().date()),
        min_value=data.index.min().date(),
        max_value=data.index.max().date(),
    )
    view_option = col2.selectbox(
        "Candlestick window",
        ["Last 90 days", "Last 180 days", "Last 365 days", "All"],
        index=1,
    )
    n_map = {"Last 90 days": 90, "Last 180 days": 180,
             "Last 365 days": 365, "All": len(data)}

    if isinstance(start_date, tuple) and len(start_date) == 2:
        view = data.loc[pd.Timestamp(start_date[0]): pd.Timestamp(end_date[1])]
    else:
        view = data
    if view.empty:
        st.warning("No data in the selected range.")
        return

    st.plotly_chart(
        charts.candlestick_chart(view, last_n=n_map[view_option]),
        use_container_width=True,
    )
    st.plotly_chart(charts.close_price_chart(view), use_container_width=True)
    st.plotly_chart(charts.volume_chart(view), use_container_width=True)


def render_technical_indicators() -> None:
    st.title("📐 Technical Indicators")
    data = load_market_data()

    st.markdown(
        "All indicators are computed from the daily close using only past "
        "and current data (no look-ahead), exactly as the prediction models see them."
    )

    tab_sma, tab_rsi, tab_macd, tab_vol = st.tabs(
        ["SMA / EMA", "RSI", "MACD", "Volatility"]
    )
    with tab_sma:
        st.plotly_chart(charts.sma_ema_chart(data), use_container_width=True)
        st.caption("SMA = simple moving average, EMA = exponential moving average "
                   "of the daily close. Golden/death-cross watchers: compare SMA 7 vs SMA 30.")
    with tab_rsi:
        st.plotly_chart(charts.rsi_chart(data), use_container_width=True)
        st.caption("RSI > 70 suggests overbought conditions; RSI < 30 suggests oversold.")
    with tab_macd:
        st.plotly_chart(charts.macd_chart(data), use_container_width=True)
        st.caption("MACD line above its signal line suggests bullish momentum.")
    with tab_vol:
        st.plotly_chart(charts.volatility_chart(data), use_container_width=True)
        st.caption("Rolling standard deviation of daily percentage returns — "
                   "higher values mean more turbulent markets.")

    st.subheader("Latest indicator values")
    from src.btc_prediction.features.build_features import features_for_last_row

    try:
        latest = features_for_last_row(data).iloc[0]
        summary = pd.DataFrame(
            {
                "Value": {
                    "Close": f"${latest['Close']:,.2f}",
                    "SMA 7 / 14 / 30": ", ".join(f"${latest[k]:,.0f}" for k in ("SMA_7", "SMA_14", "SMA_30")),
                    "EMA 12 / 26": ", ".join(f"${latest[k]:,.0f}" for k in ("EMA_12", "EMA_26")),
                    "RSI (14)": f"{latest['RSI_14']:.1f}",
                    "MACD / Signal": f"{latest['MACD']:.1f} / {latest['MACD_Signal']:.1f}",
                    "Daily return": f"{latest['Daily_Return_Pct']:+.2f}%",
                    "Volatility 7d / 30d": f"{latest['Rolling_Volatility_7']:.2f}% / {latest['Rolling_Volatility_30']:.2f}%",
                    "High-Low range": f"${latest['High_Low_Range']:,.0f} ({latest['High_Low_Range_Pct']:.2f}%)",
                    "Open-close change": f"{latest['Open_Close_Pct']:+.2f}%",
                }
            }
        )
        st.dataframe(summary, use_container_width=True, height=300)
    except ValueError as exc:
        st.warning(str(exc))
