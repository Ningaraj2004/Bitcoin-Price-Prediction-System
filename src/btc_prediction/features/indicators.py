"""Technical indicators computed from daily OHLCV data.

Every indicator in this module uses only information available up to and
including the current row's close, so appending them can never leak future
information. Lagged columns explicitly shift by at least one day.
"""

from __future__ import annotations

import pandas as pd

from config.settings import (
    CLOSE_LAGS,
    EMA_SPANS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    SMA_WINDOWS,
    VOLATILITY_WINDOWS,
    VOLUME_LAGS,
)


def add_sma(df: pd.DataFrame, windows=SMA_WINDOWS) -> pd.DataFrame:
    """Simple moving averages of the close price (include the current day)."""
    out = df.copy()
    for window in windows:
        out[f"SMA_{window}"] = out["Close"].rolling(window=window, min_periods=window).mean()
    return out


def add_ema(df: pd.DataFrame, spans=EMA_SPANS) -> pd.DataFrame:
    """Exponential moving averages of the close price."""
    out = df.copy()
    for span in spans:
        out[f"EMA_{span}"] = out["Close"].ewm(span=span, adjust=False).mean()
    return out


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Relative Strength Index using Wilder's exponential smoothing."""
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out[f"RSI_{period}"] = 100.0 - 100.0 / (1.0 + rs)
    out[f"RSI_{period}"] = out[f"RSI_{period}"].fillna(100.0).where(
        avg_loss.replace(0.0, pd.NA).notna() | avg_gain.notna(), out[f"RSI_{period}"]
    )
    # Pure-gain periods (no losses at all) give RSI = 100; handle the
    # all-flat edge case where both averages are zero.
    both_zero = (avg_gain.fillna(0.0) == 0.0) & (avg_loss.fillna(0.0) == 0.0)
    out.loc[both_zero, f"RSI_{period}"] = 50.0
    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD line, signal line (EMA of MACD) and histogram."""
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_Signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    return out


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage return of the close price."""
    out = df.copy()
    out["Daily_Return_Pct"] = out["Close"].pct_change() * 100.0
    return out


def add_rolling_volatility(df: pd.DataFrame, windows=VOLATILITY_WINDOWS) -> pd.DataFrame:
    """Rolling standard deviation of daily percentage returns (in %)."""
    out = df.copy()
    returns = out["Close"].pct_change() * 100.0
    for window in windows:
        out[f"Rolling_Volatility_{window}"] = returns.rolling(window=window, min_periods=window).std()
    return out


def add_price_range(df: pd.DataFrame) -> pd.DataFrame:
    """High-low range in absolute and percentage terms."""
    out = df.copy()
    out["High_Low_Range"] = out["High"] - out["Low"]
    out["High_Low_Range_Pct"] = (out["High"] - out["Low"]) / out["Close"] * 100.0
    return out


def add_open_close_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Open-to-close percentage change of the current day."""
    out = df.copy()
    out["Open_Close_Pct"] = (out["Close"] - out["Open"]) / out["Open"] * 100.0
    return out


def add_lags(df: pd.DataFrame, close_lags=CLOSE_LAGS, volume_lags=VOLUME_LAGS) -> pd.DataFrame:
    """Lagged closing prices and trading volumes (shifted, leak-free)."""
    out = df.copy()
    for lag in close_lags:
        out[f"Close_Lag_{lag}"] = out["Close"].shift(lag)
    for lag in volume_lags:
        out[f"Volume_Lag_{lag}"] = out["Volume"].shift(lag)
    return out


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator in a single call, returning a new frame."""
    out = add_sma(df)
    out = add_ema(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_daily_returns(out)
    out = add_rolling_volatility(out)
    out = add_price_range(out)
    out = add_open_close_pct(out)
    out = add_lags(out)
    return out
