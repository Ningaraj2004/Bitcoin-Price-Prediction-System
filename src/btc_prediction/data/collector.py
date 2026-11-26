"""Download, validate, clean and persist BTC-USD daily OHLCV data.

Design notes
------------
* Raw downloads are stored separately from cleaned/processed data
  (``data/raw`` vs ``data/processed``) so the original Yahoo Finance payload
  is always auditable.
* The incomplete candle of the current day (still forming) is excluded so
  models never train on a partial day.
* All failures are wrapped in :class:`DataDownloadError` with actionable
  messages; the dashboard can then show a helpful notice instead of crashing.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import pandas as pd

try:  # yfinance is only needed when actually downloading
    import yfinance as yf
except ImportError as _exc:  # pragma: no cover
    yf = None
    _YF_IMPORT_ERROR = _exc
else:
    _YF_IMPORT_ERROR = None

from config.settings import (
    DATA_START_DATE,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TICKER,
    ensure_directories,
    get_logger,
)

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class DataDownloadError(RuntimeError):
    """Raised when market data cannot be downloaded or is unusable."""


class DataValidationError(ValueError):
    """Raised when downloaded data fails validation."""


def _require_yfinance() -> None:
    if yf is None or _YF_IMPORT_ERROR is not None:  # pragma: no cover
        raise DataDownloadError(
            "yfinance is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        )


def fetch_ohlcv(
    ticker: str = TICKER,
    start: str = DATA_START_DATE,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Yahoo Finance symbol, default ``BTC-USD``.
    start, end:
        Date range strings (``YYYY-MM-DD``). ``end=None`` means "up to now".

    Returns
    -------
    pd.DataFrame indexed by date with columns
    ``Open, High, Low, Close, Volume``.

    Raises
    ------
    DataDownloadError
        If the network request fails or the response is empty.
    """
    _require_yfinance()
    logger.info("Downloading %s daily data from %s to %s ...", ticker, start, end or "now")
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
    except Exception as exc:  # network errors, Yahoo downtime, rate limits
        raise DataDownloadError(
            f"Could not download {ticker} data from Yahoo Finance: {exc}. "
            "Check your internet connection and try again, or reuse the cached "
            "CSV in data/raw/."
        ) from exc

    if df is None or df.empty:
        raise DataDownloadError(
            f"Yahoo Finance returned no data for {ticker} between {start} and "
            f"{end or 'now'}. Verify the ticker symbol and date range."
        )

    # Newer yfinance versions return MultiIndex columns even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, [c for c in REQUIRED_COLUMNS if c in df.columns]]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataDownloadError(
            f"Downloaded data is missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}."
        )
    logger.info("Downloaded %d daily rows.", len(df))
    return df


def validate_ohlcv(df: pd.DataFrame) -> dict:
    """Validate a raw OHLCV frame and return a quality report.

    Fatal problems raise :class:`DataValidationError`; recoverable issues are
    reported in the returned dictionary so callers can log them.
    """
    if df is None or df.empty:
        raise DataValidationError("OHLCV data is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("OHLCV data must be indexed by date.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"OHLCV data is missing columns: {missing}.")

    report = {
        "n_rows": int(len(df)),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "unsorted": bool(not df.index.is_monotonic_increasing),
        "missing_values": {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS},
        "non_positive_prices": int(
            (df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()
        ),
        "high_below_low": int((df["High"] < df["Low"]).sum()),
        "negative_volume": int((df["Volume"] < 0).sum()),
    }
    logger.info("Data quality report: %s", report)
    return report


def clean_ohlcv(df: pd.DataFrame, drop_incomplete_current_day: bool = True) -> pd.DataFrame:
    """Clean an OHLCV frame: sort, deduplicate, fill gaps, drop partial candle.

    Steps (in order):
    1. Sort chronologically.
    2. Drop duplicated dates, keeping the last occurrence.
    3. Drop rows where the whole OHLC block is missing.
    4. Forward-fill isolated NaNs, then drop any row still containing NaN.
    5. Remove the current day's still-forming candle when requested.

    The result has float columns and a sorted, unique DatetimeIndex.
    """
    if df is None or df.empty:
        raise DataValidationError("Cannot clean an empty OHLCV frame.")

    out = df.copy()
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]

    ohlc = ["Open", "High", "Low", "Close"]
    out = out.dropna(subset=ohlc, how="all")
    # Forward-fill isolated gaps (Yahoo occasionally returns empty days);
    # leading NaNs cannot be filled, so they are dropped.
    out = out[["Open", "High", "Low", "Close", "Volume"]].ffill().dropna()

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    out = out.dropna()

    if drop_incomplete_current_day:
        today_utc = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        if len(out) and out.index.max() >= today_utc:
            dropped_date = out.index.max()
            out = out[out.index < today_utc]
            logger.info(
                "Dropped incomplete current-day candle dated %s.", dropped_date.date()
            )
    return out


def save_raw(df: pd.DataFrame, path=RAW_DATA_PATH) -> None:
    """Persist the untouched download as CSV."""
    ensure_directories()
    out = df.copy()
    out.index.name = "Date"
    out.to_csv(path, index=True, index_label="Date")
    logger.info("Raw data saved to %s (%d rows).", path, len(out))


def save_processed(df: pd.DataFrame, path=PROCESSED_DATA_PATH) -> None:
    """Persist the cleaned dataset as CSV."""
    ensure_directories()
    out = df.copy()
    out.index.name = "Date"
    out.to_csv(path, index=True, index_label="Date")
    logger.info("Processed data saved to %s (%d rows).", path, len(out))


def load_processed(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset, parsing the date index."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run "
            "`python scripts/download_data.py` first, or use the dashboard's "
            "'Refresh market data' button."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    return df


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw (uncleaned) download, parsing the date index."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python scripts/download_data.py`."
        )
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def download_and_save(
    ticker: str = TICKER,
    start: str = DATA_START_DATE,
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full collection step: download, validate, clean, save raw + processed."""
    raw = fetch_ohlcv(ticker=ticker, start=start, end=end)
    validate_ohlcv(raw)
    save_raw(raw)

    clean = clean_ohlcv(raw)
    if clean.empty:
        raise DataValidationError(
            "Cleaning removed every row; the raw download appears unusable."
        )
    validate_ohlcv(clean)
    save_processed(clean)
    logger.info(
        "Collection finished: %d usable daily rows (%s to %s).",
        len(clean),
        clean.index.min().date(),
        clean.index.max().date(),
    )
    return raw, clean


def load_or_download(force_refresh: bool = False) -> pd.DataFrame:
    """Return the processed dataset, downloading only when needed.

    * ``force_refresh=True`` re-downloads from Yahoo Finance.
    * Otherwise the cached CSV is used when it exists (keeps the dashboard
      fast and resilient to Yahoo outages); if it is missing, a download is
      attempted once and the result cached for next time.
    """
    if not force_refresh and PROCESSED_DATA_PATH.exists():
        return load_processed()
    _, clean = download_and_save()
    return clean
