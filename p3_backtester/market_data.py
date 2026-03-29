"""
market_data.py — fetches and caches full OHLCV data for a backtest window.

Fetches enough extra history before start_date to provide the MIN_WARMUP_BARS
of data needed by bar_slicer.py before the first signal bar.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from p3_backtester.bar_slicer import MIN_WARMUP_BARS

# yfinance max lookback per interval (days) — mirrors p1 fetcher
_INTERVAL_LOOKBACK: dict[str, int] = {
    "1m":  6,
    "5m":  58,
    "15m": 58,
    "30m": 58,
    "1h":  720,
    "1d":  730,
    "1wk": 3650,
}

# Approximate bars per calendar day per interval (used to compute warmup window)
_BARS_PER_DAY: dict[str, float] = {
    "1m":  390,   # equities: 6.5h * 60
    "5m":  78,
    "15m": 26,
    "30m": 13,
    "1h":  6.5,
    "1d":  1,
    "1wk": 0.2,
}


class MarketDataError(Exception):
    pass


def fetch_ohlcv(ticker: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance for the backtest window, extended by enough
    history to provide MIN_WARMUP_BARS before start_date.

    Returns a DataFrame with DatetimeIndex, columns: Open, High, Low, Close, Volume.
    Raises MarketDataError if insufficient data is returned.
    """
    if interval not in _INTERVAL_LOOKBACK:
        raise MarketDataError(f"Unsupported interval '{interval}'")

    bars_per_day = _BARS_PER_DAY.get(interval, 1)
    warmup_days  = int(MIN_WARMUP_BARS / bars_per_day) + 10  # +10 day buffer for weekends/holidays

    start_dt  = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt    = datetime.strptime(end_date,   "%Y-%m-%d") + timedelta(days=1)  # inclusive end
    fetch_start = (start_dt - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    fetch_end   = end_dt.strftime("%Y-%m-%d")

    # Check yfinance lookback limit
    max_days = _INTERVAL_LOOKBACK[interval]
    total_days = (end_dt - datetime.strptime(fetch_start, "%Y-%m-%d")).days
    if total_days > max_days:
        raise MarketDataError(
            f"Requested range ({total_days} days including warmup) exceeds yfinance limit "
            f"for {interval} interval ({max_days} days). Use a longer interval or shorter date range."
        )

    df = yf.Ticker(ticker).history(start=fetch_start, end=fetch_end, interval=interval)

    if df.empty:
        raise MarketDataError(f"No data returned for {ticker} at {interval} interval")

    # Ensure tz-naive index for consistent slicing
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)

    return df


def split_backtest_window(df: pd.DataFrame, start_date: str) -> tuple[int, int]:
    """
    Returns (warmup_end_index, backtest_end_index) where:
    - warmup_end_index: first bar index on or after start_date (first valid signal bar)
    - backtest_end_index: len(df) - 1
    """
    start_dt = pd.Timestamp(start_date)
    warmup_end = int((df.index >= start_dt).argmax())
    if warmup_end == 0 and df.index[0] < start_dt:
        raise MarketDataError(f"start_date {start_date} is before available data")
    return warmup_end, len(df) - 1
