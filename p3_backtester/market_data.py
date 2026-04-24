"""
market_data.py — multi-source OHLCV fetcher with automatic fallback chain.

Source priority (per asset class):
  Equities / Indices / Commodities:
    1. Stooq (pandas_datareader) — free, no API key, 10-20 year daily history
    2. yfinance                  — fallback; 1H capped at 729 days
  Crypto:
    1. CCXT / Binance            — free, full history since listing, all intervals
    2. yfinance                  — fallback

Ticker mapping:
  Stooq uses different symbols — e.g. GC=F -> GC.F, ^GDAXI -> ^DAX
  CCXT uses "BTC/USDT" style — BTC-USD -> BTC/USDT on Binance

Environment variables (optional):
  BINANCE_API_KEY, BINANCE_API_SECRET — for higher rate limits (not required for data)
"""
import os
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from p3_backtester.bar_slicer import MIN_WARMUP_BARS

warnings.filterwarnings("ignore", category=FutureWarning)

# ── yfinance lookback limits (days) ──────────────────────────────────────────
_YF_LOOKBACK: dict[str, int] = {
    "1m":  6,
    "5m":  58,
    "15m": 58,
    "30m": 58,
    "1h":  729,
    "4h":  729,
    "1d":  10000,
    "1wk": 50000,
}

# Approximate bars per calendar day per interval
_BARS_PER_DAY: dict[str, float] = {
    "1m":  390,
    "5m":  78,
    "15m": 26,
    "30m": 13,
    "1h":  6.5,
    "4h":  1.625,  # ~6.5 / 4
    "1d":  0.69,   # ~252 trading days / 365 calendar days
    "1wk": 0.14,   # ~52 weeks / 365 calendar days
}

# ── Stooq ticker mapping ──────────────────────────────────────────────────────
# Stooq has its own symbol format; map yfinance tickers to stooq equivalents.
_STOOQ_MAP: dict[str, str] = {
    "GC=F":    "GC.F",       # Gold futures
    "SI=F":    "SI.F",       # Silver futures
    "CL=F":    "CL.F",       # WTI Crude
    "BZ=F":    "BZ.F",       # Brent Crude
    "^GSPC":   "^SPX",       # S&P 500
    "^DJI":    "^DJI",       # Dow Jones
    "^NDX":    "^NDX",       # NASDAQ 100
    "^GDAXI":  "^DAX",       # DAX
    "AAPL":    "AAPL.US",
    "MSFT":    "MSFT.US",
    "TSLA":    "TSLA.US",
}

# ── CCXT / Binance ticker mapping ─────────────────────────────────────────────
_CCXT_MAP: dict[str, str] = {
    "BTC-USD":  "BTC/USDT",
    "ETH-USD":  "ETH/USDT",
    "BNB-USD":  "BNB/USDT",
    "SOL-USD":  "SOL/USDT",
    "XRP-USD":  "XRP/USDT",
    "DOGE-USD": "DOGE/USDT",
}

_CCXT_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "1d": "1d", "1wk": "1w",
}

_STOOQ_INTERVAL_MAP: dict[str, str] = {
    "1d":  "d",
    "1wk": "w",
    "1m":  None,   # stooq does not support intraday
    "5m":  None,
    "15m": None,
    "30m": None,
    "1h":  None,
}


class MarketDataError(Exception):
    pass


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    ticker: str,
    interval: str,
    start_date: str,
    end_date: str,
    source: str = "auto",
) -> pd.DataFrame:
    """
    Fetch OHLCV data, extended by MIN_WARMUP_BARS of history before start_date.

    Parameters
    ----------
    ticker     : yfinance-style ticker (GC=F, BTC-USD, ^NDX, etc.)
    interval   : 1m, 5m, 15m, 30m, 1h, 1d, 1wk
    start_date : backtest start (warmup is added automatically)
    end_date   : backtest end (inclusive)
    source     : "auto" | "yfinance" | "stooq" | "ccxt"

    Returns DataFrame with DatetimeIndex, columns Open/High/Low/Close/Volume.
    Raises MarketDataError if no data could be fetched from any source.
    """
    fetch_start, fetch_end = _compute_fetch_range(ticker, interval, start_date, end_date)

    is_crypto = ticker in _CCXT_MAP or ticker.endswith("-USD") or "BTC" in ticker or "ETH" in ticker

    if source == "auto":
        sources = _auto_source_order(ticker, interval, is_crypto)
    elif source == "eodhd":
        # Delegate directly to core.data EODHD provider
        from core.data import _fetch_eodhd
        df = _fetch_eodhd(ticker, interval, fetch_start, fetch_end)
        return _normalise(df)
    else:
        sources = [source]

    last_error = None
    for src in sources:
        try:
            df = _fetch_from(src, ticker, interval, fetch_start, fetch_end, is_crypto)
            if df is not None and not df.empty:
                df = _normalise(df)
                if len(df) >= MIN_WARMUP_BARS:
                    return df
        except Exception as e:
            last_error = e
            continue

    raise MarketDataError(
        f"No data returned for {ticker} at {interval} from any source. "
        f"Last error: {last_error}"
    )


def split_backtest_window(df: pd.DataFrame, start_date: str) -> tuple[int, int]:
    """
    Returns (warmup_end_index, backtest_end_index) where:
    - warmup_end_index: first bar on or after start_date
    - backtest_end_index: len(df) - 1
    """
    start_dt = pd.Timestamp(start_date)
    warmup_end = int((df.index >= start_dt).argmax())
    if warmup_end == 0 and df.index[0] < start_dt:
        raise MarketDataError(f"start_date {start_date} is before available data")
    return warmup_end, len(df) - 1


def available_sources(ticker: str, interval: str) -> list[str]:
    """Return which sources are likely available for this ticker/interval combo."""
    is_crypto = ticker in _CCXT_MAP or ticker.endswith("-USD")
    return _auto_source_order(ticker, interval, is_crypto)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _auto_source_order(ticker: str, interval: str, is_crypto: bool) -> list[str]:
    """Priority order of data sources for this asset/interval."""
    if is_crypto:
        if interval in ("1d", "1wk", "1h", "30m", "15m", "5m", "1m") and ticker in _CCXT_MAP:
            return ["ccxt", "yfinance"]
        return ["yfinance"]

    # Stooq only supports daily/weekly
    stooq_ticker = _STOOQ_MAP.get(ticker)
    stooq_interval = _STOOQ_INTERVAL_MAP.get(interval)
    if stooq_ticker and stooq_interval:
        return ["stooq", "yfinance"]

    return ["yfinance"]


def _compute_fetch_range(ticker: str, interval: str, start_date: str, end_date: str) -> tuple[str, str]:
    """Compute actual fetch start (with warmup) and end."""
    bars_per_day = _BARS_PER_DAY.get(interval, 1)
    warmup_days  = int(MIN_WARMUP_BARS / bars_per_day) + 10

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d") + timedelta(days=1)

    fetch_start = (start_dt - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    fetch_end   = end_dt.strftime("%Y-%m-%d")

    # yfinance intraday limit check (1h cap at 729 days, 4h at 729 days too)
    _YF_INTRADAY = {"1h": 729, "4h": 729, "30m": 58, "15m": 58, "5m": 58, "1m": 6}
    if interval in _YF_INTRADAY:
        max_days = _YF_INTRADAY[interval]
        total_days = (end_dt - datetime.strptime(fetch_start, "%Y-%m-%d")).days
        if total_days > max_days:
            fetch_start = (end_dt - timedelta(days=max_days - 1)).strftime("%Y-%m-%d")

    return fetch_start, fetch_end


def _fetch_from(
    source: str,
    ticker: str,
    interval: str,
    fetch_start: str,
    fetch_end: str,
    is_crypto: bool,
) -> pd.DataFrame | None:
    if source == "yfinance":
        return _fetch_yfinance(ticker, interval, fetch_start, fetch_end)
    if source == "stooq":
        return _fetch_stooq(ticker, interval, fetch_start, fetch_end)
    if source == "ccxt":
        return _fetch_ccxt(ticker, interval, fetch_start, fetch_end)
    return None


def _fetch_yfinance(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, end=end, interval=interval)
    if df.empty:
        raise MarketDataError(f"yfinance returned empty data for {ticker}")
    return df


def _fetch_stooq(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch from Stooq via pandas_datareader.
    Stooq provides up to 10-20 years of daily/weekly data for free.
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        return None

    stooq_ticker   = _STOOQ_MAP.get(ticker, ticker)
    stooq_interval = _STOOQ_INTERVAL_MAP.get(interval)
    if not stooq_interval:
        return None   # stooq doesn't support intraday

    try:
        df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
    except Exception as e:
        raise MarketDataError(f"Stooq fetch failed for {stooq_ticker}: {e}")

    if df is None or df.empty:
        raise MarketDataError(f"Stooq returned empty data for {stooq_ticker}")

    # Stooq returns descending order — reverse to ascending
    df = df.sort_index(ascending=True)

    # Rename columns to match expected format
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low",
                             "Close": "Close", "Volume": "Volume"})
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    return df


def _fetch_ccxt(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch from Binance via CCXT.
    Supports all intervals; full history since asset listing.
    Paginates automatically to get full requested range.
    """
    try:
        import ccxt
    except ImportError:
        return None

    symbol = _CCXT_MAP.get(ticker)
    if not symbol:
        return None

    ccxt_interval = _CCXT_INTERVAL_MAP.get(interval)
    if not ccxt_interval:
        raise MarketDataError(f"CCXT: unsupported interval {interval}")

    api_key    = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    exchange = ccxt.binance({
        "apiKey":    api_key    or None,
        "secret":    api_secret or None,
        "enableRateLimit": True,
    })

    since_ms = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    end_ms   = int(datetime.strptime(end,   "%Y-%m-%d").timestamp() * 1000)

    all_candles = []
    limit = 1000   # Binance max per request

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, ccxt_interval, since=since_ms, limit=limit)
        except Exception as e:
            raise MarketDataError(f"CCXT Binance fetch failed for {symbol}: {e}")

        if not candles:
            break

        all_candles.extend(candles)

        last_ts = candles[-1][0]
        if last_ts >= end_ms or len(candles) < limit:
            break

        since_ms = last_ts + 1

    if not all_candles:
        raise MarketDataError(f"CCXT returned no candles for {symbol}")

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df[df.index < pd.Timestamp(end)]

    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure tz-naive DatetimeIndex and correct columns."""
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep only OHLCV columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)
    df = df[df["Close"] > 0]
    return df
