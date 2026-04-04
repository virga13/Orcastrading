"""
core/data.py — Provider-agnostic data fetcher.

Wraps p3_backtester.market_data.fetch_ohlcv and adds support for EODHD.
The active provider is selected via the DATA_PROVIDER environment variable:

    DATA_PROVIDER=yfinance   (default — free, limited intraday history)
    DATA_PROVIDER=eodhd      (recommended for intraday — requires EODHD_API_KEY)
    DATA_PROVIDER=ccxt       (crypto only)

Usage:
    from core.data import fetch
    from core.config import get_ticker

    ticker = get_ticker("gold")                       # resolves per-provider ticker
    df = fetch("gold", "15m", "2026-03-01", "2026-04-01")
"""
from __future__ import annotations

import os
import pandas as pd
from datetime import datetime, timedelta

from core.config import get_ticker, get_strategy_lookback


# ── Public API ────────────────────────────────────────────────────────────────

def fetch(
    asset_id: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    source: str | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for an asset, auto-selecting the right ticker for the provider.

    Parameters
    ----------
    asset_id   : asset id from config/assets.yaml  (e.g. "gold", "spx500")
    timeframe  : interval string                    (e.g. "1d", "15m", "1h")
    start_date : YYYY-MM-DD
    end_date   : YYYY-MM-DD
    source     : override provider  ("yfinance" | "eodhd" | "ccxt")
                 defaults to DATA_PROVIDER env var or "yfinance"

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns Open/High/Low/Close/Volume.
    """
    provider = (source or os.getenv("DATA_PROVIDER", "yfinance")).lower()
    ticker   = get_ticker(asset_id, source=provider)

    if provider == "eodhd":
        try:
            return _fetch_eodhd(ticker, timeframe, start_date, end_date)
        except RuntimeError as e:
            # Graceful fallback: EODHD failed (403 = plan limitation, 404 = bad ticker).
            # Fall through to yfinance so scanning still works while on free plan.
            import warnings
            warnings.warn(
                f"EODHD fetch failed ({e}), falling back to yfinance. "
                "Upgrade your EODHD plan at eodhd.com to unlock intraday history.",
                stacklevel=2,
            )

    # Default — delegate to the existing multi-source fetcher
    from p3_backtester.market_data import fetch_ohlcv
    return fetch_ohlcv(ticker, timeframe, start_date, end_date, source="auto")


def fetch_with_warmup(
    asset_id: str,
    strategy_id: str,
    start_date: str,
    end_date: str,
    source: str | None = None,
) -> pd.DataFrame:
    """
    Fetch data with the warmup period required by the given strategy automatically
    prepended. Useful for scanning so indicators are fully warmed up at start_date.
    """
    lookback = get_strategy_lookback(strategy_id)
    warmup_start = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=lookback)
    ).strftime("%Y-%m-%d")
    from core.config import get_strategy_timeframe
    timeframe = get_strategy_timeframe(strategy_id)
    return fetch(asset_id, timeframe, warmup_start, end_date, source=source)


# ── EODHD provider ────────────────────────────────────────────────────────────

def _fetch_eodhd(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV from EODHD Historical Data API.
    Requires EODHD_API_KEY environment variable.

    EODHD interval format:
      1d   → d (daily)
      1wk  → w (weekly)
      1h   → 1h
      30m  → 30m
      15m  → 15m
      5m   → 5m
      1m   → 1m

    Endpoint (intraday):
      https://eodhd.com/api/intraday/{ticker}?interval=15m&from=...&to=...&api_token=...
    Endpoint (daily/weekly):
      https://eodhd.com/api/eod/{ticker}?from=...&to=...&api_token=...&fmt=json

    Docs: https://eodhd.com/financial-apis/intraday-historical-data-api/
    """
    api_key = os.getenv("EODHD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "EODHD_API_KEY not set. Add it to your .env file.\n"
            "Get a free trial at https://eodhd.com/"
        )

    import urllib.request
    import json

    _EODHD_INTERVAL_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "1d": "d", "1wk": "w",
    }
    eodhd_interval = _EODHD_INTERVAL_MAP.get(interval)
    if not eodhd_interval:
        raise ValueError(f"EODHD: unsupported interval '{interval}'")

    is_intraday = interval not in ("1d", "1wk")

    if is_intraday:
        # Convert dates to unix timestamps for intraday endpoint
        from_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        to_ts   = int(datetime.strptime(end,   "%Y-%m-%d").timestamp())
        url = (
            f"https://eodhd.com/api/intraday/{ticker}"
            f"?interval={eodhd_interval}&from={from_ts}&to={to_ts}"
            f"&api_token={api_key}&fmt=json"
        )
    else:
        url = (
            f"https://eodhd.com/api/eod/{ticker}"
            f"?from={start}&to={end}&period={eodhd_interval}"
            f"&api_token={api_key}&fmt=json"
        )

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"EODHD fetch failed for {ticker}: {e}")

    if not data:
        raise RuntimeError(f"EODHD returned empty data for {ticker} {interval}")

    if is_intraday:
        # Intraday response: [{"timestamp": int, "open": f, "high": f, "low": f, "close": f, "volume": int}, ...]
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime")
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
    else:
        # Daily/weekly response: [{"date": "2024-01-02", "open": f, ...}, ...]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)
    df = df[df["Close"] > 0]
    df.sort_index(inplace=True)

    return df


# ── Utility ───────────────────────────────────────────────────────────────────

def active_provider() -> str:
    """Return the name of the currently configured data provider."""
    return os.getenv("DATA_PROVIDER", "yfinance").lower()


def provider_supports_intraday(provider: str | None = None) -> bool:
    """
    Return True if the active provider has meaningful intraday history.
    yfinance is limited to ~60 days of 15-min data; eodhd has years.
    """
    p = (provider or active_provider()).lower()
    return p in ("eodhd", "ccxt")
