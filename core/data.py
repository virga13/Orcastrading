"""
core/data.py — Provider-agnostic data fetcher.

Wraps p3_backtester.market_data.fetch_ohlcv and adds support for EODHD and Twelve Data.
The active provider is selected via the DATA_PROVIDER environment variable:

    DATA_PROVIDER=yfinance    (default — free, limited intraday history)
    DATA_PROVIDER=eodhd       (recommended for intraday — requires EODHD_API_KEY)
    DATA_PROVIDER=twelvedata  (recommended for Gold/Silver/Forex — requires TWELVEDATA_API_KEY)
    DATA_PROVIDER=ccxt        (crypto only)

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

    if provider == "twelvedata":
        try:
            return _fetch_twelvedata(ticker, timeframe, start_date, end_date)
        except RuntimeError as e:
            import warnings
            warnings.warn(
                f"Twelve Data fetch failed ({e}), falling back to yfinance.",
                stacklevel=2,
            )

    if provider == "eodhd":
        try:
            return _fetch_eodhd(ticker, timeframe, start_date, end_date)
        except RuntimeError as e:
            import warnings
            warnings.warn(
                f"EODHD fetch failed ({e}), falling back to yfinance. "
                "Upgrade your EODHD plan at eodhd.com to unlock intraday history.",
                stacklevel=2,
            )

    # Default — delegate to the existing multi-source fetcher.
    # Use the yfinance ticker specifically so XAU/USD doesn't get passed to yfinance.
    yf_ticker = get_ticker(asset_id, source="yfinance")
    from p3_backtester.market_data import fetch_ohlcv
    return fetch_ohlcv(yf_ticker, timeframe, start_date, end_date, source="auto")


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


# ── Twelve Data provider ─────────────────────────────────────────────────────

_TD_INTERVAL_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "8h": "8h", "1d": "1day", "1wk": "1week",
}


def _fetch_twelvedata(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV from Twelve Data REST API.
    Requires TWELVEDATA_API_KEY environment variable.
    Free tier: 800 credits/day, 8 credits/min.
    Docs: https://twelvedata.com/docs
    """
    import urllib.request, json as _json

    api_key = os.getenv("TWELVEDATA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TWELVEDATA_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at https://twelvedata.com/"
        )

    td_interval = _TD_INTERVAL_MAP.get(interval)
    if not td_interval:
        raise ValueError(f"Twelve Data: unsupported interval '{interval}'")

    # Calculate outputsize from date range
    from datetime import datetime as _dt
    days = (_dt.strptime(end, "%Y-%m-%d") - _dt.strptime(start, "%Y-%m-%d")).days
    # For intraday, outputsize needs more bars per day
    bars_per_day = {"1min": 390, "5min": 78, "15min": 26, "30min": 13,
                    "1h": 7, "4h": 2, "8h": 3, "1day": 1, "1week": 1}
    outputsize = min(5000, max(30, days * bars_per_day.get(td_interval, 1) + 10))

    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={urllib.request.quote(ticker, safe='/')}"
        f"&interval={td_interval}&outputsize={outputsize}"
        f"&start_date={start}&end_date={end}"
        f"&apikey={api_key}&format=JSON"
    )
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"Twelve Data fetch failed for {ticker}: {e}")

    if data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error for {ticker}: {data.get('message', data)}")

    values = data.get("values", [])
    if not values:
        raise RuntimeError(f"Twelve Data returned empty data for {ticker} {interval}")

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    else:
        df["Volume"] = 0
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    return df


def fetch_twelvedata_quotes(symbols: list[str], api_key: str) -> dict:
    """
    Fetch real-time quotes + historical close prices for Market Pulse.
    Uses Twelve Data /time_series for each symbol (daily, 70 bars).
    Returns dict: {symbol: {"price": float, "1W%": float, "1M%": float, "3M%": float}}

    Only call this for symbols confirmed on the free tier (forex/crypto pairs like XAU/USD).
    """
    import urllib.request, json as _json, time as _time

    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    results = {}
    for symbol in symbols:
        try:
            url = (
                f"https://api.twelvedata.com/time_series"
                f"?symbol={urllib.request.quote(symbol, safe='/')}"
                f"&interval=1day&outputsize=70&apikey={api_key}&format=JSON"
            )
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read())
            if data.get("status") == "error":
                results[symbol] = {"error": data.get("message", "API error")}
                continue
            values = data.get("values", [])
            if len(values) < 5:
                results[symbol] = {"error": "insufficient data"}
                continue
            closes = [float(v["close"]) for v in values]  # newest first
            price = closes[0]
            def _pct(n, _c=closes, _p=price):
                return (_p - _c[n]) / _c[n] * 100 if len(_c) > n else None
            results[symbol] = {
                "price": price,
                "1W%": _pct(5),
                "1M%": _pct(21),
                "3M%": _pct(63),
            }
            _time.sleep(0.15)  # stay under 8 credits/min on free tier
        except Exception as e:
            results[symbol] = {"error": str(e)}
    return results


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
    yfinance is limited to ~60 days of 15-min data; eodhd/twelvedata have years.
    """
    p = (provider or active_provider()).lower()
    return p in ("eodhd", "ccxt", "twelvedata")
