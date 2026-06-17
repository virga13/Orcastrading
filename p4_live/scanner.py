"""
p4_live/scanner.py — Daily live signal scanner.

Scans all configured (asset, strategy) pairs from config/assets.yaml.
Assets and strategies are managed entirely in config — no code change needed.

MTF Trend parameters are LOCKED — do not modify without re-validating the backtest.
Baselines (WR, PF, MaxDD) now live in config/assets.yaml under baseline: blocks.
"""
import ta
import pandas as pd
from datetime import date, timedelta

from p3_backtester.strategies.mtf_trend import MTFTrendStrategy
from p3_backtester.market_data import fetch_ohlcv
from core.config import get_all_assets, get_ticker, get_strategy_config, get_strategy_timeframe

# ── Deprecated singleton — kept for backfill.py import compat only ─────────────
# scan() and scan_all() now delegate to scan_strategy() / scan_all_strategies().
# Do not use _STRATEGY for new code.
_STRATEGY = MTFTrendStrategy(shorts_enabled=False)

INTERVAL = "1d"

# ── Asset registry — built dynamically from config/assets.yaml ────────────────
# Each entry: {"ticker": yfinance_ticker, "label": display_name, "baseline": {...}}
# Baselines are stored in assets.yaml under baseline: blocks and loaded here.
# To add a new asset: add it to assets.yaml (with validated baseline:), done.

def _build_asset_list() -> list[dict]:
    from core.config import get_all_assets
    result = []
    for a in get_all_assets():
        if not a.get("enabled", True):
            continue
        yf = a.get("tickers", {}).get("yfinance")
        if not yf:
            continue
        entry: dict = {"ticker": yf, "label": a.get("label", a["id"])}
        if a.get("baseline"):
            entry["baseline"] = a["baseline"]
        result.append(entry)
    return result

ASSETS   = _build_asset_list()
ASSET_MAP = {a["ticker"]: a for a in ASSETS}

def get_asset_by_id(asset_id: str) -> dict | None:
    """Look up an asset by its core.config id (e.g. 'gold', 'spx500')."""
    from core.config import get_ticker as _get_ticker
    try:
        ticker = _get_ticker(asset_id)
        return ASSET_MAP.get(ticker)
    except KeyError:
        return None

_LOOKBACK_DAYS = 400   # enough for weekly EMA50 warmup + SMA200


def _fetch_latest_bar(ticker: str) -> dict | None:
    """Fetch the most recent OHLC bar for a ticker. Returns None on failure."""
    from datetime import timedelta
    start = (date.today() - timedelta(days=10)).isoformat()
    end   = date.today().isoformat()
    try:
        df = fetch_ohlcv(ticker, INTERVAL, start, end, source="auto")
        return {
            "date":  df.index[-1].strftime("%Y-%m-%d"),
            "close": float(df["Close"].iloc[-1]),
            "high":  float(df["High"].iloc[-1]),
            "low":   float(df["Low"].iloc[-1]),
        }
    except Exception:
        return None


def _compute_tech(df: pd.DataFrame, ticker: str, interval: str = INTERVAL) -> dict:
    """Compute the tech dict that generate_setups() expects."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    atr = float(
        ta.volatility.AverageTrueRange(high, low, close, window=14)
        .average_true_range().iloc[-1]
    )
    rsi = float(
        ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    )
    adx = float(
        ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
    )
    ema20 = float(
        ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    )
    ema50 = float(
        ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    )
    price = float(close.iloc[-1])

    # Key levels — rolling swing highs/lows at 5, 10, 20, 50-bar windows
    # give 8 distinct candidates vs the prior 4, substantially improving S/R accuracy.
    sr_candidates: set[float] = set()
    for window in (5, 10, 20, 50):
        if len(high) >= window:
            sr_candidates.add(float(high.rolling(window).max().iloc[-1]))
            sr_candidates.add(float(low.rolling(window).min().iloc[-1]))
    supports     = sorted([v for v in sr_candidates if v < price], reverse=True)
    resistances  = sorted([v for v in sr_candidates if v > price])
    nearest_support    = supports[0]    if supports    else float(low.min())
    nearest_resistance = resistances[0] if resistances else float(high.max())

    return {
        "ticker":              ticker,
        "current_price":       price,
        "atr_14":              atr,
        "atr_pct":             round(atr / price * 100, 2),
        "rsi_14":              rsi,
        "adx_14":              adx,
        "ema20":               ema20,
        "ema50":               ema50,
        "nearest_support":     nearest_support,
        "nearest_resistance":  nearest_resistance,
        "interval":            interval,
    }


def scan(ticker: str, as_of: date | None = None) -> dict:
    """
    Scan one asset (by yfinance ticker) using config-driven strategies.
    Delegates to scan_strategy() for the mtf_trend strategy on the 1d timeframe.
    Kept for backward compatibility — prefer scan_strategy() for new code.
    """
    asset = ASSET_MAP.get(ticker)
    if asset is None:
        raise ValueError(f"Unknown ticker '{ticker}'. Available: {list(ASSET_MAP)}")
    # Resolve asset_id from ticker
    from core.config import get_all_assets
    asset_id = next(
        (a["id"] for a in get_all_assets()
         if a.get("tickers", {}).get("yfinance") == ticker),
        None,
    )
    if asset_id is None:
        raise ValueError(f"Cannot resolve asset_id for ticker '{ticker}'")
    return scan_strategy(asset_id, "mtf_trend", timeframe="1d", as_of=as_of)


def scan_all(as_of: date | None = None) -> list[dict]:
    """Scan all assets with all configured strategies. Delegates to scan_all_strategies()."""
    return scan_all_strategies(as_of=as_of)


# ── Multi-strategy scanner (config-driven) ────────────────────────────────────

def scan_strategy(asset_id: str, strategy_id: str, timeframe: str | None = None, as_of: date | None = None, _df=None) -> dict:
    """
    Scan one asset with one strategy. Fetches the appropriate timeframe automatically,
    or uses the explicit timeframe if provided.

    Returns a signal dict with:
      fired, ticker, label, strategy, signal_date, direction, entry_low/high,
      stop_loss, tp1, tp2, rr, confidence, rationale, price, atr, rsi
    or
      fired=False, reason=<why>
    """
    from p3_backtester.strategies.registry import build_from_config
    from core.data import fetch as _core_fetch

    # Canonical ticker for journal/dedup — always use yfinance format (GC=F, BTC-USD, etc.)
    # so backfill and live scanner records share the same ticker key.
    ticker     = get_ticker(asset_id, source="yfinance")
    timeframe  = timeframe or get_strategy_timeframe(strategy_id)
    today      = as_of or date.today()
    end_date   = today.strftime("%Y-%m-%d")

    # Determine lookback
    from core.config import get_strategy_lookback
    lookback   = get_strategy_lookback(strategy_id)
    start_date = (today - timedelta(days=lookback)).strftime("%Y-%m-%d")

    # Find label
    label = next((a["label"] for a in get_all_assets() if a["id"] == asset_id), ticker)

    try:
        # Use pre-fetched df if provided (avoids redundant API calls in bulk scans)
        if _df is not None:
            df = _df
        else:
            # Use core.data.fetch so the active DATA_PROVIDER (e.g. twelvedata) is respected
            df = _core_fetch(asset_id, timeframe, start_date, end_date)
    except Exception as e:
        return {
            "fired": False, "ticker": ticker, "label": label,
            "strategy": strategy_id, "timeframe": timeframe, "signal_date": end_date,
            "reason": str(e), "price": None, "atr": None, "rsi": None,
        }

    if len(df) < 5:
        return {
            "fired": False, "ticker": ticker, "label": label,
            "strategy": strategy_id, "timeframe": timeframe, "signal_date": end_date,
            "reason": "Insufficient data", "price": None, "atr": None, "rsi": None,
        }

    # Always compute tech from the strategy's own timeframe data.
    # Strategies that need HTF context (MTF Trend, EMA Continuation) resample
    # df internally — no need to pass a separate daily df.
    tech = _compute_tech(df, ticker, interval=timeframe)

    # Run strategy — apply any per-asset param overrides from assets.yaml
    from core.config import get_asset_strategy_params
    asset_param_overrides = get_asset_strategy_params(asset_id, strategy_id)
    strategy  = build_from_config(strategy_id, param_overrides=asset_param_overrides)
    result        = strategy.generate_setups(tech, df)
    signal_dt     = df.index[-1].strftime("%Y-%m-%d")
    signal_bar_ts = df.index[-1].isoformat()

    if result is None or not result.setups:
        return {
            "fired": False, "ticker": ticker, "label": label,
            "strategy": strategy_id, "timeframe": timeframe,
            "signal_date": signal_dt, "signal_bar_ts": signal_bar_ts,
            "reason": "No setup conditions met",
            "price": tech["current_price"], "atr": tech["atr_14"], "rsi": tech["rsi_14"],
        }

    setup = result.setups[0]
    risk  = abs(
        (setup.entry_high if setup.direction == "long" else setup.entry_low)
        - setup.stop_loss
    )
    tp2   = setup.targets[1].price if len(setup.targets) > 1 else None

    return {
        "fired":         True,
        "ticker":        ticker,
        "label":         label,
        "strategy":      strategy_id,
        "timeframe":     timeframe,
        "signal_date":   signal_dt,
        "signal_bar_ts": signal_bar_ts,
        "direction":   setup.direction,
        "entry_low":   setup.entry_low,
        "entry_high":  setup.entry_high,
        "stop_loss":   setup.stop_loss,
        "tp1":         setup.targets[0].price,
        "tp2":         tp2,
        "tp1_alloc":   setup.targets[0].allocation_pct,
        "tp2_alloc":   setup.targets[1].allocation_pct if len(setup.targets) > 1 else 0,
        "risk_pts":    round(risk, 4),
        "rr":          setup.rr_ratio,
        "confidence":  setup.confidence,
        "rationale":   setup.rationale,
        "price":       tech["current_price"],
        "atr":         tech["atr_14"],
        "rsi":         tech["rsi_14"],
        "regime":      setup.confidence_note,
    }


def scan_all_strategies(as_of: date | None = None) -> list[dict]:
    """
    Scan every (asset, strategy, timeframe) triple enabled in config/assets.yaml.
    Pre-fetches unique (asset, timeframe) pairs to avoid redundant API calls.
    Returns one result dict per triple. Used by the daily multi-strategy run.
    """
    from core.config import get_watchlist, get_strategy_lookback
    from core.data import fetch as _core_fetch

    watchlist = get_watchlist()
    today     = as_of or date.today()

    # ── Pre-fetch unique (asset_id, timeframe) data to minimise API credit usage
    data_cache: dict[tuple[str, str], object] = {}
    for asset_id, strategy_id, timeframe in watchlist:
        key = (asset_id, timeframe)
        if key in data_cache:
            continue
        lookback   = get_strategy_lookback(strategy_id)
        start_date = (today - timedelta(days=lookback)).strftime("%Y-%m-%d")
        end_date   = today.strftime("%Y-%m-%d")
        try:
            data_cache[key] = _core_fetch(asset_id, timeframe, start_date, end_date)
        except Exception as e:
            data_cache[key] = e   # store error; scan_strategy will handle it

    results = []
    for asset_id, strategy_id, timeframe in watchlist:
        cached = data_cache.get((asset_id, timeframe))
        _df = cached if not isinstance(cached, Exception) else None
        _err = cached if isinstance(cached, Exception) else None
        if _err is not None:
            results.append({
                "fired": False, "ticker": get_ticker(asset_id, source="yfinance"),
                "label": asset_id, "strategy": strategy_id, "timeframe": timeframe,
                "signal_date": today.isoformat(),
                "reason": str(_err), "price": None, "atr": None, "rsi": None,
            })
            continue
        try:
            results.append(scan_strategy(asset_id, strategy_id, timeframe=timeframe, as_of=as_of, _df=_df))
        except Exception as e:
            results.append({
                "fired": False, "ticker": get_ticker(asset_id, source="yfinance"),
                "label": asset_id, "strategy": strategy_id, "timeframe": timeframe,
                "signal_date": today.isoformat(),
                "reason": str(e), "price": None, "atr": None, "rsi": None,
            })
    return results


def _fetch_daily_for_filters(asset_id: str, today: date) -> pd.DataFrame | None:
    """Fetch daily bars for use in EMAs/regime filter when primary timeframe is intraday."""
    from core.data import fetch as _core_fetch
    try:
        start = (today - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end   = today.strftime("%Y-%m-%d")
        return _core_fetch(asset_id, "1d", start, end)
    except Exception:
        return None
