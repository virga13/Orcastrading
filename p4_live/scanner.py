"""
p4_live/scanner.py — Daily live signal scanner.

Scans all configured (asset, strategy) pairs from config/assets.yaml.
Strategies and assets are added/removed in config — no code change needed.

MTF Trend parameters are LOCKED — do not modify without re-validating the backtest.
Validated baselines (1D/1W, test window):
  SPX500 (^GSPC) 2024-03-28 to 2026-03-28: WR 42.9%  | NetPF 2.43 | MaxDD 7.0R  | Kelly 6.3%
  US30   (^DJI)  2024-03-27 to 2026-03-30: WR 51.3%  | NetPF 2.05 | MaxDD 5.0R  | Kelly 7.7%
"""
import ta
import pandas as pd
from datetime import date, timedelta

from p3_backtester.strategies.mtf_trend import MTFTrendStrategy
from p3_backtester.market_data import fetch_ohlcv
from core.config import get_all_assets, get_ticker, get_strategy_config, get_strategy_timeframe

# ── Locked strategy config — shared across all assets ─────────────────────────
_STRATEGY = MTFTrendStrategy(
    ema_fast=20,
    ema_slow=50,
    adx_threshold=15,
    adx_threshold_short=15,
    pullback_max_atr=1.5,
    min_prior_distance_atr=1.5,
    prior_lookback=5,
    rsi_max=65,
    rsi_min=35,
    sl_buffer_atr=0.15,
    sl_lookback=10,
    tp1_r=2.5,
    tp2_r=4.0,
    tp1_alloc=70,
    tp2_alloc=30,
    win_rate=0.48,
    regime_filter=True,
    shorts_enabled=False,
)

INTERVAL = "1d"

# ── Asset registry — add new assets here ──────────────────────────────────────
ASSETS = [
    {
        "ticker": "^GSPC",
        "label":  "SPX500",
        "baseline": {
            "win_rate":        0.429,
            "net_pf":          2.43,
            "avg_net_pnl_r":   0.585,
            "max_drawdown_r":  7.0,
            "kelly_25pct":     0.063,
            "n_trades":        49,
            "period":          "2024-03-28 to 2026-03-28",
        },
    },
    {
        "ticker": "^DJI",
        "label":  "US30",
        "baseline": {
            "win_rate":        0.513,
            "net_pf":          2.05,
            "avg_net_pnl_r":   0.492,
            "max_drawdown_r":  5.0,
            "kelly_25pct":     0.077,
            "n_trades":        39,
            "period":          "2024-03-27 to 2026-03-30",
        },
    },
    {
        "ticker": "BTC-USD",
        "label":  "Bitcoin",
        "baseline": {
            "win_rate":        0.250,
            "net_pf":          0.92,
            "avg_net_pnl_r":  -0.062,
            "max_drawdown_r": 16.3,
            "kelly_25pct":     0.000,
            "n_trades":        40,
            "period":          "2024-07-09 to 2026-03-30",
        },
    },
    {
        "ticker": "GC=F",
        "label":  "Gold",
        "baseline": {
            "win_rate":        0.305,
            "net_pf":          1.24,
            "avg_net_pnl_r":   0.166,
            "max_drawdown_r":  9.0,
            "kelly_25pct":     0.016,
            "n_trades":        59,
            "period":          "2024-03-13 to 2026-03-30",
        },
    },
    {
        "ticker": "SI=F",
        "label":  "Silver",
        "baseline": {
            "win_rate":        0.279,
            "net_pf":          1.02,
            "avg_net_pnl_r":   0.013,
            "max_drawdown_r": 14.9,
            "kelly_25pct":     0.004,
            "n_trades":        43,
            "period":          "2024-03-13 to 2026-03-30",
        },
    },
    {
        "ticker": "^GDAXI",
        "label":  "GER40",
        "baseline": {
            "win_rate":        0.286,
            "net_pf":          1.05,
            "avg_net_pnl_r":   0.039,
            "max_drawdown_r": 17.0,
            "kelly_25pct":     0.005,
            "n_trades":        49,
            "period":          "2024-03-11 to 2026-03-30",
        },
    },
]

# Keep a dict for fast lookup by ticker
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

    # Key levels — 20-bar swing high/low for nearest support/resistance
    high_20  = float(high.rolling(20).max().iloc[-1])
    low_20   = float(low.rolling(20).min().iloc[-1])
    high_max = float(high.max())
    low_min  = float(low.min())
    candidates = [low_20, low_min, high_20, high_max]
    supports     = [v for v in candidates if v < price]
    resistances  = [v for v in candidates if v > price]
    nearest_support    = max(supports)    if supports    else low_min
    nearest_resistance = min(resistances) if resistances else high_max

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


def _gate_failure_reason(df: pd.DataFrame, ticker: str) -> str:
    """Return a human-readable reason why no signal was generated."""
    from p3_backtester.strategies.regime_filter import classify_regime

    close = df["Close"]
    regime = classify_regime(df, INTERVAL)

    from p3_backtester.strategies.mtf_trend import _HTF_RESAMPLE
    import ta as _ta
    rule = _HTF_RESAMPLE.get(INTERVAL, "1W")
    try:
        htf_close = close.resample(rule).last().dropna()
        ema_f = float(_ta.trend.EMAIndicator(htf_close, 20).ema_indicator().iloc[-1])
        ema_s = float(_ta.trend.EMAIndicator(htf_close, 50).ema_indicator().iloc[-1])
        htf_bias = "long" if ema_f > ema_s else "short"

        htf_high = df["High"].resample(rule).max().reindex(htf_close.index).dropna()
        htf_low  = df["Low"].resample(rule).min().reindex(htf_close.index).dropna()
        n = min(len(htf_close), len(htf_high), len(htf_low))
        htf_adx = float(
            _ta.trend.ADXIndicator(
                htf_high.iloc[-n:], htf_low.iloc[-n:], htf_close.iloc[-n:], window=14
            ).adx().iloc[-1]
        ) if n >= 20 else 0.0
    except Exception:
        return "Insufficient HTF data"

    reasons = []
    if regime == "bear_trend" and htf_bias == "long":
        reasons.append(f"Regime BEAR (SMA200 filter blocking longs)")
    if htf_adx < 15:
        reasons.append(f"HTF ADX {htf_adx:.1f} below threshold (15) — market not trending")
    if htf_bias == "short":
        reasons.append(f"HTF EMA20 < EMA50 — no long bias")

    etf_ema = float(_ta.trend.EMAIndicator(close, 20).ema_indicator().iloc[-1])
    price   = float(close.iloc[-1])
    atr     = float(
        _ta.volatility.AverageTrueRange(df["High"], df["Low"], close, window=14)
        .average_true_range().iloc[-1]
    )
    dist_atr = (price - etf_ema) / atr
    if abs(dist_atr) > 1.5:
        reasons.append(f"Price {dist_atr:+.1f} ATR from EMA20 — not in retest zone (+/-1.5)")

    rsi = float(_ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])
    if htf_bias == "long" and rsi > 65:
        reasons.append(f"RSI {rsi:.0f} overbought (>65)")

    context = (
        f"HTF bias={htf_bias}  ADX={htf_adx:.1f}  "
        f"Regime={regime}  EMA20_dist={dist_atr:+.1f}ATR  RSI={rsi:.0f}"
    )
    return ("; ".join(reasons) if reasons else "Prior distance gate failed") + f"\n  [{context}]"


def scan(ticker: str, as_of: date | None = None) -> dict:
    """
    Fetch latest data and run the locked strategy for one asset.

    Returns a signal dict (fired=True/False) including everything needed
    to record the trade.
    """
    asset = ASSET_MAP.get(ticker)
    if asset is None:
        raise ValueError(f"Unknown ticker '{ticker}'. Available: {list(ASSET_MAP)}")

    label      = asset["label"]
    end_date   = (as_of or date.today()).strftime("%Y-%m-%d")
    start_date = (
        (as_of or date.today()) - timedelta(days=_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")

    try:
        df = fetch_ohlcv(ticker, INTERVAL, start_date, end_date, source="auto")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {e}")

    if len(df) < 60:
        raise RuntimeError(f"Insufficient data for {ticker}: only {len(df)} bars")

    tech   = _compute_tech(df, ticker)
    setups = _STRATEGY.generate_setups(tech, df)

    signal_date = df.index[-1].strftime("%Y-%m-%d")

    if setups is None or not setups.setups:
        return {
            "fired":       False,
            "ticker":      ticker,
            "label":       label,
            "signal_date": signal_date,
            "reason":      _gate_failure_reason(df, ticker),
            "price":       tech["current_price"],
            "atr":         tech["atr_14"],
            "rsi":         tech["rsi_14"],
        }

    setup = setups.setups[0]
    risk  = abs(
        (setup.entry_high if setup.direction == "long" else setup.entry_low)
        - setup.stop_loss
    )

    return {
        "fired":       True,
        "ticker":      ticker,
        "label":       label,
        "signal_date": signal_date,
        "direction":   setup.direction,
        "entry_low":   setup.entry_low,
        "entry_high":  setup.entry_high,
        "stop_loss":   setup.stop_loss,
        "tp1":         setup.targets[0].price,
        "tp2":         setup.targets[1].price if len(setup.targets) > 1 else None,
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


def scan_all(as_of: date | None = None) -> list[dict]:
    """Scan all registered assets with MTF Trend. Returns one result dict per asset."""
    results = []
    for asset in ASSETS:
        try:
            results.append(scan(asset["ticker"], as_of=as_of))
        except RuntimeError as e:
            results.append({
                "fired":       False,
                "ticker":      asset["ticker"],
                "label":       asset["label"],
                "signal_date": (as_of or date.today()).isoformat(),
                "reason":      str(e),
                "price":       None,
                "atr":         None,
                "rsi":         None,
            })
    return results


# ── Multi-strategy scanner (config-driven) ────────────────────────────────────

def scan_strategy(asset_id: str, strategy_id: str, as_of: date | None = None) -> dict:
    """
    Scan one asset with one strategy. Fetches the appropriate timeframe automatically.

    Returns a signal dict with:
      fired, ticker, label, strategy, signal_date, direction, entry_low/high,
      stop_loss, tp1, tp2, rr, confidence, rationale, price, atr, rsi
    or
      fired=False, reason=<why>
    """
    from p3_backtester.strategies.registry import build_from_config

    ticker     = get_ticker(asset_id)
    timeframe  = get_strategy_timeframe(strategy_id)
    today      = as_of or date.today()
    end_date   = today.strftime("%Y-%m-%d")

    # Determine lookback
    from core.config import get_strategy_lookback
    lookback   = get_strategy_lookback(strategy_id)
    start_date = (today - timedelta(days=lookback)).strftime("%Y-%m-%d")

    # Find label
    label = ticker
    for a in get_all_assets():
        if get_ticker(a["id"]) == ticker:
            label = a["label"]
            break

    try:
        df = fetch_ohlcv(ticker, timeframe, start_date, end_date, source="auto")
    except Exception as e:
        return {
            "fired": False, "ticker": ticker, "label": label,
            "strategy": strategy_id, "signal_date": end_date,
            "reason": str(e), "price": None, "atr": None, "rsi": None,
        }

    if len(df) < 5:
        return {
            "fired": False, "ticker": ticker, "label": label,
            "strategy": strategy_id, "signal_date": end_date,
            "reason": "Insufficient data", "price": None, "atr": None, "rsi": None,
        }

    # Build tech dict (daily indicators for filters)
    daily_df = df if timeframe == "1d" else _fetch_daily_for_filters(ticker, today)
    tech = _compute_tech(daily_df if daily_df is not None else df, ticker, interval=timeframe)

    # Run strategy
    strategy  = build_from_config(strategy_id)
    result    = strategy.generate_setups(tech, df)
    signal_dt = df.index[-1].strftime("%Y-%m-%d")

    if result is None or not result.setups:
        return {
            "fired": False, "ticker": ticker, "label": label,
            "strategy": strategy_id, "signal_date": signal_dt,
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
        "fired":       True,
        "ticker":      ticker,
        "label":       label,
        "strategy":    strategy_id,
        "signal_date": signal_dt,
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
    Scan every (asset, strategy) pair enabled in config/assets.yaml.
    Returns one result dict per pair. Used by the daily multi-strategy run.
    """
    from core.config import get_watchlist
    results = []
    for asset_id, strategy_id in get_watchlist():
        try:
            results.append(scan_strategy(asset_id, strategy_id, as_of=as_of))
        except Exception as e:
            results.append({
                "fired": False, "ticker": get_ticker(asset_id),
                "label": asset_id, "strategy": strategy_id,
                "signal_date": (as_of or date.today()).isoformat(),
                "reason": str(e), "price": None, "atr": None, "rsi": None,
            })
    return results


def _fetch_daily_for_filters(ticker: str, today: date) -> pd.DataFrame | None:
    """Fetch daily bars for use in EMAs/regime filter when primary timeframe is intraday."""
    try:
        start = (today - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end   = today.strftime("%Y-%m-%d")
        return fetch_ohlcv(ticker, "1d", start, end, source="auto")
    except Exception:
        return None
