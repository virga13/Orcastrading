"""
p1_analysis_engine/strategy_scanner.py — AI-powered strategy signal scanner.

Uses Claude to evaluate whether configured daily trading strategies have fired for
a given asset, based on real market data. One Claude API call per asset evaluates
all of its enabled daily strategies simultaneously.

Intraday strategies (ORB, mean_reversion, continuation) stay rule-based in P4 —
they run every 15 minutes and calling Claude that frequently would be too costly.

Usage:
    from p1_analysis_engine.strategy_scanner import scan_asset_for_strategies

    result = scan_asset_for_strategies("gold")
    for signal in result.signals:
        if signal.fired:
            print(signal.strategy_id, signal.direction, signal.entry_low)

Adding a new strategy:
    1. Add signal_conditions to config/strategies.yaml
    2. Add a Python class to p3_backtester/strategies/ for backtesting
    3. That's it — the scanner will automatically pick it up if timeframe is "1d"
"""
from __future__ import annotations

from datetime import date

import ta
import yfinance as yf

from core.config import (
    get_asset,
    get_enabled_strategies,
    get_strategy_config,
    get_strategy_timeframe,
    get_ticker,
)
from p1_analysis_engine.schema import StrategyScannerOutput


def scan_asset_for_strategies(
    asset_id: str,
    strategy_ids: list[str] | None = None,
    model: str = "claude-sonnet-4-6",
    macro_context: dict | None = None,
) -> StrategyScannerOutput:
    """
    Run P1 AI analysis for one asset against its enabled daily strategies.

    macro_context — pre-fetched macro dict (pass from scheduler to avoid
                    redundant FRED calls across assets). Fetched per-asset
                    if not provided.

    strategy_ids — override which strategies to evaluate; defaults to all enabled
                   daily (1d timeframe) strategies for the asset.

    Returns StrategyScannerOutput with one StrategySignal per evaluated strategy.
    """
    from p1_analysis_engine.synthesis.claude_synthesizer import synthesize_strategy_scan

    asset  = get_asset(asset_id)
    ticker = get_ticker(asset_id, "yfinance")
    label  = asset["label"]

    # Resolve which daily strategies to evaluate
    enabled = get_enabled_strategies(asset_id)
    daily_ids = strategy_ids or [
        sid for sid in enabled if _is_daily_strategy(sid)
    ]

    if not daily_ids:
        return StrategyScannerOutput(
            asset_id=asset_id,
            ticker=ticker,
            label=label,
            current_price=0.0,
            atr=0.0,
            rsi=0.0,
            analysis_date=date.today().isoformat(),
            market_regime="uncertain",
            signals=[],
        )

    market_data   = _fetch_market_data(ticker)
    strategy_defs = [_load_strategy_def(sid) for sid in daily_ids]

    # Macro context — fetch once per scheduler run, reuse across assets
    if macro_context is None:
        category_map = {"index": "equity", "commodity": "commodity",
                        "crypto": "crypto", "forex": "forex"}
        asset_class = category_map.get(asset.get("category", "index"), "equity")
        try:
            from p1_analysis_engine.fetchers.macro import fetch_macro
            macro_context = fetch_macro(asset_class)
        except Exception:
            macro_context = {}

    # Per-asset news (top 3 headlines only — keeps prompt tight)
    news_headlines: list[str] = []
    try:
        from p1_analysis_engine.fetchers.news import fetch_news
        raw_news = fetch_news(ticker, label)
        news_headlines = [n.get("title", "") for n in raw_news[:3] if n.get("title")]
    except Exception:
        pass

    return synthesize_strategy_scan(
        asset_id=asset_id,
        ticker=ticker,
        label=label,
        market_data=market_data,
        strategy_defs=strategy_defs,
        macro_context=macro_context,
        news_headlines=news_headlines,
        model=model,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_daily_strategy(strategy_id: str) -> bool:
    return get_strategy_timeframe(strategy_id) == "1d"


def _load_strategy_def(strategy_id: str) -> dict:
    cfg = get_strategy_config(strategy_id)
    return {
        "id":                strategy_id,
        "name":              cfg.get("name", strategy_id),
        "description":       cfg.get("description", ""),
        "signal_conditions": cfg.get("signal_conditions", "No conditions defined."),
    }


def _fetch_market_data(ticker: str) -> dict:
    """
    Fetch daily + weekly OHLCV and compute every indicator needed for strategy
    evaluation. Returns a flat dict that the synthesizer formats into a prompt.
    """
    daily = yf.Ticker(ticker).history(period="730d", interval="1d")
    if daily.empty or len(daily) < 60:
        raise RuntimeError(f"Insufficient daily data for {ticker} ({len(daily)} bars)")

    weekly = yf.Ticker(ticker).history(period="730d", interval="1wk")

    close  = daily["Close"]
    high   = daily["High"]
    low    = daily["Low"]
    volume = daily["Volume"]

    price   = float(close.iloc[-1])
    atr_ser = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    atr_val = float(atr_ser.iloc[-1])
    rsi_val = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])

    ema20_ser  = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    ema50_ser  = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    ema200_ser = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    adx_ser    = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    e20      = float(ema20_ser.iloc[-1])
    e50_drop = ema50_ser.dropna()
    e50      = float(e50_drop.iloc[-1])  if len(e50_drop) >= 50  else None
    e200_drop = ema200_ser.dropna()
    e200     = float(e200_drop.iloc[-1]) if len(e200_drop) >= 200 else None

    adx_today = float(adx_ser.iloc[-1])
    adx_prev  = float(adx_ser.iloc[-2])

    # Volume
    vol_today   = float(volume.iloc[-1])
    vol_20d_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio   = round(vol_today / vol_20d_avg, 2) if vol_20d_avg > 0 else 0.0

    # 20-day high / low (rolling max/min across all history; last value = today's window)
    high_20d = float(high.rolling(20).max().iloc[-1])
    low_20d  = float(low.rolling(20).min().iloc[-1])

    # Today's bar
    today_open = float(daily["Open"].iloc[-1])
    today_high = float(high.iloc[-1])
    today_low  = float(low.iloc[-1])
    bar_range  = today_high - today_low
    close_pct  = round((price - today_low) / bar_range, 3) if bar_range > 0 else 0.5

    # Price distance from EMA20 in ATR units (signed: positive = above)
    price_atr_from_ema20 = round((price - e20) / atr_val, 3) if atr_val > 0 else 0.0

    # 10-day swing high using the prior bar (not today — today hasn't closed fully in live scan)
    if len(high) >= 11:
        swing_high_10d       = float(high.rolling(10).max().iloc[-2])
        dist_from_swing_high = round((swing_high_10d - price) / atr_val, 3) if atr_val > 0 else 999.0
    else:
        swing_high_10d       = today_high
        dist_from_swing_high = 0.0

    # Weekly data
    weekly_ema20  = None
    weekly_ema50  = None
    weekly_rsi    = None
    weekly_trend  = "unknown"

    if not weekly.empty and len(weekly) >= 20:
        wc = weekly["Close"]
        weekly_ema20 = round(float(ta.trend.EMAIndicator(wc, window=20).ema_indicator().iloc[-1]), 4)
        weekly_rsi   = round(float(ta.momentum.RSIIndicator(wc, window=14).rsi().iloc[-1]), 2)
        if len(wc) >= 50:
            weekly_ema50 = round(float(ta.trend.EMAIndicator(wc, window=50).ema_indicator().iloc[-1]), 4)
            weekly_trend = "bullish" if weekly_ema20 > weekly_ema50 else "bearish"

    return {
        "date":                       date.today().isoformat(),
        "price":                      round(price, 4),
        "atr":                        round(atr_val, 4),
        "atr_pct":                    round((atr_val / price) * 100, 2),
        "rsi":                        round(rsi_val, 2),
        "ema20":                      round(e20, 4),
        "ema50":                      round(e50, 4)  if e50  else None,
        "ema200":                     round(e200, 4) if e200 else None,
        "adx_today":                  round(adx_today, 2),
        "adx_prev":                   round(adx_prev, 2),
        "volume_today":               int(vol_today),
        "volume_20d_avg":             int(vol_20d_avg),
        "volume_ratio":               vol_ratio,
        "high_20d":                   round(high_20d, 4),
        "low_20d":                    round(low_20d, 4),
        "today_open":                 round(today_open, 4),
        "today_high":                 round(today_high, 4),
        "today_low":                  round(today_low, 4),
        "close_pct_of_range":         close_pct,
        "price_atr_from_ema20":       price_atr_from_ema20,
        "swing_high_10d":             round(swing_high_10d, 4),
        "distance_from_swing_high_atr": dist_from_swing_high,
        "weekly_ema20":               weekly_ema20,
        "weekly_ema50":               weekly_ema50,
        "weekly_rsi":                 weekly_rsi,
        "weekly_trend":               weekly_trend,
    }
