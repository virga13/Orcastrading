"""
strategies/regime_filter.py — Market regime classification from price action.

3-state regime detector based on price vs SMA200 and SMA200 slope.
Applied only on daily/weekly intervals — intraday is too noisy for SMA200.

States
------
bull_trend   price > SMA200 × (1 + BULL_BUFFER) — clearly above the long-term average
bear_trend   price < SMA200 × (1 - BEAR_BUFFER) AND SMA200 is declining
neutral      everything else (transitioning or insufficient data)

Usage in strategy
-----------------
    from p3_backtester.strategies.regime_filter import classify_regime

    regime = classify_regime(df, interval)   # df sliced to signal bar (zero lookahead)
    if regime == "bear_trend" and htf_bias == "long":
        return None   # don't buy into a bear market
    if regime == "bull_trend" and htf_bias == "short":
        return None   # don't short a bull market

Financial rationale
-------------------
The SMA200 is the single most-watched long-term trend level used by institutional
participants. When price is substantially below a declining SMA200, every EMA retest
is a potential lower-high — not a continuation entry. The 2022 rate-hike bear wiped
all assets in the MTF Trend backtest because the strategy kept loading longs below
the SMA200. This filter eliminates those trades.
"""
import pandas as pd

_INTRADAY = {"1m", "5m", "15m", "30m", "1h"}
_SMA_PERIOD = 200
_SLOPE_LOOKBACK = 20     # bars over which to measure SMA200 slope (≈1 month on daily)
_BULL_BUFFER = 0.02      # price must be ≥2% above SMA200 to classify as bull
_BEAR_BUFFER = 0.01      # price must be ≥1% below SMA200 (lower threshold — more sensitive)


def classify_regime(df: pd.DataFrame, interval: str) -> str:
    """
    Classify the current market regime from the sliced OHLCV DataFrame.

    Parameters
    ----------
    df       : OHLCV DataFrame sliced to the signal bar (zero lookahead guaranteed by caller)
    interval : entry interval string ("1m", "5m", "15m", "30m", "1h", "1d", "1wk")

    Returns
    -------
    "bull_trend" | "bear_trend" | "neutral"
    """
    if interval in _INTRADAY:
        return "neutral"   # regime filter is noise below daily timeframe

    close = df["Close"]
    n = len(close)

    if n < _SMA_PERIOD:
        return "neutral"   # insufficient history for a meaningful SMA200

    sma200 = close.rolling(_SMA_PERIOD).mean()
    last_price = float(close.iloc[-1])
    last_sma = float(sma200.iloc[-1])

    if pd.isna(last_sma) or last_sma <= 0:
        return "neutral"

    price_vs_sma = (last_price - last_sma) / last_sma   # signed distance as fraction

    # SMA200 slope: rate of change over SLOPE_LOOKBACK bars
    slope = 0.0
    if n >= _SMA_PERIOD + _SLOPE_LOOKBACK:
        past_sma = float(sma200.iloc[-(1 + _SLOPE_LOOKBACK)])
        if not pd.isna(past_sma) and past_sma > 0:
            slope = (last_sma - past_sma) / past_sma

    # Bull: price clearly above SMA200 (regime allows longs, blocks shorts)
    if price_vs_sma > _BULL_BUFFER:
        return "bull_trend"

    # Bear: price below SMA200 AND SMA200 itself is declining
    # Requiring slope < 0 prevents false bear calls during brief dips in an uptrend
    if price_vs_sma < -_BEAR_BUFFER and slope < 0:
        return "bear_trend"

    return "neutral"
