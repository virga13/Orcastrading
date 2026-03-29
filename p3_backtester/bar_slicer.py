"""
bar_slicer.py — recomputes the same technical indicators as p1_analysis_engine/fetchers/technical.py
but on a pre-sliced DataFrame (df.iloc[:i+1]) to guarantee zero lookahead bias.

MAINTENANCE: This module must stay in sync with p1_analysis_engine/fetchers/technical.py.
Any indicator added, removed, or modified there must be mirrored here.
"""
import ta
import pandas as pd


# Minimum bars required before a signal bar is valid.
# EMA200 needs 200 bars; add headroom for ATR/ADX/Stoch warmup.
MIN_WARMUP_BARS = 220


def recompute_technical(df: pd.DataFrame, interval: str) -> dict | None:
    """
    Given a DataFrame sliced to df.iloc[:signal_bar_index+1], recompute
    all technical indicators for the last bar.

    Returns the same dict structure as fetch_technical(), plus the 'interval' key.
    Returns None if there are insufficient bars (< MIN_WARMUP_BARS).
    """
    if len(df) < MIN_WARMUP_BARS:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    current_price = float(close.iloc[-1])

    # --- EMAs ---
    ema20  = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    ema50  = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()

    e20  = float(ema20.iloc[-1])
    e50  = float(ema50.iloc[-1])  if len(ema50.dropna())  > 0 else None
    e200 = float(ema200.iloc[-1]) if len(ema200.dropna()) > 0 else None

    if e50 and e200:
        if e20 > e50 > e200:
            ema_alignment = "bullish"
        elif e20 < e50 < e200:
            ema_alignment = "bearish"
        else:
            ema_alignment = "mixed"
    else:
        ema_alignment = "mixed"

    if current_price > e20 and (e50 is None or e20 > e50):
        trend = "uptrend"
    elif current_price < e20 and (e50 is None or e20 < e50):
        trend = "downtrend"
    else:
        trend = "sideways"

    # --- RSI ---
    rsi_val = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])

    # --- MACD ---
    macd_diff = float(ta.trend.MACD(close).macd_diff().iloc[-1])
    if macd_diff > 0:
        macd_signal = "bullish"
    elif macd_diff < 0:
        macd_signal = "bearish"
    else:
        macd_signal = "neutral"

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = float(bb.bollinger_hband().iloc[-1])
    bb_lower = float(bb.bollinger_lband().iloc[-1])
    if current_price > bb_upper:
        bb_position = "above_upper"
    elif current_price >= bb_upper * 0.99:
        bb_position = "upper"
    elif current_price <= bb_lower * 1.01:
        bb_position = "lower"
    elif current_price < bb_lower:
        bb_position = "below_lower"
    else:
        bb_position = "middle"

    # --- ATR ---
    atr_val = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
    atr_pct = round((atr_val / current_price) * 100, 2)

    # --- ADX ---
    adx_val = float(ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1])

    # --- Stochastic ---
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    stoch_k = float(stoch.stoch().iloc[-1])
    stoch_d = float(stoch.stoch_signal().iloc[-1])
    if stoch_k < 20:
        stoch_signal = "oversold"
    elif stoch_k > 80:
        stoch_signal = "overbought"
    else:
        stoch_signal = "neutral"

    # --- CMF ---
    cmf_val = float(ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow().iloc[-1])
    if cmf_val > 0.1:
        cmf_signal = "accumulation"
    elif cmf_val < -0.1:
        cmf_signal = "distribution"
    else:
        cmf_signal = "neutral"

    # --- Volume trend ---
    avg_vol_20 = float(volume.rolling(20).mean().iloc[-1])
    avg_vol_5  = float(volume.rolling(5).mean().iloc[-1])
    if avg_vol_5 > avg_vol_20 * 1.1:
        volume_trend = "increasing"
    elif avg_vol_5 < avg_vol_20 * 0.9:
        volume_trend = "decreasing"
    else:
        volume_trend = "neutral"

    # --- Key levels ---
    high_20  = float(high.rolling(20).max().iloc[-1])
    low_20   = float(low.rolling(20).min().iloc[-1])
    high_52w = float(high.max())
    low_52w  = float(low.min())

    fib_range = high_52w - low_52w
    fib_levels = {
        "fib_23.6": round(high_52w - 0.236 * fib_range, 4),
        "fib_38.2": round(high_52w - 0.382 * fib_range, 4),
        "fib_50.0": round(high_52w - 0.500 * fib_range, 4),
        "fib_61.8": round(high_52w - 0.618 * fib_range, 4),
    }

    key_levels = [
        {"price": round(low_20,  4), "label": "support",    "strength": "moderate"},
        {"price": round(high_20, 4), "label": "resistance", "strength": "moderate"},
        {"price": round(high_52w,4), "label": "52w_high",   "strength": "strong"},
        {"price": round(low_52w, 4), "label": "52w_low",    "strength": "strong"},
    ]
    for label, price in fib_levels.items():
        key_levels.append({"price": price, "label": label, "strength": "weak"})

    supports     = [l["price"] for l in key_levels if l["price"] < current_price]
    resistances  = [l["price"] for l in key_levels if l["price"] > current_price]
    nearest_support    = max(supports)    if supports    else low_52w
    nearest_resistance = min(resistances) if resistances else high_52w

    return {
        "current_price":      round(current_price, 4),
        "trend":              trend,
        "rsi_14":             round(rsi_val, 2),
        "macd_signal":        macd_signal,
        "ema_alignment":      ema_alignment,
        "ema20":              round(e20,  4),
        "ema50":              round(e50,  4) if e50  else None,
        "ema200":             round(e200, 4) if e200 else None,
        "adx_14":             round(adx_val, 2),
        "atr_14":             round(atr_val, 4),
        "atr_pct":            atr_pct,
        "bb_upper":           round(bb_upper, 4),
        "bb_lower":           round(bb_lower, 4),
        "bb_position":        bb_position,
        "stoch_k":            round(stoch_k, 2),
        "stoch_d":            round(stoch_d, 2),
        "stoch_signal":       stoch_signal,
        "cmf_20":             round(cmf_val, 4),
        "cmf_signal":         cmf_signal,
        "volume_trend":       volume_trend,
        "key_levels":         key_levels,
        "nearest_support":    round(nearest_support,    4),
        "nearest_resistance": round(nearest_resistance, 4),
        "high_52w":           round(high_52w, 4),
        "low_52w":            round(low_52w,  4),
        "interval":           interval,
    }
