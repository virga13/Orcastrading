"""
tests/conftest.py — shared fixtures for the Orcastrading test suite.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env so os.getenv() works in all tests
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Dev-mode bypass for license checks
os.environ.setdefault("ORCA_DEV_MODE", "1")


# ── Synthetic OHLCV fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """
    600-bar daily OHLCV in a steady uptrend (EMA20 > EMA50 > EMA200).
    Sufficient history for all indicator warm-up periods.
    """
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    price = np.linspace(1800, 2600, n) + np.random.randn(n) * 8
    return pd.DataFrame(
        {
            "Open":   price * (1 + np.random.randn(n) * 0.001),
            "High":   price * (1 + np.abs(np.random.randn(n)) * 0.004),
            "Low":    price * (1 - np.abs(np.random.randn(n)) * 0.004),
            "Close":  price,
            "Volume": np.random.randint(200_000, 600_000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture(scope="session")
def tech_dict(synthetic_df) -> dict:
    """
    Tech dict computed from synthetic_df with conditions tuned to trigger
    pullback strategies (price forced to sit at EMA20).
    """
    import ta
    df = synthetic_df
    close = df["Close"]
    ema20 = float(ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1])
    ema50 = float(ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1])
    atr   = float(ta.volatility.AverageTrueRange(df["High"], df["Low"], close, window=14).average_true_range().iloc[-1])
    adx   = float(ta.trend.ADXIndicator(df["High"], df["Low"], close, window=14).adx().iloc[-1])
    rsi   = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
    price = ema20 + 0.1 * atr   # just above EMA20 — triggers pullback strategies

    return dict(
        current_price=price, ema20=ema20, ema50=ema50, ema200=ema50 * 0.85,
        atr_14=atr, atr_pct=round(atr / price * 100, 2),
        adx_14=max(adx, 26.0),   # ensure above all ADX thresholds
        rsi_14=min(rsi, 60.0),   # ensure below RSI max
        macd_signal="bullish", trend="uptrend",
        nearest_support=price - atr * 2,
        nearest_resistance=price + atr * 3,
        interval="1d", ticker="XAUUSD",
        stoch_k=45.0, stoch_d=43.0, volume_trend="increasing",
    )
