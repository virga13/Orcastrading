"""
p3_backtester/strategies/fibonacci_pullback.py — Fibonacci Retracement Pullback strategy.

Concept:
  After a significant directional move (swing low -> swing high for longs), price
  retraces to a key Fibonacci level (38.2%, 50%, or 61.8%). The strategy enters at
  the Fib zone expecting the trend to resume.

  Fib levels computed from the swing range:
    fib_38 = swing_high - 0.382 * swing_range
    fib_50 = swing_high - 0.500 * swing_range
    fib_62 = swing_high - 0.618 * swing_range
  (Long: look for price at these levels as support)
  (Short: mirror from swing low upward)

Entry / SL / TP:
  entry = close near Fib zone
  SL    = swing_low - sl_buffer_atr * ATR (for longs)
  TP1   = swing_high (100% retrace target)
  TP2   = swing_high + 0.618 * swing_range (161.8% extension)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import ta

from p3_backtester.strategies.base import StrategyBase
from p1_analysis_engine.schema import (
    SetupsOutput, TradingSetup, SetupTarget,
    InvalidationScenario, DecisionTreeEntry,
)

_FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
_KEY_LEVELS  = [0.382, 0.500, 0.618]  # levels we actually trade

_HTF_RESAMPLE: dict[str, str] = {
    "1m": "15min", "5m": "1h", "15m": "4h",
    "30m": "1D", "1h": "1D", "4h": "1W",
    "1d": "1W", "1wk": "1ME",
}


def _interval_trade_type(interval: str) -> str:
    return {"1m": "scalp", "5m": "scalp", "15m": "scalp",
            "30m": "intraday", "1h": "intraday",
            "1d": "swing", "1wk": "swing"}.get(interval, "swing")


def _interval_duration(interval: str) -> str:
    return {"1m": "5-30 minutes", "5m": "30-90 minutes", "15m": "1-3 hours",
            "30m": "2-6 hours", "1h": "4-12 hours",
            "1d": "3-8 days", "1wk": "3-8 weeks"}.get(interval, "varies")


def _ev(wr: float, avg_r: float) -> float:
    return round(wr * avg_r - (1 - wr) * 1.0, 4)


def _pf(wr: float, avg_r: float) -> float:
    lr = 1 - wr
    return round((wr * avg_r) / lr, 3) if lr > 1e-10 else 999.0


def _find_swing_for_long(
    high: pd.Series, low: pd.Series, lookback: int, min_swing_atr: float, atr_val: float
) -> tuple[float, float] | None:
    """
    Identify a bullish swing: swing low (earlier) -> swing high (later) -> current pullback.
    Returns (swing_low, swing_high) or None.
    """
    h = high.iloc[-lookback:]
    l = low.iloc[-lookback:]
    n = len(h)

    sh_pos = int(np.argmax(h.values))
    sh_val = float(h.iloc[sh_pos])

    if sh_pos >= n - 3:
        return None

    sl_val = float(l.iloc[:sh_pos + 1].min())
    swing_size = sh_val - sl_val

    if swing_size < min_swing_atr * atr_val or swing_size <= 0:
        return None

    return sl_val, sh_val


def _find_swing_for_short(
    high: pd.Series, low: pd.Series, lookback: int, min_swing_atr: float, atr_val: float
) -> tuple[float, float] | None:
    """
    Identify a bearish swing: swing high (earlier) -> swing low (later) -> current pullback.
    Returns (swing_low, swing_high) or None.
    """
    h = high.iloc[-lookback:]
    l = low.iloc[-lookback:]
    n = len(l)

    sl_pos = int(np.argmin(l.values))
    sl_val = float(l.iloc[sl_pos])

    if sl_pos >= n - 3:
        return None

    sh_val = float(h.iloc[:sl_pos + 1].max())
    swing_size = sh_val - sl_val

    if swing_size < min_swing_atr * atr_val or swing_size <= 0:
        return None

    return sl_val, sh_val


def _nearest_fib_level(price: float, swing_low: float, swing_high: float, levels: list[float]) -> tuple[float, float] | None:
    """Return (fib_price, fib_ratio) of the nearest key Fibonacci level to price."""
    swing_range = swing_high - swing_low
    if swing_range <= 0:
        return None
    candidates = [(swing_high - r * swing_range, r) for r in levels]
    nearest = min(candidates, key=lambda x: abs(x[0] - price))
    return nearest


class FibonacciPullbackStrategy(StrategyBase):
    """
    Fibonacci Retracement Pullback strategy.

    Waits for a significant trend move, then enters when price pulls back
    to a key Fibonacci level (38.2%, 50%, or 61.8%) with trend confirmation.
    """

    name = "fibonacci_pullback"
    description = "Fib retracement pullback — enter at 38.2/50/61.8% after a trend move"
    uses_bar_slicer = True

    def __init__(
        self,
        swing_lookback: int = 30,
        min_swing_atr: float = 3.0,
        fib_zone_atr: float = 0.5,
        ema_period: int = 50,
        adx_threshold: int = 20,
        rsi_max_long: int = 65,
        rsi_min_short: int = 35,
        sl_buffer_atr: float = 0.15,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = True,
        win_rate: float = 0.45,
    ):
        self.swing_lookback  = swing_lookback
        self.min_swing_atr   = min_swing_atr
        self.fib_zone_atr    = fib_zone_atr
        self.ema_period      = ema_period
        self.adx_threshold   = adx_threshold
        self.rsi_max_long    = rsi_max_long
        self.rsi_min_short   = rsi_min_short
        self.sl_buffer_atr   = sl_buffer_atr
        self.tp1_r           = tp1_r
        self.tp2_r           = tp2_r
        self.tp1_alloc       = tp1_alloc
        self.tp2_alloc       = tp2_alloc
        self.shorts_enabled  = shorts_enabled
        self.win_rate        = win_rate

    @property
    def params(self) -> dict:
        return {
            "swing_lookback": self.swing_lookback,
            "min_swing_atr":  self.min_swing_atr,
            "fib_zone_atr":   self.fib_zone_atr,
            "ema_period":     self.ema_period,
            "adx_threshold":  self.adx_threshold,
            "tp1_r":          self.tp1_r,
            "tp2_r":          self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = self.swing_lookback + self.ema_period + 10
        if len(df) < min_bars:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        price = float(close.iloc[-1])

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        rsi_val = float(
            ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        )
        adx_val = float(
            ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
        )
        ema_val = float(
            ta.trend.EMAIndicator(close, window=self.ema_period).ema_indicator().iloc[-1]
        )

        if adx_val < self.adx_threshold:
            return None

        direction: str | None = None
        swing_low = swing_high = 0.0
        fib_val = fib_ratio = 0.0

        if price > ema_val and rsi_val <= self.rsi_max_long:
            result = _find_swing_for_long(high, low, self.swing_lookback, self.min_swing_atr, atr_val)
            if result is not None:
                swing_low, swing_high = result
                if swing_low < price < swing_high:
                    nearest = _nearest_fib_level(price, swing_low, swing_high, _KEY_LEVELS)
                    if nearest is not None:
                        fib_val, fib_ratio = nearest
                        if abs(price - fib_val) <= self.fib_zone_atr * atr_val:
                            direction = "long"

        if direction is None and self.shorts_enabled and price < ema_val and rsi_val >= self.rsi_min_short:
            result = _find_swing_for_short(high, low, self.swing_lookback, self.min_swing_atr, atr_val)
            if result is not None:
                swing_low, swing_high = result
                if swing_low < price < swing_high:
                    nearest = _nearest_fib_level(
                        price, swing_low, swing_high,
                        [1.0 - r for r in _KEY_LEVELS]
                    )
                    if nearest is not None:
                        fib_val, fib_ratio = nearest
                        if abs(price - fib_val) <= self.fib_zone_atr * atr_val:
                            direction = "short"

        if direction is None:
            return None

        swing_range = swing_high - swing_low
        buf = self.sl_buffer_atr * atr_val

        if direction == "long":
            entry_low  = round(price - 0.2 * atr_val, 4)
            entry_high = round(price + 0.2 * atr_val, 4)
            sl         = round(swing_low - buf, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_low  = round(price - 0.2 * atr_val, 4)
            entry_high = round(price + 0.2 * atr_val, 4)
            sl         = round(swing_high + buf, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        fib_pct_str = f"{fib_ratio * 100:.1f}%"
        avg_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev    = _ev(self.win_rate, avg_r)
        pf    = _pf(self.win_rate, avg_r)
        atr_pct = tech.get("atr_pct", round(atr_val / price * 100, 3))
        trade_type = _interval_trade_type(interval)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Fib {fib_pct_str} Pullback",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Price pulled back to {fib_pct_str} Fibonacci level ({fib_val:.4f}) "
                f"after a swing move of {swing_range:.4f} ({swing_range / atr_val:.1f} ATRs). "
                f"EMA{self.ema_period}={ema_val:.4f}, ADX={adx_val:.1f}, RSI={rsi_val:.1f}."
            ),
            trigger=(
                f"Price within {self.fib_zone_atr} ATR of {fib_pct_str} Fib level {fib_val:.4f}."
            ),
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=sl,
            trailing_sl_to_breakeven=tp1,
            targets=[
                SetupTarget(price=tp1, label=f"Target 1 ({self.tp1_alloc}%)", allocation_pct=self.tp1_alloc),
                SetupTarget(price=tp2, label=f"Target 2 ({self.tp2_alloc}%)", allocation_pct=self.tp2_alloc),
            ],
            rr_ratio=round(avg_r, 2),
            win_rate_estimate=self.win_rate,
            trade_duration=_interval_duration(interval),
            ev=ev,
            profit_factor=pf,
            confidence=0.45,
            confidence_note=(
                f"Fib {fib_pct_str} at {fib_val:.4f} | Swing {swing_low:.4f}-{swing_high:.4f} | "
                f"ADX={adx_val:.1f} | RSI={rsi_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes {'below' if direction == 'long' else 'above'} swing {'low' if direction == 'long' else 'high'} {swing_low if direction == 'long' else swing_high:.4f}",
                description="Swing structure invalidated — trend reversal",
                price_trigger=sl,
                action="Exit — Fibonacci level failed as support/resistance",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Price holds {fib_pct_str} Fib level and bounces",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=price,
                ),
                DecisionTreeEntry(
                    scenario=f"Price breaks {'below' if direction == 'long' else 'above'} swing {'low' if direction == 'long' else 'high'}",
                    outcome="NO TRADE — swing structure broken",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
