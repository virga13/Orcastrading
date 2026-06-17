"""
p3_backtester/strategies/supertrend.py — Supertrend trend-following strategy.

Concept:
  The Supertrend indicator draws an ATR-based trailing band that flips between
  acting as a support (green, uptrend) and resistance (red, downtrend). A signal
  fires on the first bar where price crosses the band and the trend flips.

  Formula:
    hl2         = (High + Low) / 2
    basic_upper = hl2 + mult × ATR(period)
    basic_lower = hl2 - mult × ATR(period)
    upper_band  = min(basic_upper, prev_upper) unless close[i-1] > prev_upper
    lower_band  = max(basic_lower, prev_lower) unless close[i-1] < prev_lower
    supertrend  = lower_band (uptrend) | upper_band (downtrend)

  Flip to uptrend:   prev supertrend = upper_band AND close > upper_band
  Flip to downtrend: prev supertrend = lower_band AND close < lower_band

Entry / SL / TP:
  entry = close on crossover bar
  SL    = supertrend line value (the trailing stop) + sl_buffer × ATR
  TP1   = entry + tp1_r × risk  |  TP2 = entry + tp2_r × risk
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


def _compute_supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, mult: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns (supertrend_line, trend_direction) where trend_direction is
    +1 (uptrend / green) or -1 (downtrend / red).
    """
    atr = ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr

    n = len(close)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    supertrend  = np.full(n, np.nan)
    trend       = np.zeros(n, dtype=int)

    for i in range(1, n):
        bu = float(basic_upper.iloc[i])
        bl = float(basic_lower.iloc[i])
        cl_prev = float(close.iloc[i - 1])

        # Upper band: only tightens unless prev close broke above it
        if np.isnan(upper_band[i - 1]) or bu < upper_band[i - 1] or cl_prev > upper_band[i - 1]:
            upper_band[i] = bu
        else:
            upper_band[i] = upper_band[i - 1]

        # Lower band: only rises unless prev close broke below it
        if np.isnan(lower_band[i - 1]) or bl > lower_band[i - 1] or cl_prev < lower_band[i - 1]:
            lower_band[i] = bl
        else:
            lower_band[i] = lower_band[i - 1]

        # Determine trend direction
        if np.isnan(supertrend[i - 1]):
            supertrend[i] = upper_band[i]
            trend[i] = -1
        elif supertrend[i - 1] == upper_band[i - 1]:
            cl_cur = float(close.iloc[i])
            if cl_cur > upper_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:  # supertrend[i-1] == lower_band[i-1]
            cl_cur = float(close.iloc[i])
            if cl_cur < lower_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1

    return (
        pd.Series(supertrend, index=close.index),
        pd.Series(trend, index=close.index),
    )


class SupertrendStrategy(StrategyBase):
    """
    Supertrend crossover strategy.

    Enters on the bar where the Supertrend indicator flips direction,
    using the Supertrend line as a dynamic stop-loss.
    """

    name = "supertrend"
    description = "Supertrend crossover — ATR trailing stop flips direction"
    uses_bar_slicer = True

    def __init__(
        self,
        atr_period: int = 10,
        multiplier: float = 3.0,
        sl_buffer_atr: float = 0.05,
        htf_filter: bool = True,
        tp1_r: float = 1.5,
        tp2_r: float = 3.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = True,
        win_rate: float = 0.45,
    ):
        self.atr_period    = atr_period
        self.multiplier    = multiplier
        self.sl_buffer_atr = sl_buffer_atr
        self.htf_filter    = htf_filter
        self.tp1_r         = tp1_r
        self.tp2_r         = tp2_r
        self.tp1_alloc     = tp1_alloc
        self.tp2_alloc     = tp2_alloc
        self.shorts_enabled = shorts_enabled
        self.win_rate      = win_rate

    @property
    def params(self) -> dict:
        return {
            "atr_period":  self.atr_period,
            "multiplier":  self.multiplier,
            "htf_filter":  self.htf_filter,
            "tp1_r":       self.tp1_r,
            "tp2_r":       self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = self.atr_period * 3 + 2
        if len(df) < min_bars:
            return None

        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]
        interval = tech.get("interval", "1d")

        st_line, trend = _compute_supertrend(high, low, close, self.atr_period, self.multiplier)

        # Signal: trend must flip on the CURRENT bar (zero-lookahead)
        curr_trend = int(trend.iloc[-1])
        prev_trend = int(trend.iloc[-2])

        if curr_trend == prev_trend or curr_trend == 0 or prev_trend == 0:
            return None  # no flip

        direction = "long" if curr_trend == 1 else "short"

        if direction == "short" and not self.shorts_enabled:
            return None

        # ── HTF trend filter ────────────────────────────────────────────────────
        if self.htf_filter:
            htf_rule = _HTF_RESAMPLE.get(interval, "1W")
            try:
                htf_close = close.resample(htf_rule).last().dropna()
                if len(htf_close) >= 50:
                    htf_ema20 = float(ta.trend.EMAIndicator(htf_close, 20).ema_indicator().iloc[-1])
                    htf_ema50 = float(ta.trend.EMAIndicator(htf_close, 50).ema_indicator().iloc[-1])
                    if direction == "long" and htf_ema20 < htf_ema50:
                        return None  # HTF bearish — skip long
                    if direction == "short" and htf_ema20 > htf_ema50:
                        return None  # HTF bullish — skip short
            except Exception:
                pass

        # ── Entry / SL / TP ────────────────────────────────────────────────────
        price   = float(close.iloc[-1])
        st_val  = float(st_line.iloc[-1])
        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        buf = self.sl_buffer_atr * atr_val

        if direction == "long":
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(st_val - buf, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(st_val + buf, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        avg_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev    = _ev(self.win_rate, avg_r)
        pf    = _pf(self.win_rate, avg_r)
        atr_pct = tech.get("atr_pct", round(atr_val / price * 100, 3))
        trade_type = _interval_trade_type(interval)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Supertrend flip",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Supertrend flipped to {'UPTREND' if direction == 'long' else 'DOWNTREND'} "
                f"— price {price:.4f} crossed {'above' if direction == 'long' else 'below'} "
                f"Supertrend line {st_val:.4f} (period={self.atr_period}, mult={self.multiplier})."
            ),
            trigger=(
                f"Supertrend crossover bar — trend flipped from "
                f"{'downtrend to uptrend' if direction == 'long' else 'uptrend to downtrend'}."
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
            confidence=0.50,
            confidence_note=(
                f"Supertrend line={st_val:.4f} | ATR={atr_val:.4f} | "
                f"Trend flipped from {prev_trend:+d} to {curr_trend:+d}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Supertrend flips back {'bearish' if direction == 'long' else 'bullish'}",
                description="Trend reversal — Supertrend crossover in opposite direction",
                price_trigger=sl,
                action="Exit immediately — trend reversed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Supertrend flips {direction} and holds",
                    outcome=f"ENTER {direction.upper()} — SL at Supertrend ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=price,
                ),
                DecisionTreeEntry(
                    scenario=f"Price returns {'below' if direction == 'long' else 'above'} Supertrend ${st_val:.4f}",
                    outcome="NO TRADE — crossover failed",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
