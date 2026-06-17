"""
p3_backtester/strategies/triple_ema_cross.py — Triple EMA Cross strategy.

Concept:
  The Triple EMA Cross uses three EMAs: fast, medium, and slow. A bullish signal
  fires when the fast EMA crosses above the medium EMA while both are above the
  slow EMA. This "triple alignment" ensures the signal is a momentum continuation
  in an established trend, not a random counter-trend cross.

  Classic settings: 9/21/50 or 4/9/18 (aggressive) or 10/30/100 (conservative).

  This generates 20-40 signals/year on daily charts — high frequency relative
  to most indicators. Quality filter: requires the cross to occur in the right
  trend context (above slow EMA).

Logic:
  1. EMA(slow)[-1]: price is above slow EMA (uptrend established)
  2. EMA(medium)[-1] > EMA(slow)[-1]: medium trend aligned
  3. EMA(fast)[-2] <= EMA(medium)[-2]: fast was below medium (pullback occurred)
  4. EMA(fast)[-1] > EMA(medium)[-1]: fast crosses above medium (momentum resumes)
  5. ADX >= adx_threshold (trend has strength)
  6. RSI <= rsi_max (not overbought)

  SHORT: fast crosses below medium while both below slow EMA.

Entry / SL / TP:
  entry_low  = close
  entry_high = close + entry_buffer_atr * ATR
  SL = medium EMA - sl_buffer_atr * ATR  (cross level is the natural stop)
  TP1 = entry + tp1_r * risk
  TP2 = entry + tp2_r * risk
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


def _interval_trade_type(interval: str) -> str:
    return {"1m": "scalp", "5m": "scalp", "15m": "scalp",
            "30m": "intraday", "1h": "intraday",
            "1d": "swing", "1wk": "swing"}.get(interval, "swing")


def _interval_duration(interval: str) -> str:
    return {"1m": "5-30 min", "5m": "30-90 min", "15m": "1-3 hr",
            "30m": "2-6 hr", "1h": "4-12 hr",
            "1d": "3-8 days", "1wk": "3-8 wk"}.get(interval, "varies")


def _ev(wr: float, avg_r: float) -> float:
    return round(wr * avg_r - (1 - wr) * 1.0, 4)


def _pf(wr: float, avg_r: float) -> float:
    lr = 1 - wr
    return round((wr * avg_r) / lr, 3) if lr > 1e-10 else 999.0


class TripleEMACrossStrategy(StrategyBase):
    """
    Triple EMA Cross — enters when the fast EMA crosses the medium EMA
    while both are above the slow EMA, confirming trend continuation.
    """

    name = "triple_ema_cross"
    description = "Fast EMA crosses medium EMA while both above slow EMA — trend continuation"
    uses_bar_slicer = True

    def __init__(
        self,
        ema_fast: int = 9,
        ema_medium: int = 21,
        ema_slow: int = 50,
        adx_threshold: int = 20,
        rsi_max_long: int = 70,
        sl_buffer_atr: float = 0.1,
        entry_buffer_atr: float = 0.05,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.ema_fast           = ema_fast
        self.ema_medium         = ema_medium
        self.ema_slow           = ema_slow
        self.adx_threshold      = adx_threshold
        self.rsi_max_long       = rsi_max_long
        self.sl_buffer_atr      = sl_buffer_atr
        self.entry_buffer_atr   = entry_buffer_atr
        self.tp1_r              = tp1_r
        self.tp2_r              = tp2_r
        self.tp1_alloc          = tp1_alloc
        self.tp2_alloc          = tp2_alloc
        self.shorts_enabled     = shorts_enabled
        self.win_rate           = win_rate

    @property
    def params(self) -> dict:
        return {
            "ema_fast":       self.ema_fast,
            "ema_medium":     self.ema_medium,
            "ema_slow":       self.ema_slow,
            "adx_threshold":  self.adx_threshold,
            "tp1_r":          self.tp1_r,
            "tp2_r":          self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        if len(df) < self.ema_slow + 10:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        ema_f  = ta.trend.EMAIndicator(close, self.ema_fast).ema_indicator()
        ema_m  = ta.trend.EMAIndicator(close, self.ema_medium).ema_indicator()
        ema_s  = ta.trend.EMAIndicator(close, self.ema_slow).ema_indicator()

        ef_curr = float(ema_f.iloc[-1]); ef_prev = float(ema_f.iloc[-2])
        em_curr = float(ema_m.iloc[-1]); em_prev = float(ema_m.iloc[-2])
        es_curr = float(ema_s.iloc[-1])
        cl_curr = float(close.iloc[-1])

        if any(pd.isna(v) for v in [ef_curr, em_curr, es_curr]):
            return None

        # Bullish cross: fast crosses above medium
        long_cross  = (ef_prev <= em_prev) and (ef_curr > em_curr)
        short_cross = (ef_prev >= em_prev) and (ef_curr < em_curr)

        if not long_cross and not short_cross:
            return None
        if short_cross and not self.shorts_enabled:
            return None

        direction = "long" if long_cross else "short"

        # Slow EMA context filter
        if direction == "long" and (em_curr < es_curr or cl_curr < es_curr):
            return None
        if direction == "short" and (em_curr > es_curr or cl_curr > es_curr):
            return None

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        adx_val = float(ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1])
        if adx_val < self.adx_threshold:
            return None

        rsi_val = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
        if direction == "long" and rsi_val > self.rsi_max_long:
            return None

        buf_entry = self.entry_buffer_atr * atr_val
        buf_sl    = self.sl_buffer_atr * atr_val

        if direction == "long":
            entry_low  = round(cl_curr, 4)
            entry_high = round(cl_curr + buf_entry, 4)
            sl         = round(em_curr - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(cl_curr, 4)
            entry_low  = round(cl_curr - buf_entry, 4)
            sl         = round(em_curr + buf_sl, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        avg_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev    = _ev(self.win_rate, avg_r)
        pf    = _pf(self.win_rate, avg_r)
        atr_pct = tech.get("atr_pct", round(atr_val / cl_curr * 100, 3))
        trade_type = _interval_trade_type(interval)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Triple EMA Cross",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"EMA{self.ema_fast} ({ef_curr:.2f}) crossed "
                f"{'above' if direction == 'long' else 'below'} "
                f"EMA{self.ema_medium} ({em_curr:.2f}). "
                f"EMA{self.ema_medium} {'above' if direction == 'long' else 'below'} "
                f"EMA{self.ema_slow} ({es_curr:.2f}) — triple alignment confirmed. "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}."
            ),
            trigger=(
                f"EMA{self.ema_fast}/{self.ema_medium} cross confirmed — "
                f"enter above {entry_low:.4f}."
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
                f"EMA{self.ema_fast}={ef_curr:.2f} EMA{self.ema_medium}={em_curr:.2f} "
                f"EMA{self.ema_slow}={es_curr:.2f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=cl_curr,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"EMA{self.ema_fast} crosses back {'below' if direction == 'long' else 'above'} EMA{self.ema_medium}",
                description="Triple EMA alignment broken — cross reversed",
                price_trigger=sl,
                action="Exit — EMA cross reversed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"EMA{self.ema_fast} holds {'above' if direction == 'long' else 'below'} EMA{self.ema_medium} and price trends",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="EMA cross immediately reverses (whipsaw)",
                    outcome="NO TRADE — wait for cleaner cross",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
