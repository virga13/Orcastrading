"""
p3_backtester/strategies/roc_momentum.py — Rate of Change (ROC) Momentum.

Concept:
  Rate of Change measures the percentage change in price over N periods:
    ROC = (Close[0] - Close[-N]) / Close[-N] * 100

  When ROC crosses from negative to positive (zero-line cross), it signals
  momentum flipping bullish. Unlike RSI/Stochastic which are mean-reverting,
  ROC can sustain long trends above zero — making it a trend-following signal.

  Key advantage: ROC generates 30-60 zero-crosses per year on daily charts,
  making it one of the most signal-rich indicators. With trend and ADX filters
  to reduce whipsaws, it becomes a momentum continuation tool.

Logic:
  1. ROC(roc_period)[-2] <= 0  AND  ROC[-1] > 0  (zero-line cross up)
  2. Close > EMA(ema_period)  (uptrend context)
  3. ADX >= adx_threshold  (trend has strength)
  4. RSI <= rsi_max  (not overbought)
  5. Optionally: ROC magnitude >= min_roc_pct after cross (momentum has strength)

  SHORT: ROC crosses from positive to negative with price below EMA.

Entry / SL / TP:
  entry_low  = close
  entry_high = close + entry_buffer_atr * ATR
  SL = lowest low of sl_lookback bars - sl_buffer_atr * ATR
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


class ROCMomentumStrategy(StrategyBase):
    """
    Rate of Change zero-line cross — enters when ROC flips positive in
    an established uptrend, signalling momentum resumption.
    """

    name = "roc_momentum"
    description = "ROC zero-line crossup in uptrend with EMA and ADX gate — momentum resumption"
    uses_bar_slicer = True

    def __init__(
        self,
        roc_period: int = 10,
        min_roc_pct: float = 0.0,
        ema_period: int = 50,
        adx_threshold: int = 20,
        rsi_max_long: int = 70,
        sl_lookback: int = 5,
        entry_buffer_atr: float = 0.05,
        sl_buffer_atr: float = 0.1,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.roc_period       = roc_period
        self.min_roc_pct      = min_roc_pct
        self.ema_period       = ema_period
        self.adx_threshold    = adx_threshold
        self.rsi_max_long     = rsi_max_long
        self.sl_lookback      = sl_lookback
        self.entry_buffer_atr = entry_buffer_atr
        self.sl_buffer_atr    = sl_buffer_atr
        self.tp1_r            = tp1_r
        self.tp2_r            = tp2_r
        self.tp1_alloc        = tp1_alloc
        self.tp2_alloc        = tp2_alloc
        self.shorts_enabled   = shorts_enabled
        self.win_rate         = win_rate

    @property
    def params(self) -> dict:
        return {
            "roc_period":    self.roc_period,
            "min_roc_pct":   self.min_roc_pct,
            "ema_period":    self.ema_period,
            "adx_threshold": self.adx_threshold,
            "tp1_r":         self.tp1_r,
            "tp2_r":         self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        if len(df) < max(self.ema_period + 10, self.roc_period + 5):
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        roc = close.pct_change(self.roc_period) * 100.0
        roc_curr = float(roc.iloc[-1])
        roc_prev = float(roc.iloc[-2])

        if pd.isna(roc_curr) or pd.isna(roc_prev):
            return None

        long_cross  = (roc_prev <= 0) and (roc_curr > self.min_roc_pct)
        short_cross = (roc_prev >= 0) and (roc_curr < -self.min_roc_pct)

        if not long_cross and not short_cross:
            return None
        if short_cross and not self.shorts_enabled:
            return None

        direction = "long" if long_cross else "short"

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        curr_close = float(close.iloc[-1])
        ema_val    = float(ta.trend.EMAIndicator(close, self.ema_period).ema_indicator().iloc[-1])

        if direction == "long" and curr_close < ema_val:
            return None
        if direction == "short" and curr_close > ema_val:
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
            entry_low  = round(curr_close, 4)
            entry_high = round(curr_close + buf_entry, 4)
            sl_price   = float(low.iloc[-self.sl_lookback:].min())
            sl         = round(sl_price - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(curr_close, 4)
            entry_low  = round(curr_close - buf_entry, 4)
            sl_price   = float(high.iloc[-self.sl_lookback:].max())
            sl         = round(sl_price + buf_sl, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        avg_r   = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev      = _ev(self.win_rate, avg_r)
        pf      = _pf(self.win_rate, avg_r)
        atr_pct = tech.get("atr_pct", round(atr_val / curr_close * 100, 3))
        trade_type = _interval_trade_type(interval)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} ROC Momentum Cross",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"ROC({self.roc_period}) crossed {'above' if direction == 'long' else 'below'} zero "
                f"(prev={roc_prev:.2f}%, curr={roc_curr:.2f}%). "
                f"Price {'above' if direction == 'long' else 'below'} EMA{self.ema_period}={ema_val:.4f}. "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"Momentum flipped {'bullish' if direction == 'long' else 'bearish'} in trend."
            ),
            trigger=f"ROC turned {'positive' if direction == 'long' else 'negative'} — enter above {entry_low:.4f}.",
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
                f"ROC{self.roc_period}={roc_curr:.2f}% (crossed 0) | "
                f"EMA{self.ema_period}={ema_val:.4f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"ROC drops back {'below' if direction == 'long' else 'above'} zero",
                description="ROC momentum reversed — cross failed",
                price_trigger=sl,
                action="Exit — ROC whipsaw",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"ROC continues {'higher' if direction == 'long' else 'lower'}, price trends",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="ROC drops back below zero — whipsaw",
                    outcome="NO TRADE — wait for sustained ROC cross",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
