"""
p3_backtester/strategies/trix_signal.py — TRIX Zero-Line Cross.

Concept:
  TRIX is the 1-period percent rate-of-change of a triple-smoothed EMA:
    EMA1 = EMA(close, period)
    EMA2 = EMA(EMA1, period)
    EMA3 = EMA(EMA2, period)
    TRIX = (EMA3[-1] - EMA3[-2]) / EMA3[-2] * 100

  Triple smoothing removes short-term noise while retaining trend direction.
  A zero-line cross (TRIX flipping positive) signals sustained bullish momentum —
  stronger than ROC because three layers of smoothing filter out false crosses.

  Key advantage over ROC: far fewer whipsaws due to triple smoothing.
  Estimated signal frequency: 10-25 zero-crosses per year on daily
  (depending on TRIX period — shorter = more signals but noisier).

Logic:
  1. TRIX[-2] <= 0  AND  TRIX[-1] > 0  (zero-line cross up)
  2. close > EMA(ema_period)  (uptrend context)
  3. ADX >= adx_threshold  (trend has strength)
  4. RSI <= rsi_max_long  (not overbought)

  SHORT: TRIX crosses from positive to negative below EMA.

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


def _compute_trix(close: pd.Series, period: int) -> pd.Series:
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = ema3.pct_change() * 100.0
    return trix


class TRIXSignalStrategy(StrategyBase):
    """
    TRIX zero-line cross in uptrend — triple-smoothed ROC flip from negative
    to positive signals strong sustained momentum with minimal whipsaws.
    """

    name = "trix_signal"
    description = "TRIX zero-line cross-up in uptrend — triple-smoothed momentum resumption"
    uses_bar_slicer = True

    def __init__(
        self,
        trix_period: int = 9,
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
        self.trix_period      = trix_period
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
            "trix_period":   self.trix_period,
            "ema_period":    self.ema_period,
            "adx_threshold": self.adx_threshold,
            "tp1_r":         self.tp1_r,
            "tp2_r":         self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = max(self.ema_period + 10, self.trix_period * 3 + 10)
        if len(df) < min_bars:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        trix = _compute_trix(close, self.trix_period)
        trix_curr = float(trix.iloc[-1])
        trix_prev = float(trix.iloc[-2])

        if pd.isna(trix_curr) or pd.isna(trix_prev):
            return None

        long_cross  = (trix_prev <= 0) and (trix_curr > 0)
        short_cross = (trix_prev >= 0) and (trix_curr < 0)

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

        avg_r      = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev         = _ev(self.win_rate, avg_r)
        pf         = _pf(self.win_rate, avg_r)
        atr_pct    = tech.get("atr_pct", round(atr_val / curr_close * 100, 3))
        trade_type = _interval_trade_type(interval)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} TRIX Zero-Line Cross",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"TRIX({self.trix_period}) crossed {'above' if direction == 'long' else 'below'} zero "
                f"(prev={trix_prev:.4f}%, curr={trix_curr:.4f}%). "
                f"Price {'above' if direction == 'long' else 'below'} EMA{self.ema_period}={ema_val:.4f}. "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"Triple-smoothed momentum flipped {'bullish' if direction == 'long' else 'bearish'}."
            ),
            trigger=f"TRIX turned {'positive' if direction == 'long' else 'negative'} — enter above {entry_low:.4f}.",
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
                f"TRIX{self.trix_period}={trix_curr:.4f}% (crossed 0) | "
                f"EMA{self.ema_period}={ema_val:.4f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"TRIX drops back {'below' if direction == 'long' else 'above'} zero",
                description="TRIX momentum reversed — cross failed",
                price_trigger=sl,
                action="Exit — TRIX reversal",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"TRIX continues {'higher' if direction == 'long' else 'lower'}, price trends",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="TRIX drops back below zero — cross failed",
                    outcome="NO TRADE — wait for sustained TRIX cross",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
