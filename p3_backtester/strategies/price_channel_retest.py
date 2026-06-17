"""
p3_backtester/strategies/price_channel_retest.py — Price Channel Retest (Breakout Pullback).

Concept:
  After a Donchian/N-day high breakout, price often pulls back to retest the
  old resistance level (now support). Entering on the RETEST rather than the
  initial breakout gives a better R:R — tighter SL (just below the old
  channel high) and the trend is already established.

  This avoids the "chasing" problem of breakout entries and instead catches
  the continuation move after the retest confirms support.

  Signal: price broke above N-day high within the last M bars, then pulled
  back to within atr_zone of the breakout level, and now closes above it again.

  Estimated frequency: 15-30 signals/year on daily (depends on N and retest tolerance).

Logic:
  1. breakout_high = max close over lookback_period N bars (excluding current)
     Price broke above breakout_high within the last retest_window bars
  2. Current close is within retest_atr_zone * ATR of breakout_high (pullback to zone)
  3. Current close > breakout_high (re-cleared the level)
  4. close > EMA(ema_period)  (uptrend context)
  5. ADX >= adx_threshold  (trend still has strength)
  6. RSI <= rsi_max_long  (not overbought)

Entry / SL / TP:
  entry_low  = close
  entry_high = close + entry_buffer_atr * ATR
  SL = breakout_high - sl_buffer_atr * ATR  (below the retest support)
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


class PriceChannelRetestStrategy(StrategyBase):
    """
    Retest of a broken N-day high — enters on pullback to old resistance
    (now support) after initial breakout, for better R:R than chasing.
    """

    name = "price_channel_retest"
    description = "Retest entry after N-day high breakout — continuation after pullback to support"
    uses_bar_slicer = True

    def __init__(
        self,
        lookback_period: int = 20,
        retest_window: int = 10,
        retest_atr_zone: float = 1.0,
        ema_period: int = 50,
        adx_threshold: int = 15,
        rsi_max_long: int = 70,
        entry_buffer_atr: float = 0.05,
        sl_buffer_atr: float = 0.15,
        tp1_r: float = 2.0,
        tp2_r: float = 5.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.lookback_period  = lookback_period
        self.retest_window    = retest_window
        self.retest_atr_zone  = retest_atr_zone
        self.ema_period       = ema_period
        self.adx_threshold    = adx_threshold
        self.rsi_max_long     = rsi_max_long
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
            "lookback_period": self.lookback_period,
            "retest_window":   self.retest_window,
            "retest_atr_zone": self.retest_atr_zone,
            "ema_period":      self.ema_period,
            "adx_threshold":   self.adx_threshold,
            "tp1_r":           self.tp1_r,
            "tp2_r":           self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = max(self.ema_period + 10, self.lookback_period + self.retest_window + 10)
        if len(df) < min_bars:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        curr_close = float(close.iloc[-1])

        # The reference breakout level: N-day high excluding the current bar and the retest window
        ref_end   = len(df) - 1 - self.retest_window
        ref_start = ref_end - self.lookback_period
        if ref_start < 0:
            return None

        breakout_high = float(close.iloc[ref_start:ref_end].max())
        breakout_low  = float(close.iloc[ref_start:ref_end].min())

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        zone = self.retest_atr_zone * atr_val

        # LONG: check breakout above N-day high then retest
        long_signal  = False
        short_signal = False

        # Did price break above the channel high during the retest window?
        window_closes = close.iloc[-(self.retest_window):].values
        window_highs  = high.iloc[-(self.retest_window):].values
        window_lows   = low.iloc[-(self.retest_window):].values

        broke_above = any(c > breakout_high for c in window_closes[:-1])
        pulled_back = abs(curr_close - breakout_high) <= zone
        reconfirmed = curr_close > breakout_high

        if broke_above and pulled_back and reconfirmed:
            long_signal = True

        # SHORT: mirror logic
        if self.shorts_enabled:
            broke_below = any(c < breakout_low for c in window_closes[:-1])
            pulled_back_short = abs(curr_close - breakout_low) <= zone
            reconfirmed_short = curr_close < breakout_low
            if broke_below and pulled_back_short and reconfirmed_short:
                short_signal = True

        if not long_signal and not short_signal:
            return None

        direction = "long" if long_signal else "short"

        ema_val = float(ta.trend.EMAIndicator(close, self.ema_period).ema_indicator().iloc[-1])
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
            sl         = round(breakout_high - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(curr_close, 4)
            entry_low  = round(curr_close - buf_entry, 4)
            sl         = round(breakout_low + buf_sl, 4)
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

        level_str = f"{breakout_high:.4f}" if direction == "long" else f"{breakout_low:.4f}"

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Price Channel Retest",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"{self.lookback_period}-day channel {'high' if direction == 'long' else 'low'} "
                f"at {level_str} was broken and retested. "
                f"Old resistance now confirmed as support. "
                f"EMA{self.ema_period}={ema_val:.4f}, ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"Continuation entry with tight SL at the breakout level."
            ),
            trigger=f"Retest of {level_str} holding — enter above {entry_low:.4f}.",
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
            confidence=0.55,
            confidence_note=(
                f"Retest of {self.lookback_period}d channel level {level_str} | "
                f"EMA{self.ema_period}={ema_val:.4f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes {'below' if direction == 'long' else 'above'} {level_str} — retest failed",
                description="Old breakout level rejected as support",
                price_trigger=sl,
                action="Exit — retest failure",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Price holds above retest level and continues {'up' if direction == 'long' else 'down'}",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="Retest level breaks — resistance holds",
                    outcome="NO TRADE — breakout was false; wait for new structure",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
