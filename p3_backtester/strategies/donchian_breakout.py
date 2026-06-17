"""
p3_backtester/strategies/donchian_breakout.py — Donchian Channel Breakout (Turtle Trading).

Concept:
  The Donchian Channel plots the highest high and lowest low over a lookback period.
  A breakout above the upper band signals a long entry; below the lower band, short.
  This is the core of the original Turtle Trading system (Richard Dennis, 1983),
  one of the most thoroughly documented trend-following systems in trading history.

  Key differences vs MomentumBreakout:
  - No volume requirement — pure price channel only
  - Exit rule: price crosses the exit_period channel (shorter than entry_period)
  - ATR-based position sizing baked into the signal (original turtle used 1N = 1 ATR)

Logic:
  1. Current close > highest high of last entry_period bars (excluding current) -> long
  2. Current close < lowest  low  of last entry_period bars (excluding current) -> short
  3. Trend filter: daily EMA50 direction for long/short gate
  4. ADX >= adx_threshold (trend has strength — avoid flat channels)
  5. Skip if price already ran more than max_breakout_atr ATRs past the channel edge
     (late entry filter — don't chase extended moves)

Entry / SL / TP:
  entry      = close (breakout bar)
  SL (long)  = highest high of entry_period - exit_period * ATR  [turtle-style]
              or swing low of last sl_lookback bars (whichever is wider for safety)
  TP1        = entry + tp1_r * risk
  TP2        = entry + tp2_r * risk
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


class DonchianBreakoutStrategy(StrategyBase):
    """
    Donchian Channel Breakout — Turtle Trading variant.

    Enters when price closes above (long) or below (short) the N-period
    highest high / lowest low. Inspired by the original Turtle Trading rules
    adapted for modern multi-asset portfolios.
    """

    name = "donchian_breakout"
    description = "Donchian channel breakout — N-period high/low breakout (Turtle Trading)"
    uses_bar_slicer = True

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        adx_threshold: int = 20,
        max_breakout_atr: float = 2.0,
        sl_buffer_atr: float = 0.1,
        htf_filter: bool = True,
        tp1_r: float = 2.0,
        tp2_r: float = 5.0,
        tp1_alloc: int = 50,
        tp2_alloc: int = 50,
        shorts_enabled: bool = True,
        win_rate: float = 0.40,
    ):
        self.entry_period      = entry_period
        self.exit_period       = exit_period
        self.adx_threshold     = adx_threshold
        self.max_breakout_atr  = max_breakout_atr
        self.sl_buffer_atr     = sl_buffer_atr
        self.htf_filter        = htf_filter
        self.tp1_r             = tp1_r
        self.tp2_r             = tp2_r
        self.tp1_alloc         = tp1_alloc
        self.tp2_alloc         = tp2_alloc
        self.shorts_enabled    = shorts_enabled
        self.win_rate          = win_rate

    @property
    def params(self) -> dict:
        return {
            "entry_period":  self.entry_period,
            "exit_period":   self.exit_period,
            "adx_threshold": self.adx_threshold,
            "tp1_r":         self.tp1_r,
            "tp2_r":         self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = self.entry_period + 20
        if len(df) < min_bars:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        # Channel computed on bars BEFORE current bar (zero-lookahead)
        channel_high = float(high.iloc[-(self.entry_period + 1):-1].max())
        channel_low  = float(low.iloc[-(self.entry_period + 1):-1].min())
        curr_close   = float(close.iloc[-1])

        long_breakout  = curr_close > channel_high
        short_breakout = curr_close < channel_low

        if not long_breakout and not short_breakout:
            return None

        if short_breakout and not self.shorts_enabled:
            return None

        direction = "long" if long_breakout else "short"

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        # Late-entry filter: skip if price already ran too far past the channel
        breakout_dist = (curr_close - channel_high) if direction == "long" else (channel_low - curr_close)
        if breakout_dist > self.max_breakout_atr * atr_val:
            return None

        adx_val = float(
            ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
        )
        if adx_val < self.adx_threshold:
            return None

        # HTF trend alignment
        if self.htf_filter:
            htf_rule = _HTF_RESAMPLE.get(interval, "1W")
            try:
                htf_close = close.resample(htf_rule).last().dropna()
                if len(htf_close) >= 50:
                    htf_ema20 = float(ta.trend.EMAIndicator(htf_close, 20).ema_indicator().iloc[-1])
                    htf_ema50 = float(ta.trend.EMAIndicator(htf_close, 50).ema_indicator().iloc[-1])
                    if direction == "long"  and htf_ema20 < htf_ema50:
                        return None
                    if direction == "short" and htf_ema20 > htf_ema50:
                        return None
            except Exception:
                pass

        # Exit channel (shorter period) used as SL basis
        exit_high = float(high.iloc[-(self.exit_period + 1):-1].max())
        exit_low  = float(low.iloc[-(self.exit_period + 1):-1].min())
        buf = self.sl_buffer_atr * atr_val

        if direction == "long":
            entry_low  = round(curr_close, 4)
            entry_high = round(curr_close, 4)
            sl         = round(exit_low - buf, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_low  = round(curr_close, 4)
            entry_high = round(curr_close, 4)
            sl         = round(exit_high + buf, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        avg_r      = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev         = _ev(self.win_rate, avg_r)
        pf         = _pf(self.win_rate, avg_r)
        price      = curr_close
        atr_pct    = tech.get("atr_pct", round(atr_val / price * 100, 3))
        trade_type = _interval_trade_type(interval)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Donchian {self.entry_period}-bar Breakout",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Price {curr_close:.4f} broke {'above' if direction == 'long' else 'below'} "
                f"{self.entry_period}-bar Donchian {'high' if direction == 'long' else 'low'} "
                f"at {channel_high if direction == 'long' else channel_low:.4f}. "
                f"Breakout dist: {breakout_dist:.4f} ({breakout_dist/atr_val:.1f} ATRs). "
                f"ADX={adx_val:.1f}."
            ),
            trigger=(
                f"Close above {self.entry_period}-period channel {'high' if direction == 'long' else 'low'} "
                f"{channel_high if direction == 'long' else channel_low:.4f}."
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
                f"Donchian {self.entry_period}-bar channel: H={channel_high:.4f} L={channel_low:.4f} | "
                f"ADX={adx_val:.1f} | ATR={atr_val:.4f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes {'below' if direction == 'long' else 'above'} {self.exit_period}-bar exit channel",
                description="Breakout failed — price returned inside the channel",
                price_trigger=sl,
                action="Exit — Donchian breakout reversed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Price holds above {self.entry_period}-bar high and trends",
                    outcome=f"ENTER {direction.upper()} — SL at {self.exit_period}-bar low ${sl:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=price,
                ),
                DecisionTreeEntry(
                    scenario="Price reverses back inside channel within 1-2 bars",
                    outcome="NO TRADE — false breakout, wait for re-test",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
