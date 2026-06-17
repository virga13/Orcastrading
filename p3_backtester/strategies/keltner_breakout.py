"""
p3_backtester/strategies/keltner_breakout.py — Keltner Channel Breakout.

Concept:
  Keltner Channels are ATR-based envelopes around an EMA:
    KC_mid   = EMA(ema_period)
    KC_upper = EMA + atr_mult * ATR(atr_period)
    KC_lower = EMA - atr_mult * ATR(atr_period)

  When price closes ABOVE the upper Keltner Channel, it indicates strong
  momentum breakout — price has broken decisively above the volatility band.
  This is distinct from BB Squeeze (which detects low-vol coiling) — KC
  breakout detects the high-momentum expansion phase directly.

  Signal frequency: ~20-40 signals/year on daily with atr_mult=1.5-2.0.
  Tighter multipliers generate more signals; wider ones filter to only
  the strongest breakouts.

Logic:
  1. close[-1] > EMA(ema_period)[-1] + atr_mult * ATR[-1]  (KC breakout)
  2. Optionally: close[-2] <= KC_upper[-2]  (fresh cross, not deep inside)
  3. ADX >= adx_threshold  (trend has strength)
  4. RSI <= rsi_max_long  (not overbought at extreme)
  5. Optionally: volume[-1] > volume_mult * SMA_volume  (expansion confirms)

  SHORT: price closes below lower KC, below EMA.

Entry / SL / TP:
  entry = close  (market-order zone)
  SL    = EMA - sl_buffer_atr * ATR  (below the KC mid = channel lost)
  TP1   = entry + tp1_r * risk
  TP2   = entry + tp2_r * risk
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


class KeltnerBreakoutStrategy(StrategyBase):
    """
    Keltner Channel upper-band breakout — enters when price closes above
    the upper KC band in an established trend, signalling momentum expansion.
    """

    name = "keltner_breakout"
    description = "KC upper-band close breakout with ADX filter — momentum expansion entry"
    uses_bar_slicer = True

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 14,
        atr_mult: float = 1.5,
        require_fresh_cross: bool = True,
        adx_threshold: int = 20,
        rsi_max_long: int = 75,
        entry_buffer_atr: float = 0.0,
        sl_buffer_atr: float = 0.2,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.ema_period        = ema_period
        self.atr_period        = atr_period
        self.atr_mult          = atr_mult
        self.require_fresh_cross = require_fresh_cross
        self.adx_threshold     = adx_threshold
        self.rsi_max_long      = rsi_max_long
        self.entry_buffer_atr  = entry_buffer_atr
        self.sl_buffer_atr     = sl_buffer_atr
        self.tp1_r             = tp1_r
        self.tp2_r             = tp2_r
        self.tp1_alloc         = tp1_alloc
        self.tp2_alloc         = tp2_alloc
        self.shorts_enabled    = shorts_enabled
        self.win_rate          = win_rate

    @property
    def params(self) -> dict:
        return {
            "ema_period":    self.ema_period,
            "atr_mult":      self.atr_mult,
            "adx_threshold": self.adx_threshold,
            "tp1_r":         self.tp1_r,
            "tp2_r":         self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        if len(df) < max(self.ema_period + 10, self.atr_period + 5, 50):
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        atr_series = ta.volatility.AverageTrueRange(
            high, low, close, window=self.atr_period
        ).average_true_range()
        ema_series = ta.trend.EMAIndicator(close, self.ema_period).ema_indicator()

        atr_curr = float(atr_series.iloc[-1])
        atr_prev = float(atr_series.iloc[-2])
        ema_curr = float(ema_series.iloc[-1])
        ema_prev = float(ema_series.iloc[-2])

        if atr_curr <= 0 or pd.isna(atr_curr):
            return None

        kc_upper_curr = ema_curr + self.atr_mult * atr_curr
        kc_lower_curr = ema_curr - self.atr_mult * atr_curr
        kc_upper_prev = ema_prev + self.atr_mult * atr_prev
        kc_lower_prev = ema_prev - self.atr_mult * atr_prev

        curr_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])

        long_signal  = curr_close > kc_upper_curr
        short_signal = curr_close < kc_lower_curr

        if not long_signal and not short_signal:
            return None
        if short_signal and not self.shorts_enabled:
            return None

        if self.require_fresh_cross:
            if long_signal and prev_close > kc_upper_prev:
                return None
            if short_signal and prev_close < kc_lower_prev:
                return None

        direction = "long" if long_signal else "short"

        adx_val = float(ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1])
        if adx_val < self.adx_threshold:
            return None

        rsi_val = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
        if direction == "long" and rsi_val > self.rsi_max_long:
            return None

        buf_entry = self.entry_buffer_atr * atr_curr
        buf_sl    = self.sl_buffer_atr * atr_curr

        if direction == "long":
            entry_low  = round(curr_close, 4)
            entry_high = round(curr_close + buf_entry, 4)
            sl         = round(ema_curr - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(curr_close, 4)
            entry_low  = round(curr_close - buf_entry, 4)
            sl         = round(ema_curr + buf_sl, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        avg_r      = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev         = _ev(self.win_rate, avg_r)
        pf         = _pf(self.win_rate, avg_r)
        atr_pct    = tech.get("atr_pct", round(atr_curr / curr_close * 100, 3))
        trade_type = _interval_trade_type(interval)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Keltner Channel Breakout",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Price closed {'above' if direction == 'long' else 'below'} "
                f"Keltner Channel ({self.atr_mult}×ATR) at {kc_upper_curr if direction == 'long' else kc_lower_curr:.4f}. "
                f"EMA{self.ema_period}={ema_curr:.4f}. ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"Momentum expansion beyond volatility band."
            ),
            trigger=f"KC breakout confirmed — enter above {entry_low:.4f}.",
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
                f"KC{self.ema_period}/{self.atr_mult}x breakout | "
                f"EMA={ema_curr:.4f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes back {'below' if direction == 'long' else 'above'} KC mid (EMA)",
                description="Breakout failed — momentum exhausted",
                price_trigger=ema_curr,
                action="Exit — KC breakout reversal",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Momentum continues, price holds above KC upper",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="Price retreats back below KC — false breakout",
                    outcome="NO TRADE — wait for sustained breakout",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_curr:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
