"""
p3_backtester/strategies/cci_reversal.py — CCI Reversal strategy.

Concept:
  The Commodity Channel Index measures how far price has deviated from its
  statistical average, in units of mean deviation. Unlike RSI/Stochastic which
  are bounded (0-100), CCI is unbounded — it can go to +/-300 in strong trends.

  Formula: CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
  where Typical Price = (H + L + C) / 3

  The classic oversold zone is below -100; overbought is above +100.
  A reversal signal fires when CCI crosses back above -100 from below (or below
  +100 from above for shorts). This confirms oversold conditions are lifting.

  Key advantage over Williams %R: CCI responds faster to volatile moves and
  generates 25-40+ signals/year on daily, even with trend filters.

Logic:
  1. CCI[-2] <= oversold_level (e.g. -100)  AND  CCI[-1] > oversold_level
     (CCI just crossed up out of oversold zone)
  2. Price > EMA(ema_period)  (uptrend context)
  3. ADX >= adx_threshold  (trend has strength)
  4. RSI <= rsi_max  (not overbought)

  SHORT: CCI crosses below overbought_level, price < EMA.

Entry / SL / TP:
  entry_low  = curr_close
  entry_high = curr_close + entry_buffer_atr * ATR
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


class CCIReversalStrategy(StrategyBase):
    """
    CCI Reversal — enters when CCI crosses back above oversold threshold
    in an established uptrend, signalling oversold momentum lifting.
    """

    name = "cci_reversal"
    description = "CCI oversold cross-up in uptrend with EMA and ADX gate"
    uses_bar_slicer = True

    def __init__(
        self,
        cci_period: int = 20,
        oversold_level: float = -100.0,
        overbought_level: float = 100.0,
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
        self.cci_period        = cci_period
        self.oversold_level    = oversold_level
        self.overbought_level  = overbought_level
        self.ema_period        = ema_period
        self.adx_threshold     = adx_threshold
        self.rsi_max_long      = rsi_max_long
        self.sl_lookback       = sl_lookback
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
            "cci_period":       self.cci_period,
            "oversold_level":   self.oversold_level,
            "ema_period":       self.ema_period,
            "adx_threshold":    self.adx_threshold,
            "tp1_r":            self.tp1_r,
            "tp2_r":            self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        if len(df) < max(self.ema_period + 10, self.cci_period + 5):
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        cci = ta.trend.CCIIndicator(high, low, close, window=self.cci_period).cci()
        cci_curr = float(cci.iloc[-1])
        cci_prev = float(cci.iloc[-2])

        if pd.isna(cci_curr) or pd.isna(cci_prev):
            return None

        long_cross  = (cci_prev <= self.oversold_level)  and (cci_curr > self.oversold_level)
        short_cross = (cci_prev >= self.overbought_level) and (cci_curr < self.overbought_level)

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

        avg_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev    = _ev(self.win_rate, avg_r)
        pf    = _pf(self.win_rate, avg_r)
        atr_pct = tech.get("atr_pct", round(atr_val / curr_close * 100, 3))
        trade_type = _interval_trade_type(interval)
        label = "oversold" if direction == "long" else "overbought"
        level = self.oversold_level if direction == "long" else self.overbought_level

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} CCI Reversal",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"CCI({self.cci_period}) crossed {'above' if direction == 'long' else 'below'} "
                f"{label} level {level:.0f} "
                f"(prev={cci_prev:.0f}, curr={cci_curr:.0f}). "
                f"Price {'above' if direction == 'long' else 'below'} EMA{self.ema_period}={ema_val:.4f}. "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"Oversold extreme lifting in established trend."
            ),
            trigger=(
                f"CCI exited {label} zone — "
                f"enter {'above' if direction == 'long' else 'below'} {entry_low:.4f}."
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
                f"CCI={cci_curr:.0f} (crossed {level:.0f}) | "
                f"EMA{self.ema_period}={ema_val:.4f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"CCI drops back {'below' if direction == 'long' else 'above'} {level:.0f}",
                description="CCI re-entered extreme zone — reversal failed",
                price_trigger=sl,
                action="Exit — CCI reversal rejected",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"CCI continues {'higher' if direction == 'long' else 'lower'}, price trends",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario=f"CCI drops back into {label} territory",
                    outcome="NO TRADE — CCI reversal failed, wait for re-cross",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
