"""
p3_backtester/strategies/parabolic_sar_flip.py — Parabolic SAR Flip strategy.

Concept:
  The Parabolic SAR (Stop and Reverse) plots dots above/below price to indicate
  trend direction. When the SAR flips from above price (bearish) to below price
  (bullish), it signals a potential trend reversal or resumption of an uptrend.

  This is one of the most signal-rich daily indicators (flips ~15-25x per year),
  but prone to whipsaws in choppy markets. We gate it with ADX and EMA to filter
  only the high-quality flips.

Logic:
  1. SAR was above price on bar[-2] (bearish SAR)
  2. SAR is below price on bar[-1] (bullish SAR) — flip occurred
  3. Price > EMA (long-term trend alignment)
  4. ADX >= adx_threshold (avoid choppy markets)
  5. RSI < rsi_max (not already overbought)

  SAR computation: step parameter accelerates from initial_step each new extreme,
  capped at max_step. Standard settings: step=0.02, max=0.20.

Entry / SL / TP:
  entry_low  = curr_close
  entry_high = curr_close + entry_buffer_atr * ATR
  SL = SAR value (the just-flipped dot is a natural stop level)
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


def _compute_psar(high: pd.Series, low: pd.Series,
                  step: float = 0.02, max_step: float = 0.20) -> pd.Series:
    """Compute Parabolic SAR. Returns series of SAR values."""
    psar = pd.Series(index=high.index, dtype=float)
    bull = True
    af = step
    ep = float(low.iloc[0])
    hp = float(high.iloc[0])
    lp = float(low.iloc[0])
    psar.iloc[0] = lp

    for i in range(1, len(high)):
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        prev_sar = float(psar.iloc[i - 1])

        if bull:
            new_sar = prev_sar + af * (hp - prev_sar)
            new_sar = min(new_sar, float(low.iloc[i - 1]),
                          float(low.iloc[i - 2]) if i >= 2 else float(low.iloc[i - 1]))
            if l < new_sar:
                bull = False
                new_sar = hp
                lp = l
                af = step
                ep = l
            else:
                if h > hp:
                    hp = h
                    af = min(af + step, max_step)
                    ep = hp
        else:
            new_sar = prev_sar + af * (lp - prev_sar)
            new_sar = max(new_sar, float(high.iloc[i - 1]),
                          float(high.iloc[i - 2]) if i >= 2 else float(high.iloc[i - 1]))
            if h > new_sar:
                bull = True
                new_sar = lp
                hp = h
                af = step
                ep = h
            else:
                if l < lp:
                    lp = l
                    af = min(af + step, max_step)
                    ep = lp

        psar.iloc[i] = new_sar

    return psar


class ParabolicSARFlipStrategy(StrategyBase):
    """
    Parabolic SAR Flip — enters when SAR flips from above-price (bearish)
    to below-price (bullish), confirming a trend direction change.
    """

    name = "parabolic_sar_flip"
    description = "Parabolic SAR flip from bearish to bullish with EMA and ADX gate"
    uses_bar_slicer = True

    def __init__(
        self,
        sar_step: float = 0.02,
        sar_max_step: float = 0.20,
        ema_period: int = 50,
        adx_threshold: int = 20,
        rsi_max_long: int = 70,
        entry_buffer_atr: float = 0.05,
        sl_buffer_atr: float = 0.15,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.sar_step           = sar_step
        self.sar_max_step       = sar_max_step
        self.ema_period         = ema_period
        self.adx_threshold      = adx_threshold
        self.rsi_max_long       = rsi_max_long
        self.entry_buffer_atr   = entry_buffer_atr
        self.sl_buffer_atr      = sl_buffer_atr
        self.tp1_r              = tp1_r
        self.tp2_r              = tp2_r
        self.tp1_alloc          = tp1_alloc
        self.tp2_alloc          = tp2_alloc
        self.shorts_enabled     = shorts_enabled
        self.win_rate           = win_rate

    @property
    def params(self) -> dict:
        return {
            "sar_step":       self.sar_step,
            "sar_max_step":   self.sar_max_step,
            "ema_period":     self.ema_period,
            "adx_threshold":  self.adx_threshold,
            "tp1_r":          self.tp1_r,
            "tp2_r":          self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        if len(df) < max(self.ema_period + 10, 50):
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        psar = _compute_psar(high, low, self.sar_step, self.sar_max_step)

        curr_close  = float(close.iloc[-1])
        curr_sar    = float(psar.iloc[-1])
        prev_close  = float(close.iloc[-2])
        prev_sar    = float(psar.iloc[-2])

        # Long flip: SAR was above prev_close (bearish), now below curr_close (bullish)
        long_flip  = (prev_sar > prev_close) and (curr_sar < curr_close)
        short_flip = (prev_sar < prev_close) and (curr_sar > curr_close)

        if not long_flip and not short_flip:
            return None
        if short_flip and not self.shorts_enabled:
            return None

        direction = "long" if long_flip else "short"

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

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
            # SL at the SAR value (natural reversal level) with buffer
            sl         = round(curr_sar - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(curr_close, 4)
            entry_low  = round(curr_close - buf_entry, 4)
            sl         = round(curr_sar + buf_sl, 4)
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
        sar_dist_r = round(abs(curr_close - curr_sar) / atr_val, 2)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Parabolic SAR Flip",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Parabolic SAR flipped {'above' if direction == 'short' else 'below'} price "
                f"(SAR={curr_sar:.4f}, price={curr_close:.4f}, dist={sar_dist_r} ATRs). "
                f"Price {'above' if direction == 'long' else 'below'} EMA{self.ema_period}={ema_val:.4f}. "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"SAR flip signals trend direction change."
            ),
            trigger=(
                f"SAR now {'below' if direction == 'long' else 'above'} price — "
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
                f"SAR={curr_sar:.4f} ({sar_dist_r} ATRs from price) | "
                f"EMA{self.ema_period}={ema_val:.4f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"SAR flips back {'above' if direction == 'long' else 'below'} price",
                description="SAR reversed — prior flip was a whipsaw",
                price_trigger=sl,
                action="Exit — SAR whipsaw confirmed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"SAR holds {'below' if direction == 'long' else 'above'} price and trend develops",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="Price immediately reverses and SAR flips back",
                    outcome="NO TRADE — SAR whipsaw, wait for next setup",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
