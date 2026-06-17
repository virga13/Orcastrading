"""
p3_backtester/strategies/rsi_divergence.py — RSI Divergence strategy.

Concept:
  Detects classic divergence between price action and the RSI oscillator.
  Divergence signals loss of momentum and potential trend reversal.

  Bullish divergence (long signal):
    Price makes a lower low, but RSI makes a higher low.
    => Selling pressure is weakening — reversal to the upside likely.

  Bearish divergence (short signal):
    Price makes a higher high, but RSI makes a lower high.
    => Buying pressure is weakening — reversal to the downside likely.

Pivot detection:
  A price pivot low is a bar whose low is lower than the N bars before AND after it.
  Two most recent pivots are compared for divergence.

Entry / SL / TP:
  entry = close on divergence bar (close is past the pivot; divergence confirmed)
  SL    = recent swing low - buffer (long) | recent swing high + buffer (short)
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


def _find_pivot_lows(series: pd.Series, order: int) -> list[int]:
    """Return indices (relative to series) of pivot lows within series."""
    vals = series.values
    pivots = []
    for i in range(order, len(vals) - order):
        window = vals[i - order: i + order + 1]
        if vals[i] == min(window):
            pivots.append(i)
    return pivots


def _find_pivot_highs(series: pd.Series, order: int) -> list[int]:
    """Return indices (relative to series) of pivot highs within series."""
    vals = series.values
    pivots = []
    for i in range(order, len(vals) - order):
        window = vals[i - order: i + order + 1]
        if vals[i] == max(window):
            pivots.append(i)
    return pivots


class RSIDivergenceStrategy(StrategyBase):
    """
    RSI Divergence strategy.

    Fires when price and RSI diverge at recent swing pivots, signalling
    a momentum shift and potential trend reversal.
    """

    name = "rsi_divergence"
    description = "RSI divergence — price/RSI diverge at swing pivots, signals momentum reversal"
    uses_bar_slicer = True

    def __init__(
        self,
        rsi_period: int = 14,
        pivot_order: int = 5,
        lookback: int = 60,
        rsi_oversold: float = 45.0,
        rsi_overbought: float = 55.0,
        max_pivot_gap: int = 40,
        sl_buffer_atr: float = 0.15,
        tp1_r: float = 1.5,
        tp2_r: float = 3.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = True,
        htf_filter: bool = False,
        win_rate: float = 0.45,
    ):
        self.rsi_period     = rsi_period
        self.pivot_order    = pivot_order
        self.lookback       = lookback
        self.rsi_oversold   = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.max_pivot_gap  = max_pivot_gap
        self.sl_buffer_atr  = sl_buffer_atr
        self.tp1_r          = tp1_r
        self.tp2_r          = tp2_r
        self.tp1_alloc      = tp1_alloc
        self.tp2_alloc      = tp2_alloc
        self.shorts_enabled = shorts_enabled
        self.htf_filter     = htf_filter
        self.win_rate       = win_rate

    @property
    def params(self) -> dict:
        return {
            "rsi_period":  self.rsi_period,
            "pivot_order": self.pivot_order,
            "lookback":    self.lookback,
            "tp1_r":       self.tp1_r,
            "tp2_r":       self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = self.lookback + self.rsi_period + 10
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

        rsi_series = ta.momentum.RSIIndicator(close, window=self.rsi_period).rsi()

        low_window  = low.iloc[-self.lookback:]
        high_window = high.iloc[-self.lookback:]
        rsi_window  = rsi_series.iloc[-self.lookback:]

        if rsi_window.isna().all():
            return None

        direction: str | None = None
        sl = tp1 = tp2 = 0.0
        pivot1_price = pivot2_price = 0.0
        pivot1_rsi   = pivot2_rsi   = 0.0

        # ── Bullish divergence: lower price low + higher RSI low ──────────────
        pl_idxs = _find_pivot_lows(low_window, self.pivot_order)
        if len(pl_idxs) >= 2:
            p2_pos = pl_idxs[-1]
            for p1_pos in reversed(pl_idxs[:-1]):
                if p2_pos - p1_pos > self.max_pivot_gap:
                    break
                prc1 = float(low_window.iloc[p1_pos])
                prc2 = float(low_window.iloc[p2_pos])
                rsi1 = float(rsi_window.iloc[p1_pos])
                rsi2 = float(rsi_window.iloc[p2_pos])

                if pd.isna(rsi1) or pd.isna(rsi2):
                    continue

                # Divergence: price lower low + RSI higher low
                if prc2 < prc1 and rsi2 > rsi1 and rsi2 <= self.rsi_oversold:
                    direction = "long"
                    pivot1_price, pivot2_price = prc1, prc2
                    pivot1_rsi,   pivot2_rsi   = rsi1, rsi2
                    break

        # ── Bearish divergence: higher price high + lower RSI high ────────────
        if direction is None and self.shorts_enabled:
            ph_idxs = _find_pivot_highs(high_window, self.pivot_order)
            if len(ph_idxs) >= 2:
                p2_pos = ph_idxs[-1]
                for p1_pos in reversed(ph_idxs[:-1]):
                    if p2_pos - p1_pos > self.max_pivot_gap:
                        break
                    prc1 = float(high_window.iloc[p1_pos])
                    prc2 = float(high_window.iloc[p2_pos])
                    rsi1 = float(rsi_window.iloc[p1_pos])
                    rsi2 = float(rsi_window.iloc[p2_pos])

                    if pd.isna(rsi1) or pd.isna(rsi2):
                        continue

                    if prc2 > prc1 and rsi2 < rsi1 and rsi2 >= self.rsi_overbought:
                        direction = "short"
                        pivot1_price, pivot2_price = prc1, prc2
                        pivot1_rsi,   pivot2_rsi   = rsi1, rsi2
                        break

        if direction is None:
            return None

        # ── HTF trend alignment (optional) ────────────────────────────────────
        if self.htf_filter:
            htf_rule = _HTF_RESAMPLE.get(interval, "1W")
            try:
                htf_close = close.resample(htf_rule).last().dropna()
                if len(htf_close) >= 50:
                    htf_ema20 = float(ta.trend.EMAIndicator(htf_close, 20).ema_indicator().iloc[-1])
                    htf_ema50 = float(ta.trend.EMAIndicator(htf_close, 50).ema_indicator().iloc[-1])
                    if direction == "long" and htf_ema20 < htf_ema50:
                        return None
                    if direction == "short" and htf_ema20 > htf_ema50:
                        return None
            except Exception:
                pass

        buf = self.sl_buffer_atr * atr_val

        if direction == "long":
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(float(low.iloc[-self.pivot_order * 2:].min()) - buf, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(float(high.iloc[-self.pivot_order * 2:].max()) + buf, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        avg_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev    = _ev(self.win_rate, avg_r)
        pf    = _pf(self.win_rate, avg_r)
        atr_pct = tech.get("atr_pct", round(atr_val / price * 100, 3))
        div_type = "Bullish" if direction == "long" else "Bearish"
        trade_type = _interval_trade_type(interval)

        setup = TradingSetup(
            name="Setup A",
            label=f"{div_type} RSI Divergence",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"{div_type} RSI divergence detected. "
                f"Price pivots: {pivot1_price:.4f} -> {pivot2_price:.4f} "
                f"({'lower low' if direction == 'long' else 'higher high'}). "
                f"RSI pivots: {pivot1_rsi:.1f} -> {pivot2_rsi:.1f} "
                f"({'higher low' if direction == 'long' else 'lower high'}). "
                f"Momentum weakening — {'reversal to upside' if direction == 'long' else 'reversal to downside'} expected."
            ),
            trigger=(
                f"{div_type} divergence confirmed at current bar. "
                f"RSI={pivot2_rsi:.1f} ({'oversold zone' if direction == 'long' else 'overbought zone'})."
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
                f"RSI divergence: price {pivot1_price:.4f}->{pivot2_price:.4f}, "
                f"RSI {pivot1_rsi:.1f}->{pivot2_rsi:.1f} | ATR={atr_val:.4f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price breaks {'below' if direction == 'long' else 'above'} recent pivot {'low' if direction == 'long' else 'high'}",
                description="Divergence failed — momentum continued in prior direction",
                price_trigger=sl,
                action="Exit — divergence invalidated",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"{div_type} divergence holds and price reverses",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=price,
                ),
                DecisionTreeEntry(
                    scenario=f"Price continues {'down' if direction == 'long' else 'up'} through pivot",
                    outcome="NO TRADE — divergence failed",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
