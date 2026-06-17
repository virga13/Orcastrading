"""
p3_backtester/strategies/consecutive_highs.py — Consecutive Higher Closes.

Concept:
  N consecutive bars each closing higher than the previous close signals
  persistent buying pressure and momentum. Unlike Heikin Ashi (which
  smooths candles), this uses raw closing prices — avoiding any lookahead
  or transformation bias.

  The pattern is a natural "staircase" — markets that grind higher in
  controlled fashion with each close above the last are in momentum mode.
  Adding an EMA filter ensures we're in an established uptrend, and a
  max-RSI cap prevents buying into extended rallies.

  Signal frequency: ~25-50 signals/year on daily depending on N and ADX floor.

Logic:
  1. Last N closes are each strictly higher than the previous:
       close[-k] > close[-(k+1)] for k in 1..N
  2. close[-1] > EMA(ema_period)[-1]  (uptrend)
  3. ADX >= adx_threshold  (trend has strength)
  4. RSI <= rsi_max_long  (not overbought)
  5. Optionally: close[-N] > close[-(N+1)] * (1 + min_gain_pct/100)
     — total move over N bars exceeds minimum threshold

  SHORT: N consecutive closes each lower than previous, price below EMA.

Entry / SL / TP:
  entry_low  = close
  entry_high = close + entry_buffer_atr * ATR
  SL = lowest low of (N + sl_extra) bars - sl_buffer_atr * ATR
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


class ConsecutiveHighsStrategy(StrategyBase):
    """
    N consecutive higher closes above EMA — persistent momentum continuation.
    """

    name = "consecutive_highs"
    description = "N consecutive higher closes in uptrend — momentum staircase continuation"
    uses_bar_slicer = True

    def __init__(
        self,
        n_bars: int = 3,
        ema_period: int = 50,
        adx_threshold: int = 20,
        rsi_max_long: int = 70,
        min_gain_pct: float = 0.0,
        sl_lookback_extra: int = 2,
        entry_buffer_atr: float = 0.05,
        sl_buffer_atr: float = 0.1,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.n_bars            = n_bars
        self.ema_period        = ema_period
        self.adx_threshold     = adx_threshold
        self.rsi_max_long      = rsi_max_long
        self.min_gain_pct      = min_gain_pct
        self.sl_lookback_extra = sl_lookback_extra
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
            "n_bars":        self.n_bars,
            "ema_period":    self.ema_period,
            "adx_threshold": self.adx_threshold,
            "tp1_r":         self.tp1_r,
            "tp2_r":         self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = max(self.ema_period + 10, self.n_bars + self.sl_lookback_extra + 5)
        if len(df) < min_bars:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        closes = [float(close.iloc[-(k + 1)]) for k in range(self.n_bars + 1)]

        long_ok  = all(closes[k] > closes[k + 1] for k in range(self.n_bars))
        short_ok = all(closes[k] < closes[k + 1] for k in range(self.n_bars))

        if not long_ok and not short_ok:
            return None
        if short_ok and not self.shorts_enabled:
            return None

        direction = "long" if long_ok else "short"

        if self.min_gain_pct > 0:
            total_move = abs(closes[0] - closes[self.n_bars]) / closes[self.n_bars] * 100
            if total_move < self.min_gain_pct:
                return None

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
        sl_lookback = self.n_bars + self.sl_lookback_extra

        if direction == "long":
            entry_low  = round(curr_close, 4)
            entry_high = round(curr_close + buf_entry, 4)
            sl_price   = float(low.iloc[-sl_lookback:].min())
            sl         = round(sl_price - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(curr_close, 4)
            entry_low  = round(curr_close - buf_entry, 4)
            sl_price   = float(high.iloc[-sl_lookback:].max())
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
            label=f"{'Long' if direction == 'long' else 'Short'} Consecutive Higher Closes ({self.n_bars} bars)",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"{self.n_bars} consecutive {'higher' if direction == 'long' else 'lower'} closes "
                f"({', '.join(f'{c:.2f}' for c in reversed(closes[:self.n_bars + 1]))}). "
                f"Price {'above' if direction == 'long' else 'below'} EMA{self.ema_period}={ema_val:.4f}. "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"Persistent {'buying' if direction == 'long' else 'selling'} momentum."
            ),
            trigger=f"{self.n_bars} consecutive closes {'up' if direction == 'long' else 'down'} — enter above {entry_low:.4f}.",
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
                f"{self.n_bars} consec {'higher' if direction == 'long' else 'lower'} closes | "
                f"EMA{self.ema_period}={ema_val:.4f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes {'lower' if direction == 'long' else 'higher'} — streak broken",
                description="Consecutive pattern broken — momentum stalled",
                price_trigger=sl,
                action="Exit — streak reversal",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Momentum continues, price closes {'higher' if direction == 'long' else 'lower'} again",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="Streak broken — close reversal",
                    outcome="NO TRADE — wait for pattern to re-establish",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
