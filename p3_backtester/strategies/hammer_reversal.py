"""
p3_backtester/strategies/hammer_reversal.py — Hammer / Pin Bar Reversal.

Concept:
  A hammer (bullish) or shooting star (bearish) is a candlestick with:
  - A small real body (close near the top for hammers, near the bottom for stars)
  - A long lower wick (hammer) or upper wick (shooting star)
  - Little or no opposite wick

  The pattern signals rejection: sellers drove price down aggressively during the
  bar but bulls recovered, closing near the top — demonstrating buying pressure.
  When this appears near a key level (EMA support) in a trend, it's high-probability.

  Signal count: 15-30 per year on daily — more than PSAR/Williams %R but
  quality-filtered (EMA proximity + ADX + wick ratio).

Logic (hammer / bullish pin bar):
  1. Long lower wick: lower_wick >= wick_ratio * bar_range  (e.g. 60% of the range)
  2. Small body: body_size <= max_body_ratio * bar_range  (e.g. 40% of range)
  3. Small upper wick: upper_wick <= max_upper_wick_ratio * bar_range  (e.g. 20%)
  4. Close is in upper portion of bar (close >= low + 0.5 * bar_range)
  5. Price is within ema_zone_atr * ATR of the EMA (at support level)
  6. ADX >= adx_threshold  (trend has strength)
  7. RSI <= rsi_max  (not overbought)

  SHORT (shooting star): mirrors hammer — long upper wick, small body near bottom.

Entry / SL / TP:
  entry_low  = bar_high (breakout above the hammer)
  entry_high = bar_high + entry_buffer_atr * ATR
  SL = bar_low - sl_buffer_atr * ATR  (wick defines the risk)
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


class HammerReversalStrategy(StrategyBase):
    """
    Hammer / Pin Bar Reversal — enters on breakout above a long-wick rejection
    candle near EMA support in an established trend.
    """

    name = "hammer_reversal"
    description = "Hammer/pin bar rejection candle near EMA support with ADX gate"
    uses_bar_slicer = True

    def __init__(
        self,
        wick_ratio: float = 0.60,
        max_body_ratio: float = 0.40,
        max_upper_wick_ratio: float = 0.25,
        ema_period: int = 50,
        ema_zone_atr: float = 2.0,
        adx_threshold: int = 20,
        rsi_max_long: int = 70,
        min_bar_atr: float = 0.5,
        entry_buffer_atr: float = 0.05,
        sl_buffer_atr: float = 0.1,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.wick_ratio           = wick_ratio
        self.max_body_ratio       = max_body_ratio
        self.max_upper_wick_ratio = max_upper_wick_ratio
        self.ema_period           = ema_period
        self.ema_zone_atr         = ema_zone_atr
        self.adx_threshold        = adx_threshold
        self.rsi_max_long         = rsi_max_long
        self.min_bar_atr          = min_bar_atr
        self.entry_buffer_atr     = entry_buffer_atr
        self.sl_buffer_atr        = sl_buffer_atr
        self.tp1_r                = tp1_r
        self.tp2_r                = tp2_r
        self.tp1_alloc            = tp1_alloc
        self.tp2_alloc            = tp2_alloc
        self.shorts_enabled       = shorts_enabled
        self.win_rate             = win_rate

    @property
    def params(self) -> dict:
        return {
            "wick_ratio":     self.wick_ratio,
            "ema_period":     self.ema_period,
            "ema_zone_atr":   self.ema_zone_atr,
            "adx_threshold":  self.adx_threshold,
            "tp1_r":          self.tp1_r,
            "tp2_r":          self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        if len(df) < max(self.ema_period + 10, 30):
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        open_  = df["Open"]
        interval = tech.get("interval", "1d")

        bar_high  = float(high.iloc[-1])
        bar_low   = float(low.iloc[-1])
        bar_close = float(close.iloc[-1])
        bar_open  = float(open_.iloc[-1])

        bar_range = bar_high - bar_low
        if bar_range <= 0:
            return None

        body_top    = max(bar_open, bar_close)
        body_bottom = min(bar_open, bar_close)
        body_size   = body_top - body_bottom
        lower_wick  = body_bottom - bar_low
        upper_wick  = bar_high - body_top

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        # Bar must be meaningful size
        if bar_range < self.min_bar_atr * atr_val:
            return None

        # Hammer conditions (bullish)
        is_hammer = (
            lower_wick >= self.wick_ratio * bar_range
            and body_size <= self.max_body_ratio * bar_range
            and upper_wick <= self.max_upper_wick_ratio * bar_range
            and bar_close >= bar_low + 0.5 * bar_range
        )

        # Shooting star (bearish)
        is_star = (
            upper_wick >= self.wick_ratio * bar_range
            and body_size <= self.max_body_ratio * bar_range
            and lower_wick <= self.max_upper_wick_ratio * bar_range
            and bar_close <= bar_low + 0.5 * bar_range
        )

        if not is_hammer and not is_star:
            return None
        if is_star and not self.shorts_enabled:
            return None

        direction = "long" if is_hammer else "short"

        curr_close = bar_close
        ema_val    = float(ta.trend.EMAIndicator(close, self.ema_period).ema_indicator().iloc[-1])

        # EMA proximity filter: bar must be near the EMA
        dist_to_ema = abs(curr_close - ema_val)
        if dist_to_ema > self.ema_zone_atr * atr_val:
            return None

        if direction == "long" and curr_close < ema_val - self.ema_zone_atr * atr_val:
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
            entry_low  = round(bar_high, 4)
            entry_high = round(bar_high + buf_entry, 4)
            sl         = round(bar_low - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(bar_low, 4)
            entry_low  = round(bar_low - buf_entry, 4)
            sl         = round(bar_high + buf_sl, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        wick_r = round(lower_wick / atr_val if direction == "long" else upper_wick / atr_val, 2)
        avg_r  = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev     = _ev(self.win_rate, avg_r)
        pf     = _pf(self.win_rate, avg_r)
        atr_pct = tech.get("atr_pct", round(atr_val / curr_close * 100, 3))
        trade_type = _interval_trade_type(interval)
        pattern = "Hammer" if direction == "long" else "Shooting Star"

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} {pattern} Pin Bar",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"{pattern} formed near EMA{self.ema_period}={ema_val:.4f} "
                f"(dist={dist_to_ema/atr_val:.2f} ATRs). "
                f"{'Lower' if direction == 'long' else 'Upper'} wick={wick_r} ATRs "
                f"({(lower_wick if direction == 'long' else upper_wick)/bar_range:.0%} of range). "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"Price rejection signals {'buying' if direction == 'long' else 'selling'} pressure."
            ),
            trigger=(
                f"Price breaks {'above' if direction == 'long' else 'below'} "
                f"{'hammer high' if direction == 'long' else 'star low'} "
                f"{'%.4f' % entry_low}."
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
                f"{pattern}: wick={wick_r} ATRs | body={body_size/bar_range:.0%} of range | "
                f"EMA{self.ema_period} dist={dist_to_ema/atr_val:.2f} ATRs | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes {'below' if direction == 'long' else 'above'} pin bar {'low' if direction == 'long' else 'high'}",
                description="Pin bar range violated — rejection pattern failed",
                price_trigger=sl,
                action="Exit — pin bar rejection failed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Price breaks {'above' if direction == 'long' else 'below'} pin bar and holds",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="Price fails to break pin bar high/low",
                    outcome="NO TRADE — wait for breakout",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
