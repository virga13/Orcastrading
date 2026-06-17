"""
p3_backtester/strategies/inside_bar.py — Inside Bar Breakout strategy.

Concept:
  An inside bar is a bar whose high-low range is completely contained within
  the previous bar's (the "mother bar") high-low range. It represents a
  consolidation / indecision pause in the market. When price subsequently
  breaks out above (long) or below (short) the inside bar, it signals
  continuation of the mother bar's directional move.

  Setup (long):
    1. Previous bar is a large "mother bar" (range >= min_mother_atr * ATR)
    2. Current bar is an inside bar (high < mother_high AND low > mother_low)
    3. The mother bar is a bullish bar (close > open) [for longs]
    4. HTF EMA20 > EMA50 (trend alignment)
    5. ADX >= adx_threshold (trend has enough strength)
    6. RSI not overbought

  Entry / SL / TP:
    entry_high = inside_bar_high + entry_buffer_atr * ATR  (breakout long)
    entry_low  = inside_bar_high                            (tight entry zone)
    SL         = inside_bar_low - sl_buffer_atr * ATR
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


class InsideBarStrategy(StrategyBase):
    """
    Inside Bar Breakout strategy.

    Enters on breakout above/below an inside bar after a strong mother bar,
    treating the inside bar as a defined-risk consolidation level.
    """

    name = "inside_bar"
    description = "Inside bar breakout — consolidation pause after a strong directional bar"
    uses_bar_slicer = True

    def __init__(
        self,
        min_mother_atr: float = 1.0,
        adx_threshold: int = 20,
        rsi_max_long: int = 70,
        rsi_min_short: int = 30,
        require_mother_direction: bool = True,
        htf_filter: bool = True,
        entry_buffer_atr: float = 0.05,
        sl_buffer_atr: float = 0.1,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = True,
        win_rate: float = 0.45,
    ):
        self.min_mother_atr          = min_mother_atr
        self.adx_threshold           = adx_threshold
        self.rsi_max_long            = rsi_max_long
        self.rsi_min_short           = rsi_min_short
        self.require_mother_direction = require_mother_direction
        self.htf_filter              = htf_filter
        self.entry_buffer_atr        = entry_buffer_atr
        self.sl_buffer_atr           = sl_buffer_atr
        self.tp1_r                   = tp1_r
        self.tp2_r                   = tp2_r
        self.tp1_alloc               = tp1_alloc
        self.tp2_alloc               = tp2_alloc
        self.shorts_enabled          = shorts_enabled
        self.win_rate                = win_rate

    @property
    def params(self) -> dict:
        return {
            "min_mother_atr": self.min_mother_atr,
            "adx_threshold":  self.adx_threshold,
            "htf_filter":     self.htf_filter,
            "tp1_r":          self.tp1_r,
            "tp2_r":          self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        if len(df) < 30:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        open_  = df["Open"]
        interval = tech.get("interval", "1d")

        # Current bar is bar[-1], mother bar is bar[-2]
        curr_high  = float(high.iloc[-1])
        curr_low   = float(low.iloc[-1])
        curr_close = float(close.iloc[-1])
        curr_open  = float(open_.iloc[-1])
        moth_high  = float(high.iloc[-2])
        moth_low   = float(low.iloc[-2])
        moth_close = float(close.iloc[-2])
        moth_open  = float(open_.iloc[-2])

        # Check inside bar condition
        if not (curr_high < moth_high and curr_low > moth_low):
            return None

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        mother_range = moth_high - moth_low
        if mother_range < self.min_mother_atr * atr_val:
            return None

        adx_val = float(
            ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
        )
        if adx_val < self.adx_threshold:
            return None

        rsi_val = float(
            ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        )

        mother_bullish = moth_close > moth_open
        mother_bearish = moth_close < moth_open

        direction: str | None = None

        if rsi_val <= self.rsi_max_long:
            if not self.require_mother_direction or mother_bullish:
                direction = "long"

        if direction is None and self.shorts_enabled and rsi_val >= self.rsi_min_short:
            if not self.require_mother_direction or mother_bearish:
                direction = "short"

        if direction is None:
            return None

        # ── HTF trend alignment ────────────────────────────────────────────────
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

        buf_entry = self.entry_buffer_atr * atr_val
        buf_sl    = self.sl_buffer_atr * atr_val

        if direction == "long":
            entry_low  = round(curr_high, 4)
            entry_high = round(curr_high + buf_entry, 4)
            sl         = round(curr_low - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(curr_low, 4)
            entry_low  = round(curr_low - buf_entry, 4)
            sl         = round(curr_high + buf_sl, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        avg_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev    = _ev(self.win_rate, avg_r)
        pf    = _pf(self.win_rate, avg_r)
        price   = curr_close
        atr_pct = tech.get("atr_pct", round(atr_val / price * 100, 3))
        trade_type = _interval_trade_type(interval)
        ib_range_r = round(mother_range / atr_val, 2)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Inside Bar Breakout",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Inside bar formed within {'bullish' if mother_bullish else 'bearish'} mother bar "
                f"(range={mother_range:.4f}, {ib_range_r} ATRs). "
                f"Inside bar: H={curr_high:.4f} L={curr_low:.4f}. "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"Breakout {'above' if direction == 'long' else 'below'} inside bar signals continuation."
            ),
            trigger=(
                f"Price breaks {'above' if direction == 'long' else 'below'} "
                f"inside bar {'high' if direction == 'long' else 'low'} "
                f"{'%.4f' % entry_low if direction == 'long' else '%.4f' % entry_high}."
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
                f"Mother bar range={ib_range_r} ATRs | "
                f"Inside bar H={curr_high:.4f} L={curr_low:.4f} | "
                f"ADX={adx_val:.1f} | RSI={rsi_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes {'below' if direction == 'long' else 'above'} inside bar {'low' if direction == 'long' else 'high'}",
                description="Inside bar range violated — breakout failed",
                price_trigger=sl,
                action="Exit — inside bar breakout rejected",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Price breaks {'above' if direction == 'long' else 'below'} inside bar and holds",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario=f"Price fails breakout and returns inside the bar range",
                    outcome="NO TRADE — false breakout",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
