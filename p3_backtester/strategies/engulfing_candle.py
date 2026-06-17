"""
p3_backtester/strategies/engulfing_candle.py — Engulfing Candle pattern strategy.

Concept:
  A bullish engulfing pattern occurs when a bearish bar (close < open) is followed
  by a bullish bar (close > open) whose body completely engulfs the previous bar's body.
  It signals that bulls took over decisively from bears within a single bar.

  Bearish engulfing is the mirror: a bullish bar followed by a bearish bar whose body
  completely engulfs the prior bullish bar's body.

  The pattern is most reliable when it occurs at a key level:
    - Near the EMA (trend resumption setup)
    - At a swing low support / swing high resistance
    - With above-average volume (conviction)

  This strategy implements the EMA-confirmation variant — the engulfing must occur
  near (within ema_zone_atr ATRs of) the EMA20 to qualify as a trend continuation.

Logic:
  Bullish engulfing at EMA (long):
    1. Previous bar: close < open (bearish bar)
    2. Current bar:  close > open (bullish bar) AND
                     open  <= prev_open (opens inside or below prev bar) AND
                     close >= prev_close (closes above prev bar's open — full engulf)
                     [body engulf: curr_open <= prev_close AND curr_close >= prev_open]
    3. Price within ema_zone_atr * ATR of EMA20 (at the EMA level)
    4. ADX >= adx_threshold (trend active)
    5. RSI not overbought (< rsi_max)
    6. Volume >= volume_mult * avg 20-bar volume (conviction)

Entry / SL / TP:
  entry     = close of engulfing bar
  SL (long) = low of the two-bar pattern - sl_buffer_atr * ATR
  TP1       = entry + tp1_r * risk
  TP2       = entry + tp2_r * risk
"""
from __future__ import annotations

import pandas as pd
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


class EngulfingCandleStrategy(StrategyBase):
    """
    Engulfing Candle pattern at EMA.

    Fires when a bullish (long) or bearish (short) engulfing candle forms
    near the EMA20, confirming a trend continuation after a brief pullback.
    """

    name = "engulfing_candle"
    description = "Bullish/bearish engulfing candle near EMA — trend continuation after pullback"
    uses_bar_slicer = True

    def __init__(
        self,
        ema_period: int = 20,
        ema_zone_atr: float = 1.5,
        adx_threshold: int = 20,
        rsi_max_long: int = 65,
        rsi_min_short: int = 35,
        volume_mult: float = 1.0,
        require_close_above_prev_open: bool = True,
        htf_filter: bool = True,
        sl_buffer_atr: float = 0.1,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = True,
        win_rate: float = 0.45,
    ):
        self.ema_period                  = ema_period
        self.ema_zone_atr                = ema_zone_atr
        self.adx_threshold               = adx_threshold
        self.rsi_max_long                = rsi_max_long
        self.rsi_min_short               = rsi_min_short
        self.volume_mult                 = volume_mult
        self.require_close_above_prev_open = require_close_above_prev_open
        self.htf_filter                  = htf_filter
        self.sl_buffer_atr               = sl_buffer_atr
        self.tp1_r                       = tp1_r
        self.tp2_r                       = tp2_r
        self.tp1_alloc                   = tp1_alloc
        self.tp2_alloc                   = tp2_alloc
        self.shorts_enabled              = shorts_enabled
        self.win_rate                    = win_rate

    @property
    def params(self) -> dict:
        return {
            "ema_period":    self.ema_period,
            "ema_zone_atr":  self.ema_zone_atr,
            "adx_threshold": self.adx_threshold,
            "tp1_r":         self.tp1_r,
            "tp2_r":         self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        if len(df) < self.ema_period + 25:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        open_  = df["Open"]
        interval = tech.get("interval", "1d")

        curr_open  = float(open_.iloc[-1])
        curr_close = float(close.iloc[-1])
        curr_high  = float(high.iloc[-1])
        curr_low   = float(low.iloc[-1])
        prev_open  = float(open_.iloc[-2])
        prev_close = float(close.iloc[-2])
        prev_low   = float(low.iloc[-2])
        prev_high  = float(high.iloc[-2])

        prev_bearish = prev_close < prev_open
        prev_bullish = prev_close > prev_open
        curr_bullish = curr_close > curr_open
        curr_bearish = curr_close < curr_open

        # Body engulf check (body of current must contain body of previous)
        bull_engulf = (
            prev_bearish and curr_bullish
            and curr_open  <= prev_close   # current opens at or below prev close
            and curr_close >= prev_open    # current closes at or above prev open
        )
        bear_engulf = (
            prev_bullish and curr_bearish
            and curr_open  >= prev_close   # current opens at or above prev close
            and curr_close <= prev_open    # current closes at or below prev open
        )

        if not bull_engulf and not bear_engulf:
            return None

        if bear_engulf and not self.shorts_enabled:
            return None

        direction = "long" if bull_engulf else "short"

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        ema_val = float(
            ta.trend.EMAIndicator(close, window=self.ema_period).ema_indicator().iloc[-1]
        )
        price = curr_close

        # EMA proximity check — pattern must be near the EMA
        ema_dist = abs(price - ema_val)
        if ema_dist > self.ema_zone_atr * atr_val:
            return None

        # Trend direction via EMA
        if direction == "long"  and price < ema_val * 0.99:
            return None
        if direction == "short" and price > ema_val * 1.01:
            return None

        adx_val = float(
            ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
        )
        if adx_val < self.adx_threshold:
            return None

        rsi_val = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
        if direction == "long"  and rsi_val > self.rsi_max_long:
            return None
        if direction == "short" and rsi_val < self.rsi_min_short:
            return None

        # Volume filter
        if self.volume_mult > 1.0 and "Volume" in df.columns:
            avg_vol = float(df["Volume"].iloc[-21:-1].mean())
            curr_vol = float(df["Volume"].iloc[-1])
            if avg_vol > 0 and curr_vol < self.volume_mult * avg_vol:
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

        buf = self.sl_buffer_atr * atr_val
        pattern_low  = min(curr_low, prev_low)
        pattern_high = max(curr_high, prev_high)

        if direction == "long":
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(pattern_low - buf, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(pattern_high + buf, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        avg_r      = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev         = _ev(self.win_rate, avg_r)
        pf         = _pf(self.win_rate, avg_r)
        atr_pct    = tech.get("atr_pct", round(atr_val / price * 100, 3))
        trade_type = _interval_trade_type(interval)
        pat_name   = "Bullish Engulfing" if direction == "long" else "Bearish Engulfing"

        setup = TradingSetup(
            name="Setup A",
            label=f"{pat_name} at EMA{self.ema_period}",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"{pat_name} pattern at EMA{self.ema_period} ({ema_val:.4f}). "
                f"{'Bullish' if direction == 'long' else 'Bearish'} bar "
                f"O={curr_open:.4f} C={curr_close:.4f} fully engulfed previous "
                f"{'bearish' if direction == 'long' else 'bullish'} bar "
                f"O={prev_open:.4f} C={prev_close:.4f}. "
                f"EMA dist={ema_dist:.4f} ({ema_dist/atr_val:.1f} ATRs). ADX={adx_val:.1f}."
            ),
            trigger=(
                f"{pat_name} confirmed on bar close. "
                f"EMA{self.ema_period} within {self.ema_zone_atr} ATRs."
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
                f"{pat_name} | EMA{self.ema_period}={ema_val:.4f} "
                f"(dist={ema_dist/atr_val:.1f} ATR) | ADX={adx_val:.1f} | RSI={rsi_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes {'below' if direction == 'long' else 'above'} pattern {'low' if direction == 'long' else 'high'} {pattern_low if direction == 'long' else pattern_high:.4f}",
                description="Engulfing pattern failed — price broke the two-bar low/high",
                price_trigger=sl,
                action="Exit — pattern invalidated",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"{pat_name} holds and price continues {'up' if direction == 'long' else 'down'}",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=price,
                ),
                DecisionTreeEntry(
                    scenario=f"Price closes {'below' if direction == 'long' else 'above'} the two-bar pattern range",
                    outcome="NO TRADE — engulfing pattern failed",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
