"""
p3_backtester/strategies/bb_bounce.py — Bollinger Band Bounce strategy.

Concept:
  Bollinger Bands (BB) are a mean-reversion and volatility tool. When price
  dips to or below the lower band and then closes back above it, it signals
  an oversold condition being corrected. In an uptrend, these dips-to-lower-BB
  are high-quality pullback entries.

  Distinct from BB Squeeze (which triggers on volatility expansion): this
  strategy trades the BOUNCE off the lower band itself.

  The key filter: price must be in an uptrend (above EMA) so we're trading
  pullbacks, not counter-trend fades in a downtrend.

Logic (bullish bounce):
  1. bar[-2]: close <= lower_band (or low touches lower_band)  — at the band
  2. bar[-1]: close > lower_band  — bounced back above
  3. Price > EMA(ema_period)  — uptrend context
  4. ADX >= adx_threshold
  5. RSI <= rsi_max  (not overbought — this is a pullback entry)

  SHORT: price bounces down from upper BB in downtrend.

Entry / SL / TP:
  entry_low  = close (bounce bar)
  entry_high = close + entry_buffer_atr * ATR
  SL = lower_band - sl_buffer_atr * ATR  (band is the support level)
  TP1 = entry + tp1_r * risk  (middle band is natural TP1)
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


class BBBounceStrategy(StrategyBase):
    """
    Bollinger Band Bounce — enters when price bounces back above the lower BB
    in an established uptrend, treating the bounce as a pullback entry.
    """

    name = "bb_bounce"
    description = "Bollinger Band lower-band bounce in uptrend — pullback entry with mean-reversion trigger"
    uses_bar_slicer = True

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ema_period: int = 50,
        adx_threshold: int = 20,
        rsi_max_long: int = 65,
        require_low_touch: bool = False,
        entry_buffer_atr: float = 0.05,
        sl_buffer_atr: float = 0.15,
        tp1_r: float = 1.5,
        tp2_r: float = 3.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.bb_period          = bb_period
        self.bb_std             = bb_std
        self.ema_period         = ema_period
        self.adx_threshold      = adx_threshold
        self.rsi_max_long       = rsi_max_long
        self.require_low_touch  = require_low_touch
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
            "bb_period":      self.bb_period,
            "bb_std":         self.bb_std,
            "ema_period":     self.ema_period,
            "adx_threshold":  self.adx_threshold,
            "tp1_r":          self.tp1_r,
            "tp2_r":          self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        if len(df) < max(self.ema_period + 10, self.bb_period + 5):
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        bb_ind    = ta.volatility.BollingerBands(close, self.bb_period, self.bb_std)
        lower_bb  = bb_ind.bollinger_lband()
        upper_bb  = bb_ind.bollinger_hband()
        mid_bb    = bb_ind.bollinger_mavg()

        lb_curr = float(lower_bb.iloc[-1])
        lb_prev = float(lower_bb.iloc[-2])
        ub_curr = float(upper_bb.iloc[-1])
        mb_curr = float(mid_bb.iloc[-1])
        cl_curr = float(close.iloc[-1])
        cl_prev = float(close.iloc[-2])
        lo_prev = float(low.iloc[-2])

        if any(pd.isna(v) for v in [lb_curr, lb_prev, cl_curr]):
            return None

        # Bounce: previous bar at/below lower BB, current closes above
        if self.require_low_touch:
            prev_at_band = lo_prev <= lb_prev
        else:
            prev_at_band = cl_prev <= lb_prev

        long_bounce  = prev_at_band and (cl_curr > lb_curr)

        # Short bounce: previous bar at/above upper BB, current closes below
        hi_prev = float(high.iloc[-2])
        if self.require_low_touch:
            prev_at_upper = hi_prev >= float(upper_bb.iloc[-2])
        else:
            prev_at_upper = cl_prev >= float(upper_bb.iloc[-2])
        short_bounce = prev_at_upper and (cl_curr < ub_curr)

        if not long_bounce and not short_bounce:
            return None
        if short_bounce and not self.shorts_enabled:
            return None

        direction = "long" if long_bounce else "short"

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        ema_val = float(ta.trend.EMAIndicator(close, self.ema_period).ema_indicator().iloc[-1])
        if direction == "long" and cl_curr < ema_val:
            return None
        if direction == "short" and cl_curr > ema_val:
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
            entry_low  = round(cl_curr, 4)
            entry_high = round(cl_curr + buf_entry, 4)
            sl         = round(lb_curr - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(cl_curr, 4)
            entry_low  = round(cl_curr - buf_entry, 4)
            sl         = round(ub_curr + buf_sl, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)

        avg_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev    = _ev(self.win_rate, avg_r)
        pf    = _pf(self.win_rate, avg_r)
        atr_pct = tech.get("atr_pct", round(atr_val / cl_curr * 100, 3))
        trade_type = _interval_trade_type(interval)
        mb_dist_r = round((mb_curr - cl_curr) / atr_val, 2) if direction == "long" else round((cl_curr - mb_curr) / atr_val, 2)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} BB Lower Band Bounce",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Price bounced from {'lower' if direction == 'long' else 'upper'} "
                f"Bollinger Band ({lb_curr if direction == 'long' else ub_curr:.4f}). "
                f"Previous close {'at/below' if direction == 'long' else 'at/above'} band. "
                f"Middle BB={mb_curr:.4f} ({mb_dist_r:.1f} ATRs away — natural TP1). "
                f"EMA{self.ema_period}={ema_val:.4f}, ADX={adx_val:.1f}, RSI={rsi_val:.1f}."
            ),
            trigger=f"Bounce confirmed — enter above {entry_low:.4f}.",
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
                f"Lower BB={lb_curr:.4f} | Middle BB={mb_curr:.4f} | "
                f"EMA{self.ema_period}={ema_val:.4f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=cl_curr,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes back {'below' if direction == 'long' else 'above'} the {'lower' if direction == 'long' else 'upper'} BB",
                description="BB bounce failed — price re-entered band zone",
                price_trigger=sl,
                action="Exit — BB bounce rejected",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario="Price holds above lower BB and moves toward middle band",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f} (near mid-BB)",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="Price drops back below lower band",
                    outcome="NO TRADE — BB bounce failed, re-assess",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
