"""
p3_backtester/strategies/adx_di_crossover.py — ADX DI+ / DI- Crossover.

Concept:
  The ADX indicator produces three lines: ADX (trend strength), DI+ (bullish
  directional pressure), and DI- (bearish directional pressure). When DI+
  crosses above DI-, bullish momentum is taking over. Wilder's original system
  traded every DI cross, but the classic refinement is to only take crosses where:
  1. ADX is rising (trend is gaining strength, not fading)
  2. ADX is above a minimum floor (some trend exists)
  3. Price is above a longer-term EMA (we're in an uptrend context)

  This generates 10-25 signals per year on daily charts — enough to validate.

Logic:
  1. DI+[-2] <= DI-[-2] AND DI+[-1] > DI-[-1] (bullish cross just happened)
  2. ADX is rising: ADX[-1] > ADX[-adx_lookback] (momentum increasing)
  3. ADX[-1] >= adx_floor (some trend present — avoids flat markets)
  4. Close > EMA(ema_period) (uptrend context)
  5. RSI <= rsi_max (not overbought)

  SHORT: DI- crosses above DI+ with falling ADX, price < EMA.

Entry / SL / TP:
  entry_low  = curr_close
  entry_high = curr_close + entry_buffer_atr * ATR
  SL         = lowest low of sl_lookback bars - sl_buffer_atr * ATR
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


def _interval_trade_type(interval: str) -> str:
    return {"1m": "scalp", "5m": "scalp", "15m": "scalp",
            "30m": "intraday", "1h": "intraday",
            "1d": "swing", "1wk": "swing"}.get(interval, "swing")


def _interval_duration(interval: str) -> str:
    return {"1m": "5-30 min", "5m": "30-90 min", "15m": "1-3 hr",
            "30m": "2-6 hr", "1h": "4-12 hr",
            "1d": "3-10 days", "1wk": "3-8 wk"}.get(interval, "varies")


def _ev(wr: float, avg_r: float) -> float:
    return round(wr * avg_r - (1 - wr) * 1.0, 4)


def _pf(wr: float, avg_r: float) -> float:
    lr = 1 - wr
    return round((wr * avg_r) / lr, 3) if lr > 1e-10 else 999.0


class ADXDICrossoverStrategy(StrategyBase):
    """
    ADX DI+ / DI- crossover with rising ADX — enters when bullish directional
    pressure overtakes bearish pressure in an established or strengthening trend.
    """

    name = "adx_di_crossover"
    description = "ADX DI+/DI- crossover with rising ADX and EMA trend filter"
    uses_bar_slicer = True

    def __init__(
        self,
        adx_period: int = 14,
        adx_floor: float = 15.0,
        adx_lookback: int = 3,
        require_rising_adx: bool = True,
        ema_period: int = 50,
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
        self.adx_period         = adx_period
        self.adx_floor          = adx_floor
        self.adx_lookback       = adx_lookback
        self.require_rising_adx = require_rising_adx
        self.ema_period         = ema_period
        self.rsi_max_long       = rsi_max_long
        self.sl_lookback        = sl_lookback
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
            "adx_period":         self.adx_period,
            "adx_floor":          self.adx_floor,
            "require_rising_adx": self.require_rising_adx,
            "ema_period":         self.ema_period,
            "tp1_r":              self.tp1_r,
            "tp2_r":              self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        need = max(self.ema_period + 10, self.adx_period * 3 + self.adx_lookback + 5)
        if len(df) < need:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        adx_ind = ta.trend.ADXIndicator(high, low, close, window=self.adx_period)
        adx     = adx_ind.adx()
        di_plus = adx_ind.adx_pos()
        di_minus= adx_ind.adx_neg()

        adx_curr  = float(adx.iloc[-1])
        adx_prev  = float(adx.iloc[-(self.adx_lookback + 1)])
        dip_curr  = float(di_plus.iloc[-1])
        dip_prev  = float(di_plus.iloc[-2])
        dim_curr  = float(di_minus.iloc[-1])
        dim_prev  = float(di_minus.iloc[-2])

        if any(pd.isna(v) for v in [adx_curr, dip_curr, dim_curr, dip_prev, dim_prev]):
            return None

        # Bull cross: DI+ crosses above DI-
        long_cross  = (dip_prev <= dim_prev) and (dip_curr > dim_curr)
        short_cross = (dip_prev >= dim_prev) and (dip_curr < dim_curr)

        if not long_cross and not short_cross:
            return None
        if short_cross and not self.shorts_enabled:
            return None

        direction = "long" if long_cross else "short"

        # ADX floor filter
        if adx_curr < self.adx_floor:
            return None

        # Rising ADX filter
        if self.require_rising_adx and adx_curr <= adx_prev:
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

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} ADX DI Crossover",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"DI+ crossed {'above' if direction == 'long' else 'below'} DI- "
                f"(DI+={dip_curr:.1f}, DI-={dim_curr:.1f}). "
                f"ADX={adx_curr:.1f} ({'rising' if adx_curr > adx_prev else 'flat'} from {adx_prev:.1f}). "
                f"Price {'above' if direction == 'long' else 'below'} EMA{self.ema_period}={ema_val:.4f}. "
                f"RSI={rsi_val:.1f}. Bullish directional pressure taking over."
            ),
            trigger=(
                f"DI+ now {'above' if direction == 'long' else 'below'} DI- — "
                f"enter on next bar above {entry_low:.4f}."
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
                f"DI+={dip_curr:.1f} vs DI-={dim_curr:.1f} | "
                f"ADX={adx_curr:.1f} (was {adx_prev:.1f}) | "
                f"EMA{self.ema_period}={ema_val:.4f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"DI- crosses back {'above' if direction == 'long' else 'below'} DI+",
                description="ADX DI cross reversed — trend direction flip failed",
                price_trigger=sl,
                action="Exit — DI crossover failed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"DI+ holds {'above' if direction == 'long' else 'below'} DI- and ADX continues rising",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="DI lines immediately re-cross (whipsaw)",
                    outcome="NO TRADE — wait for cleaner DI separation",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
