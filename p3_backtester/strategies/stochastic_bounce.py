"""
p3_backtester/strategies/stochastic_bounce.py — Stochastic Oscillator Bounce.

Concept:
  The Stochastic Oscillator (%K and %D) measures where the current close sits
  within the recent high-low range, normalised to 0-100. Values below 20 indicate
  oversold (price near the low of the range); above 80 indicate overbought.

  Bullish bounce (long):
    - %K crosses ABOVE %D from the oversold zone (%K was < oversold_level)
    - Price is above EMA50 (overall uptrend — bounce expected to resume)
    - ADX >= adx_threshold (enough trend to expect a resumption, not just noise)

  Bearish bounce (short):
    - %K crosses BELOW %D from the overbought zone (%K was > overbought_level)
    - Price is below EMA50 (overall downtrend)
    - ADX >= adx_threshold

  The combination of trend filter + stochastic cross is known as the "Stochastic
  in trend" setup — one of the most commonly taught patterns in technical analysis.

Entry / SL / TP:
  entry = close on cross bar
  SL    = recent swing low (for longs) - sl_buffer_atr * ATR
  TP1   = entry + tp1_r * risk
  TP2   = entry + tp2_r * risk
"""
from __future__ import annotations

import pandas as pd
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
    return {"1m": "5-30 minutes", "5m": "30-90 minutes", "15m": "1-3 hours",
            "30m": "2-6 hours", "1h": "4-12 hours",
            "1d": "3-8 days", "1wk": "3-8 weeks"}.get(interval, "varies")


def _ev(wr: float, avg_r: float) -> float:
    return round(wr * avg_r - (1 - wr) * 1.0, 4)


def _pf(wr: float, avg_r: float) -> float:
    lr = 1 - wr
    return round((wr * avg_r) / lr, 3) if lr > 1e-10 else 999.0


class StochasticBounceStrategy(StrategyBase):
    """
    Stochastic Oscillator Bounce.

    Enters when %K crosses %D from oversold (long) or overbought (short),
    filtered by trend direction via EMA50 and ADX strength.
    """

    name = "stochastic_bounce"
    description = "Stochastic %K/%D cross from oversold/overbought with trend filter"
    uses_bar_slicer = True

    def __init__(
        self,
        stoch_k: int = 14,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
        oversold_level: float = 20.0,
        overbought_level: float = 80.0,
        ema_period: int = 50,
        adx_threshold: int = 20,
        sl_lookback: int = 10,
        sl_buffer_atr: float = 0.15,
        tp1_r: float = 1.5,
        tp2_r: float = 3.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = True,
        win_rate: float = 0.45,
    ):
        self.stoch_k          = stoch_k
        self.stoch_d          = stoch_d
        self.stoch_smooth     = stoch_smooth
        self.oversold_level   = oversold_level
        self.overbought_level = overbought_level
        self.ema_period       = ema_period
        self.adx_threshold    = adx_threshold
        self.sl_lookback      = sl_lookback
        self.sl_buffer_atr    = sl_buffer_atr
        self.tp1_r            = tp1_r
        self.tp2_r            = tp2_r
        self.tp1_alloc        = tp1_alloc
        self.tp2_alloc        = tp2_alloc
        self.shorts_enabled   = shorts_enabled
        self.win_rate         = win_rate

    @property
    def params(self) -> dict:
        return {
            "stoch_k":         self.stoch_k,
            "stoch_d":         self.stoch_d,
            "oversold_level":  self.oversold_level,
            "overbought_level":self.overbought_level,
            "ema_period":      self.ema_period,
            "adx_threshold":   self.adx_threshold,
            "tp1_r":           self.tp1_r,
            "tp2_r":           self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = self.stoch_k + self.stoch_d + self.ema_period + 10
        if len(df) < min_bars:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        stoch = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close,
            window=self.stoch_k,
            smooth_window=self.stoch_d,
        )
        k_series = stoch.stoch()
        d_series = stoch.stoch_signal()

        if k_series.isna().iloc[-1] or d_series.isna().iloc[-1]:
            return None

        k_curr = float(k_series.iloc[-1])
        k_prev = float(k_series.iloc[-2])
        d_curr = float(d_series.iloc[-1])
        d_prev = float(d_series.iloc[-2])

        # Cross detection
        bull_cross = k_prev <= d_prev and k_curr > d_curr  # %K crossed above %D
        bear_cross = k_prev >= d_prev and k_curr < d_curr  # %K crossed below %D

        if not bull_cross and not bear_cross:
            return None

        # Oversold/overbought zone check — cross must originate from the extreme zone
        if bull_cross and k_prev > self.oversold_level:
            return None
        if bear_cross and k_prev < self.overbought_level:
            return None

        if bear_cross and not self.shorts_enabled:
            return None

        direction = "long" if bull_cross else "short"

        # EMA trend filter
        price   = float(close.iloc[-1])
        ema_val = float(ta.trend.EMAIndicator(close, window=self.ema_period).ema_indicator().iloc[-1])

        if direction == "long"  and price < ema_val:
            return None
        if direction == "short" and price > ema_val:
            return None

        # ADX strength
        adx_val = float(
            ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
        )
        if adx_val < self.adx_threshold:
            return None

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        buf = self.sl_buffer_atr * atr_val

        if direction == "long":
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(float(low.iloc[-self.sl_lookback:].min()) - buf, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(float(high.iloc[-self.sl_lookback:].max()) + buf, 4)
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
        zone_str   = "oversold" if direction == "long" else "overbought"

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Stochastic Bounce ({zone_str})",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Stochastic %K({self.stoch_k}) crossed {'above' if direction == 'long' else 'below'} "
                f"%D({self.stoch_d}) from {zone_str} zone. "
                f"%K={k_curr:.1f}, %D={d_curr:.1f} (prev: %K={k_prev:.1f}, %D={d_prev:.1f}). "
                f"EMA{self.ema_period}={ema_val:.4f}, ADX={adx_val:.1f}."
            ),
            trigger=(
                f"Stochastic cross: %K {k_prev:.1f} -> {k_curr:.1f} crossed "
                f"{'above' if direction == 'long' else 'below'} %D from {zone_str} zone."
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
                f"Stoch %K={k_curr:.1f} %D={d_curr:.1f} | "
                f"EMA{self.ema_period}={ema_val:.4f} | ADX={adx_val:.1f} | ATR={atr_val:.4f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Stochastic re-enters {zone_str} zone or price breaks {'below' if direction == 'long' else 'above'} SL",
                description="Bounce failed — momentum did not follow through",
                price_trigger=sl,
                action="Exit — stochastic bounce invalidated",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Stochastic continues rising from {zone_str} and price trends",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=price,
                ),
                DecisionTreeEntry(
                    scenario=f"Stochastic re-crosses back into {zone_str} zone",
                    outcome="NO TRADE — false cross, wait for re-test",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
