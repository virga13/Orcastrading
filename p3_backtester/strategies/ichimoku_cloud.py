"""
p3_backtester/strategies/ichimoku_cloud.py — Ichimoku Cloud Breakout.

Concept:
  The Ichimoku Kinko Hyo system uses five lines to define trend, momentum,
  and support/resistance simultaneously:
    Tenkan-sen  (conversion)  = (9-period high + 9-period low) / 2
    Kijun-sen   (base)        = (26-period high + 26-period low) / 2
    Senkou A    (leading A)   = (Tenkan + Kijun) / 2, shifted 26 bars forward
    Senkou B    (leading B)   = (52-period high + 52-period low) / 2, shifted 26 bars
    Chikou      (lagging)     = current close shifted 26 bars back

  The "Kumo" (cloud) is the area between Senkou A and Senkou B.
  A price above the cloud signals an uptrend; breakout from below to above
  is the classic entry.

  Note: We use the CURRENT (unshifted) Senkou A/B computed at bar[-1] to
  represent the cloud NOW — this avoids lookahead bias from the standard
  26-bar forward shift.

Logic (bullish cloud breakout):
  1. Close[-2] <= max(SenkouA[-2], SenkouB[-2]) — was IN or below cloud
  2. Close[-1] > max(SenkouA[-1], SenkouB[-1]) — now ABOVE cloud
  3. Tenkan >= Kijun (bullish TK cross or Tenkan above Kijun)
  4. ADX >= adx_threshold (trend strength)
  5. RSI <= rsi_max

  OR (simpler trend continuation):
  1. Close is above the cloud for the last confirm_bars bars
  2. Tenkan crosses above Kijun (fresh TK cross)
  3. ADX >= adx_threshold

Entry / SL / TP:
  entry = close (confirmation bar)
  SL    = top of cloud (Kijun-sen or cloud top) - sl_buffer_atr * ATR
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


def _interval_trade_type(interval: str) -> str:
    return {"1m": "scalp", "5m": "scalp", "15m": "scalp",
            "30m": "intraday", "1h": "intraday",
            "1d": "swing", "1wk": "swing"}.get(interval, "swing")


def _interval_duration(interval: str) -> str:
    return {"1m": "5-30 min", "5m": "30-90 min", "15m": "1-3 hr",
            "30m": "2-6 hr", "1h": "4-12 hr",
            "1d": "5-15 days", "1wk": "4-12 wk"}.get(interval, "varies")


def _ev(wr: float, avg_r: float) -> float:
    return round(wr * avg_r - (1 - wr) * 1.0, 4)


def _pf(wr: float, avg_r: float) -> float:
    lr = 1 - wr
    return round((wr * avg_r) / lr, 3) if lr > 1e-10 else 999.0


def _ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
              tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    """Compute unshifted Ichimoku lines (no lookahead)."""
    t_high = high.rolling(tenkan).max()
    t_low  = low.rolling(tenkan).min()
    tenkan_s = (t_high + t_low) / 2.0

    k_high = high.rolling(kijun).max()
    k_low  = low.rolling(kijun).min()
    kijun_s = (k_high + k_low) / 2.0

    senkou_a = (tenkan_s + kijun_s) / 2.0

    sb_high = high.rolling(senkou_b).max()
    sb_low  = low.rolling(senkou_b).min()
    senkou_b_s = (sb_high + sb_low) / 2.0

    return tenkan_s, kijun_s, senkou_a, senkou_b_s


class IchimokuCloudStrategy(StrategyBase):
    """
    Ichimoku Cloud Breakout — enters on Tenkan/Kijun cross while price is
    above the Kumo cloud, or on a fresh cloud breakout from below.
    """

    name = "ichimoku_cloud"
    description = "Ichimoku Tenkan/Kijun cross above the cloud with ADX gate"
    uses_bar_slicer = True

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        require_above_cloud: bool = True,
        adx_threshold: int = 20,
        rsi_max_long: int = 70,
        entry_buffer_atr: float = 0.05,
        sl_buffer_atr: float = 0.15,
        tp1_r: float = 2.0,
        tp2_r: float = 5.0,
        tp1_alloc: int = 50,
        tp2_alloc: int = 50,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.tenkan_period      = tenkan_period
        self.kijun_period       = kijun_period
        self.senkou_b_period    = senkou_b_period
        self.require_above_cloud= require_above_cloud
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
            "tenkan_period":       self.tenkan_period,
            "kijun_period":        self.kijun_period,
            "require_above_cloud": self.require_above_cloud,
            "adx_threshold":       self.adx_threshold,
            "tp1_r":               self.tp1_r,
            "tp2_r":               self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        need = self.senkou_b_period + self.kijun_period + 10
        if len(df) < need:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        tenkan, kijun, senkou_a, senkou_b = _ichimoku(
            high, low, close,
            self.tenkan_period, self.kijun_period, self.senkou_b_period
        )

        tk_curr  = float(tenkan.iloc[-1])
        tk_prev  = float(tenkan.iloc[-2])
        kj_curr  = float(kijun.iloc[-1])
        kj_prev  = float(kijun.iloc[-2])
        sa_curr  = float(senkou_a.iloc[-1])
        sb_curr  = float(senkou_b.iloc[-1])
        cl_curr  = float(close.iloc[-1])

        if any(pd.isna(v) for v in [tk_curr, kj_curr, sa_curr, sb_curr]):
            return None

        cloud_top_curr    = max(sa_curr, sb_curr)
        cloud_bottom_curr = min(sa_curr, sb_curr)

        # Bullish TK cross: Tenkan crosses above Kijun
        long_tk_cross  = (tk_prev <= kj_prev) and (tk_curr > kj_curr)
        short_tk_cross = (tk_prev >= kj_prev) and (tk_curr < kj_curr)

        if not long_tk_cross and not short_tk_cross:
            return None
        if short_tk_cross and not self.shorts_enabled:
            return None

        direction = "long" if long_tk_cross else "short"

        # Cloud position filter
        if self.require_above_cloud:
            if direction == "long" and cl_curr <= cloud_top_curr:
                return None
            if direction == "short" and cl_curr >= cloud_bottom_curr:
                return None

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
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
            # SL below the Kijun (base line — natural Ichimoku stop) or cloud top
            sl_level   = min(kj_curr, cloud_bottom_curr)
            sl         = round(sl_level - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(cl_curr, 4)
            entry_low  = round(cl_curr - buf_entry, 4)
            sl_level   = max(kj_curr, cloud_top_curr)
            sl         = round(sl_level + buf_sl, 4)
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
        cloud_dist = round((cl_curr - cloud_top_curr) / atr_val, 2) if direction == "long" else round((cloud_bottom_curr - cl_curr) / atr_val, 2)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} Ichimoku TK Cross",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Tenkan ({tk_curr:.2f}) crossed {'above' if direction == 'long' else 'below'} "
                f"Kijun ({kj_curr:.2f}). "
                f"Price {'above' if direction == 'long' else 'below'} cloud "
                f"(top={cloud_top_curr:.2f}, dist={cloud_dist:.1f} ATRs). "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}."
            ),
            trigger=f"TK cross confirmed — enter above {entry_low:.4f}.",
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
                f"Tenkan={tk_curr:.2f} Kijun={kj_curr:.2f} | "
                f"Cloud top={cloud_top_curr:.2f} | ADX={adx_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=cl_curr,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Tenkan crosses back {'below' if direction == 'long' else 'above'} Kijun",
                description="TK cross reversed — Ichimoku setup invalidated",
                price_trigger=sl,
                action="Exit — TK cross reversed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario="Tenkan holds above Kijun, price stays above cloud",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="TK cross reverses or price drops into cloud",
                    outcome="NO TRADE — setup failed",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
