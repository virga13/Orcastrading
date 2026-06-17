"""
p3_backtester/strategies/heikin_ashi_trend.py — Heikin Ashi Trend Continuation.

Concept:
  Heikin Ashi candles (HA) smooth price action by averaging OHLC values across bars,
  making trends and reversals easier to read. A clean HA uptrend is identified by:
    - HA close > HA open for N consecutive bars (bullish HA bodies)
    - HA low == HA open (no lower wicks — pure bullish momentum)
  When these conditions are met after a pullback (price dips near EMA then recovers),
  it signals trend continuation with defined risk.

Logic:
  1. Compute Heikin Ashi OHLC from real OHLC bars
  2. Require consecutive_bull_bars of HA close > HA open
  3. Optionally require no lower wicks on the most recent N bars (require_no_lower_wick)
  4. Price > EMA (trend alignment), ADX >= threshold
  5. RSI < rsi_max (not overbought)

Entry / SL / TP:
  entry_low  = curr_close (at close of signal bar)
  entry_high = curr_close + entry_buffer_atr * ATR
  SL = lowest HA low of the last ha_sl_lookback HA bars - sl_buffer_atr * ATR
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


def _compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha["ha_open"]  = ha_open
    ha["ha_close"] = ha_close
    ha["ha_high"]  = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha["ha_low"]   = pd.concat([df["Low"],  ha_open, ha_close], axis=1).min(axis=1)
    return ha


class HeikinAshiTrendStrategy(StrategyBase):
    """
    Heikin Ashi trend continuation — enters when HA candles show clean
    consecutive bullish momentum above a trend EMA with ADX confirmation.
    """

    name = "heikin_ashi_trend"
    description = "Heikin Ashi consecutive bullish bars above EMA with ADX"
    uses_bar_slicer = True

    def __init__(
        self,
        consecutive_bull_bars: int = 3,
        require_no_lower_wick: bool = True,
        ema_period: int = 50,
        adx_threshold: int = 20,
        rsi_max_long: int = 70,
        ha_sl_lookback: int = 3,
        entry_buffer_atr: float = 0.05,
        sl_buffer_atr: float = 0.1,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = False,
        win_rate: float = 0.45,
    ):
        self.consecutive_bull_bars  = consecutive_bull_bars
        self.require_no_lower_wick  = require_no_lower_wick
        self.ema_period             = ema_period
        self.adx_threshold          = adx_threshold
        self.rsi_max_long           = rsi_max_long
        self.ha_sl_lookback         = ha_sl_lookback
        self.entry_buffer_atr       = entry_buffer_atr
        self.sl_buffer_atr          = sl_buffer_atr
        self.tp1_r                  = tp1_r
        self.tp2_r                  = tp2_r
        self.tp1_alloc              = tp1_alloc
        self.tp2_alloc              = tp2_alloc
        self.shorts_enabled         = shorts_enabled
        self.win_rate               = win_rate

    @property
    def params(self) -> dict:
        return {
            "consecutive_bull_bars": self.consecutive_bull_bars,
            "require_no_lower_wick": self.require_no_lower_wick,
            "ema_period":            self.ema_period,
            "adx_threshold":         self.adx_threshold,
            "tp1_r":                 self.tp1_r,
            "tp2_r":                 self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        need = max(self.ema_period + 10, 50 + self.consecutive_bull_bars)
        if len(df) < need:
            return None

        high   = df["High"]
        low    = df["Low"]
        close  = df["Close"]
        interval = tech.get("interval", "1d")

        ha = _compute_heikin_ashi(df)
        ha_close = ha["ha_close"]
        ha_open  = ha["ha_open"]
        ha_low   = ha["ha_low"]

        # Check last N bars are all HA bullish (close > open)
        n = self.consecutive_bull_bars
        last_n_bull = all(
            float(ha_close.iloc[-(k+1)]) > float(ha_open.iloc[-(k+1)])
            for k in range(n)
        )
        if not last_n_bull:
            # Try short: all HA bearish
            if not self.shorts_enabled:
                return None
            last_n_bear = all(
                float(ha_close.iloc[-(k+1)]) < float(ha_open.iloc[-(k+1)])
                for k in range(n)
            )
            if not last_n_bear:
                return None
            direction = "short"
        else:
            direction = "long"

        # Require no lower wick (for longs): ha_low == ha_open on the last N bars
        if self.require_no_lower_wick:
            if direction == "long":
                for k in range(n):
                    ha_low_k  = float(ha_low.iloc[-(k+1)])
                    ha_open_k = float(ha_open.iloc[-(k+1)])
                    if abs(ha_low_k - ha_open_k) > 1e-6:
                        return None
            else:
                ha_high = ha["ha_high"]
                for k in range(n):
                    ha_high_k = float(ha_high.iloc[-(k+1)])
                    ha_open_k = float(ha_open.iloc[-(k+1)])
                    if abs(ha_high_k - ha_open_k) > 1e-6:
                        return None

        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        ema_val = float(ta.trend.EMAIndicator(close, self.ema_period).ema_indicator().iloc[-1])
        curr_close = float(close.iloc[-1])

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

        # SL = lowest HA low of last ha_sl_lookback bars - buffer
        sl_ha_lows = [float(ha_low.iloc[-(k+1)]) for k in range(self.ha_sl_lookback)]
        ha_sl_level = min(sl_ha_lows)

        buf_entry = self.entry_buffer_atr * atr_val
        buf_sl    = self.sl_buffer_atr * atr_val

        if direction == "long":
            entry_low  = round(curr_close, 4)
            entry_high = round(curr_close + buf_entry, 4)
            sl         = round(ha_sl_level - buf_sl, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
        else:
            entry_high = round(curr_close, 4)
            entry_low  = round(curr_close - buf_entry, 4)
            ha_high    = ha["ha_high"]
            ha_sl_highs = [float(ha_high.iloc[-(k+1)]) for k in range(self.ha_sl_lookback)]
            ha_sl_level_s = max(ha_sl_highs)
            sl         = round(ha_sl_level_s + buf_sl, 4)
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
            label=f"{'Long' if direction == 'long' else 'Short'} Heikin Ashi Trend",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"{n} consecutive {'bullish' if direction == 'long' else 'bearish'} Heikin Ashi bars"
                f"{' with no lower wicks' if self.require_no_lower_wick else ''}. "
                f"Price {'above' if direction == 'long' else 'below'} EMA{self.ema_period}={ema_val:.4f}. "
                f"ADX={adx_val:.1f}, RSI={rsi_val:.1f}. "
                f"Clean HA momentum confirms trend continuation."
            ),
            trigger=(
                f"Price holds {'above' if direction == 'long' else 'below'} "
                f"{'%.4f' % entry_low} at next bar open — or enters on close fill."
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
                f"HA consecutive bars={n} | "
                f"EMA{self.ema_period}={ema_val:.4f} | "
                f"ADX={adx_val:.1f} | RSI={rsi_val:.1f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=curr_close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Price closes {'below' if direction == 'long' else 'above'} EMA{self.ema_period} or HA flips",
                description="Heikin Ashi trend broken — setup invalidated",
                price_trigger=sl,
                action="Exit — HA trend failed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"HA trend continues, price holds {'above' if direction == 'long' else 'below'} EMA",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=float(entry_high if direction == "long" else entry_low),
                ),
                DecisionTreeEntry(
                    scenario="HA reversal bar appears (opposite color candle)",
                    outcome="NO TRADE — wait for re-setup",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
