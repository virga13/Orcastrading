"""
p3_backtester/strategies/macd_crossover.py — MACD Signal-Line Crossover strategy.

Concept:
  The classic MACD (Moving Average Convergence/Divergence) fires a trade
  when the MACD line (fast EMA - slow EMA) crosses its signal line (EMA of MACD).
  This version adds quality filters to avoid whipsaws:

  1. MACD/Signal crossover on the current bar (prev MACD vs signal side flips)
  2. Histogram confirms direction (histogram sign matches crossover direction)
  3. RSI not overbought/oversold in crossover direction
  4. Volume expansion (optional): volume >= volume_mult × N-bar average
  5. HTF EMA20/EMA50 trend alignment (optional)
  6. Zero-line filter (optional): MACD must already be above/below zero for stronger confirmation

Entry / SL / TP:
  entry = close on crossover bar (market order)
  SL    = N-bar swing low (long) | N-bar swing high (short)
  TP1   = entry + tp1_r × risk  |  TP2 = entry + tp2_r × risk
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


class MACDCrossoverStrategy(StrategyBase):
    """
    MACD signal-line crossover with quality filters.

    Fires when MACD line crosses its signal line, with optional RSI,
    volume, HTF trend, and zero-line filters to reduce whipsaws.
    """

    name = "macd_crossover"
    description = "MACD crossover — signal-line cross with RSI + volume + HTF filters"
    uses_bar_slicer = True

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        rsi_max_long: int = 65,
        rsi_min_short: int = 35,
        require_above_zero: bool = False,
        volume_mult: float = 1.0,       # 1.0 = no volume filter
        htf_filter: bool = True,
        sl_lookback: int = 10,
        sl_buffer_atr: float = 0.1,
        tp1_r: float = 1.5,
        tp2_r: float = 3.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = True,
        win_rate: float = 0.45,
    ):
        self.macd_fast         = macd_fast
        self.macd_slow         = macd_slow
        self.macd_signal       = macd_signal
        self.rsi_period        = rsi_period
        self.rsi_max_long      = rsi_max_long
        self.rsi_min_short     = rsi_min_short
        self.require_above_zero = require_above_zero
        self.volume_mult       = volume_mult
        self.htf_filter        = htf_filter
        self.sl_lookback       = sl_lookback
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
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "rsi_max_long": self.rsi_max_long,
            "htf_filter": self.htf_filter,
            "tp1_r": self.tp1_r,
            "tp2_r": self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = self.macd_slow + self.macd_signal + 5
        if len(df) < min_bars:
            return None

        close    = df["Close"]
        high     = df["High"]
        low      = df["Low"]
        interval = tech.get("interval", "1d")

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_ind  = ta.trend.MACD(close, window_fast=self.macd_fast,
                                   window_slow=self.macd_slow,
                                   window_sign=self.macd_signal)
        macd_line = macd_ind.macd()
        sig_line  = macd_ind.macd_signal()
        histogram = macd_ind.macd_diff()

        if len(macd_line.dropna()) < 3:
            return None

        macd_curr = float(macd_line.iloc[-1])
        macd_prev = float(macd_line.iloc[-2])
        sig_curr  = float(sig_line.iloc[-1])
        sig_prev  = float(sig_line.iloc[-2])
        hist_curr = float(histogram.iloc[-1])

        if any(pd.isna(v) for v in [macd_curr, macd_prev, sig_curr, sig_prev]):
            return None

        # ── Crossover detection ────────────────────────────────────────────────
        long_cross  = macd_prev <= sig_prev and macd_curr > sig_curr
        short_cross = macd_prev >= sig_prev and macd_curr < sig_curr

        if not long_cross and not short_cross:
            return None

        direction = "long" if long_cross else "short"
        if direction == "short" and not self.shorts_enabled:
            return None

        # ── Histogram confirmation — must agree with crossover ─────────────────
        if direction == "long" and hist_curr <= 0:
            return None
        if direction == "short" and hist_curr >= 0:
            return None

        # ── Zero-line filter (optional) ────────────────────────────────────────
        if self.require_above_zero:
            if direction == "long" and macd_curr <= 0:
                return None
            if direction == "short" and macd_curr >= 0:
                return None

        # ── RSI filter ─────────────────────────────────────────────────────────
        rsi_s   = ta.momentum.RSIIndicator(close, window=self.rsi_period).rsi()
        rsi_val = float(rsi_s.iloc[-1])
        if pd.isna(rsi_val):
            return None
        if direction == "long" and rsi_val > self.rsi_max_long:
            return None
        if direction == "short" and rsi_val < self.rsi_min_short:
            return None

        # ── Volume filter ──────────────────────────────────────────────────────
        if self.volume_mult > 1.0:
            vol     = float(df["Volume"].iloc[-1])
            avg_vol = float(df["Volume"].iloc[-21:-1].mean()) if len(df) > 21 else 0.0
            if avg_vol > 0 and vol < self.volume_mult * avg_vol:
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

        # ── Entry / SL / TP ────────────────────────────────────────────────────
        price   = float(close.iloc[-1])
        atr_val = tech.get("atr_14") or float(
            ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        )
        atr_val = float(atr_val)
        if atr_val <= 0 or pd.isna(atr_val):
            return None

        n = min(self.sl_lookback, len(df) - 1)
        buf = self.sl_buffer_atr * atr_val

        if direction == "long":
            swing_low  = float(low.iloc[-n - 1:-1].min())
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(swing_low - buf, 4)
            risk       = entry_high - sl
            if risk <= 0 or risk / price > 0.15:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
            inv_level = swing_low
        else:
            swing_high = float(high.iloc[-n - 1:-1].max())
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(swing_high + buf, 4)
            risk       = sl - entry_low
            if risk <= 0 or risk / price > 0.15:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)
            inv_level = swing_high

        avg_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev    = _ev(self.win_rate, avg_r)
        pf    = _pf(self.win_rate, avg_r)
        atr_pct = tech.get("atr_pct", round(atr_val / price * 100, 3))
        trade_type = _interval_trade_type(interval)

        zero_note = " | zero-line filter ON" if self.require_above_zero else ""
        htf_note  = " | HTF aligned" if self.htf_filter else ""

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} MACD crossover",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"MACD({self.macd_fast},{self.macd_slow},{self.macd_signal}) crossed "
                f"{'above' if direction == 'long' else 'below'} signal line. "
                f"MACD={macd_curr:.5f}, Signal={sig_curr:.5f}, Hist={hist_curr:.5f}. "
                f"RSI={rsi_val:.0f}.{htf_note}{zero_note}"
            ),
            trigger=(
                f"MACD line crossed {'above' if direction == 'long' else 'below'} signal line "
                f"at price {price:.4f}."
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
            confidence=0.48,
            confidence_note=(
                f"MACD={macd_curr:.5f} | Sig={sig_curr:.5f} | Hist={hist_curr:.5f} | RSI={rsi_val:.0f}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"MACD crosses back {'below' if direction == 'long' else 'above'} signal line",
                description="MACD reversal — crossover signal failed",
                price_trigger=sl,
                action="Exit immediately — MACD momentum reversed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"MACD crossover holds and histogram {'expands positive' if direction == 'long' else 'expands negative'}",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=price,
                ),
                DecisionTreeEntry(
                    scenario=f"MACD crosses back {'below' if direction == 'long' else 'above'} signal line",
                    outcome="NO TRADE — crossover reversed",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
