"""
p3_backtester/strategies/bb_squeeze.py — Bollinger Band Squeeze strategy.

Concept (TTM Squeeze style):
  When Bollinger Bands (20, 2σ) contract INSIDE Keltner Channels (20, 1.5×ATR),
  the market is in a low-volatility squeeze. When BB expands back outside KC,
  a directional breakout is underway. We enter in the direction price breaks out.

Entry conditions:
  1. Squeeze was active for >= min_squeeze_bars consecutive bars
     (BB fully inside KC on all prior bars)
  2. Current bar is the first bar where BB expands outside KC (squeeze fires)
  3. Direction: close > BB midline → long; close < BB midline → short

Entry / SL / TP:
  entry = close of the breakout bar (market order)
  SL    = KC lower band (long) | KC upper band (short) - sl_buffer × ATR
  TP1   = entry + tp1_r × risk
  TP2   = entry + tp2_r × risk
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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    return ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()


def _ev(wr: float, avg_r: float) -> float:
    return round(wr * avg_r - (1 - wr) * 1.0, 4)


def _pf(wr: float, avg_r: float) -> float:
    lr = 1 - wr
    return round((wr * avg_r) / lr, 3) if lr > 1e-10 else 999.0


class BBSqueezeStrategy(StrategyBase):
    """
    Bollinger Band Squeeze breakout.

    Fires when BB bands expand outside Keltner Channels after a squeeze,
    entering in the direction price breaks relative to the BB midline.
    """

    name = "bb_squeeze"
    description = "BB Squeeze breakout — Bollinger inside Keltner → breakout direction"
    uses_bar_slicer = True

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        min_squeeze_bars: int = 5,
        sl_buffer_atr: float = 0.1,
        tp1_r: float = 1.5,
        tp2_r: float = 3.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        shorts_enabled: bool = True,
        win_rate: float = 0.45,
    ):
        self.bb_period       = bb_period
        self.bb_std          = bb_std
        self.kc_period       = kc_period
        self.kc_mult         = kc_mult
        self.min_squeeze_bars = min_squeeze_bars
        self.sl_buffer_atr   = sl_buffer_atr
        self.tp1_r           = tp1_r
        self.tp2_r           = tp2_r
        self.tp1_alloc       = tp1_alloc
        self.tp2_alloc       = tp2_alloc
        self.shorts_enabled  = shorts_enabled
        self.win_rate        = win_rate

    @property
    def params(self) -> dict:
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "kc_mult": self.kc_mult,
            "min_squeeze_bars": self.min_squeeze_bars,
            "tp1_r": self.tp1_r,
            "tp2_r": self.tp2_r,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        min_bars = max(self.bb_period, self.kc_period) + self.min_squeeze_bars + 2
        if len(df) < min_bars:
            return None

        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]
        interval = tech.get("interval", "1d")

        # ── Bollinger Bands ────────────────────────────────────────────────────
        bb = ta.volatility.BollingerBands(close, window=self.bb_period, window_dev=self.bb_std)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_mid   = bb.bollinger_mavg()

        # ── Keltner Channel ────────────────────────────────────────────────────
        atr_s   = _atr(high, low, close, self.kc_period)
        kc_mid  = ta.trend.EMAIndicator(close, window=self.kc_period).ema_indicator()
        kc_upper = kc_mid + self.kc_mult * atr_s
        kc_lower = kc_mid - self.kc_mult * atr_s

        # ── Squeeze detection ─────────────────────────────────────────────────
        # squeeze ON = BB fully inside KC
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # Current bar must NOT be in squeeze (BB just expanded outside KC)
        if squeeze.iloc[-1]:
            return None

        # Count consecutive squeeze bars immediately before current bar
        consec = 0
        for j in range(2, len(squeeze) + 1):
            if squeeze.iloc[-j]:
                consec += 1
            else:
                break

        if consec < self.min_squeeze_bars:
            return None

        # ── Direction ─────────────────────────────────────────────────────────
        price    = float(close.iloc[-1])
        midline  = float(bb_mid.iloc[-1])
        atr_val  = float(atr_s.iloc[-1])
        kc_up    = float(kc_upper.iloc[-1])
        kc_lo    = float(kc_lower.iloc[-1])

        if pd.isna(midline) or pd.isna(atr_val) or atr_val <= 0:
            return None

        if price > midline:
            direction = "long"
        elif price < midline and self.shorts_enabled:
            direction = "short"
        else:
            return None

        # ── Entry / SL / TP ───────────────────────────────────────────────────
        buf = self.sl_buffer_atr * atr_val

        if direction == "long":
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(kc_lo - buf, 4)
            risk       = entry_high - sl
            if risk <= 0:
                return None
            tp1 = round(entry_high + self.tp1_r * risk, 4)
            tp2 = round(entry_high + self.tp2_r * risk, 4)
            trail_at = tp1
            inv_cond = f"Close back below KC midline ${kc_mid.iloc[-1]:.4f}"
        else:
            entry_low  = round(price, 4)
            entry_high = round(price, 4)
            sl         = round(kc_up + buf, 4)
            risk       = sl - entry_low
            if risk <= 0:
                return None
            tp1 = round(entry_low - self.tp1_r * risk, 4)
            tp2 = round(entry_low - self.tp2_r * risk, 4)
            trail_at = tp1
            inv_cond = f"Close back above KC midline ${kc_mid.iloc[-1]:.4f}"

        avg_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev    = _ev(self.win_rate, avg_r)
        pf    = _pf(self.win_rate, avg_r)

        atr_pct = tech.get("atr_pct", round(atr_val / price * 100, 3))
        trade_type = _interval_trade_type(interval)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if direction == 'long' else 'Short'} BB squeeze breakout",
            direction=direction,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"BB Squeeze fired after {consec} consecutive squeeze bars. "
                f"BB ({self.bb_period}, {self.bb_std}) expanded outside KC ({self.kc_period}, {self.kc_mult}x). "
                f"Price {price:.4f} {'above' if direction == 'long' else 'below'} BB midline {midline:.4f}."
            ),
            trigger=(
                f"Squeeze release bar — BB expanded outside Keltner Channel. "
                f"Enter at close {price:.4f}."
            ),
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=sl,
            trailing_sl_to_breakeven=trail_at,
            targets=[
                SetupTarget(price=tp1, label=f"Target 1 ({self.tp1_alloc}%)", allocation_pct=self.tp1_alloc),
                SetupTarget(price=tp2, label=f"Target 2 ({self.tp2_alloc}%)", allocation_pct=self.tp2_alloc),
            ],
            rr_ratio=round(avg_r, 2),
            win_rate_estimate=self.win_rate,
            trade_duration=_interval_duration(interval),
            ev=ev,
            profit_factor=pf,
            confidence=0.50,
            confidence_note=(
                f"BB({self.bb_period},{self.bb_std}) mid={midline:.4f} | "
                f"KC({self.kc_period},{self.kc_mult}) [{kc_lo:.4f}–{kc_up:.4f}] | "
                f"Squeeze bars: {consec}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=inv_cond,
                description="Squeeze breakout has failed — price returned to range",
                price_trigger=sl,
                action="Exit immediately — breakout invalidated",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Squeeze fires, price {'above' if direction == 'long' else 'below'} midline",
                    outcome=f"ENTER {direction.upper()} — SL ${sl:.4f}, TP1 ${tp1:.4f}",
                    direction=direction,
                    setup_name="Setup A",
                    entry_price=price,
                ),
                DecisionTreeEntry(
                    scenario=f"Price reverses back {'below' if direction == 'long' else 'above'} ${sl:.4f}",
                    outcome="NO TRADE — breakout failed",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr_val:.4f} ({atr_pct}% of price). "
                + ("REDUCE 50% — high vol." if float(atr_pct) > 2 else "Normal sizing.")
            ),
        )
