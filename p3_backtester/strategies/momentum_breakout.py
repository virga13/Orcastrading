"""
p3_backtester/strategies/momentum_breakout.py — Daily Momentum Breakout strategy.

Fires when price closes above a multi-week consolidation high on volume expansion.

Entry conditions:
  1. Price closes above the N-day high (default N=20) — consolidation breakout
  2. Breakout bar volume >= volume_multiplier × N-day average volume
  3. Bar closes in the top close_top_pct of the bar's range  (conviction close, not a wick)
  4. ADX(14) was < adx_max_before on the PREVIOUS bar  (was consolidating, not already trending)
  5. RSI(14) is in [rsi_min, rsi_max] at breakout  (momentum, not already overbought)
  6. Weekly EMA20 > EMA50  (higher-timeframe trend aligned)

Stop-loss: below the N-day consolidation low
TP1: entry + 1 × consolidation range height  (measured move)
TP2: entry + 2 × consolidation range height
"""
from __future__ import annotations

import pandas as pd
import ta

from p3_backtester.strategies.base import StrategyBase
from p1_analysis_engine.schema import (
    SetupsOutput, TradingSetup, SetupTarget, InvalidationScenario, DecisionTreeEntry,
)


# HTF resampling rules per entry interval
_HTF_RESAMPLE: dict[str, str] = {
    "1m": "15min", "5m": "1h", "15m": "4h",
    "30m": "1D", "1h": "1D", "4h": "1W",
    "1d": "1W", "1wk": "1ME",
}


class MomentumBreakoutStrategy(StrategyBase):
    name = "momentum_breakout"
    description = "N-period high breakout with volume expansion and HTF bias — any timeframe"

    def __init__(
        self,
        lookback_periods: int     = 20,   # was lookback_days — now interval-agnostic
        volume_multiplier: float  = 1.5,
        close_top_pct: float      = 0.25,
        adx_max_before: int       = 25,
        rsi_min: int              = 50,
        rsi_max: int              = 70,
        tp1_alloc: int            = 70,
        tp2_alloc: int            = 30,
        # Legacy compat
        lookback_days: int        = 0,
    ):
        self.lookback_periods   = lookback_periods if not lookback_days else lookback_days
        self.volume_multiplier  = volume_multiplier
        self.close_top_pct      = close_top_pct
        self.adx_max_before     = adx_max_before
        self.rsi_min            = rsi_min
        self.rsi_max            = rsi_max
        self.tp1_alloc          = tp1_alloc
        self.tp2_alloc          = tp2_alloc

    @property
    def params(self) -> dict:
        return {
            "lookback_periods":  self.lookback_periods,
            "volume_multiplier": self.volume_multiplier,
            "close_top_pct":     self.close_top_pct,
            "adx_max_before":    self.adx_max_before,
            "rsi_min":           self.rsi_min,
            "rsi_max":           self.rsi_max,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        """
        Scan the current bar (df.iloc[-1]) for a momentum breakout signal.
        Works on any timeframe — lookback_periods and HTF are interval-agnostic.
        """
        n = self.lookback_periods
        if len(df) < n + 2:
            return None

        bar      = df.iloc[-1]
        prev_bar = df.iloc[-2]
        hist     = df.iloc[-(n + 1):-1]   # N bars BEFORE current bar (zero-lookahead)

        close    = float(bar["Close"])
        high     = float(bar["High"])
        low      = float(bar["Low"])
        volume   = float(bar["Volume"])
        bar_size = high - low

        # ── Condition 1: close above N-day high ───────────────────────────────
        n_day_high = float(hist["High"].max())
        n_day_low  = float(hist["Low"].min())
        if close <= n_day_high:
            return None

        # ── Condition 2: volume >= multiplier × N-day avg volume ──────────────
        avg_volume = float(hist["Volume"].mean())
        if avg_volume > 0 and volume < self.volume_multiplier * avg_volume:
            return None

        # ── Condition 3: strong close — in top X% of bar range ────────────────
        if bar_size > 0:
            close_position = (close - low) / bar_size   # 1.0 = close at high
            if close_position < (1.0 - self.close_top_pct):
                return None

        # ── Condition 4: ADX on PREVIOUS bar < adx_max_before ────────────────
        #    (confirms price was in consolidation, not already trending)
        #    Use the full df for ADX so the indicator has enough warm-up bars.
        adx_series = ta.trend.ADXIndicator(
            df["High"], df["Low"], df["Close"], window=14,
        ).adx()
        if len(adx_series) >= 2:
            prev_adx = adx_series.iloc[-2]
            if pd.notna(prev_adx) and float(prev_adx) >= self.adx_max_before:
                return None

        # ── Condition 5: RSI in momentum zone ────────────────────────────────
        rsi_series = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        rsi = float(rsi_series.iloc[-1])
        if not (self.rsi_min <= rsi <= self.rsi_max):
            return None

        # ── Condition 6: HTF EMA20 > EMA50 (derived from entry interval) ────────
        interval = tech.get("interval", "1d")
        htf_rule = _HTF_RESAMPLE.get(interval, "1W")
        try:
            htf = df["Close"].resample(htf_rule).last().dropna()
            if len(htf) >= 50:
                ema20_h = float(ta.trend.EMAIndicator(htf, 20).ema_indicator().iloc[-1])
                ema50_h = float(ta.trend.EMAIndicator(htf, 50).ema_indicator().iloc[-1])
                if ema20_h <= ema50_h:
                    return None
        except Exception:
            pass   # if resampling fails (e.g. sub-daily without enough history), skip HTF filter

        # ── All conditions passed — build setup ───────────────────────────────
        consolidation_range = n_day_high - n_day_low
        sl   = n_day_low
        tp1  = close + consolidation_range
        tp2  = close + 2.0 * consolidation_range
        risk = close - sl

        if risk <= 0:
            return None

        rr = (abs(close - tp1) * (self.tp1_alloc / 100)
              + abs(close - tp2) * (self.tp2_alloc / 100)) / risk
        vol_ratio = volume / avg_volume if avg_volume > 0 else 0.0

        rationale = (
            f"Momentum breakout — close {close:.2f} above {n}-period high {n_day_high:.2f} | "
            f"Vol {vol_ratio:.1f}× avg | RSI {rsi:.0f} | "
            f"Consolidation range {n_day_low:.2f}–{n_day_high:.2f} "
            f"({consolidation_range:.2f} pts) | "
            f"Measured move target {tp1:.2f}"
        )

        ticker = tech.get("ticker", "")

        setup = TradingSetup(
            name="MOM-A",
            label=f"Momentum Breakout LONG",
            direction="long",
            trade_type="swing",
            status="ACTIVE",
            priority="primary",
            rationale=rationale,
            trigger=f"Close above {n}-day high {n_day_high:.2f} on volume > {self.volume_multiplier}× avg",
            entry_low=close,
            entry_high=close,
            stop_loss=sl,
            trailing_sl_to_breakeven=close + 0.5 * consolidation_range,
            targets=[
                SetupTarget(price=tp1, label="TP1 (measured move)", allocation_pct=self.tp1_alloc),
                SetupTarget(price=tp2, label="TP2 (2× range)",      allocation_pct=self.tp2_alloc),
            ],
            rr_ratio=round(rr, 2),
            win_rate_estimate=0.40,
            trade_duration="1-4 weeks",
            ev=round((rr * 0.40 - 1.0 * 0.60) * risk, 2),
            profit_factor=round(rr * 0.40 / (1.0 * 0.60), 2) if rr > 0 else 0.0,
            confidence=min(0.90, 0.50 + 0.08 * (vol_ratio - 1.0)),
            confidence_note=(
                f"Vol {vol_ratio:.1f}× avg | RSI {rsi:.0f} | "
                f"ADX was below {self.adx_max_before} (consolidation confirmed)"
            ),
        )

        return SetupsOutput(
            asset=ticker,
            current_price=close,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Daily close back below {n_day_high:.2f} (breakout level)",
                description="Failed breakout — price returned into the consolidation range",
                price_trigger=n_day_high,
                action="Exit immediately; breakout has failed",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Close above {n_day_high:.2f} on volume > {self.volume_multiplier}× avg",
                    outcome=f"Enter LONG at {close:.2f}, SL {sl:.2f}, TP1 {tp1:.2f}",
                    direction="long",
                    setup_name="MOM-A",
                    entry_price=close,
                ),
            ],
            position_sizing_note=(
                f"Risk 1R per trade. SL at consolidation low {sl:.2f} "
                f"({risk / close:.2%} of price). "
                f"Measured move TP1 = {tp1:.2f} ({tp1 / close - 1:.2%} upside)."
            ),
        )
