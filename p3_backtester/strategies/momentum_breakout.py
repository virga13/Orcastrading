"""
p3_backtester/strategies/momentum_breakout.py — Momentum Breakout / Breakdown strategy.

LONG: fires when price closes above N-period consolidation high on volume expansion.
SHORT: fires when price closes below N-period consolidation low on volume expansion.

Entry conditions (long / mirrored for short):
  1. Close above N-day high / below N-day low  — consolidation breakout/breakdown
  2. Volume >= volume_multiplier × N-day average volume
  3. Bar closes in top (long) / bottom (short) close_top_pct of range — conviction close
  4. ADX(14) was < adx_max_before on the PREVIOUS bar — was consolidating, not trending
  5. RSI(14) in [rsi_min, rsi_max] for longs / [rsi_min_short, rsi_max_short] for shorts
  6. HTF EMA20 > EMA50 for longs / EMA20 < EMA50 for shorts

Stop-loss (long): below N-day low   |   (short): above N-day high
TP1: entry ± 1 × consolidation range (measured move)
TP2: entry ± 2 × consolidation range
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
    description = "N-period high/low breakout with volume expansion and HTF bias — any timeframe"

    def __init__(
        self,
        lookback_periods: int     = 20,   # was lookback_days — now interval-agnostic
        volume_multiplier: float  = 1.5,
        close_top_pct: float      = 0.25,
        adx_max_before: int       = 25,
        rsi_min: int              = 50,
        rsi_max: int              = 70,
        rsi_min_short: int        = 30,   # RSI floor for shorts (not oversold)
        rsi_max_short: int        = 50,   # RSI ceiling for shorts (momentum)
        tp1_alloc: int            = 70,
        tp2_alloc: int            = 30,
        shorts_enabled: bool      = True,
        # Legacy compat
        lookback_days: int        = 0,
    ):
        self.lookback_periods   = lookback_periods if not lookback_days else lookback_days
        self.volume_multiplier  = volume_multiplier
        self.close_top_pct      = close_top_pct
        self.adx_max_before     = adx_max_before
        self.rsi_min            = rsi_min
        self.rsi_max            = rsi_max
        self.rsi_min_short      = rsi_min_short
        self.rsi_max_short      = rsi_max_short
        self.tp1_alloc          = tp1_alloc
        self.tp2_alloc          = tp2_alloc
        self.shorts_enabled     = shorts_enabled

    @property
    def params(self) -> dict:
        return {
            "lookback_periods":  self.lookback_periods,
            "volume_multiplier": self.volume_multiplier,
            "close_top_pct":     self.close_top_pct,
            "adx_max_before":    self.adx_max_before,
            "rsi_min":           self.rsi_min,
            "rsi_max":           self.rsi_max,
            "shorts_enabled":    self.shorts_enabled,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        """
        Scan the current bar for a momentum breakout (long) or breakdown (short).
        Returns the first direction whose conditions all pass; long is checked first.
        """
        n = self.lookback_periods
        if len(df) < n + 2:
            return None

        bar  = df.iloc[-1]
        hist = df.iloc[-(n + 1):-1]   # N bars BEFORE current bar (zero-lookahead)

        close    = float(bar["Close"])
        high     = float(bar["High"])
        low      = float(bar["Low"])
        volume   = float(bar["Volume"])
        bar_size = high - low

        n_day_high = float(hist["High"].max())
        n_day_low  = float(hist["Low"].min())
        avg_volume = float(hist["Volume"].mean())

        # ── Shared indicators (computed once) ─────────────────────────────────
        adx_series = ta.trend.ADXIndicator(
            df["High"], df["Low"], df["Close"], window=14,
        ).adx()
        prev_adx = float(adx_series.iloc[-2]) if len(adx_series) >= 2 else float("nan")

        rsi_series = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        rsi = float(rsi_series.iloc[-1])

        interval = tech.get("interval", "1d")
        htf_rule = _HTF_RESAMPLE.get(interval, "1W")
        htf_ema20 = htf_ema50 = None
        try:
            htf = df["Close"].resample(htf_rule).last().dropna()
            if len(htf) >= 50:
                htf_ema20 = float(ta.trend.EMAIndicator(htf, 20).ema_indicator().iloc[-1])
                htf_ema50 = float(ta.trend.EMAIndicator(htf, 50).ema_indicator().iloc[-1])
        except Exception:
            pass

        vol_ratio = volume / avg_volume if avg_volume > 0 else 0.0
        ticker    = tech.get("ticker", "")

        # ── Try LONG (breakout above N-day high) ──────────────────────────────
        long_ok = (
            close > n_day_high                                          # 1. breakout
            and (avg_volume <= 0 or volume >= self.volume_multiplier * avg_volume)  # 2. volume
            and (bar_size <= 0 or (close - low) / bar_size >= (1.0 - self.close_top_pct))  # 3. top close
            and (pd.isna(prev_adx) or prev_adx < self.adx_max_before)  # 4. was consolidating
            and self.rsi_min <= rsi <= self.rsi_max                     # 5. momentum RSI
            and (htf_ema20 is None or htf_ema20 > htf_ema50)           # 6. HTF bullish
        )

        # ── Try SHORT (breakdown below N-day low) ─────────────────────────────
        short_ok = (
            self.shorts_enabled
            and close < n_day_low                                       # 1. breakdown
            and (avg_volume <= 0 or volume >= self.volume_multiplier * avg_volume)  # 2. volume
            and (bar_size <= 0 or (high - close) / bar_size >= (1.0 - self.close_top_pct))  # 3. bottom close
            and (pd.isna(prev_adx) or prev_adx < self.adx_max_before)  # 4. was consolidating
            and self.rsi_min_short <= rsi <= self.rsi_max_short         # 5. bearish RSI zone
            and (htf_ema20 is None or htf_ema20 < htf_ema50)           # 6. HTF bearish
        )

        if not long_ok and not short_ok:
            return None

        direction = "long" if long_ok else "short"
        consolidation_range = n_day_high - n_day_low

        if direction == "long":
            sl   = n_day_low
            tp1  = close + consolidation_range
            tp2  = close + 2.0 * consolidation_range
            risk = close - sl
            trail_level = close + 0.5 * consolidation_range
            inv_level   = n_day_high
            inv_desc    = "Failed breakout — price returned into the consolidation range"
            inv_action  = "Exit immediately; breakout has failed"
            trigger_txt = f"Close above {n}-period high {n_day_high:.2f} on volume > {self.volume_multiplier}× avg"
            rationale   = (
                f"Momentum breakout — close {close:.2f} above {n}-period high {n_day_high:.2f} | "
                f"Vol {vol_ratio:.1f}× avg | RSI {rsi:.0f} | "
                f"Consolidation range {n_day_low:.2f}–{n_day_high:.2f} ({consolidation_range:.2f} pts)"
            )
            sizing_note = (
                f"Risk 1R per trade. SL at consolidation low {sl:.2f} "
                f"({risk / close:.2%} of price). Measured move TP1 = {tp1:.2f}."
            )
        else:
            sl   = n_day_high
            tp1  = close - consolidation_range
            tp2  = close - 2.0 * consolidation_range
            risk = sl - close
            trail_level = close - 0.5 * consolidation_range
            inv_level   = n_day_low
            inv_desc    = "Failed breakdown — price reclaimed the consolidation range"
            inv_action  = "Exit immediately; breakdown has failed"
            trigger_txt = f"Close below {n}-period low {n_day_low:.2f} on volume > {self.volume_multiplier}× avg"
            rationale   = (
                f"Momentum breakdown — close {close:.2f} below {n}-period low {n_day_low:.2f} | "
                f"Vol {vol_ratio:.1f}× avg | RSI {rsi:.0f} | "
                f"Consolidation range {n_day_low:.2f}–{n_day_high:.2f} ({consolidation_range:.2f} pts)"
            )
            sizing_note = (
                f"Risk 1R per trade. SL at consolidation high {sl:.2f} "
                f"({risk / close:.2%} of price). Measured move TP1 = {tp1:.2f}."
            )

        if risk <= 0:
            return None
        if risk / close > 0.20:
            return None

        rr = (abs(close - tp1) * (self.tp1_alloc / 100)
              + abs(close - tp2) * (self.tp2_alloc / 100)) / risk
        if rr < 0.8:
            return None

        setup = TradingSetup(
            name="MOM-A",
            label=f"Momentum {'Breakout LONG' if direction == 'long' else 'Breakdown SHORT'}",
            direction=direction,
            trade_type="swing",
            status="ACTIVE",
            priority="primary",
            rationale=rationale,
            trigger=trigger_txt,
            entry_low=close,
            entry_high=close,
            stop_loss=sl,
            trailing_sl_to_breakeven=trail_level,
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
                condition=f"Daily close {'below' if direction=='long' else 'above'} {inv_level:.2f}",
                description=inv_desc,
                price_trigger=inv_level,
                action=inv_action,
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=trigger_txt,
                    outcome=f"Enter {direction.upper()} at {close:.2f}, SL {sl:.2f}, TP1 {tp1:.2f}",
                    direction=direction,
                    setup_name="MOM-A",
                    entry_price=close,
                ),
            ],
            position_sizing_note=sizing_note,
        )
