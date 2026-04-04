"""
p3_backtester/strategies/orb.py — Opening Range Breakout strategy.

Timeframe-agnostic: works on any intraday interval (1m, 5m, 15m, 30m, 1h).
Opening range duration is defined in minutes (opening_range_minutes=30),
and the bar count is derived at runtime from the interval in the tech dict.

Opening range: first bars covering opening_range_minutes of the session.
  e.g. opening_range_minutes=30 → 2 bars on 15m, 6 bars on 5m, 30 bars on 1m

Signal fires when a bar CLOSES outside the range with volume and VWAP confirmation.

Conditions (long):
  1. Bar closes above opening range high
  2. Bar volume >= volume_multiplier × avg opening-range bar volume
  3. Bar close > session VWAP  (require_vwap_filter=True)
  4. Bar body >= body_pct_min × bar total size  (conviction close, not a wick)
  5. Current bar is within trade_window_minutes after opening range ends
  6. Opening range size is within [range_min_atr_factor, range_max_atr_factor] × daily ATR
  7. RSI < rsi_max_long  (not overbought)
  8. Daily EMA20 > EMA50  (daily_trend_filter=True)  → only take aligned long

Conditions (short): mirror image, with VWAP and trend filter inverted.

Stop-loss: below opening range low  (for longs)
TP1: entry + 1 × range height
TP2: entry + 2 × range height
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
import ta

from p3_backtester.strategies.base import StrategyBase
from p1_analysis_engine.schema import (
    SetupsOutput, TradingSetup, SetupTarget, InvalidationScenario, DecisionTreeEntry,
)


# Map interval string → bar duration in minutes
_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "2m": 2, "5m": 5, "10m": 10, "15m": 15,
    "30m": 30, "1h": 60, "2h": 120, "4h": 240,
}


def _bars_for_minutes(interval: str, minutes: int) -> int:
    """Return how many bars of `interval` cover `minutes` of trading time."""
    bar_mins = _INTERVAL_MINUTES.get(interval, 15)
    return max(1, round(minutes / bar_mins))


class ORBStrategy(StrategyBase):
    name = "orb"
    description = "Opening Range Breakout — timeframe-agnostic, duration defined in minutes"

    def __init__(
        self,
        opening_range_minutes: int  = 30,   # replaces opening_range_bars — time-based
        trade_window_minutes: int   = 90,   # replaces trade_window_bars — time-based
        volume_multiplier: float    = 1.5,
        range_min_atr_factor: float = 0.3,
        range_max_atr_factor: float = 2.5,
        body_pct_min: float         = 0.5,
        rsi_max_long: int           = 70,
        rsi_min_short: int          = 30,
        tp1_r: float                = 1.0,
        tp2_r: float                = 2.0,
        tp1_alloc: int              = 70,
        tp2_alloc: int              = 30,
        sl_buffer_factor: float     = 0.1,
        long_enabled: bool          = True,
        short_enabled: bool         = True,
        require_vwap_filter: bool   = True,
        daily_trend_filter: bool    = True,
        max_gap_pct: float          = 0.01,
        # Legacy compat — if set, override the minute-based defaults
        opening_range_bars: int     = 0,
        trade_window_bars: int      = 0,
    ):
        self.opening_range_minutes = opening_range_minutes
        self.trade_window_minutes  = trade_window_minutes
        self.volume_multiplier     = volume_multiplier
        self.range_min_atr_factor  = range_min_atr_factor
        self.range_max_atr_factor  = range_max_atr_factor
        self.body_pct_min          = body_pct_min
        self.rsi_max_long          = rsi_max_long
        self.rsi_min_short         = rsi_min_short
        self.tp1_r                 = tp1_r
        self.tp2_r                 = tp2_r
        self.tp1_alloc             = tp1_alloc
        self.tp2_alloc             = tp2_alloc
        self.sl_buffer_factor      = sl_buffer_factor
        self.long_enabled          = long_enabled
        self.short_enabled         = short_enabled
        self.require_vwap_filter   = require_vwap_filter
        self.daily_trend_filter    = daily_trend_filter
        self.max_gap_pct           = max_gap_pct
        # Legacy bar-count overrides (takes priority if non-zero)
        self._legacy_or_bars       = opening_range_bars
        self._legacy_tw_bars       = trade_window_bars

    def _resolve_bars(self, interval: str) -> tuple[int, int]:
        """Return (opening_range_bars, trade_window_bars) for the given interval."""
        if self._legacy_or_bars > 0:
            or_bars = self._legacy_or_bars
        else:
            or_bars = _bars_for_minutes(interval, self.opening_range_minutes)
        if self._legacy_tw_bars > 0:
            tw_bars = self._legacy_tw_bars
        else:
            tw_bars = _bars_for_minutes(interval, self.trade_window_minutes)
        return or_bars, tw_bars

    @property
    def params(self) -> dict:
        return {
            "opening_range_minutes": self.opening_range_minutes,
            "trade_window_minutes":  self.trade_window_minutes,
            "volume_multiplier":     self.volume_multiplier,
            "tp1_r":                 self.tp1_r,
            "tp2_r":                 self.tp2_r,
            "require_vwap_filter":   self.require_vwap_filter,
            "daily_trend_filter":    self.daily_trend_filter,
        }

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        """
        Scan the most recent session in `df` for an ORB signal.

        Parameters
        ----------
        tech : dict
            Must include 'ticker', 'current_price', 'atr_14', 'rsi_14'.
            Optionally 'ema20', 'ema50' for daily_trend_filter.
        df   : pd.DataFrame
            15-min OHLCV bars (tz-naive, exchange-local time).
            Should include at least today's session bars + enough history
            for daily ATR (opening_range_bars + trade_window_bars is min).

        Returns
        -------
        SetupsOutput | None
        """
        interval = tech.get("interval", "15m")
        or_bars, tw_bars = self._resolve_bars(interval)

        if len(df) < or_bars + 1:
            return None

        # ── Isolate today's session ───────────────────────────────────────────
        last_date  = df.index[-1].normalize()
        today_mask = df.index.normalize() == last_date
        today      = df[today_mask].copy()

        if len(today) < or_bars + 1:
            return None

        # ── Gap filter ────────────────────────────────────────────────────────
        # Skip when the overnight gap distorts the opening range.
        # A gap > max_gap_pct means price opened far from where it closed —
        # the opening range forms at an artificial level, not a meaningful equilibrium.
        if self.max_gap_pct > 0:
            yesterday_bars = df[df.index.normalize() < last_date]
            if len(yesterday_bars) > 0:
                yesterday_close = float(yesterday_bars["Close"].iloc[-1])
                today_open      = float(today["Open"].iloc[0])
                if yesterday_close > 0:
                    gap_pct = abs(today_open - yesterday_close) / yesterday_close
                    if gap_pct > self.max_gap_pct:
                        return None   # overnight gap — skip this session

        # ── Opening range ─────────────────────────────────────────────────────
        or_bars_df = today.iloc[:or_bars]
        or_high    = float(or_bars_df["High"].max())
        or_low     = float(or_bars_df["Low"].min())
        or_range   = or_high - or_low

        if or_range <= 0:
            return None

        # Range size filter vs daily ATR
        daily_atr = float(tech.get("atr_14", or_range))
        if daily_atr > 0:
            if or_range < self.range_min_atr_factor * daily_atr:
                return None   # range too tight — false-break territory
            if or_range > self.range_max_atr_factor * daily_atr:
                return None   # range too wide — poor R/R

        avg_or_volume = float(or_bars_df["Volume"].mean()) if or_bars_df["Volume"].mean() > 0 else 1.0

        # ── Session VWAP ──────────────────────────────────────────────────────
        today = _compute_vwap(today)

        # ── Daily trend filter ────────────────────────────────────────────────
        daily_ema20 = tech.get("ema20")
        daily_ema50 = tech.get("ema50")
        if self.daily_trend_filter and daily_ema20 and daily_ema50:
            daily_bullish = daily_ema20 > daily_ema50
        else:
            daily_bullish = True   # no filter if EMAs not in tech dict

        rsi = float(tech.get("rsi_14", 50))

        # ── Scan trade window bars ────────────────────────────────────────────
        trade_bars = today.iloc[or_bars: or_bars + tw_bars]

        for _, bar in trade_bars.iterrows():
            bar_close  = float(bar["Close"])
            bar_open   = float(bar["Open"])
            bar_high   = float(bar["High"])
            bar_low    = float(bar["Low"])
            bar_volume = float(bar["Volume"])
            bar_vwap   = float(bar.get("vwap", bar_close))
            bar_size   = bar_high - bar_low
            body       = abs(bar_close - bar_open)
            body_pct   = body / bar_size if bar_size > 0 else 0.0

            # ─ LONG ──────────────────────────────────────────────────────────
            if self.long_enabled and daily_bullish:
                if (
                    bar_close > or_high
                    and bar_volume >= self.volume_multiplier * avg_or_volume
                    and body_pct >= self.body_pct_min
                    and rsi <= self.rsi_max_long
                    and (not self.require_vwap_filter or bar_close > bar_vwap)
                ):
                    return self._build_setup(
                        direction="long",
                        entry=bar_close,
                        or_high=or_high,
                        or_low=or_low,
                        or_range=or_range,
                        vwap=bar_vwap,
                        bar_volume=bar_volume,
                        avg_or_volume=avg_or_volume,
                        rsi=rsi,
                        tech=tech,
                    )

            # ─ SHORT ─────────────────────────────────────────────────────────
            if self.short_enabled and not daily_bullish:
                if (
                    bar_close < or_low
                    and bar_volume >= self.volume_multiplier * avg_or_volume
                    and body_pct >= self.body_pct_min
                    and rsi >= self.rsi_min_short
                    and (not self.require_vwap_filter or bar_close < bar_vwap)
                ):
                    return self._build_setup(
                        direction="short",
                        entry=bar_close,
                        or_high=or_high,
                        or_low=or_low,
                        or_range=or_range,
                        vwap=bar_vwap,
                        bar_volume=bar_volume,
                        avg_or_volume=avg_or_volume,
                        rsi=rsi,
                        tech=tech,
                    )

        return None   # no signal fired during trade window

    # ── Setup builder ─────────────────────────────────────────────────────────

    def _build_setup(
        self,
        direction: str,
        entry: float,
        or_high: float,
        or_low: float,
        or_range: float,
        vwap: float,
        bar_volume: float,
        avg_or_volume: float,
        rsi: float,
        tech: dict,
    ) -> SetupsOutput:
        ticker = tech.get("ticker", "")

        if direction == "long":
            sl    = or_low - self.sl_buffer_factor * or_range
            tp1   = entry + self.tp1_r * or_range
            tp2   = entry + self.tp2_r * or_range
            be    = entry + 0.5 * or_range   # breakeven SL trigger
        else:
            sl    = or_high + self.sl_buffer_factor * or_range
            tp1   = entry - self.tp1_r * or_range
            tp2   = entry - self.tp2_r * or_range
            be    = entry - 0.5 * or_range

        risk   = abs(entry - sl)
        rr     = (abs(entry - tp1) * (self.tp1_alloc / 100)
                  + abs(entry - tp2) * (self.tp2_alloc / 100)) / risk if risk > 0 else 0.0
        vol_ratio = bar_volume / avg_or_volume if avg_or_volume > 0 else 0.0

        rationale = (
            f"ORB {direction.upper()} — range {or_low:.2f}–{or_high:.2f} "
            f"(size {or_range:.2f}) | "
            f"Breakout at {entry:.2f} (close above {'range high' if direction == 'long' else 'range low'}) | "
            f"Vol {vol_ratio:.1f}× avg | VWAP {vwap:.2f} | RSI {rsi:.0f}"
        )

        setup = TradingSetup(
            name="ORB-A",
            label=f"ORB {direction.upper()}",
            direction=direction,
            trade_type="intraday",
            status="ACTIVE",
            priority="primary",
            rationale=rationale,
            trigger=f"Close {'above' if direction == 'long' else 'below'} opening range "
                    f"({'high' if direction == 'long' else 'low'}) {or_high:.2f if direction == 'long' else or_low:.2f}",
            entry_low=entry,
            entry_high=entry,
            stop_loss=sl,
            trailing_sl_to_breakeven=be,
            targets=[
                SetupTarget(price=tp1, label="TP1", allocation_pct=self.tp1_alloc),
                SetupTarget(price=tp2, label="TP2", allocation_pct=self.tp2_alloc),
            ],
            rr_ratio=round(rr, 2),
            win_rate_estimate=0.45,
            trade_duration="intraday",
            ev=round((rr * 0.45 - 1.0 * 0.55) * risk, 2),
            profit_factor=round(rr * 0.45 / (1.0 * 0.55), 2) if rr > 0 else 0.0,
            confidence=min(0.95, 0.5 + 0.1 * (vol_ratio - 1.0)),
            confidence_note=f"Vol {vol_ratio:.1f}× OR avg | RSI {rsi:.0f} | VWAP aligned",
        )

        return SetupsOutput(
            asset=ticker,
            current_price=entry,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"{'Low' if direction == 'long' else 'High'} closes back inside opening range",
                description="Breakout failure — price returned into the range",
                price_trigger=or_high if direction == "long" else or_low,
                action="Exit immediately, wait for next session",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Price breaks {'above' if direction == 'long' else 'below'} range on volume",
                    outcome=f"Enter {direction.upper()} at market, SL {sl:.2f}",
                    direction=direction,
                    setup_name="ORB-A",
                    entry_price=entry,
                ),
            ],
            position_sizing_note=f"Risk 1R per trade. Range height = {or_range:.2f}. "
                                  f"SL = {sl:.2f} ({abs(entry - sl) / tech.get('current_price', entry):.2%} of price).",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'vwap' column to df, computed cumulatively from the first bar."""
    df = df.copy()
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"].replace(0, 1)   # avoid div-by-zero on zero-volume bars
    cum_vp  = (typical * vol).cumsum()
    cum_vol = vol.cumsum()
    df["vwap"] = cum_vp / cum_vol
    return df
