"""
strategies/gold_scalper.py — Gold Scalper intraday strategy.

Converted from BRK Gold Scalper v2 (Pine Script v6).
Designed for M5 entry bars with M15 signal layer and M30 macro filter.

Composite score system (0–10, threshold default 6):
  [2]  M30 macro trend  — close > EMA50 > EMA200 and RSI > 50
  [2]  M15 EMA fresh cross, or [1] M15 EMA trend alignment
  [1]  RSI-M15 in momentum zone (45–68 long / 32–55 short)
  [1]  M5 EMA8 > EMA21 (entry-level structure)
  [1]  Candle pattern on M5 (pin / engulf / marubozu / 3-bar continuation)
  [1]  ADX-M15 >= threshold and DI directional bias matches
  [1]  VWAP bias — daily anchor alignment
  [1]  Volatility regime — ATR ratio 0.7–2.0× vs 50-period ATR MA

SL:  sl_mult  × ATR from entry (ATR-scaled, adapts to volatility)
TP1: tp1_mult × ATR, tp1_alloc% of position → move SL to breakeven
TP2: tp2_mult × ATR, tp2_alloc% of position (free ride after TP1)

⚠️  Set scanner timeframe to 5m for correct MTF behaviour.
⚠️  Intended for XAU/USD (Gold). Verify ATR scale with your broker.
"""
from __future__ import annotations

import pandas as pd
import ta

from p3_backtester.strategies.base import StrategyBase
from p1_analysis_engine.schema import (
    TradingSetup, SetupTarget, InvalidationScenario,
    DecisionTreeEntry, SetupsOutput,
)

_M15 = "15min"
_M30 = "30min"


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return ta.trend.EMAIndicator(series, window=period).ema_indicator()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    return ta.momentum.RSIIndicator(series, window=period).rsi()


def _adx_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> tuple[float, float, float]:
    """Returns (adx, di_plus, di_minus). Returns (0, 0, 0) on failure."""
    try:
        ind = ta.trend.ADXIndicator(high, low, close, window=period)
        return (
            float(ind.adx().iloc[-1]),
            float(ind.adx_pos().iloc[-1]),
            float(ind.adx_neg().iloc[-1]),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()


def _vwap_daily(df: pd.DataFrame) -> float:
    """Daily-anchored VWAP from intraday df. Falls back to close when no volume."""
    if "Volume" not in df.columns or df["Volume"].sum() < 1:
        return float(df["Close"].iloc[-1])
    try:
        hlc3      = (df["High"] + df["Low"] + df["Close"]) / 3
        today     = df.index[-1].date()
        mask      = pd.Series(df.index.date, index=df.index) == today
        vol_today = df["Volume"][mask]
        if vol_today.sum() < 1:
            return float(df["Close"].iloc[-1])
        return float((hlc3[mask] * vol_today).sum() / vol_today.sum())
    except Exception:
        return float(df["Close"].iloc[-1])


def _vwap_daily_series(df: pd.DataFrame) -> pd.Series:
    """Daily-anchored VWAP as a full Series aligned to df.index (for pre-computation)."""
    try:
        hlc3  = (df["High"] + df["Low"] + df["Close"]) / 3
        vol   = df["Volume"] if ("Volume" in df.columns and df["Volume"].sum() > 0) \
                else pd.Series(1.0, index=df.index)
        tpv   = hlc3 * vol
        dates = pd.Series(df.index.date, index=df.index)
        result = pd.Series(index=df.index, dtype=float)
        for day, grp_idx in dates.groupby(dates).groups.items():
            cv = vol.loc[grp_idx].cumsum()
            ct = tpv.loc[grp_idx].cumsum()
            result.loc[grp_idx] = ct / cv.where(cv > 0, other=1.0)
        return result.fillna(df["Close"])
    except Exception:
        return df["Close"].copy()


# ── Candle pattern detection ──────────────────────────────────────────────────

def _candle_pattern(df: pd.DataFrame, direction: str) -> bool:
    """
    Detect bullish/bearish candle patterns on the last 4 bars.
    Patterns: hammer/pin bar, engulfing, marubozu, 3-bar continuation.
    """
    if len(df) < 4:
        return False

    o1, c1 = float(df["Open"].iloc[-1]), float(df["Close"].iloc[-1])
    h1, l1 = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1])
    o2, c2 = float(df["Open"].iloc[-2]), float(df["Close"].iloc[-2])
    o3, c3 = float(df["Open"].iloc[-3]), float(df["Close"].iloc[-3])
    o4, c4 = float(df["Open"].iloc[-4]), float(df["Close"].iloc[-4])

    body1  = abs(c1 - o1)
    range1 = h1 - l1
    if range1 < 1e-8:
        return False

    lower1 = min(o1, c1) - l1
    upper1 = h1 - max(o1, c1)

    if direction == "long":
        pin    = lower1 > range1 * 0.60 and body1 < range1 * 0.35 and c1 > o1
        engulf = c1 > o1 and c2 < o2 and c1 > o2 and o1 < c2
        marub  = c1 > o1 and body1 > range1 * 0.80 and lower1 < range1 * 0.10 and upper1 < range1 * 0.10
        cont3  = c1 > o1 and c2 > o2 and c3 > o3 and c1 > c2 and c2 > c3
    else:
        pin    = upper1 > range1 * 0.60 and body1 < range1 * 0.35 and c1 < o1
        engulf = c1 < o1 and c2 > o2 and c1 < o2 and o1 > c2
        marub  = c1 < o1 and body1 > range1 * 0.80 and upper1 < range1 * 0.10 and lower1 < range1 * 0.10
        cont3  = c1 < o1 and c2 < o2 and c3 < o3 and c1 < c2 and c2 < c3

    return pin or engulf or marub or cont3


# ── Resample helpers ──────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, rule: str, min_bars: int) -> pd.DataFrame | None:
    try:
        htf = df.resample(rule).agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna(subset=["Close"])
        return htf if len(htf) >= min_bars else None
    except Exception:
        return None


# ── Strategy ──────────────────────────────────────────────────────────────────

class GoldScalperStrategy(StrategyBase):
    """
    Gold Scalper — BRK Gold Scalper v2 ported to Python/Orcastrading.

    Composite 0–10 score with a configurable minimum threshold (default 6).
    Multi-timeframe: M5 entry → M15 signal → M30 macro.
    ATR-scaled SL and TP with breakeven move after TP1.
    """

    name            = "Gold Scalper"
    description     = (
        "BRK Gold Scalper v2: M30 macro + M15 ADX/EMA signal + M5 entry. "
        "Composite 6/10 score gate. ATR-scaled SL/TP. XAU/USD optimised."
    )
    uses_bar_slicer = False  # computes all MTF indicators internally; skips recompute_technical()

    def __init__(
        self,
        # Score gate
        min_score:          int   = 6,
        # M5 EMA periods
        ema_f5:             int   = 8,
        ema_s5:             int   = 21,
        # M15 EMA periods
        ema_f15:            int   = 21,
        ema_s15:            int   = 50,
        # M30 EMA periods (macro)
        ema_f30:            int   = 50,
        ema_s30:            int   = 200,
        # ADX (evaluated on M15)
        adx_period:         int   = 14,
        adx_min:            int   = 22,
        # RSI
        rsi_period:         int   = 14,
        rsi_long_min:       int   = 45,
        rsi_long_max:       int   = 68,
        rsi_short_min:      int   = 32,
        rsi_short_max:      int   = 55,
        # Volatility regime
        vol_low:            float = 0.7,
        vol_high:           float = 2.0,
        atr_ma_period:      int   = 50,
        # SL / TP (× ATR)
        sl_mult:            float = 1.5,
        tp1_mult:           float = 1.0,
        tp2_mult:           float = 2.5,
        tp1_alloc:          int   = 50,
        tp2_alloc:          int   = 50,
        # Session / daily filters (v2.4)
        sess_start_utc:     int   = 8,
        sess_end_utc:       int   = 20,
        avoid_friday:       bool  = True,
        max_trades_per_day: int   = 3,
        # Filters
        use_vwap:           bool  = True,
        shorts_enabled:     bool  = True,
        # Historical win-rate estimate
        win_rate:           float = 0.50,
    ):
        self.min_score          = min_score
        self.ema_f5             = ema_f5
        self.ema_s5             = ema_s5
        self.ema_f15            = ema_f15
        self.ema_s15            = ema_s15
        self.ema_f30            = ema_f30
        self.ema_s30            = ema_s30
        self.adx_period         = adx_period
        self.adx_min            = adx_min
        self.rsi_period         = rsi_period
        self.rsi_long_min       = rsi_long_min
        self.rsi_long_max       = rsi_long_max
        self.rsi_short_min      = rsi_short_min
        self.rsi_short_max      = rsi_short_max
        self.vol_low            = vol_low
        self.vol_high           = vol_high
        self.atr_ma_period      = atr_ma_period
        self.sl_mult            = sl_mult
        self.tp1_mult           = tp1_mult
        self.tp2_mult           = tp2_mult
        self.tp1_alloc          = tp1_alloc
        self.tp2_alloc          = tp2_alloc
        self.sess_start_utc     = sess_start_utc
        self.sess_end_utc       = sess_end_utc
        self.avoid_friday       = avoid_friday
        self.max_trades_per_day = max_trades_per_day
        self.use_vwap           = use_vwap
        self.shorts_enabled     = shorts_enabled
        self.win_rate           = win_rate
        self._daily_counts: dict[str, int] = {}
        # Pre-computed indicator series (set by prime(df) for large datasets)
        self._m15_pre:    pd.DataFrame | None = None
        self._m30_pre:    pd.DataFrame | None = None
        self._m5_atr:     pd.Series | None    = None
        self._m5_atr_ma:  pd.Series | None    = None
        self._m5_ema_f:   pd.Series | None    = None
        self._m5_ema_s:   pd.Series | None    = None
        self._m5_vwap:    pd.Series | None    = None

    def prime(self, df: pd.DataFrame) -> None:
        """Pre-compute all indicator series from the full 5m df once.

        Eliminates O(n²) per-bar resampling/computation. After priming,
        generate_setups() does O(1) index lookups instead of full-series ops.
        """
        # M5 indicators
        atr_s          = _atr(df["High"], df["Low"], df["Close"])
        self._m5_atr   = atr_s
        self._m5_atr_ma = atr_s.rolling(self.atr_ma_period).mean()
        self._m5_ema_f = _ema(df["Close"], self.ema_f5)
        self._m5_ema_s = _ema(df["Close"], self.ema_s5)
        self._m5_vwap  = _vwap_daily_series(df) if self.use_vwap else df["Close"].copy()

        # M15 indicators
        m15 = _resample(df, _M15, min_bars=self.ema_s15 + 10)
        if m15 is not None:
            m15 = m15.copy()
            m15["_ema_f"] = _ema(m15["Close"], self.ema_f15)
            m15["_ema_s"] = _ema(m15["Close"], self.ema_s15)
            m15["_rsi"]   = _rsi(m15["Close"], self.rsi_period)
            try:
                adx_ind         = ta.trend.ADXIndicator(m15["High"], m15["Low"], m15["Close"], window=self.adx_period)
                m15["_adx"]     = adx_ind.adx()
                m15["_diplus"]  = adx_ind.adx_pos()
                m15["_diminus"] = adx_ind.adx_neg()
            except Exception:
                m15["_adx"] = m15["_diplus"] = m15["_diminus"] = 0.0
            self._m15_pre = m15

        # M30 indicators
        m30 = _resample(df, _M30, min_bars=self.ema_s30 + 10)
        if m30 is not None:
            m30 = m30.copy()
            m30["_ema_f"] = _ema(m30["Close"], self.ema_f30)
            m30["_ema_s"] = _ema(m30["Close"], self.ema_s30)
            m30["_rsi"]   = _rsi(m30["Close"], self.rsi_period)
            self._m30_pre = m30

    @property
    def params(self) -> dict:
        return {
            "min_score":          self.min_score,
            "adx_min":            self.adx_min,
            "sl_mult":            self.sl_mult,
            "tp1_mult":           self.tp1_mult,
            "tp2_mult":           self.tp2_mult,
            "sess_start_utc":     self.sess_start_utc,
            "sess_end_utc":       self.sess_end_utc,
            "avoid_friday":       self.avoid_friday,
            "max_trades_per_day": self.max_trades_per_day,
            "use_vwap":           self.use_vwap,
            "shorts_enabled":     self.shorts_enabled,
        }

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        price    = tech["current_price"]
        interval = tech.get("interval", "5m")

        # Need enough M5 bars to build M30 EMA200 (200 bars × 6 M5-per-M30 = 1200 M5 bars minimum)
        min_m5 = self.ema_s30 * 6 + self.atr_ma_period + 10
        if len(df) < min_m5:
            return None

        # ── Session / daily filters (v2.4) ────────────────────────────────────
        bar_ts = df.index[-1]
        try:
            bar_utc = bar_ts.tz_localize("UTC") if bar_ts.tzinfo is None else bar_ts.tz_convert("UTC")
        except Exception:
            bar_utc = bar_ts
        h_utc = bar_utc.hour

        if not (self.sess_start_utc <= h_utc < self.sess_end_utc):
            return None

        if self.avoid_friday and bar_utc.dayofweek == 4 and h_utc >= 18:
            return None

        date_key = bar_utc.strftime("%Y-%m-%d")
        if self._daily_counts.get(date_key, 0) >= self.max_trades_per_day:
            return None

        # ── M5: ATR, EMAs, VWAP ──────────────────────────────────────────────
        if self._m5_atr is not None:
            # Fast path: O(1) lookup into pre-computed series
            try:
                atr_v    = float(self._m5_atr.loc[bar_ts])
                atr_ma_v = float(self._m5_atr_ma.loc[bar_ts])
                ema_f5_v = float(self._m5_ema_f.loc[bar_ts])
                ema_s5_v = float(self._m5_ema_s.loc[bar_ts])
                vwap_v   = float(self._m5_vwap.loc[bar_ts])
            except KeyError:
                return None
        else:
            # Slow path: recompute from growing slice (no prime() call)
            atr_s    = _atr(df["High"], df["Low"], df["Close"])
            atr_ma_s = atr_s.rolling(self.atr_ma_period).mean()
            atr_v    = float(atr_s.iloc[-1])
            atr_ma_v = float(atr_ma_s.iloc[-1])
            ema_f5_s = _ema(df["Close"], self.ema_f5)
            ema_s5_s = _ema(df["Close"], self.ema_s5)
            ema_f5_v = float(ema_f5_s.iloc[-1])
            ema_s5_v = float(ema_s5_s.iloc[-1])
            vwap_v   = _vwap_daily(df) if self.use_vwap else price

        if atr_v < 1e-8 or atr_ma_v < 1e-8:
            return None
        if pd.isna(ema_f5_v) or pd.isna(ema_s5_v):
            return None

        vol_ratio = atr_v / atr_ma_v
        vol_ok    = self.vol_low <= vol_ratio <= self.vol_high

        # ── M15: EMA cross / trend, RSI, ADX ─────────────────────────────────
        if self._m15_pre is not None:
            # Fast path: slice pre-computed frame up to bar_ts.
            # Use bar_ts (same tz as source df) not bar_utc to avoid tz mismatch.
            m15 = self._m15_pre.loc[:bar_ts]
            if len(m15) < self.ema_s15 + 10:
                return None
            ema_f15_v = float(m15["_ema_f"].iloc[-1])
            ema_s15_v = float(m15["_ema_s"].iloc[-1])
            ema_f15_p = float(m15["_ema_f"].iloc[-2]) if len(m15) >= 2 else ema_f15_v
            ema_s15_p = float(m15["_ema_s"].iloc[-2]) if len(m15) >= 2 else ema_s15_v
            rsi_15_v  = float(m15["_rsi"].iloc[-1])
            adx_v     = float(m15["_adx"].iloc[-1])
            di_plus_v = float(m15["_diplus"].iloc[-1])
            di_minus_v= float(m15["_diminus"].iloc[-1])
        else:
            # Slow path: resample on-the-fly (57-day window, acceptable)
            m15 = _resample(df, _M15, min_bars=self.ema_s15 + 10)
            if m15 is None:
                return None
            ema_f15_s  = _ema(m15["Close"], self.ema_f15)
            ema_s15_s  = _ema(m15["Close"], self.ema_s15)
            rsi_15_s   = _rsi(m15["Close"], self.rsi_period)
            ema_f15_v  = float(ema_f15_s.iloc[-1])
            ema_s15_v  = float(ema_s15_s.iloc[-1])
            ema_f15_p  = float(ema_f15_s.iloc[-2]) if len(ema_f15_s) >= 2 else ema_f15_v
            ema_s15_p  = float(ema_s15_s.iloc[-2]) if len(ema_s15_s) >= 2 else ema_s15_v
            rsi_15_v   = float(rsi_15_s.iloc[-1])
            n15 = min(len(m15), 120)
            adx_v, di_plus_v, di_minus_v = _adx_di(
                m15["High"].iloc[-n15:], m15["Low"].iloc[-n15:],
                m15["Close"].iloc[-n15:], self.adx_period,
            )

        if any(pd.isna(v) for v in [ema_f15_v, ema_s15_v, rsi_15_v, adx_v]):
            return None

        # ── M30: macro trend ──────────────────────────────────────────────────
        if self._m30_pre is not None:
            m30 = self._m30_pre.loc[:bar_ts]
            if len(m30) < self.ema_s30 + 10:
                return None
            ema_f30_v = float(m30["_ema_f"].iloc[-1])
            ema_s30_v = float(m30["_ema_s"].iloc[-1])
            rsi_30_v  = float(m30["_rsi"].iloc[-1])
            m30_close = float(m30["Close"].iloc[-1])
        else:
            m30 = _resample(df, _M30, min_bars=self.ema_s30 + 10)
            if m30 is None:
                return None
            ema_f30_v = float(_ema(m30["Close"], self.ema_f30).iloc[-1])
            ema_s30_v = float(_ema(m30["Close"], self.ema_s30).iloc[-1])
            rsi_30_v  = float(_rsi(m30["Close"], self.rsi_period).iloc[-1])
            m30_close = float(m30["Close"].iloc[-1])

        if any(pd.isna(v) for v in [ema_f30_v, ema_s30_v, rsi_30_v]):
            return None

        # ── Score components ──────────────────────────────────────────────────
        bull_m30    = m30_close > ema_f30_v and ema_f30_v > ema_s30_v and rsi_30_v > 50
        bear_m30    = m30_close < ema_f30_v and ema_f30_v < ema_s30_v and rsi_30_v < 50

        cross_bull  = ema_f15_v > ema_s15_v and ema_f15_p <= ema_s15_p
        cross_bear  = ema_f15_v < ema_s15_v and ema_f15_p >= ema_s15_p
        trend_bull  = ema_f15_v > ema_s15_v
        trend_bear  = ema_f15_v < ema_s15_v

        adx_trending = adx_v >= self.adx_min
        adx_bull     = adx_trending and di_plus_v  > di_minus_v
        adx_bear     = adx_trending and di_minus_v > di_plus_v

        # ── Bull score ────────────────────────────────────────────────────────
        bull_score = 0
        bull_score += 2 if bull_m30 else 0
        bull_score += 2 if cross_bull else (1 if trend_bull else 0)
        bull_score += 1 if self.rsi_long_min  <= rsi_15_v <= self.rsi_long_max  else 0
        bull_score += 1 if ema_f5_v > ema_s5_v else 0
        bull_score += 1 if _candle_pattern(df, "long") else 0
        bull_score += 1 if adx_bull else 0
        bull_score += 1 if (not self.use_vwap or price > vwap_v) else 0
        bull_score += 1 if vol_ok else 0

        # ── Bear score ────────────────────────────────────────────────────────
        bear_score = 0
        bear_score += 2 if bear_m30 else 0
        bear_score += 2 if cross_bear else (1 if trend_bear else 0)
        bear_score += 1 if self.rsi_short_min <= rsi_15_v <= self.rsi_short_max else 0
        bear_score += 1 if ema_f5_v < ema_s5_v else 0
        bear_score += 1 if _candle_pattern(df, "short") else 0
        bear_score += 1 if adx_bear else 0
        bear_score += 1 if (not self.use_vwap or price < vwap_v) else 0
        bear_score += 1 if vol_ok else 0

        # ── Direction selection ───────────────────────────────────────────────
        if bull_score >= bear_score and bull_score >= self.min_score:
            direction, score = "long", bull_score
        elif self.shorts_enabled and bear_score >= self.min_score:
            direction, score = "short", bear_score
        else:
            return None

        # ── Entry / SL / TP ───────────────────────────────────────────────────
        if direction == "long":
            entry_low  = round(price - 0.1 * atr_v, 4)
            entry_high = round(price, 4)
            sl         = round(price - self.sl_mult  * atr_v, 4)
            tp1        = round(price + self.tp1_mult * atr_v, 4)
            tp2        = round(price + self.tp2_mult * atr_v, 4)
        else:
            entry_low  = round(price, 4)
            entry_high = round(price + 0.1 * atr_v, 4)
            sl         = round(price + self.sl_mult  * atr_v, 4)
            tp1        = round(price - self.tp1_mult * atr_v, 4)
            tp2        = round(price - self.tp2_mult * atr_v, 4)

        risk_pts = abs(price - sl)
        if risk_pts < 1e-8:
            return None

        avg_r  = self.tp1_mult * (self.tp1_alloc / 100) + self.tp2_mult * (self.tp2_alloc / 100)
        ev     = round((self.win_rate * avg_r) - ((1 - self.win_rate) * 1.0), 4)
        pf_val = round((self.win_rate * avg_r) / max(1 - self.win_rate, 1e-10), 3)
        conf   = min(0.38 + score * 0.04 + (0.06 if adx_trending else 0.0), 0.85)

        m30_state    = "BULL" if bull_m30 else ("BEAR" if bear_m30 else "FLAT")
        cross_note   = "fresh cross" if (cross_bull if direction == "long" else cross_bear) else "trend aligned"
        vwap_side    = "above" if price > vwap_v else "below"
        vol_state    = f"{vol_ratio:.2f}× {'OK' if vol_ok else 'EXTREME'}"

        setup = TradingSetup(
            name  = "Setup A",
            label = f"{'Long' if direction == 'long' else 'Short'} scalp — {score}/10 score (M5/M15/M30)",
            direction  = direction,
            trade_type = "scalp",
            status     = "PRIMARY WATCH",
            priority   = "primary",
            rationale  = (
                f"Composite score {score}/10 (threshold {self.min_score}). "
                f"M30 macro: {m30_state} "
                f"(EMA{self.ema_f30}={ema_f30_v:,.2f} {'>' if bull_m30 else '<'} EMA{self.ema_s30}={ema_s30_v:,.2f}, RSI={rsi_30_v:.0f}). "
                f"M15 EMA: {cross_note} "
                f"(EMA{self.ema_f15}={ema_f15_v:,.2f} vs EMA{self.ema_s15}={ema_s15_v:,.2f}). "
                f"ADX(M15)={adx_v:.1f}, DI+={di_plus_v:.1f}/DI-={di_minus_v:.1f}. "
                f"RSI(M15)={rsi_15_v:.0f}. "
                f"Vol regime={vol_state}. "
                f"VWAP={vwap_v:,.2f} (price {vwap_side})."
            ),
            trigger = (
                f"Enter {direction} at market. "
                f"Score {score}/{self.min_score} threshold met. "
                f"Zone ${entry_low:,.4f}–${entry_high:,.4f}."
            ),
            entry_low              = entry_low,
            entry_high             = entry_high,
            stop_loss              = sl,
            trailing_sl_to_breakeven = tp1,
            targets = [
                SetupTarget(
                    price        = tp1,
                    label        = f"TP1 {self.tp1_alloc}% @ {self.tp1_mult:.1f}× ATR — move SL to breakeven",
                    allocation_pct = self.tp1_alloc,
                ),
                SetupTarget(
                    price        = tp2,
                    label        = f"TP2 {self.tp2_alloc}% @ {self.tp2_mult:.1f}× ATR (free ride)",
                    allocation_pct = self.tp2_alloc,
                ),
            ],
            rr_ratio           = round(avg_r, 2),
            win_rate_estimate  = self.win_rate,
            trade_duration     = "30–120 minutes",
            ev                 = ev,
            profit_factor      = pf_val,
            confidence         = round(conf, 2),
            confidence_note    = (
                f"Score {score}/10 | M30 {m30_state} | ADX {adx_v:.0f} "
                f"| Vol {vol_state} | RSI-M15 {rsi_15_v:.0f} "
                f"| M5-EMA {'bull' if ema_f5_v > ema_s5_v else 'bear'} "
                f"| VWAP {vwap_side}"
            ),
        )

        self._daily_counts[date_key] = self._daily_counts.get(date_key, 0) + 1

        return SetupsOutput(
            asset          = tech.get("ticker", "UNKNOWN"),
            current_price  = price,
            setups         = [setup],
            invalidation   = InvalidationScenario(
                condition      = f"Price {'falls' if direction == 'long' else 'rises'} to SL ${sl:,.4f}",
                description    = f"ATR stop: {self.sl_mult:.1f}× ATR from entry",
                price_trigger  = sl,
                action         = "Exit immediately — stop triggered",
            ),
            decision_tree  = [
                DecisionTreeEntry(
                    scenario   = f"Score {score}/10 holds on next M5 bar",
                    outcome    = (
                        f"ENTER {direction.upper()} — SL ${sl:,.4f}, "
                        f"TP1 ${tp1:,.4f} ({self.tp1_mult:.1f}×ATR)"
                    ),
                    direction  = direction,
                    setup_name = "Setup A",
                    entry_price = entry_high if direction == "long" else entry_low,
                ),
                DecisionTreeEntry(
                    scenario   = f"Score drops below {self.min_score} (M30 flips or ADX weakens)",
                    outcome    = "NO TRADE — conditions no longer met",
                    direction  = "no_trade",
                    setup_name = "NO TRADE",
                ),
            ],
            position_sizing_note = (
                f"ATR(M5)={atr_v:,.2f} ({tech.get('atr_pct', 0):.2f}% of price). "
                f"SL={self.sl_mult:.1f}×ATR={self.sl_mult * atr_v:,.2f}. "
                f"Vol regime {vol_state} — "
                f"{'normal sizing.' if vol_ok else 'EXTREME volatility: skip or halve size.'}"
            ),
        )
