"""
strategies/mtf_trend.py — Multi-Timeframe Trend Following strategy.

Concept (from quantitative framework):
  A) Momentum / Trend-Following (Primary)
     - Higher timeframe (HTF) provides directional bias via EMA20/50 alignment
     - Entry timeframe (ETF) provides precise entry on EMA20 retest
     - RSI confirms momentum: not overbought on long entry, not oversold on short
     - Stop: structural swing low/high (NOT fixed ATR)
     - Target: 2.5R minimum (TP1 70%), 4.0R (TP2 30%)

Timeframe pairs (auto-derived from entry interval):
  Entry 15m  -> Bias 1H    (58-day yfinance cap — short window)
  Entry 1H   -> Bias 1D    (729-day window — recommended)
  Entry 1D   -> Bias 1W    (10-year window — long-term)

All HTF computation resamples from the already-sliced df (df.iloc[:i+1]),
so there is zero lookahead bias.

Gates (all must pass):
  1. HTF EMA20 > EMA50 = long bias, EMA20 < EMA50 = short bias
  2. HTF ADX >= adx_threshold (longs) or adx_threshold_short (shorts)
  3. ETF price within pullback_max_atr of EMA20 (retest arrived)
  4. Prior distance: price was >= min_prior_distance_atr away from EMA20 IN THE
     DIRECTION OF THE TRADE in the last N bars (confirms genuine pullback, not
     price bouncing against the EMA from the wrong side)
  5. RSI filter: <= rsi_max for longs, >= rsi_min for shorts
  6. Regime filter: SMA200-based — no longs in bear_trend, no shorts in bull_trend

Changes from v1
---------------
- Gate 4 fix: abs() replaced with directional check — long signals require price
  to have been ABOVE EMA20 (not just "away from it"), preventing bounce signals
  in declining price action from masquerading as pullback entries.
- Gate 2 asymmetry: shorts use adx_threshold_short (default 15) vs longs at 20.
  Bear markets form with lower ADX initially; 20 was too strict and suppressed
  early short entries entirely.
- Gate 6 (new): SMA200 regime filter. Blocks longs when price is in a confirmed
  bear regime and blocks shorts in a confirmed bull regime. This is the primary
  fix for 2022-style catastrophic drawdowns where the strategy kept buying into
  a declining market.
- Confidence score updated to incorporate regime alignment.
"""
import ta
import pandas as pd

from p3_backtester.strategies.base import StrategyBase
from p3_backtester.strategies.regime_filter import classify_regime
from p1_analysis_engine.schema import (
    TradingSetup, SetupTarget, InvalidationScenario,
    DecisionTreeEntry, SetupsOutput,
)


# ── HTF resample rules ────────────────────────────────────────────────────────
_HTF_RESAMPLE: dict[str, str] = {
    "1m":  "15min",
    "5m":  "1h",
    "15m": "1h",
    "30m": "4h",
    "1h":  "1D",
    "1d":  "1W",
    "1wk": "1ME",   # month-end
}

_HTF_LABEL: dict[str, str] = {
    "15min": "15m", "1h": "1H", "4h": "4H", "1D": "1D", "1W": "1W", "1ME": "1M",
}


def _interval_trade_type(interval: str) -> str:
    return {
        "1m": "scalp", "5m": "scalp", "15m": "scalp",
        "30m": "intraday", "1h": "intraday",
        "1d": "swing", "1wk": "swing",
    }.get(interval, "swing")


def _interval_duration(interval: str) -> str:
    return {
        "1m": "5-30 minutes", "5m": "30-90 minutes", "15m": "1-3 hours",
        "30m": "2-6 hours", "1h": "4-12 hours",
        "1d": "3-8 days", "1wk": "3-8 weeks",
    }.get(interval, "varies")


def _find_swing_low(df: pd.DataFrame, lookback: int = 15) -> float | None:
    """
    Most recent swing low in the last `lookback` bars.
    Swing low: Low[i] <= Low[i-1] AND Low[i] <= Low[i+1].
    Falls back to rolling minimum if no swing found.
    """
    lows = df["Low"]
    n = len(lows)
    for i in range(n - 2, max(n - lookback - 2, 1), -1):
        if float(lows.iloc[i]) <= float(lows.iloc[i - 1]) and \
           float(lows.iloc[i]) <= float(lows.iloc[i + 1]):
            return float(lows.iloc[i])
    # Fallback: rolling minimum of last `lookback` bars
    return float(lows.iloc[max(0, n - lookback):n].min())


def _find_swing_high(df: pd.DataFrame, lookback: int = 15) -> float | None:
    """
    Most recent swing high in the last `lookback` bars.
    """
    highs = df["High"]
    n = len(highs)
    for i in range(n - 2, max(n - lookback - 2, 1), -1):
        if float(highs.iloc[i]) >= float(highs.iloc[i - 1]) and \
           float(highs.iloc[i]) >= float(highs.iloc[i + 1]):
            return float(highs.iloc[i])
    return float(highs.iloc[max(0, n - lookback):n].max())


class MTFTrendStrategy(StrategyBase):
    """
    Multi-Timeframe Trend Following.

    Gates (all must pass):
      1. HTF EMA20 > EMA50 = long bias, EMA20 < EMA50 = short bias
      2. HTF ADX >= adx_threshold (longs) / adx_threshold_short (shorts)
      3. ETF price within pullback_max_atr of EMA20 (retest arrived)
      4. Prior distance (directional): price was >= min_prior_distance_atr ABOVE
         EMA20 for longs (BELOW for shorts) in the last N bars — genuine pullback
      5. RSI filter: <= rsi_max for longs, >= rsi_min for shorts
      6. Regime filter: no longs in bear_trend, no shorts in bull_trend (SMA200)

    Entry zone: [EMA20 - 0.3*ATR, EMA20 + 0.2*ATR] for longs (mirror for shorts)
    Stop: structural swing low/high - sl_buffer_atr * ATR
    TP1: tp1_r * risk (tp1_alloc%), TP2: tp2_r * risk (tp2_alloc%)
    """

    name = "MTF Trend"
    description = "Multi-timeframe: HTF EMA20/50 bias + ETF EMA20 retest entry, structural SL, SMA200 regime filter"

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        adx_threshold: int = 20,
        adx_threshold_short: int = 15,
        pullback_max_atr: float = 1.2,
        min_prior_distance_atr: float = 1.5,
        prior_lookback: int = 5,
        rsi_max: int = 65,
        rsi_min: int = 35,
        sl_buffer_atr: float = 0.15,
        sl_lookback: int = 15,
        tp1_r: float = 2.5,
        tp2_r: float = 4.0,
        tp1_alloc: int = 70,
        tp2_alloc: int = 30,
        win_rate: float = 0.48,
        regime_filter: bool = True,
        shorts_enabled: bool = True,
    ):
        self.ema_fast               = ema_fast
        self.ema_slow               = ema_slow
        self.adx_threshold          = adx_threshold
        self.adx_threshold_short    = adx_threshold_short
        self.pullback_max_atr       = pullback_max_atr
        self.min_prior_distance_atr = min_prior_distance_atr
        self.prior_lookback         = prior_lookback
        self.rsi_max                = rsi_max
        self.rsi_min                = rsi_min
        self.sl_buffer_atr          = sl_buffer_atr
        self.sl_lookback            = sl_lookback
        self.tp1_r                  = tp1_r
        self.tp2_r                  = tp2_r
        self.tp1_alloc              = tp1_alloc
        self.tp2_alloc              = tp2_alloc
        self.win_rate               = win_rate
        self.regime_filter          = regime_filter
        self.shorts_enabled         = shorts_enabled

    @property
    def params(self) -> dict:
        return {
            "ema_fast":            self.ema_fast,
            "ema_slow":            self.ema_slow,
            "adx_threshold":       self.adx_threshold,
            "adx_threshold_short": self.adx_threshold_short,
            "tp1_r":               self.tp1_r,
            "rsi_max":             self.rsi_max,
            "regime_filter":       self.regime_filter,
            "shorts_enabled":      self.shorts_enabled,
        }

    def _htf_bias(
        self, df: pd.DataFrame, interval: str
    ) -> tuple[str | None, float, float, float]:
        """
        Resample df to HTF, compute EMA20/50 alignment and ADX.

        Returns (bias, htf_adx, htf_ema_fast, htf_ema_slow)
          bias: "long" | "short" | None (insufficient data or conflicted)
        """
        rule = _HTF_RESAMPLE.get(interval, "1D")
        try:
            htf_close = df["Close"].resample(rule).last().dropna()
            htf_high  = df["High"].resample(rule).max().reindex(htf_close.index).dropna()
            htf_low   = df["Low"].resample(rule).min().reindex(htf_close.index).dropna()
        except Exception:
            return None, 0.0, 0.0, 0.0

        min_bars = self.ema_slow + 20
        if len(htf_close) < min_bars:
            return None, 0.0, 0.0, 0.0

        ema_f_s = ta.trend.EMAIndicator(htf_close, window=self.ema_fast).ema_indicator()
        ema_s_s = ta.trend.EMAIndicator(htf_close, window=self.ema_slow).ema_indicator()

        ema_f_val = float(ema_f_s.iloc[-1])
        ema_s_val = float(ema_s_s.iloc[-1])

        if pd.isna(ema_f_val) or pd.isna(ema_s_val):
            return None, 0.0, 0.0, 0.0

        # HTF ADX
        htf_adx = 0.0
        n = min(len(htf_close), len(htf_high), len(htf_low))
        if n >= 20:
            try:
                htf_adx = float(
                    ta.trend.ADXIndicator(
                        htf_high.iloc[-n:], htf_low.iloc[-n:], htf_close.iloc[-n:], window=14
                    ).adx().iloc[-1]
                )
            except Exception:
                pass

        if ema_f_val > ema_s_val:
            return "long", htf_adx, ema_f_val, ema_s_val
        elif ema_f_val < ema_s_val:
            return "short", htf_adx, ema_f_val, ema_s_val
        return None, htf_adx, ema_f_val, ema_s_val

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        price    = tech["current_price"]
        atr      = tech["atr_14"]
        rsi      = tech["rsi_14"]
        interval = tech.get("interval", "1h")

        if atr < 1e-10:
            return None

        # ── Gate 1 + 2: HTF bias and ADX ─────────────────────────────────────
        # Use precomputed values when available (optimizer precomputes once per bar
        # since these are param-independent, giving ~50× speedup on grid search).
        if "htf_bias" in tech:
            htf_bias  = tech["htf_bias"]
            htf_adx   = tech["htf_adx"]
            htf_ema_f = tech["htf_ema_f"]
            htf_ema_s_val = tech["htf_ema_s"]
        else:
            htf_bias, htf_adx, htf_ema_f, htf_ema_s_val = self._htf_bias(df, interval)
        if htf_bias is None:
            return None

        # Directional gate: suppress shorts when running long-only
        if not self.shorts_enabled and htf_bias == "short":
            return None

        # Asymmetric ADX threshold: shorts require less trend strength to fire
        # (bear markets often start with lower ADX as the trend is forming)
        adx_min = self.adx_threshold if htf_bias == "long" else self.adx_threshold_short
        if htf_adx < adx_min:
            return None

        # ── Gate 6: Regime filter (SMA200) ───────────────────────────────────
        # Applied before computing ETF indicators to fail fast.
        regime = "neutral"
        if "regime" in tech:
            regime = tech["regime"]
        elif self.regime_filter:
            regime = classify_regime(df, interval)
        if self.regime_filter:
            if regime == "bear_trend" and htf_bias == "long":
                return None   # never buy into a confirmed bear market
            if regime == "bull_trend" and htf_bias == "short":
                return None   # never short a confirmed bull market

        # ── Gate 3: ETF EMA20 proximity ───────────────────────────────────────
        close = df["Close"]
        if len(close) < self.ema_fast + 5:
            return None

        if "etf_ema_val" in tech and "etf_ema_tail" in tech:
            etf_ema_val  = tech["etf_ema_val"]
            etf_ema_tail = tech["etf_ema_tail"]  # list, [-1]=current, [-j]=j bars ago
            etf_ema_s    = None  # won't use series path
        else:
            etf_ema_s    = ta.trend.EMAIndicator(close, window=self.ema_fast).ema_indicator()
            etf_ema_val  = float(etf_ema_s.iloc[-1])
            etf_ema_tail = None
        if pd.isna(etf_ema_val):
            return None

        dist_pts = price - etf_ema_val
        dist_atr = dist_pts / atr

        # Must be within pullback_max_atr of EMA20
        if abs(dist_atr) > self.pullback_max_atr:
            return None

        # Must be on the correct side — long bias needs price at or above EMA.
        # Allow up to 0.5 ATR on the wrong side: volatile assets often overshoot
        # the EMA briefly before reversing. Beyond 0.5 ATR signals a trend break.
        if htf_bias == "long"  and dist_pts < -0.5 * atr:
            return None
        if htf_bias == "short" and dist_pts >  0.5 * atr:
            return None

        # ── Gate 4: Prior distance — directional (v1 bug fix) ────────────────
        # Long:  price must have been ABOVE EMA20 by min_prior_distance_atr in
        #        the last N bars. This confirms we're entering a genuine pullback
        #        from above, not a bounce attempt against overhead resistance.
        # Short: price must have been BELOW EMA20 by min_prior_distance_atr —
        #        confirms the rally that is now fading was real.
        prior_valid = False
        for j in range(2, self.prior_lookback + 2):
            if etf_ema_tail is not None:
                if j > len(etf_ema_tail) or len(close) < j:
                    break
                p_ema = etf_ema_tail[-j]
            else:
                if len(etf_ema_s) < j or len(close) < j:
                    break
                p_ema = float(etf_ema_s.iloc[-j])
            p_price = float(close.iloc[-j])
            if htf_bias == "long":
                # Prior bar was clearly ABOVE EMA (directional, not abs)
                if p_price - p_ema >= atr * self.min_prior_distance_atr:
                    prior_valid = True
                    break
            else:
                # Prior bar was clearly BELOW EMA
                if p_ema - p_price >= atr * self.min_prior_distance_atr:
                    prior_valid = True
                    break
        if not prior_valid:
            return None

        # ── Gate 5: RSI momentum filter ───────────────────────────────────────
        if htf_bias == "long"  and rsi > self.rsi_max:
            return None   # overbought — pullback hasn't cooled yet
        if htf_bias == "short" and rsi < self.rsi_min:
            return None   # oversold — don't short into an extreme

        # ── Structural stop ───────────────────────────────────────────────────
        if htf_bias == "long":
            swing = _find_swing_low(df, lookback=self.sl_lookback)
            if swing is None:
                return None
            sl = round(swing - self.sl_buffer_atr * atr, 4)

            entry_low  = round(etf_ema_val - atr * 0.30, 4)
            entry_high = round(etf_ema_val + atr * 0.20, 4)
            fill_ref   = entry_high   # pessimistic fill
            risk_ref   = fill_ref - sl
            if risk_ref <= 1e-10:
                return None

            tp1 = round(fill_ref + self.tp1_r * risk_ref, 4)
            tp2 = round(fill_ref + self.tp2_r * risk_ref, 4)
        else:
            swing = _find_swing_high(df, lookback=self.sl_lookback)
            if swing is None:
                return None
            sl = round(swing + self.sl_buffer_atr * atr, 4)

            entry_high = round(etf_ema_val + atr * 0.30, 4)
            entry_low  = round(etf_ema_val - atr * 0.20, 4)
            fill_ref   = entry_low    # pessimistic fill
            risk_ref   = sl - fill_ref
            if risk_ref <= 1e-10:
                return None

            tp1 = round(fill_ref - self.tp1_r * risk_ref, 4)
            tp2 = round(fill_ref - self.tp2_r * risk_ref, 4)

        avg_r  = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev     = round((self.win_rate * avg_r) - ((1 - self.win_rate) * 1.0), 4)
        pf_val = round((self.win_rate * avg_r) / max(1 - self.win_rate, 1e-10), 3)

        # Confidence score: baseline from ADX strength, boosted when regime aligns
        regime_boost = {"bull_trend": 0.05, "neutral": 0.0, "bear_trend": 0.05}
        # In bear_trend we only reach here on short signals — same boost applies
        conf = min(
            0.38
            + (htf_adx - adx_min) * 0.007
            + regime_boost.get(regime, 0.0),
            0.80,
        )

        rule      = _HTF_RESAMPLE.get(interval, "1D")
        htf_label = _HTF_LABEL.get(rule, rule)
        trade_type = _interval_trade_type(interval)

        regime_note = {
            "bull_trend": "Regime: BULL (SMA200 confirmed).",
            "bear_trend": "Regime: BEAR (SMA200 confirmed).",
            "neutral":    "Regime: NEUTRAL (transitioning).",
        }.get(regime, "")

        setup = TradingSetup(
            name="Setup A",
            label=(
                f"{'Long' if htf_bias == 'long' else 'Short'} EMA{self.ema_fast} retest "
                f"({interval} entry / {htf_label} bias)"
            ),
            direction=htf_bias,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"HTF ({htf_label}) EMA{self.ema_fast}({htf_ema_f:,.2f}) "
                f"{'>' if htf_bias=='long' else '<'} "
                f"EMA{self.ema_slow}({htf_ema_s_val:,.2f}) "
                f"-> {htf_bias} bias (ADX {htf_adx:.0f}). "
                f"Price retesting ETF EMA{self.ema_fast} at {etf_ema_val:,.4f} ({dist_atr:+.2f} ATR). "
                f"RSI {rsi:.0f} confirms momentum. "
                f"Structural SL at swing {'low' if htf_bias == 'long' else 'high'} {swing:,.4f}. "
                f"{regime_note}"
            ),
            trigger=(
                f"Price enters EMA{self.ema_fast} retest zone "
                f"${entry_low:,.4f}-${entry_high:,.4f}"
            ),
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=sl,
            trailing_sl_to_breakeven=tp1,
            targets=[
                SetupTarget(
                    price=tp1,
                    label=f"TP1 {self.tp1_alloc}% @ {self.tp1_r:.1f}R",
                    allocation_pct=self.tp1_alloc,
                ),
                SetupTarget(
                    price=tp2,
                    label=f"TP2 {self.tp2_alloc}% @ {self.tp2_r:.1f}R",
                    allocation_pct=self.tp2_alloc,
                ),
            ],
            rr_ratio=round(avg_r, 2),
            win_rate_estimate=self.win_rate,
            trade_duration=_interval_duration(interval),
            ev=ev,
            profit_factor=pf_val,
            confidence=round(conf, 2),
            confidence_note=(
                f"HTF EMA{self.ema_fast}={htf_ema_f:,.2f} | HTF ADX {htf_adx:.0f} "
                f"| ETF dist {dist_atr:+.2f}ATR | RSI {rsi:.0f} | SL={sl:,.4f} "
                f"| {regime_note}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Close {'below' if htf_bias == 'long' else 'above'} ${sl:,.4f}",
                description="Structural swing level broken",
                price_trigger=sl,
                action="Stand aside — swing structure invalidated",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Price retests EMA{self.ema_fast} and holds",
                    outcome=(
                        f"ENTER {htf_bias.upper()} — SL ${sl:,.4f}, "
                        f"TP1 ${tp1:,.4f} ({self.tp1_r:.1f}R)"
                    ),
                    direction=htf_bias,
                    setup_name="Setup A",
                    entry_price=entry_high if htf_bias == "long" else entry_low,
                ),
                DecisionTreeEntry(
                    scenario=f"Price breaks {'below' if htf_bias == 'long' else 'above'} ${sl:,.4f}",
                    outcome="NO TRADE — structure broken",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr:,.4f} ({tech['atr_pct']}% of price). "
                f"Structural risk={risk_ref:,.4f}. "
                + ("REDUCE 50% — high vol." if tech["atr_pct"] > 2 else "Normal sizing.")
            ),
        )
