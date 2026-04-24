"""
strategies/ema_continuation.py — EMA Continuation strategy (tightened).

Concept: price trends away from EMA, then pulls back TO it — we enter on that
pullback in the trend direction.

Three additional quality gates (vs original):

  1. Prior distance gate
     Price must have been >= min_prior_distance_atr ATR away from EMA in the
     last `prior_lookback` bars. This ensures we are catching a genuine
     PULLBACK, not price that has been hovering near the EMA all session.

  2. Daily regime filter (long-only mode)
     Resample the intraday df to daily bars, compute EMA200 on daily close.
     Long setups only when daily close > daily EMA200.
     Short setups only when daily close < daily EMA200.
     When insufficient daily history, filter is skipped (not applied).

  3. ADX threshold raised to 25 (was 20)
     Stronger trending requirement. ADX 20-25 is often noise on Gold.

Entry / SL / TP unchanged from the base version.
"""
import ta
import pandas as pd

from p3_backtester.strategies.base import StrategyBase
from p1_analysis_engine.schema import (
    TradingSetup, SetupTarget, InvalidationScenario,
    DecisionTreeEntry, SetupsOutput,
)


def _rr(entry: float, sl: float, tp: float) -> float:
    risk   = abs(entry - sl)
    reward = abs(tp - entry)
    return reward / risk if risk > 1e-10 else 0.0


def _ev(win_rate: float, avg_reward_r: float) -> float:
    return round((win_rate * avg_reward_r) - ((1 - win_rate) * 1.0), 4)


def _pf(win_rate: float, avg_reward_r: float) -> float:
    loss_rate = 1 - win_rate
    return round((win_rate * avg_reward_r) / loss_rate, 3) if loss_rate > 1e-10 else 999.0


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


class EMAContinuationStrategy(StrategyBase):
    """
    Pullback-to-EMA trend continuation — tightened version.

    Three gates beyond the basic version:
      1. Prior distance — confirms price genuinely pulled back to the EMA
      2. Daily regime  — only trade in direction of daily EMA200 trend
      3. Higher ADX    — 25 default vs 20 in original
    """

    name = "EMA Continuation"
    description = "Pullback to EMA in trend direction (ADX-filtered, regime-aware)"

    def __init__(
        self,
        ema_period: int = 20,
        adx_threshold: int = 25,
        pullback_max_atr: float = 1.5,
        min_prior_distance_atr: float = 1.5,
        prior_lookback: int = 5,
        use_daily_regime: bool = True,
        win_rate: float = 0.50,
        min_rr: float = 1.4,
        tp1_r: float = 1.5,
        tp2_r: float = 3.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        atr_pct_max: float = 999.0,  # skip high-vol bars (e.g. 1.2 = skip if ATR > 1.2% of price)
    ):
        self.ema_period              = ema_period
        self.adx_threshold           = adx_threshold
        self.pullback_max_atr        = pullback_max_atr
        self.min_prior_distance_atr  = min_prior_distance_atr
        self.prior_lookback          = prior_lookback
        self.use_daily_regime        = use_daily_regime
        self.win_rate                = win_rate
        self.min_rr                  = min_rr
        self.tp1_r                   = tp1_r
        self.tp2_r                   = tp2_r
        self.tp1_alloc               = tp1_alloc
        self.tp2_alloc               = tp2_alloc
        self.atr_pct_max             = atr_pct_max

    @property
    def params(self) -> dict:
        return {
            "ema_period":    self.ema_period,
            "adx_threshold": self.adx_threshold,
        }

    def _get_ema_series(self, df: pd.DataFrame) -> pd.Series | None:
        close = df["Close"]
        if len(close) < self.ema_period:
            return None
        series = ta.trend.EMAIndicator(close, window=self.ema_period).ema_indicator()
        return series

    def _check_prior_distance(self, df: pd.DataFrame, atr: float) -> bool:
        """
        True if price was >= min_prior_distance_atr away from EMA in at
        least one of the last `prior_lookback` bars (excluding current bar).
        This confirms price genuinely pulled back — not just hovered near EMA.
        """
        close = df["Close"]
        ema_s = self._get_ema_series(df)
        if ema_s is None or len(ema_s.dropna()) < self.prior_lookback + 1:
            return False

        for j in range(2, self.prior_lookback + 2):
            if len(ema_s) < j:
                break
            prior_ema   = float(ema_s.iloc[-j])
            prior_price = float(close.iloc[-j])
            if abs(prior_price - prior_ema) >= atr * self.min_prior_distance_atr:
                return True
        return False

    def _daily_regime(self, df: pd.DataFrame) -> str | None:
        """
        Returns "long", "short", or None (filter skipped — insufficient data).
        Resamples intraday df to daily, computes EMA200 on daily close.
        """
        if not self.use_daily_regime:
            return None

        try:
            # Resample to business day (handles equities + commodities)
            daily_close = df["Close"].resample("1D").last().dropna()
            if len(daily_close) < 200:
                return None   # not enough daily history — skip filter

            ema200 = ta.trend.EMAIndicator(daily_close, window=200).ema_indicator()
            ema200_val  = float(ema200.iloc[-1])
            current_val = float(daily_close.iloc[-1])

            if pd.isna(ema200_val):
                return None

            return "long" if current_val > ema200_val else "short"
        except Exception:
            return None

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        price    = tech["current_price"]
        atr      = tech["atr_14"]
        adx      = tech["adx_14"]
        interval = tech.get("interval", "1d")

        # ── Gate 1: ADX threshold ─────────────────────────────────────────────
        if adx < self.adx_threshold:
            return None

        # ── Gate 1b: Volatility cap — skip crash/expansion bars ──────────────
        atr_pct = tech.get("atr_pct", 0.0)
        if atr_pct > self.atr_pct_max:
            return None

        ema_s = self._get_ema_series(df)
        if ema_s is None:
            return None
        ema_val = float(ema_s.iloc[-1])
        if pd.isna(ema_val):
            return None

        dist_from_ema = price - ema_val
        dist_atr = dist_from_ema / atr if atr > 1e-10 else 0.0

        # ── Gate 2: Price close enough to EMA (pullback arrived) ─────────────
        if abs(dist_atr) > self.pullback_max_atr:
            return None

        # ── Gate 3: Prior distance — confirmed genuine pullback ───────────────
        if not self._check_prior_distance(df, atr):
            return None

        # Bias from EMA side
        if dist_from_ema > 0:
            bias = "long"
        elif dist_from_ema < 0:
            bias = "short"
        else:
            return None

        # ── Gate 4: Volume filter — low-volume EMA touches are noise ─────────
        # Only applies when volume data is available (indices may have Volume=0).
        vol = float(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].iloc[-21:-1].mean()) if len(df) > 21 else 0.0
        if avg_vol > 0 and vol < 0.7 * avg_vol:
            return None

        # ── Gate 5: Daily regime filter ───────────────────────────────────────
        regime = self._daily_regime(df)
        if regime is not None and regime != bias:
            return None   # trading against the daily macro trend — skip

        trade_type   = _interval_trade_type(interval)
        avg_reward_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev  = _ev(self.win_rate, avg_reward_r)
        pf  = _pf(self.win_rate, avg_reward_r)
        buf = atr * 1.0

        if bias == "long":
            entry_low  = round(ema_val - atr * 0.30, 4)
            entry_high = round(ema_val + atr * 0.20, 4)
            sl         = round(ema_val - buf, 4)
            risk       = entry_high - sl
            if risk < 1e-10:
                return None
            tp1 = round(entry_high + risk * self.tp1_r, 4)
            tp2 = round(entry_high + risk * self.tp2_r, 4)
            if _rr(entry_high, sl, tp1) < self.min_rr:
                return None
        else:
            entry_high = round(ema_val + atr * 0.30, 4)
            entry_low  = round(ema_val - atr * 0.20, 4)
            sl         = round(ema_val + buf, 4)
            risk       = sl - entry_low
            if risk < 1e-10:
                return None
            tp1 = round(entry_low - risk * self.tp1_r, 4)
            tp2 = round(entry_low - risk * self.tp2_r, 4)
            if _rr(entry_low, sl, tp1) < self.min_rr:
                return None

        confidence = min(0.45 + (adx - self.adx_threshold) * 0.005, 0.75)
        regime_note = f" | Regime: {regime} daily EMA200" if regime else ""

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long' if bias == 'long' else 'Short'} EMA{self.ema_period} continuation",
            direction=bias,
            trade_type=trade_type,
            status="PRIMARY WATCH",
            priority="primary",
            rationale=(
                f"Price {'above' if bias=='long' else 'below'} EMA{self.ema_period} "
                f"({ema_val:,.4f}), confirmed pullback from {abs(dist_atr):.1f}ATR distance. "
                f"ADX {adx:.0f} confirms trend strength.{regime_note}"
            ),
            trigger=(
                f"Price at EMA{self.ema_period} zone "
                f"${entry_low:,.4f}–${entry_high:,.4f}"
            ),
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=sl,
            trailing_sl_to_breakeven=tp1,
            targets=[
                SetupTarget(price=tp1, label=f"Target 1 ({self.tp1_alloc}%)", allocation_pct=self.tp1_alloc),
                SetupTarget(price=tp2, label=f"Target 2 ({self.tp2_alloc}%)", allocation_pct=self.tp2_alloc),
            ],
            rr_ratio=round(avg_reward_r, 2),
            win_rate_estimate=self.win_rate,
            trade_duration=_interval_duration(interval),
            ev=ev,
            profit_factor=pf,
            confidence=round(confidence, 2),
            confidence_note=(
                f"EMA{self.ema_period}={ema_val:,.2f} | ADX {adx:.0f} "
                f"| dist {dist_atr:+.1f}ATR{regime_note}"
            ),
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Close {'below' if bias=='long' else 'above'} ${sl:,.4f}",
                description=f"EMA{self.ema_period} structure broken",
                price_trigger=sl,
                action="Stand aside — wait for EMA to re-establish",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Price touches EMA{self.ema_period} zone and holds",
                    outcome=f"ENTER — SL ${sl:,.4f}, TP1 ${tp1:,.4f}",
                    direction=bias,
                    setup_name="Setup A",
                    entry_price=entry_high if bias == "long" else entry_low,
                ),
                DecisionTreeEntry(
                    scenario=f"Price breaks {'below' if bias=='long' else 'above'} ${sl:,.4f}",
                    outcome="NO TRADE — EMA broken",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr:,.4f} ({tech['atr_pct']}% of price). "
                + ("REDUCE 50% — high vol." if tech["atr_pct"] > 2 else "Normal sizing.")
            ),
        )
