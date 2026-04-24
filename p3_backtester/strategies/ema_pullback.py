"""
strategies/ema_pullback.py — EMA Pullback strategy.

A deeper variant of EMA Continuation. Price has been in a trend,
pulls back further (past the EMA), and we enter on the bounce.

Difference from EMA Continuation:
  - Entry zone is BELOW the EMA for longs (deeper pullback)
  - RSI filter: requires RSI < 50 on longs (confirms pullback, not reversal)
  - Wider stop: EMA + 1.5*ATR
  - Lower default win rate (0.45) — counter-EMA entries are harder
  - Only fires when price is close to key support/resistance (within 1 ATR)

Use case: for assets that overshoot the EMA on pullbacks before bouncing.
"""
import ta
import pandas as pd

from p3_backtester.strategies.base import StrategyBase
from p1_analysis_engine.schema import (
    TradingSetup, SetupTarget, InvalidationScenario,
    DecisionTreeEntry, SetupsOutput,
)


def _rr(entry, sl, tp):
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return reward / risk if risk > 1e-10 else 0.0


def _ev(win_rate, avg_reward_r):
    return round((win_rate * avg_reward_r) - ((1 - win_rate) * 1.0), 4)


def _pf(win_rate, avg_reward_r):
    lr = 1 - win_rate
    return round((win_rate * avg_reward_r) / lr, 3) if lr > 1e-10 else 999.0


def _interval_trade_type(interval):
    return {
        "1m": "scalp", "5m": "scalp", "15m": "scalp",
        "30m": "intraday", "1h": "intraday",
        "1d": "swing", "1wk": "swing",
    }.get(interval, "swing")


def _interval_duration(interval):
    return {
        "1m": "5-30 minutes", "5m": "30-90 minutes", "15m": "1-3 hours",
        "30m": "2-6 hours", "1h": "4-12 hours",
        "1d": "3-8 days", "1wk": "3-8 weeks",
    }.get(interval, "varies")


class EMAPullbackStrategy(StrategyBase):
    """
    Deeper pullback past the EMA with bounce entry.
    Requires price to overshoot EMA into a support/resistance zone.
    """

    name = "EMA Pullback"
    description = "Deep pullback past EMA with bounce at S/R (RSI-filtered)"

    def __init__(
        self,
        ema_period: int = 20,
        adx_threshold: int = 18,
        win_rate: float = 0.45,
        min_rr: float = 1.5,
        rsi_max: int = 55,
        tp1_r: float = 1.5,
        tp2_r: float = 3.0,
        tp1_alloc: int = 60,
        tp2_alloc: int = 40,
        atr_pct_max: float = 999.0,          # skip extreme-vol bars
        require_rsi_divergence: bool = False, # require bullish RSI divergence vs prev swing low
        div_lookback: int = 10,              # bars to look back for divergence comparison
        require_pin_bar: bool = False,        # require rejection candle (lower shadow >= pin_shadow_atr)
        pin_shadow_atr: float = 0.8,         # lower shadow must be >= this × ATR
        candle_close_pct_max: float = 1.0,   # max (close-low)/(high-low) — wins close near low (0.24)
        require_cmf_neutral: bool = False,    # skip if CMF shows clear accumulation/distribution
        require_stoch_not_overbought: bool = False,  # skip if stoch > 80 (WR 7% when overbought)
        require_ema_not_bullish: bool = False,       # skip if EMA20>50>200 (WR 5% in bull alignment)
    ):
        self.ema_period                  = ema_period
        self.adx_threshold               = adx_threshold
        self.win_rate                    = win_rate
        self.min_rr                      = min_rr
        self.rsi_max                     = rsi_max
        self.tp1_r                       = tp1_r
        self.tp2_r                       = tp2_r
        self.tp1_alloc                   = tp1_alloc
        self.tp2_alloc                   = tp2_alloc
        self.atr_pct_max                 = atr_pct_max
        self.require_rsi_divergence      = require_rsi_divergence
        self.div_lookback                = div_lookback
        self.require_pin_bar             = require_pin_bar
        self.pin_shadow_atr              = pin_shadow_atr
        self.candle_close_pct_max        = candle_close_pct_max
        self.require_cmf_neutral         = require_cmf_neutral
        self.require_stoch_not_overbought = require_stoch_not_overbought
        self.require_ema_not_bullish     = require_ema_not_bullish

    @property
    def params(self) -> dict:
        return {
            "ema_period":    self.ema_period,
            "adx_threshold": self.adx_threshold,
        }

    def _check_rsi_divergence(self, df: pd.DataFrame, bias: str) -> bool:
        """
        Bullish divergence for longs: price at new low vs lookback but RSI higher.
        Bearish divergence for shorts: price at new high vs lookback but RSI lower.
        """
        if len(df) < self.div_lookback + 2:
            return False
        close  = df["Close"].values
        rsi_s  = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        if rsi_s.isna().all():
            return False

        window = self.div_lookback
        current_close = float(close[-1])
        current_rsi   = float(rsi_s.iloc[-1])
        past_closes   = close[-window - 1 : -1]
        past_rsi      = rsi_s.iloc[-window - 1 : -1].values

        if bias == "long":
            # Find the prior lowest-price bar in the window
            idx_low = int(past_closes.argmin())
            prior_close = float(past_closes[idx_low])
            prior_rsi   = float(past_rsi[idx_low])
            # Divergence: price made a new low but RSI did not
            return current_close < prior_close and current_rsi > prior_rsi
        else:
            idx_high = int(past_closes.argmax())
            prior_close = float(past_closes[idx_high])
            prior_rsi   = float(past_rsi[idx_high])
            return current_close > prior_close and current_rsi < prior_rsi

    def _check_pin_bar(self, df: pd.DataFrame, bias: str, atr: float) -> bool:
        """
        Bullish pin bar: lower shadow (Close - Low) >= pin_shadow_atr × ATR.
        Bearish pin bar: upper shadow (High - Close) >= pin_shadow_atr × ATR.
        """
        bar   = df.iloc[-1]
        if bias == "long":
            lower_shadow = float(bar["Close"]) - float(bar["Low"])
            return lower_shadow >= self.pin_shadow_atr * atr
        else:
            upper_shadow = float(bar["High"]) - float(bar["Close"])
            return upper_shadow >= self.pin_shadow_atr * atr

    def _get_ema(self, df: pd.DataFrame) -> float | None:
        close = df["Close"]
        if len(close) < self.ema_period:
            return None
        series = ta.trend.EMAIndicator(close, window=self.ema_period).ema_indicator()
        val = series.iloc[-1]
        return float(val) if pd.notna(val) else None

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        price    = tech["current_price"]
        atr      = tech["atr_14"]
        adx      = tech["adx_14"]
        rsi      = tech["rsi_14"]
        support  = tech["nearest_support"]
        resist   = tech["nearest_resistance"]
        interval = tech.get("interval", "1d")

        if adx < self.adx_threshold:
            return None

        # Volatility cap — skip chaotic/crash bars
        atr_pct = tech.get("atr_pct", 0.0)
        if atr_pct > self.atr_pct_max:
            return None

        ema_val = self._get_ema(df)
        if ema_val is None:
            return None

        dist_from_ema = price - ema_val
        dist_atr = dist_from_ema / atr if atr > 1e-10 else 0.0

        # Long setup: price has pulled below EMA, near support, RSI < 55
        # Short setup: price has rallied above EMA, near resistance, RSI > 45
        rsi_min_short = 100 - self.rsi_max   # symmetric: rsi_max=55 → short requires rsi>45
        if dist_atr < -0.1 and dist_atr > -2.5 and rsi < self.rsi_max:
            # Price is below EMA — long bounce setup
            bias = "long"
            level = support
        elif dist_atr > 0.1 and dist_atr < 2.5 and rsi > rsi_min_short:
            # Price is above EMA — short fade setup
            bias = "short"
            level = resist
        else:
            return None

        # Level must be close (within 1 ATR) to current price for bounce validity
        if abs(price - level) > atr * 1.2:
            return None

        # Optional: require bullish/bearish RSI divergence vs prior swing
        if self.require_rsi_divergence and not self._check_rsi_divergence(df, bias):
            return None

        # Optional: require rejection candle at the level (pin bar)
        if self.require_pin_bar and not self._check_pin_bar(df, bias, atr):
            return None

        # Optional: candle close position — wins close near low (0.24 median), losses near high (0.58)
        if self.candle_close_pct_max < 1.0:
            bar = df.iloc[-1]
            h, l, c = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
            close_pct = (c - l) / (h - l) if (h - l) > 1e-10 else 0.5
            if close_pct > self.candle_close_pct_max:
                return None

        # Optional: skip if CMF shows strong accumulation or distribution (WR 8% / 0%)
        if self.require_cmf_neutral:
            cmf = tech.get("cmf_signal", "neutral")
            if cmf in ("accumulation", "distribution"):
                return None

        # Optional: skip if stochastic overbought (WR 7% when stoch > 80)
        if self.require_stoch_not_overbought:
            if tech.get("stoch_signal") == "overbought":
                return None

        # Optional: skip if EMA20 > EMA50 > EMA200 alignment (WR 5% in full bull alignment)
        if self.require_ema_not_bullish:
            if tech.get("ema_alignment") == "bullish":
                return None

        trade_type   = _interval_trade_type(interval)
        avg_reward_r = self.tp1_r * (self.tp1_alloc / 100) + self.tp2_r * (self.tp2_alloc / 100)
        ev = _ev(self.win_rate, avg_reward_r)
        pf = _pf(self.win_rate, avg_reward_r)
        buf = atr * 1.5

        if bias == "long":
            entry_low  = round(level - atr * 0.10, 4)
            entry_high = round(level + atr * 0.20, 4)
            sl         = round(level - buf, 4)
            risk       = entry_high - sl
            if risk < 1e-10:
                return None
            tp1 = round(entry_high + risk * self.tp1_r, 4)
            tp2 = round(entry_high + risk * self.tp2_r, 4)
            if _rr(entry_high, sl, tp1) < self.min_rr:
                return None
        else:
            entry_high = round(level + atr * 0.10, 4)
            entry_low  = round(level - atr * 0.20, 4)
            sl         = round(level + buf, 4)
            risk       = sl - entry_low
            if risk < 1e-10:
                return None
            tp1 = round(entry_low - risk * self.tp1_r, 4)
            tp2 = round(entry_low - risk * self.tp2_r, 4)
            if _rr(entry_low, sl, tp1) < self.min_rr:
                return None

        # Coefficient 0.018 → ADX=35 produces confidence=0.706 (just above 0.70 threshold)
        # Cap at 0.75 keeps all signals in half-Kelly tier — appropriate for bounce strategy.
        confidence = min(0.40 + (adx - self.adx_threshold) * 0.018, 0.75)

        setup = TradingSetup(
            name="Setup A",
            label=f"{'Long bounce' if bias=='long' else 'Short fade'} at ${level:,.4f} (EMA{self.ema_period} overshoot)",
            direction=bias,
            trade_type=trade_type,
            status="ACTIVE",
            priority="primary",
            rationale=(
                f"Price overshot EMA{self.ema_period} ({ema_val:,.4f}) by {abs(dist_atr):.1f} ATR. "
                f"Bounce entry at {'support' if bias=='long' else 'resistance'} ${level:,.4f}. "
                f"RSI {rsi:.0f}, ADX {adx:.0f}."
            ),
            trigger=f"Price tests ${level:,.4f} and prints {'bullish' if bias=='long' else 'bearish'} reversal candle",
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
            confidence_note=f"EMA{self.ema_period}={ema_val:,.2f} | RSI {rsi:.0f} | dist {dist_atr:+.1f}ATR",
        )

        return SetupsOutput(
            asset=tech.get("ticker", "UNKNOWN"),
            current_price=price,
            setups=[setup],
            invalidation=InvalidationScenario(
                condition=f"Close {'below' if bias=='long' else 'above'} ${sl:,.4f}",
                description=f"Key level broken — bounce invalidated",
                price_trigger=sl,
                action="Stand aside",
            ),
            decision_tree=[
                DecisionTreeEntry(
                    scenario=f"Price tests ${level:,.4f} and holds",
                    outcome=f"ENTER — SL ${sl:,.4f}",
                    direction=bias,
                    setup_name="Setup A",
                    entry_price=entry_high if bias == "long" else entry_low,
                ),
                DecisionTreeEntry(
                    scenario=f"Price breaks ${sl:,.4f}",
                    outcome="NO TRADE — level broken",
                    direction="no_trade",
                    setup_name="NO TRADE",
                ),
            ],
            position_sizing_note=(
                f"ATR={atr:,.4f}. "
                + ("REDUCE 50% — high vol." if tech["atr_pct"] > 2 else "Normal sizing.")
            ),
        )
