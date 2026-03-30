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
    ):
        self.ema_period    = ema_period
        self.adx_threshold = adx_threshold
        self.win_rate      = win_rate
        self.min_rr        = min_rr

    @property
    def params(self) -> dict:
        return {
            "ema_period":    self.ema_period,
            "adx_threshold": self.adx_threshold,
        }

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

        ema_val = self._get_ema(df)
        if ema_val is None:
            return None

        dist_from_ema = price - ema_val
        dist_atr = dist_from_ema / atr if atr > 1e-10 else 0.0

        # Long setup: price has pulled below EMA, near support, RSI < 55
        # Short setup: price has rallied above EMA, near resistance, RSI > 45
        if dist_atr < -0.1 and dist_atr > -2.5 and rsi < 55:
            # Price is below EMA — long bounce setup
            bias = "long"
            level = support
        elif dist_atr > 0.1 and dist_atr < 2.5 and rsi > 45:
            # Price is above EMA — short fade setup
            bias = "short"
            level = resist
        else:
            return None

        # Level must be close (within 1 ATR) to current price for bounce validity
        if abs(price - level) > atr * 1.2:
            return None

        trade_type   = _interval_trade_type(interval)
        avg_reward_r = 1.5 * 0.6 + 3.0 * 0.4
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
            tp1 = round(entry_high + risk * 1.5, 4)
            tp2 = round(entry_high + risk * 3.0, 4)
            if _rr(entry_high, sl, tp1) < self.min_rr:
                return None
        else:
            entry_high = round(level + atr * 0.10, 4)
            entry_low  = round(level - atr * 0.20, 4)
            sl         = round(level + buf, 4)
            risk       = sl - entry_low
            if risk < 1e-10:
                return None
            tp1 = round(entry_low - risk * 1.5, 4)
            tp2 = round(entry_low - risk * 3.0, 4)
            if _rr(entry_low, sl, tp1) < self.min_rr:
                return None

        confidence = min(0.40 + (adx - self.adx_threshold) * 0.004, 0.65)

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
                SetupTarget(price=tp1, label="Target 1 (60%)", allocation_pct=60),
                SetupTarget(price=tp2, label="Target 2 (40%)", allocation_pct=40),
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
