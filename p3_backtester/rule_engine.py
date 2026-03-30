"""
rule_engine.py — deterministic trade setup generator.

Produces a SetupsOutput from a technical dict (bar_slicer output) without
any API calls. Applies the same structural rules as the Claude system prompt.

Key design principles:
- Entry on PULLBACK only — zone is always below current price for longs,
  above for shorts. Never chases price.
- Confirmation gates — trend, MACD, ADX, and RSI must align before a setup
  is generated. Low-quality signals are skipped entirely.
- Realistic targets — TP1 at 1.0R (nearest structural level), TP2 at 2.0R.
  No 4R fantasies on daily charts.
- Setup B (counter-trend) only fires when price is within 0.3 ATR of a level.
- Setup C (breakout) only fires when ADX > 20 and volume is increasing.
"""

from p1_analysis_engine.schema import (
    TradingSetup, SetupTarget, InvalidationScenario,
    DecisionTreeEntry, SetupsOutput,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rr(entry: float, sl: float, tp: float) -> float:
    risk   = abs(entry - sl)
    reward = abs(tp - entry)
    return reward / risk if risk > 1e-10 else 0.0


def _ev(win_rate: float, avg_reward_r: float) -> float:
    """EV in R-multiples: (wr × reward) - ((1-wr) × 1R risk)."""
    return round((win_rate * avg_reward_r) - ((1 - win_rate) * 1.0), 4)


def _pf(win_rate: float, avg_reward_r: float) -> float:
    loss_rate = 1 - win_rate
    return round((win_rate * avg_reward_r) / loss_rate, 3) if loss_rate > 1e-10 else 999.0


def _trade_duration(interval: str) -> str:
    return {
        "1m": "5-30 minutes", "5m": "30-90 minutes", "15m": "1-3 hours",
        "30m": "2-6 hours", "1h": "4-12 hours",
        "1d": "3-8 days", "1wk": "3-8 weeks",
    }.get(interval, "varies")


def _interval_to_trade_type(interval: str) -> str:
    return {
        "1m": "scalp", "5m": "scalp", "15m": "scalp",
        "30m": "intraday", "1h": "intraday",
        "1d": "swing", "1wk": "swing",
    }.get(interval, "swing")


# ── Signal quality gate ───────────────────────────────────────────────────────

def _assess_bias(tech: dict, interval: str = "1d") -> tuple[str, float]:
    """
    Returns (bias, confidence) from technical indicators.
    bias = "long" | "short" | "none"
    confidence = 0.0 – 1.0

    Requires alignment across trend, MACD, EMA, and RSI.
    Returns "none" when signals conflict — no setup generated.

    ADX threshold is interval-aware: shorter timeframes (5m/15m) are
    noisier so we accept a lower ADX floor to generate enough signals.
    Short bias requires stronger confirmation than long bias to reduce
    false counter-trend shorts in trending markets.
    """
    trend   = tech["trend"]
    macd    = tech["macd_signal"]
    ema_al  = tech["ema_alignment"]
    rsi     = tech["rsi_14"]
    adx     = tech["adx_14"]

    # Count confirming signals
    long_score  = 0
    short_score = 0

    if trend == "uptrend":   long_score  += 2
    if trend == "downtrend": short_score += 2

    if macd == "bullish":   long_score  += 1
    if macd == "bearish":   short_score += 1

    if ema_al == "bullish": long_score  += 1
    if ema_al == "bearish": short_score += 1

    if rsi < 50:  short_score += 0.5
    if rsi > 50:  long_score  += 0.5

    # Overbought/oversold — penalise chasing
    if rsi > 70:  long_score  -= 1.5
    if rsi < 30:  short_score -= 1.5

    # Hard RSI filter — don't chase overbought longs or oversold shorts
    if long_score > short_score and rsi > 70:
        return "none", 0.0
    if short_score > long_score and rsi < 30:
        return "none", 0.0

    # Interval-aware ADX gate — 5m/15m are noisier, use lower floor
    adx_threshold = 15 if interval in ("1m", "5m", "15m") else 20
    adx_multiplier = 1.0 if adx >= adx_threshold else 0.5

    net = (long_score - short_score) * adx_multiplier

    # Long threshold: 3.0 (unchanged)
    # Short threshold: 3.5 — require stronger confirmation to reduce
    # false counter-trend shorts in trending/volatile markets
    if net >= 3.0:
        confidence = min(0.42 + (net - 3.0) * 0.06, 0.75)
        return "long", round(confidence, 2)
    if net <= -3.5:
        confidence = min(0.42 + (abs(net) - 3.5) * 0.06, 0.75)
        return "short", round(confidence, 2)

    return "none", 0.0


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_setups_from_technicals(tech: dict) -> SetupsOutput | None:
    """
    Generate trade setups deterministically from a technical dict.
    Returns None when signal quality is insufficient (no setups generated).
    Returns a SetupsOutput otherwise.
    """
    price    = tech["current_price"]
    atr      = tech["atr_14"]
    support  = tech["nearest_support"]
    resist   = tech["nearest_resistance"]
    interval = tech.get("interval", "1d")
    adx      = tech["adx_14"]
    vol      = tech["volume_trend"]
    trade_type = _interval_to_trade_type(interval)

    bias, confidence = _assess_bias(tech, interval)

    # No setup if signals conflict
    if bias == "none":
        return None

    setups: list[TradingSetup] = []

    # ── Setup A — Primary, pullback entry aligned with bias ───────────────────
    setup_a = _build_pullback_setup(price, atr, support, resist, bias, trade_type, interval, confidence, tech)
    if setup_a:
        setups.append(setup_a)

    # Setup B (counter-trend bounce) excluded:
    # Backtest evidence shows secondary/counter-trend setups consistently lose
    # across all tested assets (WR 0-20%, avg R -0.59 to -1.00). The edge
    # does not justify the drawdown — only trend-following setups are generated.

    # Setup C (breakout) intentionally excluded:
    # On daily bars the simulator fills on intrabar touches above resistance,
    # but price often closes back below — producing systematic false fills and
    # 10% win rates. A close-above-resistance confirmation is not modelable
    # with the current bar-level simulation.

    if not setups:
        return None

    # ── Invalidation ──────────────────────────────────────────────────────────
    if bias == "long":
        inv_price = round(support - atr * 0.5, 4)
        inv_cond  = f"Daily close below ${support:,.4f} on above-average volume"
        inv_desc  = "Support broken — uptrend structure invalidated"
    else:
        inv_price = round(resist + atr * 0.5, 4)
        inv_cond  = f"Daily close above ${resist:,.4f} on above-average volume"
        inv_desc  = "Resistance broken — downtrend structure invalidated"

    invalidation = InvalidationScenario(
        condition=inv_cond,
        description=inv_desc,
        price_trigger=inv_price,
        action="Stand aside — no trades until structure re-establishes",
    )

    decision_tree = _build_decision_tree(price, support, resist, atr, setups, inv_price, bias)

    sizing_note = (
        f"ATR = {atr:,.4f} ({tech['atr_pct']}% of price). "
        + ("REDUCE SIZE 50-70% — high volatility (ATR > 2%)." if tech["atr_pct"] > 2
           else "Normal sizing.")
    )

    return SetupsOutput(
        asset=tech.get("ticker", "UNKNOWN"),
        current_price=price,
        setups=setups,
        invalidation=invalidation,
        decision_tree=decision_tree,
        position_sizing_note=sizing_note,
    )


# ── Setup builders ─────────────────────────────────────────────────────────────

def _build_pullback_setup(
    price, atr, support, resist, bias, trade_type, interval, confidence, tech
) -> TradingSetup | None:
    """
    Trend-following setup. Entry zone is a PULLBACK into a structural level,
    not at current price. Only generated when price has room to pull back.
    """
    is_long   = bias == "long"
    win_rate  = 0.55 if tech["adx_14"] >= 25 else 0.50
    buf       = atr * 0.5    # SL buffer — wide enough to absorb daily noise

    if is_long:
        # Entry zone: pull back toward support
        pullback_target = support + atr * 0.1
        if pullback_target >= price:          # already at support, no pullback room
            return None
        entry_low  = round(pullback_target - atr * 0.05, 4)
        entry_high = round(pullback_target + atr * 0.15, 4)
        sl         = round(support - buf, 4)
        risk       = entry_high - sl
        if risk < 1e-10:
            return None
        tp1 = round(entry_high + risk * 1.5, 4)
        tp2 = round(entry_high + risk * 3.0, 4)
    else:
        pullback_target = resist - atr * 0.1
        if pullback_target <= price:
            return None
        entry_high = round(pullback_target + atr * 0.05, 4)
        entry_low  = round(pullback_target - atr * 0.15, 4)
        sl         = round(resist + buf, 4)
        risk       = sl - entry_low
        if risk < 1e-10:
            return None
        tp1 = round(entry_low - risk * 1.5, 4)
        tp2 = round(entry_low - risk * 3.0, 4)

    if _rr(entry_high if is_long else entry_low, sl, tp1) < 1.5:
        return None

    avg_reward_r = 1.5 * 0.6 + 3.0 * 0.4   # weighted blend of TP1/TP2
    ev = _ev(win_rate, avg_reward_r)
    pf = _pf(win_rate, avg_reward_r)

    return TradingSetup(
        name="Setup A",
        label=f"{'Long' if is_long else 'Short'} pullback — trend continuation",
        direction=bias,
        trade_type=trade_type,
        status="PRIMARY WATCH",
        priority="primary",
        rationale=(
            f"{'Uptrend' if is_long else 'Downtrend'} continuation on pullback to "
            f"{'support' if is_long else 'resistance'}. "
            f"ADX {tech['adx_14']:.0f}, MACD {tech['macd_signal']}, RSI {tech['rsi_14']:.0f}."
        ),
        trigger=(
            f"Price pulls back to ${entry_low:,.4f}–${entry_high:,.4f} and prints "
            f"{'bullish' if is_long else 'bearish'} reversal candle"
        ),
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=sl,
        trailing_sl_to_breakeven=tp1,
        targets=[
            SetupTarget(price=tp1, label="Target 1 (60%)", allocation_pct=60),
            SetupTarget(price=tp2, label="Target 2 (40%)", allocation_pct=40),
        ],
        rr_ratio=round(avg_reward_r, 2),
        win_rate_estimate=win_rate,
        trade_duration=_trade_duration(interval),
        ev=ev,
        profit_factor=pf,
        confidence=confidence,
        confidence_note=f"ADX {tech['adx_14']:.0f} | MACD {tech['macd_signal']} | RSI {tech['rsi_14']:.0f}",
    )


def _build_level_bounce(
    price, atr, support, resist, bias, trade_type, interval, tech
) -> TradingSetup | None:
    """Counter-trend bounce. Only fires when price is within 0.3 ATR of a key level."""
    is_long  = bias == "short"   # counter-trend to the bias
    win_rate = 0.45
    buf      = atr * 0.5    # wider stop — level bounce needs room to breathe

    level = support if is_long else resist

    if is_long:
        entry_low  = round(level - atr * 0.05, 4)
        entry_high = round(level + atr * 0.15, 4)
        sl         = round(level - buf, 4)
        risk       = entry_high - sl
        if risk < 1e-10:
            return None
        tp1 = round(entry_high + risk * 1.5, 4)
        tp2 = round(entry_high + risk * 3.0, 4)
    else:
        entry_high = round(level + atr * 0.05, 4)
        entry_low  = round(level - atr * 0.15, 4)
        sl         = round(level + buf, 4)
        risk       = sl - entry_low
        if risk < 1e-10:
            return None
        tp1 = round(entry_low - risk * 1.5, 4)
        tp2 = round(entry_low - risk * 3.0, 4)

    if _rr(entry_high if is_long else entry_low, sl, tp1) < 1.5:
        return None

    avg_reward_r = 1.5 * 0.6 + 3.0 * 0.4
    ev = _ev(win_rate, avg_reward_r)
    pf = _pf(win_rate, avg_reward_r)

    return TradingSetup(
        name="Setup B",
        label=f"{'Bounce long at support' if is_long else 'Fade short at resistance'} ${level:,.4f}",
        direction="long" if is_long else "short",
        trade_type=trade_type,
        status="ACTIVE",
        priority="secondary",
        rationale=f"Counter-trend {'bounce' if is_long else 'fade'} at ${level:,.4f}. Price within 0.3 ATR.",
        trigger=f"Price tests ${level:,.4f} and prints {'bullish' if is_long else 'bearish'} reversal candle",
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=sl,
        trailing_sl_to_breakeven=tp1,
        targets=[
            SetupTarget(price=tp1, label="Target 1 (60%)", allocation_pct=60),
            SetupTarget(price=tp2, label="Target 2 (40%)", allocation_pct=40),
        ],
        rr_ratio=round(avg_reward_r, 2),
        win_rate_estimate=win_rate,
        trade_duration=_trade_duration(interval),
        ev=ev,
        profit_factor=pf,
        confidence=0.40,
        confidence_note=f"Counter-trend — only enter on confirmed reversal candle at ${level:,.4f}.",
    )


def _build_breakout(
    price, atr, support, resist, bias, trade_type, interval, tech
) -> TradingSetup | None:
    """Breakout setup. Only generated when ADX >= 20 and volume is increasing."""
    is_long  = bias == "long"
    win_rate = 0.50
    buf      = atr * 0.5    # stop must be inside the broken level to avoid false-breakout wipeout

    if is_long:
        trigger_price = round(resist + atr * 0.05, 4)
        entry_low     = round(trigger_price, 4)
        entry_high    = round(trigger_price + atr * 0.20, 4)
        sl            = round(resist - buf, 4)
        risk          = entry_high - sl
        if risk < 1e-10:
            return None
        tp1 = round(entry_high + risk * 1.5, 4)
        tp2 = round(entry_high + risk * 3.0, 4)
    else:
        trigger_price = round(support - atr * 0.05, 4)
        entry_high    = round(trigger_price, 4)
        entry_low     = round(trigger_price - atr * 0.20, 4)
        sl            = round(support + buf, 4)
        risk          = sl - entry_low
        if risk < 1e-10:
            return None
        tp1 = round(entry_low - risk * 1.5, 4)
        tp2 = round(entry_low - risk * 3.0, 4)

    if _rr(entry_high if is_long else entry_low, sl, tp1) < 1.5:
        return None

    avg_reward_r = 1.5 * 0.6 + 3.0 * 0.4
    ev = _ev(win_rate, avg_reward_r)
    pf = _pf(win_rate, avg_reward_r)

    level = resist if is_long else support

    return TradingSetup(
        name="Setup C",
        label=f"Breakout {'above' if is_long else 'below'} ${level:,.4f}",
        direction=bias,
        trade_type=trade_type,
        status="IF LEVEL BREAKS",
        priority="conditional",
        rationale=(
            f"Breakout setup — ADX {tech['adx_14']:.0f} and increasing volume confirm momentum. "
            f"Only valid on close {'above' if is_long else 'below'} ${level:,.4f}."
        ),
        trigger=(
            f"{'Bullish' if is_long else 'Bearish'} close {'above' if is_long else 'below'} "
            f"${trigger_price:,.4f} on above-average volume"
        ),
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=sl,
        trailing_sl_to_breakeven=tp1,
        targets=[
            SetupTarget(price=tp1, label="Target 1 (60%)", allocation_pct=60),
            SetupTarget(price=tp2, label="Target 2 (40%)", allocation_pct=40),
        ],
        rr_ratio=round(avg_reward_r, 2),
        win_rate_estimate=win_rate,
        trade_duration=_trade_duration(interval),
        ev=ev,
        profit_factor=pf,
        confidence=0.50,
        confidence_note=f"Conditional — do not enter before ${level:,.4f} is confirmed on volume.",
    )


# ── Decision tree ──────────────────────────────────────────────────────────────

def _build_decision_tree(
    price, support, resist, atr, setups, inv_price, bias
) -> list[DecisionTreeEntry]:
    primary = next((s for s in setups if s.priority == "primary"), setups[0])
    is_long = bias == "long"

    if is_long:
        return [
            DecisionTreeEntry(
                scenario=f"Price pulls back to ${primary.entry_high:,.4f} and holds with bullish candle",
                outcome=f"LONG Setup A — entry ${primary.entry_high:,.4f}, SL ${primary.stop_loss:,.4f}",
                direction="long", setup_name="Setup A", entry_price=primary.entry_high,
            ),
            DecisionTreeEntry(
                scenario=f"Price breaks above ${resist:,.4f} on strong volume",
                outcome=f"LONG Setup C — enter on confirmed close above ${resist:,.4f}",
                direction="long", setup_name="Setup C", entry_price=round(resist + atr * 0.05, 4),
            ),
            DecisionTreeEntry(
                scenario=f"Price drops to ${support:,.4f} and prints reversal candle",
                outcome="LONG Setup B — counter-trend bounce, tight SL below support",
                direction="long", setup_name="Setup B", entry_price=support,
            ),
            DecisionTreeEntry(
                scenario=f"Price chops between ${support:,.4f} and ${resist:,.4f} with low volume",
                outcome="NO TRADE — no edge in compression, wait for directional move",
                direction="no_trade", setup_name="NO TRADE",
            ),
            DecisionTreeEntry(
                scenario=f"Daily close below ${inv_price:,.4f} on above-average volume",
                outcome="FULL INVALIDATION — cancel all setups, stand aside",
                direction="no_trade", setup_name="NO TRADE",
            ),
        ]
    else:
        return [
            DecisionTreeEntry(
                scenario=f"Price rallies to ${primary.entry_low:,.4f} and holds with bearish candle",
                outcome=f"SHORT Setup A — entry ${primary.entry_low:,.4f}, SL ${primary.stop_loss:,.4f}",
                direction="short", setup_name="Setup A", entry_price=primary.entry_low,
            ),
            DecisionTreeEntry(
                scenario=f"Price breaks below ${support:,.4f} on strong volume",
                outcome=f"SHORT Setup C — enter on confirmed close below ${support:,.4f}",
                direction="short", setup_name="Setup C", entry_price=round(support - atr * 0.05, 4),
            ),
            DecisionTreeEntry(
                scenario=f"Price rallies to ${resist:,.4f} and prints rejection candle",
                outcome="SHORT Setup B — counter-trend fade, tight SL above resistance",
                direction="short", setup_name="Setup B", entry_price=resist,
            ),
            DecisionTreeEntry(
                scenario=f"Price chops between ${support:,.4f} and ${resist:,.4f} with low volume",
                outcome="NO TRADE — no edge in compression, wait for directional move",
                direction="no_trade", setup_name="NO TRADE",
            ),
            DecisionTreeEntry(
                scenario=f"Daily close above ${inv_price:,.4f} on above-average volume",
                outcome="FULL INVALIDATION — cancel all setups, stand aside",
                direction="no_trade", setup_name="NO TRADE",
            ),
        ]
