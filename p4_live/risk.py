"""
p4_live/risk.py — Position sizing, circuit breaker, and strategy health.

R (Risk unit) explained
-----------------------
R is account-size agnostic: 1R = whatever dollar amount you choose to risk per trade.
  If you risk 1% of a $500 account → 1R = $5
  If you risk 1% of a $1,000 account → 1R = $10

Why R matters:
  +2.5R means you made 2.5× your risk, regardless of account size.
  -7.0R max drawdown → at 1% risk per trade you'd lose 7% of capital in the worst streak.
  This lets the backtester's numbers apply equally to a $100 or $100,000 account.

Position sizing formula (apply to any signal):
  dollar_risk  = account_size × risk_pct
  point_risk   = entry_high - stop_loss       (printed in every alert)
  position_size = dollar_risk / point_risk

  Example — Gold, $500 account, 0.8% risk:
    dollar_risk   = $500 × 0.008 = $4.00
    point_risk    = $4,650 - $4,480 = $170
    position_size = $4.00 / $170 ≈ 0.024 oz equivalent
"""
from __future__ import annotations
import math
from datetime import date

# ── Correlation groups — assets whose signals should not be stacked at full size ─
# Rationale: taking SPX + DOW + DAX simultaneously means 3× exposure to one macro move.
# Apply a Kelly multiplier: 1 correlated open → 50%, 2+ open → 25%.
_CORRELATION_GROUPS: list[list[str]] = [
    ["^GSPC", "^DJI", "^GDAXI"],   # equity indices: ρ ≈ 0.75–0.95
    ["GC=F", "SI=F"],               # precious metals: ρ ≈ 0.80
]


def correlation_penalty(ticker: str, open_positions: list[str]) -> float:
    """
    Return a Kelly multiplier based on how many correlated tickers are already open.

    open_positions: list of tickers with an open (pending or filled) trade.
    """
    for group in _CORRELATION_GROUPS:
        if ticker not in group:
            continue
        n_correlated = sum(1 for t in open_positions if t in group and t != ticker)
        if n_correlated == 1:
            return 0.5
        if n_correlated >= 2:
            return 0.25
        return 1.0   # no correlated positions open
    return 1.0   # not in any correlation group


def vix_multiplier(vix_level: float | None) -> float:
    """
    Hard position size multiplier based on VIX regime.

    VIX < 15  → 1.0   (calm — full size)
    15–25     → 0.75  (caution — reduce 25%)
    25–35     → 0.50  (stress — half size)
    > 35      → 0.0   (crisis — skip trend strategies entirely)

    Returns 1.0 if vix_level is None (no data).
    """
    if vix_level is None:
        return 1.0
    if vix_level < 15:
        return 1.0
    if vix_level < 25:
        return 0.75
    if vix_level < 35:
        return 0.50
    return 0.0


# ── Backtest baselines per (ticker, strategy) ─────────────────────────────────
# Kelly fractions are at 25% of full Kelly (standard conservative sizing).
# Source: walk-forward test results stored in p4_live/scanner.py ASSETS.
_BASELINES: dict[tuple[str, str], dict] = {
    ("^GSPC",  "mtf_trend"): {"kelly_25pct": 0.063, "max_drawdown_r": 7.0,  "win_rate": 0.429},
    ("^DJI",   "mtf_trend"): {"kelly_25pct": 0.077, "max_drawdown_r": 5.0,  "win_rate": 0.513},
    ("BTC-USD","mtf_trend"): {"kelly_25pct": 0.000, "max_drawdown_r": 16.3, "win_rate": 0.250},
    ("GC=F",   "mtf_trend"): {"kelly_25pct": 0.016, "max_drawdown_r": 9.0,  "win_rate": 0.305},
    ("SI=F",   "mtf_trend"): {"kelly_25pct": 0.004, "max_drawdown_r": 14.9, "win_rate": 0.279},
    ("^GDAXI", "mtf_trend"): {"kelly_25pct": 0.005, "max_drawdown_r": 17.0, "win_rate": 0.286},
}

# Conservative default for strategies with no validated baseline yet
_DEFAULT_KELLY = 0.01   # 1% risk — safe starting point


def get_baseline(ticker: str, strategy: str) -> dict | None:
    return _BASELINES.get((ticker, strategy))


# ── Position sizing ───────────────────────────────────────────────────────────

def position_sizing(
    signal: dict,
    open_positions: list[str] | None = None,
    vix_level: float | None = None,
) -> dict:
    """
    Compute risk percentage and position sizing guidance for a signal.

    open_positions: list of tickers with open trades (used for correlation cap).
    vix_level:      current VIX reading (applies hard size multiplier).

    Returns:
        kelly_pct       — full 25%-Kelly recommendation (0.0 if no baseline)
        suggested_pct   — final recommendation after confidence/VIX/correlation adjustments
        point_risk      — entry_high - stop_loss (the divisor for position sizing)
        skip            — True if signal should be skipped
        sizing_note     — human-readable explanation
        sizing_examples — list of dicts [{"account": 500, "dollar_risk": ..., ...}]
    """
    ticker     = signal.get("ticker", "")
    strategy   = signal.get("strategy", "")
    confidence = signal.get("confidence") or 0.75
    entry_high = signal.get("entry_high") or 0.0
    stop_loss  = signal.get("stop_loss") or 0.0
    point_risk = round(entry_high - stop_loss, 4) if entry_high and stop_loss else None

    baseline   = get_baseline(ticker, strategy)
    kelly_pct  = baseline["kelly_25pct"] if baseline else _DEFAULT_KELLY

    # Kelly=0 means backtest says don't trade this asset/strategy
    if kelly_pct == 0.0:
        return {
            "kelly_pct":       0.0,
            "suggested_pct":   0.0,
            "point_risk":      point_risk,
            "skip":            True,
            "sizing_note":     "Kelly=0% - backtest does not support trading this asset with this strategy",
            "sizing_examples": [],
        }

    # VIX hard multiplier — checked first (macro regime override)
    vix_mult = vix_multiplier(vix_level)
    if vix_mult == 0.0:
        return {
            "kelly_pct":       kelly_pct,
            "suggested_pct":   0.0,
            "point_risk":      point_risk,
            "skip":            True,
            "sizing_note":     f"VIX={vix_level:.0f} > 35 - crisis regime, trend strategies paused",
            "sizing_examples": [],
        }

    # Confidence gate: full / half / skip
    if confidence >= 0.85:
        conf_mult = 1.0
        gate_note = "Full Kelly - all conditions clearly met"
    elif confidence >= 0.70:
        conf_mult = 0.5
        gate_note = "Half Kelly - borderline signal"
    else:
        return {
            "kelly_pct":       kelly_pct,
            "suggested_pct":   0.0,
            "point_risk":      point_risk,
            "skip":            True,
            "sizing_note":     f"Skipped - confidence {confidence:.0%} below 70% threshold",
            "sizing_examples": [],
        }

    # Correlation cap — scale down when correlated positions are already open
    corr_mult = correlation_penalty(ticker, open_positions or [])

    suggested_pct = round(kelly_pct * conf_mult * vix_mult * corr_mult, 4)

    # Build human-readable note with all applied adjustments
    adjustments = [gate_note]
    if vix_mult < 1.0:
        adjustments.append(f"VIX={vix_level:.0f} ({vix_mult:.0%} size)")
    if corr_mult < 1.0:
        adjustments.append(f"Correlated open positions ({corr_mult:.0%} size)")

    note = (
        f"Kelly (25%): {kelly_pct:.1%}  |  "
        + "  |  ".join(adjustments)
        + f"  |  Suggested: {suggested_pct:.1%}"
    )

    # Build sizing examples for common account sizes
    examples = []
    if point_risk and point_risk > 0:
        for acct in [100, 500, 1_000, 5_000, 10_000]:
            dollar_risk = round(acct * suggested_pct, 2)
            position_sz = round(dollar_risk / point_risk, 4) if dollar_risk > 0 else 0
            examples.append({
                "account":     acct,
                "dollar_risk": dollar_risk,
                "position":    position_sz,
            })

    return {
        "kelly_pct":       kelly_pct,
        "suggested_pct":   suggested_pct,
        "point_risk":      point_risk,
        "skip":            False,
        "sizing_note":     note,
        "sizing_examples": examples,
    }


# ── Drawdown circuit breaker ──────────────────────────────────────────────────

def check_circuit_breaker(ticker: str, strategy: str) -> tuple[bool, float, float]:
    """
    Compare live running drawdown vs backtest max drawdown.

    Returns:
        breached        — True if live DD >= backtest max DD
        live_dd_r       — current peak-to-trough drawdown in R
        max_allowed_r   — backtest max drawdown (the threshold)

    Edge cases:
      - No baseline → never breaches (returns False, 0.0, inf)
      - No closed trades → no data, returns False
      - All wins so far → drawdown = 0
    """
    from p4_live.journal import get_closed_trades

    baseline = get_baseline(ticker, strategy)
    if not baseline:
        return False, 0.0, math.inf

    max_allowed = baseline["max_drawdown_r"]

    trades = [
        t for t in get_closed_trades()
        if t["ticker"] == ticker
        and t["strategy"] == strategy
        and t["pnl_r"] is not None
    ]

    if not trades:
        return False, 0.0, max_allowed

    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t["pnl_r"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    breached = max_dd >= max_allowed
    return breached, round(max_dd, 2), max_allowed


# ── Strategy degradation detection ───────────────────────────────────────────

def check_degradation(ticker: str, strategy: str, min_trades: int = 20) -> dict:
    """
    Binomial significance test: is the live win rate consistent with the backtest?

    Uses one-tailed normal approximation to the binomial (valid for n >= 20).
    Returns a dict with:
        enough_data   — bool, False if fewer than min_trades closed
        n             — closed trade count
        live_wr       — live win rate
        baseline_wr   — backtest win rate
        z_score       — how many std devs below baseline
        p_value       — probability of observing this WR or lower if baseline holds
        degraded      — True if p_value < 0.05 and live_wr < baseline_wr

    Edge cases:
      - No baseline → returns {enough_data: False, reason: "no baseline"}
      - n < min_trades → returns {enough_data: False}
      - all trades lost → z_score is strongly negative, p_value near 0
    """
    from p4_live.journal import get_closed_trades

    baseline = get_baseline(ticker, strategy)
    if not baseline:
        return {"enough_data": False, "reason": "no validated baseline for this strategy"}

    baseline_wr = baseline["win_rate"]

    trades = [
        t for t in get_closed_trades()
        if t["ticker"] == ticker
        and t["strategy"] == strategy
        and t["pnl_r"] is not None
    ]

    n = len(trades)
    if n < min_trades:
        return {
            "enough_data": False,
            "n":           n,
            "min_trades":  min_trades,
            "reason":      f"need {min_trades} trades, have {n}",
        }

    wins     = sum(1 for t in trades if t["pnl_r"] > 0)
    live_wr  = wins / n

    # Normal approximation to binomial
    std     = math.sqrt(baseline_wr * (1 - baseline_wr) / n)
    z_score = (live_wr - baseline_wr) / std if std > 0 else 0.0

    # One-tailed p-value: P(observing live_wr or lower | true_wr = baseline_wr)
    p_value = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

    degraded = p_value < 0.05 and live_wr < baseline_wr

    return {
        "enough_data":  True,
        "n":            n,
        "live_wr":      round(live_wr, 3),
        "baseline_wr":  round(baseline_wr, 3),
        "z_score":      round(z_score, 2),
        "p_value":      round(p_value, 3),
        "degraded":     degraded,
    }


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def monte_carlo(pnl_r_list: list[float], n_simulations: int = 10_000) -> dict:
    """
    Reshuffle the observed trade sequence N times to estimate the distribution
    of outcomes. Answers the question: "was our backtest lucky or typical?"

    Returns:
        n_trades          — number of trades in the sequence
        n_simulations     — simulations run
        final_equity      — percentile distribution of final equity (R)
        max_drawdown      — percentile distribution of max drawdown (R)
        p_ruin            — probability of drawdown exceeding 2× observed max DD
        median_final_r    — median final equity across simulations
        worst5_dd_r       — 5th percentile max drawdown (realistic worst case)
        best95_final_r    — 95th percentile final equity (realistic best case)

    Edge cases:
      - Empty list → returns empty result
      - Single trade → returns trivial result
      - All identical pnl_r values → all simulations identical
    """
    import random

    if not pnl_r_list:
        return {"n_trades": 0, "n_simulations": 0, "error": "no trades provided"}

    n = len(pnl_r_list)
    observed_max_dd = _compute_max_dd(pnl_r_list)

    finals = []
    drawdowns = []

    rng = random.Random(42)  # fixed seed for reproducibility
    for _ in range(n_simulations):
        shuffled = pnl_r_list[:]
        rng.shuffle(shuffled)
        finals.append(sum(shuffled))
        drawdowns.append(_compute_max_dd(shuffled))

    finals.sort()
    drawdowns.sort()

    ruin_threshold = observed_max_dd * 2.0
    p_ruin = sum(1 for dd in drawdowns if dd >= ruin_threshold) / n_simulations

    def pct(lst, p):
        idx = max(0, min(len(lst) - 1, int(p * len(lst))))
        return round(lst[idx], 2)

    return {
        "n_trades":        n,
        "n_simulations":   n_simulations,
        "observed_final_r": round(sum(pnl_r_list), 2),
        "observed_max_dd":  round(observed_max_dd, 2),
        "median_final_r":   pct(finals, 0.50),
        "worst5_final_r":   pct(finals, 0.05),
        "best95_final_r":   pct(finals, 0.95),
        "median_max_dd":    pct(drawdowns, 0.50),
        "worst5_dd_r":      pct(drawdowns, 0.95),   # 95th pct of DD = worst realistic DD
        "best5_dd_r":       pct(drawdowns, 0.05),   # lucky scenario
        "p_ruin":           round(p_ruin, 3),
    }


def _compute_max_dd(pnl_r_list: list[float]) -> float:
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for r in pnl_r_list:
        equity += r
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd
