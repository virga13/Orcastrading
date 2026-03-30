"""
walk_forward.py — walk-forward validation and Kelly position sizing.

Walk-forward splits the backtest window into three non-overlapping periods:
  Train      (60%) — optimize strategy parameters on this data
  Validation (20%) — tune parameters, check for overfitting
  Test       (20%) — held-out, never touched until final evaluation

The test window result is the only number you should report to stakeholders.
If train >> test in performance, the strategy is overfit.

Kelly criterion:
  Full Kelly: k = (p*b - q) / b
    where p = win rate, q = 1-p, b = avg_win_R / avg_loss_R
  Use 25% Kelly as the conservative starting fraction.
  Kelly > 0.5 is capped at 0.5 (never risk more than 50% of bankroll).
"""

from __future__ import annotations
import pandas as pd
from p3_backtester.schema import RunStats


# ── Walk-forward split ────────────────────────────────────────────────────────

def compute_walk_forward_dates(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    train_pct: float = 0.60,
    val_pct:   float = 0.20,
) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    """
    Split the backtest window into train / validation / test date ranges.

    Parameters
    ----------
    df         : full OHLCV DataFrame (DatetimeIndex)
    start_date : backtest start (ISO date string)
    end_date   : backtest end   (ISO date string)
    train_pct  : fraction of bars for training (default 0.60)
    val_pct    : fraction of bars for validation (default 0.20)
                 test_pct = 1 - train_pct - val_pct

    Returns
    -------
    (train_range, val_range, test_range)
    each range is (start_date_str, end_date_str)
    """
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)

    mask = (df.index >= start_ts) & (df.index <= end_ts)
    window_bars = df[mask]

    n = len(window_bars)
    if n < 30:
        raise ValueError(f"Insufficient bars ({n}) in window for walk-forward split")

    train_end_pos = int(n * train_pct)
    val_end_pos   = int(n * (train_pct + val_pct))

    train_end_date = window_bars.index[train_end_pos - 1].strftime("%Y-%m-%d")
    val_end_date   = window_bars.index[val_end_pos  - 1].strftime("%Y-%m-%d")

    # val starts day after train ends
    val_start_date  = window_bars.index[train_end_pos].strftime("%Y-%m-%d")
    test_start_date = window_bars.index[val_end_pos].strftime("%Y-%m-%d")

    train_range = (start_date,      train_end_date)
    val_range   = (val_start_date,  val_end_date)
    test_range  = (test_start_date, end_date)

    return train_range, val_range, test_range


def format_walk_forward_ranges(
    train: tuple[str, str],
    val:   tuple[str, str],
    test:  tuple[str, str],
) -> str:
    return (
        f"  Train      : {train[0]} → {train[1]}  (60%)\n"
        f"  Validation : {val[0]}   → {val[1]}  (20%)\n"
        f"  Test       : {test[0]}  → {test[1]}  (20%) ← real answer"
    )


# ── Kelly position sizing ─────────────────────────────────────────────────────

def kelly_fraction(
    win_rate:   float,
    avg_win_r:  float,
    avg_loss_r: float = 1.0,
) -> float:
    """
    Full Kelly criterion.

    k = (p*b - q) / b
    where b = avg_win_r / avg_loss_r

    Returns 0.0 if the strategy has negative expectancy.
    """
    if win_rate <= 0 or avg_win_r <= 0:
        return 0.0
    p = win_rate
    q = 1.0 - win_rate
    b = avg_win_r / max(avg_loss_r, 1e-10)
    k = (p * b - q) / b
    return max(0.0, min(k, 0.5))   # cap at 50%


def conservative_kelly(
    win_rate:   float,
    avg_win_r:  float,
    avg_loss_r: float = 1.0,
    fraction:   float = 0.25,
) -> float:
    """
    25% Kelly (conservative starting point).
    Scale up to 50% Kelly once live results confirm the model.
    """
    return round(kelly_fraction(win_rate, avg_win_r, avg_loss_r) * fraction, 4)


def kelly_from_stats(stats: RunStats, kelly_frac: float = 0.25) -> float:
    """Compute conservative Kelly from a RunStats object."""
    filled = stats.total_trades_filled
    if filled == 0:
        return 0.0

    wins   = stats.by_direction.get("long", None)  # placeholder — use overall stats
    # Use overall stats directly
    win_rate  = stats.actual_win_rate
    avg_pnl_r = stats.actual_avg_pnl_r

    # Estimate avg_win and avg_loss from win rate + avg pnl
    # EV = wr * avg_win - (1-wr) * avg_loss = avg_pnl_r
    # Assume avg_loss ≈ 1.0R (stop was hit)
    avg_loss_r = 1.0
    if win_rate > 1e-10:
        avg_win_r = (avg_pnl_r + (1 - win_rate) * avg_loss_r) / win_rate
    else:
        avg_win_r = 0.0

    return conservative_kelly(win_rate, avg_win_r, avg_loss_r, fraction=kelly_frac)


def sizing_recommendation(stats: RunStats) -> str:
    """Return a human-readable position sizing recommendation."""
    k25 = kelly_from_stats(stats, 0.25)
    k50 = kelly_from_stats(stats, 0.50)

    if k25 <= 0:
        return "No edge detected — do not trade this strategy."

    risk_pct_25 = round(k25 * 100, 1)
    risk_pct_50 = round(k50 * 100, 1)

    lines = [
        f"25% Kelly (conservative start): risk {risk_pct_25}% of account per trade",
        f"50% Kelly (after live confirmation): risk {risk_pct_50}% per trade",
        f"Based on: WR={stats.actual_win_rate:.1%}, avgR={stats.actual_avg_pnl_r:+.2f}R, "
        f"PF={stats.actual_profit_factor:.2f}x, n={stats.total_trades_filled} trades",
    ]

    if stats.total_trades_filled < 50:
        lines.append(
            f"WARNING: only {stats.total_trades_filled} trades — "
            "insufficient sample size. Use minimum position size until n >= 100."
        )

    return "\n".join(lines)
