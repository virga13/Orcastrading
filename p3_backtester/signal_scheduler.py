"""
signal_scheduler.py — determines which bar indices in the OHLCV DataFrame
should trigger a P1 signal generation call.
"""
import pandas as pd
from p3_backtester.bar_slicer import MIN_WARMUP_BARS


def get_signal_indices(
    df: pd.DataFrame,
    mode: str,
    first_valid_index: int,
    every_n_bars: int = 10,
) -> list[int]:
    """
    Returns a sorted list of bar indices at which to generate P1 signals.

    Parameters
    ----------
    df               : full OHLCV DataFrame (index = DatetimeIndex)
    mode             : "every-n-bars" | "session-open" | "key-level-touch"
    first_valid_index: first index on or after the backtest start_date (from split_backtest_window)
    every_n_bars     : spacing for every-n-bars mode

    The minimum valid index is max(MIN_WARMUP_BARS, first_valid_index) to ensure
    EMA200 and all other indicators have enough history.
    """
    min_index = max(MIN_WARMUP_BARS, first_valid_index)

    if mode == "every-n-bars":
        return _every_n(df, min_index, every_n_bars)
    elif mode == "session-open":
        return _session_open(df, min_index)
    elif mode == "key-level-touch":
        return _key_level_touch(df, min_index)
    else:
        raise ValueError(f"Unknown signal mode: {mode}")


def _every_n(df: pd.DataFrame, min_index: int, n: int) -> list[int]:
    indices = []
    i = min_index
    while i < len(df) - 1:  # -1: need at least one forward bar for simulation
        indices.append(i)
        i += n
    return indices


def _session_open(df: pd.DataFrame, min_index: int) -> list[int]:
    """
    Emits a signal at the first bar of each trading day (date change in the index).
    For crypto (24/7) this is midnight UTC; for equities this is the first bar
    after the overnight gap.
    """
    indices = []
    prev_date = None
    for i in range(min_index, len(df) - 1):
        bar_date = df.index[i].date()
        if bar_date != prev_date:
            indices.append(i)
            prev_date = bar_date
    return indices


def _key_level_touch(df: pd.DataFrame, min_index: int) -> list[int]:
    """
    Emits a signal when price crosses a rolling 20-bar support or resistance level.
    Uses a 3-bar cooldown after each trigger to avoid signal clustering.
    """
    indices = []
    cooldown = 0
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    for i in range(min_index, len(df) - 1):
        if cooldown > 0:
            cooldown -= 1
            continue

        # Rolling 20-bar high/low as S/R (computed only on bars up to i, no lookahead)
        window = close.iloc[max(0, i - 20):i]
        if len(window) < 20:
            continue

        r_level = float(high.iloc[max(0, i - 20):i].max())
        s_level = float(low.iloc[max(0, i - 20):i].min())
        curr    = float(close.iloc[i])
        prev    = float(close.iloc[i - 1])

        touched_resistance = prev < r_level <= curr
        touched_support    = prev > s_level >= curr

        if touched_resistance or touched_support:
            indices.append(i)
            cooldown = 3

    return indices
