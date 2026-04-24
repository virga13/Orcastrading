"""
p4_live/backfill.py — Replay scanner historically and simulate trade outcomes.

Iterates over every trading day in [start_date, end_date], runs the locked
strategy bar-by-bar (zero lookahead), records signals that fired, then
simulates fills and closes using subsequent OHLCV data — identical logic
to the p3 backtester's trade_simulator.

Outcome mapping (backtester → journal):
  WIN          → win
  PARTIAL_WIN  → partial_win
  LOSS         → loss
  BREAKEVEN    → breakeven
  EXPIRED      → expired (no fill within timeout)
"""
from __future__ import annotations

import pandas as pd
from datetime import date, timedelta
from rich.console import Console
from rich.progress import track

from p3_backtester.market_data import fetch_ohlcv
from p3_backtester.strategies.mtf_trend import MTFTrendStrategy
from p4_live.scanner import _STRATEGY, ASSETS, INTERVAL, _LOOKBACK_DAYS
from p4_live.journal import (
    record_signal, record_expired, record_fill_close_direct,
)

_ENTRY_TIMEOUT = 20   # bars before a pending signal expires
_COST_R        = 0.0  # no transaction cost in forward-test journal

console = Console(legacy_windows=False)


def _simulate_outcome(
    setup,
    signal_bar_idx: int,
    df: pd.DataFrame,
) -> dict | None:
    """
    Simulate fill + outcome for a setup, returning a dict with fill/exit details,
    or None if expired (no fill within timeout).
    Uses the same pessimistic fill logic as the p3 backtester.
    """
    is_long    = setup.direction == "long"
    fill_price = None
    fill_idx   = None
    scan_end   = min(signal_bar_idx + 1 + _ENTRY_TIMEOUT, len(df))

    for i in range(signal_bar_idx + 1, scan_end):
        hi = float(df["High"].iloc[i])
        lo = float(df["Low"].iloc[i])
        if is_long and hi >= setup.entry_low and lo <= setup.entry_high:
            fill_price = min(setup.entry_high, hi)
            fill_idx   = i
            break
        elif not is_long and lo <= setup.entry_high and hi >= setup.entry_low:
            fill_price = max(setup.entry_low, lo)
            fill_idx   = i
            break

    if fill_price is None:
        return None   # expired

    risk     = abs(fill_price - setup.stop_loss)
    eff_sl   = setup.stop_loss
    targets  = sorted(setup.targets, key=lambda t: t.price if is_long else -t.price)
    rem_alloc = 100
    realized  = 0.0
    outcome   = None
    exit_bar  = fill_idx
    exit_price = fill_price

    # trailing SL to breakeven
    be_price = getattr(setup, "trailing_sl_to_breakeven", None)
    be_active = False

    for i in range(fill_idx + 1, len(df)):
        hi = float(df["High"].iloc[i])
        lo = float(df["Low"].iloc[i])

        # activate breakeven SL
        if be_price and not be_active:
            if (is_long and hi >= be_price) or (not is_long and lo <= be_price):
                eff_sl   = fill_price
                be_active = True

        # SL check first (conservative)
        sl_hit = (is_long and lo <= eff_sl) or (not is_long and hi >= eff_sl)
        if sl_hit:
            sl_pnl = (eff_sl - fill_price) * (rem_alloc / 100)
            if not is_long:
                sl_pnl = -sl_pnl
            realized += sl_pnl
            exit_price = eff_sl
            exit_bar   = i
            outcome    = "PARTIAL_WIN" if realized > 0 else ("BREAKEVEN" if abs(realized) < 1e-8 else "LOSS")
            break

        # TP checks
        for t in targets:
            if getattr(t, "_hit", False):
                continue
            tp_hit = (is_long and hi >= t.price) or (not is_long and lo <= t.price)
            if tp_hit:
                pnl = (t.price - fill_price) * (t.allocation_pct / 100)
                if not is_long:
                    pnl = -pnl
                t._hit = True      # type: ignore[attr-defined]
                realized  += pnl
                rem_alloc -= t.allocation_pct
                exit_price = t.price

        if rem_alloc <= 0:
            exit_bar = i
            outcome  = "WIN"
            break

    if outcome is None:
        # End of data — close at last close
        last_close = float(df["Close"].iloc[-1])
        final_pnl  = (last_close - fill_price) * (rem_alloc / 100)
        if not is_long:
            final_pnl = -final_pnl
        realized  += final_pnl
        exit_price = last_close
        exit_bar   = len(df) - 1
        outcome    = "WIN" if realized > 0 else ("BREAKEVEN" if abs(realized) < 1e-8 else "LOSS")

    pnl_r = round(realized / risk, 3) if risk > 1e-10 else 0.0

    _OUTCOME_MAP = {"WIN": "win", "PARTIAL_WIN": "partial_win",
                    "LOSS": "loss", "BREAKEVEN": "breakeven"}

    return {
        "fill_price":   fill_price,
        "fill_date":    df.index[fill_idx].strftime("%Y-%m-%d"),
        "exit_price":   exit_price,
        "exit_date":    df.index[exit_bar].strftime("%Y-%m-%d"),
        "exit_bar_idx": exit_bar,   # index into df (df_forward) — used for stacking guard
        "outcome":      _OUTCOME_MAP.get(outcome, outcome.lower()),
        "pnl_r":        pnl_r,
    }


def backfill(
    start_date: str,
    end_date: str,
    tickers: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """
    Replay the scanner for [start_date, end_date] and populate the journal.
    """
    from p4_live.scanner import ASSET_MAP

    targets = tickers or [a["ticker"] for a in ASSETS]

    for ticker in targets:
        asset = ASSET_MAP[ticker]
        label = asset["label"]
        console.print(f"\n[bold]{label} ({ticker})[/bold]  {start_date} → {end_date}")

        # Fetch full data once (with warmup)
        warmup_start = (
            date.fromisoformat(start_date) - timedelta(days=_LOOKBACK_DAYS)
        ).isoformat()
        try:
            df_full = fetch_ohlcv(ticker, INTERVAL, warmup_start, end_date, source="auto")
        except Exception as e:
            console.print(f"  [red]Data fetch failed:[/red] {e}")
            continue

        # Find bar indices that fall within [start_date, end_date]
        bar_dates  = df_full.index.normalize()
        start_dt   = pd.Timestamp(start_date)
        end_dt     = pd.Timestamp(end_date)
        in_window  = (bar_dates >= start_dt) & (bar_dates <= end_dt)
        window_idx = [i for i, m in enumerate(in_window) if m]

        if not window_idx:
            console.print("  [dim]No bars in range.[/dim]")
            continue

        signals         = 0
        fills           = 0
        expired         = 0
        skipped         = 0
        exit_bar_cutoff = -1   # absolute index in df_full; skip bars up to this

        for bar_i in track(window_idx, description=f"  {label}"):
            # One trade at a time — skip if a previous trade hasn't closed yet
            if bar_i <= exit_bar_cutoff:
                skipped += 1
                continue

            # Slice to simulate "as of close of bar_i" — zero lookahead
            df_slice  = df_full.iloc[: bar_i + 1].copy()
            signal_dt = df_full.index[bar_i].strftime("%Y-%m-%d")

            tech = {
                "ticker":        ticker,
                "current_price": float(df_slice["Close"].iloc[-1]),
                "atr_14":        float(
                    __import__("ta").volatility.AverageTrueRange(
                        df_slice["High"], df_slice["Low"], df_slice["Close"], window=14
                    ).average_true_range().iloc[-1]
                ),
                "atr_pct":       0.0,
                "rsi_14":        float(
                    __import__("ta").momentum.RSIIndicator(
                        df_slice["Close"], window=14
                    ).rsi().iloc[-1]
                ),
                "interval": INTERVAL,
            }

            setups = _STRATEGY.generate_setups(tech, df_slice)
            if setups is None or not setups.setups:
                continue

            setup = setups.setups[0]
            risk  = abs(
                (setup.entry_high if setup.direction == "long" else setup.entry_low)
                - setup.stop_loss
            )

            signal = {
                "fired":       True,
                "signal_date": signal_dt,
                "ticker":      ticker,
                "label":       label,
                "direction":   setup.direction,
                "entry_low":   setup.entry_low,
                "entry_high":  setup.entry_high,
                "stop_loss":   setup.stop_loss,
                "tp1":         setup.targets[0].price,
                "tp2":         setup.targets[1].price if len(setup.targets) > 1 else None,
                "tp1_alloc":   setup.targets[0].allocation_pct,
                "tp2_alloc":   setup.targets[1].allocation_pct if len(setup.targets) > 1 else 0,
                "risk_pts":    round(risk, 4),
                "rr":          setup.rr_ratio,
                "confidence":  setup.confidence,
                "rationale":   setup.rationale,
                "price":       tech["current_price"],
                "atr":         tech["atr_14"],
                "rsi":         tech["rsi_14"],
                "regime":      setup.confidence_note,
            }

            if dry_run:
                signals += 1
                console.print(f"  [dim]DRY RUN {signal_dt} {label}: {setup.direction} "
                               f"entry {setup.entry_low:.0f}-{setup.entry_high:.0f}[/dim]")
                continue

            recorded = record_signal(signal)
            if not recorded:
                continue   # already in journal

            signals += 1

            # Simulate outcome using bars AFTER the signal bar
            df_forward = df_full.iloc[bar_i:]   # includes signal bar + everything after
            result = _simulate_outcome(setup, 0, df_forward)

            if result is None:
                record_expired(signal_dt, ticker)
                expired += 1
                # No fill — lock out for the timeout window
                exit_bar_cutoff = bar_i + _ENTRY_TIMEOUT
            else:
                record_fill_close_direct(
                    signal_date=signal_dt,
                    ticker=ticker,
                    fill_price=result["fill_price"],
                    fill_date=result["fill_date"],
                    outcome=result["outcome"],
                    exit_price=result["exit_price"],
                    exit_date=result["exit_date"],
                    pnl_r=result["pnl_r"],
                    strategy="mtf_trend",
                )
                fills += 1
                # Lock out until this trade's exit bar
                exit_bar_cutoff = bar_i + result["exit_bar_idx"]

        console.print(
            f"  Done: {signals} signals, {fills} filled "
            f"({expired} expired, {skipped} zone-skipped)\n"
        )
