"""
run_mtf_optimization.py — Walk-forward grid search for MTFTrendStrategy.

Tests every combination of key parameters including shorts_enabled=True/False
so we know whether enabling shorts improves or degrades each asset.

Usage:
    python run_mtf_optimization.py                          # Gold + US30 + SPX500, 1d
    python run_mtf_optimization.py --ticker ^DJI            # US30 only
    python run_mtf_optimization.py --ticker GC=F --top 15  # Gold top-15
    python run_mtf_optimization.py --start 2019-01-01       # extend history

Walk-forward split: 60% train / 20% val / 20% test (OOS).
Ranked by val+test profit factor so parameters generalise outside training data.
"""
import sys
import io
import argparse
import itertools
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

import ta
import pandas as pd

from p3_backtester.market_data import fetch_ohlcv, split_backtest_window
from p3_backtester.bar_slicer import recompute_technical
from p3_backtester.signal_scheduler import get_signal_indices
from p3_backtester.trade_simulator import simulate_trade
from p3_backtester.strategies.mtf_trend import MTFTrendStrategy
from p3_backtester.strategies.regime_filter import classify_regime

console = Console()

MIN_TEST_FILLS  = 5    # minimum filled trades in OOS test window
MIN_TOTAL_FILLS = 15   # minimum across whole dataset (avoid overfitting to tiny n)


# ── Parameter grid ────────────────────────────────────────────────────────────
# Only the highest-leverage knobs — keep total combos < 500 so it runs in < 2 min.
GRID = {
    "adx_threshold":          [12, 15, 20, 25],       # long entry quality gate
    "adx_threshold_short":    [10, 12, 15],            # short entry quality gate
    "pullback_max_atr":       [1.0, 1.5, 2.0],         # EMA proximity window
    "tp1_r":                  [2.0, 2.5, 3.0],         # TP1 multiple
    "tp2_r":                  [4.0, 5.0, 6.0],         # TP2 multiple
    "tp1_alloc":              [60, 70],                 # % closed at TP1
    "shorts_enabled":         [True, False],            # key question
}
# Fixed params (not varied — secondary impact, pre-validated)
FIXED = {
    "ema_fast":               20,
    "ema_slow":               50,
    "min_prior_distance_atr": 0.8,
    "prior_lookback":         10,
    "rsi_max":                70,
    "rsi_min":                30,
    "sl_buffer_atr":          0.15,
    "sl_lookback":            15,
    "regime_filter":          True,    # always on — protects both longs and shorts
    "win_rate":               0.48,
}


# ── Pre-computation ───────────────────────────────────────────────────────────

_PRECOMPUTE_STRATEGY = MTFTrendStrategy()  # default ema_fast=20, ema_slow=50


def precompute_bars(
    df: pd.DataFrame,
    signal_indices: list[int],
    ticker: str,
    interval: str,
) -> list[tuple[int, dict, pd.DataFrame]]:
    """
    Compute technical indicators for every signal bar once.
    Also precomputes HTF bias, regime, and ETF EMA — these are param-independent
    so computing them here (once per bar) instead of inside the combo loop gives
    a ~50x speedup on large datasets like BTC which has 365 trading days/year.
    """
    out = []
    for i in signal_indices:
        df_slice = df.iloc[: i + 1].copy()
        tech = recompute_technical(df_slice, interval)
        if tech is None:
            continue
        tech["ticker"]   = ticker
        tech["interval"] = interval

        # Precompute param-independent expensive gates (called 1× per bar vs N_combos×)
        htf_bias, htf_adx, htf_ema_f, htf_ema_s = _PRECOMPUTE_STRATEGY._htf_bias(df_slice, interval)
        tech["htf_bias"]  = htf_bias
        tech["htf_adx"]   = htf_adx
        tech["htf_ema_f"] = htf_ema_f
        tech["htf_ema_s"] = htf_ema_s
        tech["regime"]    = classify_regime(df_slice, interval)

        # ETF EMA20 value + tail for prior-distance gate (ema_fast=20 is fixed in FIXED params)
        try:
            etf_ema = ta.trend.EMAIndicator(df_slice["Close"], window=20).ema_indicator()
            ema_arr = etf_ema.values
            tail_len = min(14, len(ema_arr))
            tech["etf_ema_val"]  = float(ema_arr[-1]) if tail_len > 0 and not pd.isna(ema_arr[-1]) else float("nan")
            tech["etf_ema_tail"] = [float(v) for v in ema_arr[-tail_len:]]  # oldest→newest
        except Exception:
            tech["etf_ema_val"]  = float("nan")
            tech["etf_ema_tail"] = []

        out.append((i, tech, df_slice))
    return out


# ── In-memory backtest ────────────────────────────────────────────────────────

def run_inmem(
    strategy: MTFTrendStrategy,
    precomputed: list[tuple[int, dict, pd.DataFrame]],
    df: pd.DataFrame,
) -> list[dict]:
    trades = []
    for i, tech, df_slice in precomputed:
        try:
            result = strategy.generate_setups(tech, df_slice)
        except Exception:
            continue
        if result is None:
            continue
        for setup in result.setups:
            if setup.priority != "primary":
                continue
            try:
                tr = simulate_trade(
                    setup=setup,
                    signal_bar_index=i,
                    signal_bar_time=df.index[i].isoformat(),
                    df=df,
                    entry_timeout_bars=20,
                    claude_confidence=setup.confidence,
                    cost_r=0.0,   # gross R — avoid noise from cost assumptions
                )
            except Exception:
                continue
            trades.append({
                "bar_index":  i,
                "bar_time":   df.index[i],
                "direction":  setup.direction,
                "outcome":    tr.outcome,
                "pnl_r":      tr.pnl_r,
                "net_pnl_r":  tr.net_pnl_r,
            })
    return trades


# ── Statistics helpers ────────────────────────────────────────────────────────

def _pf(trades: list[dict]) -> tuple[float, float, int]:
    """(profit_factor, win_rate, n_filled)"""
    filled = [t for t in trades if t["outcome"] != "EXPIRED"]
    if not filled:
        return 0.0, 0.0, 0
    wins   = [t for t in filled if t["outcome"] in ("WIN", "PARTIAL_WIN")]
    losses = [t for t in filled if t["outcome"] == "LOSS"]
    gp = sum(t["pnl_r"] for t in wins)
    gl = abs(sum(t["pnl_r"] for t in losses))
    pf = round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = round(len(wins) / len(filled), 3)
    return pf, wr, len(filled)


def _avg_r(trades: list[dict]) -> float:
    filled = [t for t in trades if t["outcome"] != "EXPIRED"]
    if not filled:
        return 0.0
    return round(sum(t["pnl_r"] for t in filled) / len(filled), 3)


def _shorts_pf(trades: list[dict]) -> tuple[float, int]:
    """Profit factor for short trades only."""
    shorts = [t for t in trades if t["direction"] == "short" and t["outcome"] != "EXPIRED"]
    if not shorts:
        return 0.0, 0
    wins = [t for t in shorts if t["outcome"] in ("WIN", "PARTIAL_WIN")]
    losses = [t for t in shorts if t["outcome"] == "LOSS"]
    gp = sum(t["pnl_r"] for t in wins)
    gl = abs(sum(t["pnl_r"] for t in losses))
    pf = round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0)
    return pf, len(shorts)


def split3(trades: list[dict], df: pd.DataFrame) -> tuple[list, list, list]:
    n = len(df)
    t1 = df.index[int(n * 0.60) - 1]
    t2 = df.index[int(n * 0.80) - 1]
    return (
        [t for t in trades if t["bar_time"] <= t1],
        [t for t in trades if t1 < t["bar_time"] <= t2],
        [t for t in trades if t["bar_time"] > t2],
    )


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid(
    df: pd.DataFrame,
    precomputed: list,
    ticker: str,
    interval: str,
) -> list[dict]:
    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))

    n = len(df)
    t1 = df.index[int(n * 0.60) - 1]
    t2 = df.index[int(n * 0.80) - 1]
    console.print(
        f"\n  Grid: [bold]{len(combos)}[/bold] combos — "
        f"train ≤{t1.date()}  val ≤{t2.date()}  test >{t2.date()}"
    )

    results = []
    skipped_guard  = 0
    skipped_sample = 0

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # tp2_r must exceed tp1_r
        if params["tp2_r"] <= params["tp1_r"]:
            skipped_guard += 1
            continue
        # adx_short should be ≤ adx_long (shorts enter earlier in trend formation)
        if params["adx_threshold_short"] > params["adx_threshold"]:
            skipped_guard += 1
            continue

        params["tp2_alloc"] = 100 - params["tp1_alloc"]
        params = {**FIXED, **params}

        strategy = MTFTrendStrategy(**params)
        trades   = run_inmem(strategy, precomputed, df)
        train, val, test = split3(trades, df)

        tr_pf, tr_wr, tr_n = _pf(train)
        va_pf, va_wr, va_n = _pf(val)
        te_pf, te_wr, te_n = _pf(test)
        oos_pf, oos_wr, oos_n = _pf(val + test)

        total_n = tr_n + va_n + te_n
        if te_n < MIN_TEST_FILLS or total_n < MIN_TOTAL_FILLS:
            skipped_sample += 1
            continue

        short_pf_full, n_shorts = _shorts_pf(trades)

        results.append({
            "params":       params,
            "shorts":       params["shorts_enabled"],
            "train_pf":     tr_pf,  "train_wr": tr_wr,  "train_n":  tr_n,
            "val_pf":       va_pf,  "val_wr":   va_wr,  "val_n":    va_n,
            "test_pf":      te_pf,  "test_wr":  te_wr,  "test_n":   te_n,
            "oos_pf":       oos_pf, "oos_wr":   oos_wr, "oos_n":    oos_n,
            "avg_r_test":   _avg_r(test),
            "short_pf":     short_pf_full,
            "n_shorts":     n_shorts,
            "total_n":      total_n,
        })

        if (idx + 1) % 100 == 0 or idx + 1 == len(combos):
            console.print(
                f"  [dim]{idx+1}/{len(combos)} scanned — "
                f"{len(results)} valid, {skipped_guard} skipped (guard), "
                f"{skipped_sample} skipped (sample)[/dim]"
            )

    results.sort(key=lambda r: r["oos_pf"], reverse=True)
    return results


# ── Display ───────────────────────────────────────────────────────────────────

def _c(v, hi=1.5, lo=1.0):
    return "green" if v >= hi else ("yellow" if v >= lo else "red")


def display(ticker: str, results: list[dict], top_n: int = 10):
    if not results:
        console.print(f"[red]No valid combos for {ticker}.[/red]")
        return

    # ── Overall top-N ─────────────────────────────────────────────────────────
    t = Table(
        title=f"[bold]{ticker}[/bold] — Top {min(top_n, len(results))} by OOS Profit Factor",
        box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1),
    )
    for col, w, just in [
        ("#",      4,  "right"),  ("Shorts", 7, "center"),
        ("ADX-L",  6,  "right"),  ("ADX-S",  5, "right"),
        ("Pull",   5,  "right"),  ("TP1r",   5, "right"),
        ("TP2r",   5,  "right"),  ("TP1%",   5, "right"),
        ("Tr PF",  7,  "right"),  ("OOS PF", 8, "right"),
        ("OOS WR", 7,  "right"),  ("OOS n",  6, "right"),
        ("Te PF",  7,  "right"),  ("Te R",   6, "right"),
    ]:
        t.add_column(col, width=w, justify=just)

    for rank, r in enumerate(results[:top_n], 1):
        p  = r["params"]
        oo = r["oos_pf"]
        te = r["test_pf"]
        t.add_row(
            str(rank),
            "[green]YES[/green]" if r["shorts"] else "[dim]no[/dim]",
            str(p["adx_threshold"]),
            str(p["adx_threshold_short"]),
            f"{p['pullback_max_atr']:.1f}",
            f"{p['tp1_r']:.1f}",
            f"{p['tp2_r']:.1f}",
            str(p["tp1_alloc"]),
            f"[{_c(r['train_pf'])}]{r['train_pf']:.2f}[/{_c(r['train_pf'])}]",
            f"[{_c(oo)}]{oo:.2f}[/{_c(oo)}]",
            f"{r['oos_wr']:.1%}",
            str(r["oos_n"]),
            f"[{_c(te)}]{te:.2f}[/{_c(te)}]",
            f"{r['avg_r_test']:+.2f}R",
        )
    console.print(t)

    # ── Shorts-only breakdown ─────────────────────────────────────────────────
    shorts_results = [r for r in results if r["shorts"] and r["n_shorts"] >= 5]
    longs_results  = [r for r in results if not r["shorts"]]

    best_with    = shorts_results[0] if shorts_results else None
    best_without = longs_results[0]  if longs_results  else None

    console.print()
    console.print("[bold]Shorts verdict:[/bold]")
    if best_with and best_without:
        diff = best_with["oos_pf"] - best_without["oos_pf"]
        sign = "+" if diff >= 0 else ""
        color = "green" if diff >= 0.1 else ("yellow" if diff >= -0.1 else "red")
        console.print(
            f"  Best with shorts:    OOS PF [bold]{best_with['oos_pf']:.2f}[/bold]  "
            f"(short trades in full dataset: {best_with['n_shorts']}, "
            f"short PF: {best_with['short_pf']:.2f})"
        )
        console.print(
            f"  Best without shorts: OOS PF [bold]{best_without['oos_pf']:.2f}[/bold]"
        )
        console.print(
            f"  Delta: [{color}]{sign}{diff:.2f}[/{color}] "
            f"({'enable shorts' if diff >= 0.1 else 'disable shorts' if diff <= -0.1 else 'negligible difference'})"
        )
    elif best_without:
        console.print(f"  No valid short-enabled combos met sample threshold (n≥5 shorts, n≥{MIN_TEST_FILLS} test fills).")
        console.print(f"  Best longs-only: OOS PF {best_without['oos_pf']:.2f}")

    # ── Recommended params ────────────────────────────────────────────────────
    best = results[0]
    p = best["params"]
    console.print(
        f"\n[bold green]Recommended params for {ticker} / mtf_trend:[/bold green]\n"
        f"  adx_threshold:          {p['adx_threshold']}\n"
        f"  adx_threshold_short:    {p['adx_threshold_short']}\n"
        f"  pullback_max_atr:       {p['pullback_max_atr']}\n"
        f"  tp1_r:                  {p['tp1_r']}\n"
        f"  tp2_r:                  {p['tp2_r']}\n"
        f"  tp1_alloc:              {p['tp1_alloc']}\n"
        f"  tp2_alloc:              {100 - p['tp1_alloc']}\n"
        f"  shorts_enabled:         {p['shorts_enabled']}\n"
        f"  OOS PF: [bold]{best['oos_pf']:.2f}[/bold]  "
        f"test WR: {best['test_wr']:.1%}  test n: {best['test_n']}\n"
        f"  [dim]Apply to config/assets.yaml (per-asset overrides) or "
        f"config/strategies.yaml (global defaults).[/dim]"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

ASSET_DEFAULTS = {
    "GC=F":  ("gold",   "2019-01-01"),
    "^DJI":  ("us30",   "2019-01-01"),
    "^GSPC": ("spx500", "2019-01-01"),
    "^NDX":  ("nas100", "2019-01-01"),
    "SI=F":  ("silver", "2019-01-01"),
}


def main():
    parser = argparse.ArgumentParser(
        description="MTFTrendStrategy parameter optimizer with shorts analysis"
    )
    parser.add_argument("--ticker",   default=None,
                        help="Specific ticker to optimize (default: Gold + US30 + SPX500)")
    parser.add_argument("--interval", default="1d",   help="Candlestick interval (default: 1d)")
    parser.add_argument("--start",    default="2019-01-01")
    parser.add_argument("--end",      default=None,   help="End date (default: today)")
    parser.add_argument("--top",      type=int, default=10)
    args = parser.parse_args()

    from datetime import date
    end = args.end or date.today().strftime("%Y-%m-%d")

    # Determine which tickers to run
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = ["GC=F", "^DJI", "^GSPC"]

    console.print()
    console.print(Panel(
        "[bold gold1]MTFTrend Optimizer[/bold gold1]\n"
        f"[dim]Interval: {args.interval}  |  {args.start} → {end}  |  "
        f"{len(list(itertools.product(*GRID.values())))} raw combos[/dim]",
        box=box.ROUNDED, padding=(0, 4),
    ))

    for ticker in tickers:
        asset_label = ASSET_DEFAULTS.get(ticker, (ticker,))[0]
        start = ASSET_DEFAULTS.get(ticker, (None, args.start))[1]
        console.print(f"\n[bold]{'─'*60}[/bold]")
        console.print(f"[bold cyan]{asset_label.upper()} ({ticker})[/bold cyan]  "
                      f"{start} → {end}  interval={args.interval}")

        console.print("  Fetching OHLCV...")
        try:
            df = fetch_ohlcv(ticker, args.interval, start, end)
        except Exception as e:
            console.print(f"  [red]Data error: {e}[/red]")
            continue
        console.print(f"  {len(df)} bars loaded")

        first_valid, _ = split_backtest_window(df, start)
        sig_idx = get_signal_indices(df, "session-open", first_valid, every_n_bars=10)
        console.print(f"  {len(sig_idx)} signal bars")

        console.print("  Pre-computing technicals...")
        precomp = precompute_bars(df, sig_idx, ticker, args.interval)
        console.print(f"  {len(precomp)} bars with valid technicals")

        results = run_grid(df, precomp, ticker, args.interval)
        console.print(f"\n  [bold]{len(results)} valid combos[/bold] met sample threshold.")

        display(ticker, results, top_n=args.top)

    console.print(
        "\n[dim]To apply: update per-asset 'params:' overrides in config/assets.yaml, "
        "or global 'params:' in config/strategies.yaml.[/dim]\n"
    )


if __name__ == "__main__":
    main()
