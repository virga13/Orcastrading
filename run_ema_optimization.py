"""
run_ema_optimization.py — Walk-forward grid search for EMA Continuation & EMA Pullback.

Approach:
  - Fetches Gold 1d OHLCV once, then re-runs signal detection + simulation in memory
    for every parameter combination (NO cache writes — independent of pass1 cache).
  - Walk-forward split: train 60% / val 20% / test 20%
  - Ranks by test-window profit factor (OOS performance).
  - Prints top-5 combos per strategy and recommended params.

Usage:
    python run_ema_optimization.py
    python run_ema_optimization.py --strategy ema_continuation
    python run_ema_optimization.py --strategy ema_pullback
    python run_ema_optimization.py --ticker GC=F --start 2021-01-01 --end 2026-04-15
"""
import sys
import io
import argparse
import itertools
import math
from datetime import datetime
from pathlib import Path

# ── UTF-8 stdout ──────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from p3_backtester.market_data import fetch_ohlcv
from p3_backtester.bar_slicer import recompute_technical
from p3_backtester.signal_scheduler import get_signal_indices
from p3_backtester.trade_simulator import simulate_trade
from p3_backtester.strategies.ema_continuation import EMAContinuationStrategy
from p3_backtester.strategies.ema_pullback import EMAPullbackStrategy

console = Console()
MIN_TRADES_TEST = 5    # minimum filled trades in OOS test window
MIN_TRADES_FULL = 10   # minimum filled trades across full dataset

# Features captured per trade for the strategy learner
TRADE_FEATURE_KEYS = [
    "adx", "rsi", "atr_pct", "ema_alignment",
    "stoch_signal", "volume_trend", "bb_position", "trend",
]


# ── Parameter grids ───────────────────────────────────────────────────────────

GRID_CONTINUATION = {
    "adx_threshold":   [18, 22, 25, 30],
    "tp1_r":           [2.0, 2.5, 3.0, 3.5],
    "tp2_r":           [4.0, 5.0, 6.0],
    "tp1_alloc":       [60, 70],
    "atr_pct_max":     [1.0, 1.2, 1.5, 999.0],
    # tp2_alloc is derived as 100 - tp1_alloc
}

GRID_PULLBACK = {
    "adx_threshold":          [15, 18, 22, 25],
    "tp1_r":                  [2.0, 2.5, 3.0, 3.5],
    "tp2_r":                  [4.0, 5.0, 6.0],
    "tp1_alloc":              [60, 70],
    "require_rsi_divergence": [False, True],
    "require_pin_bar":        [False, True],
}


# ── In-memory backtest ────────────────────────────────────────────────────────

def precompute_technicals(
    df: pd.DataFrame,
    signal_indices: list[int],
    ticker: str,
    interval: str,
) -> list[tuple[int, dict, pd.DataFrame]]:
    """
    Compute technical indicators for every signal bar once.
    Returns list of (bar_index, technical_dict, df_slice) tuples.
    Call this once before the grid search loop.
    """
    result = []
    for i in signal_indices:
        df_slice  = df.iloc[: i + 1].copy()
        technical = recompute_technical(df_slice, interval)
        if technical is None:
            continue
        technical["ticker"] = ticker

        close  = df_slice["Close"]
        high   = df_slice["High"]
        low    = df_slice["Low"]

        # Williams %R (14-bar) — more sensitive oversold detector than stochastic
        try:
            wrl = float(ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1])
            technical["williams_r_sig"] = "oversold" if wrl < -80 else ("overbought" if wrl > -20 else "neutral")
        except Exception:
            technical["williams_r_sig"] = "neutral"

        # Consecutive bearish bars before this signal bar
        try:
            n_consec = 0
            closes = close.values
            opens  = df_slice["Open"].values
            for j in range(len(closes) - 2, max(len(closes) - 12, -1), -1):
                if closes[j] < opens[j]:
                    n_consec += 1
                else:
                    break
            technical["consec_bearish"] = n_consec
        except Exception:
            technical["consec_bearish"] = 0

        # ATR ratio: current ATR / 20-bar average ATR (>1 = expanding, <1 = contracting)
        try:
            atr_series = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
            atr_avg20  = float(atr_series.rolling(20).mean().iloc[-1])
            technical["atr_ratio"] = round(float(atr_series.iloc[-1]) / atr_avg20, 3) if atr_avg20 > 0 else 1.0
        except Exception:
            technical["atr_ratio"] = 1.0

        # Candle close position: (close - low) / (high - low) — 1.0 = closed at top
        try:
            h, l, c = float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
            technical["candle_close_pct"] = round((c - l) / (h - l), 3) if (h - l) > 0 else 0.5
        except Exception:
            technical["candle_close_pct"] = 0.5

        result.append((i, technical, df_slice))
    return result


def run_backtest_inmem(
    strategy,
    precomputed: list[tuple[int, dict, pd.DataFrame]],
    df: pd.DataFrame,
) -> list[dict]:
    """
    Run a full backtest in memory using pre-computed technicals.
    Skips recompute_technical — each combo only calls generate_setups + simulate_trade.
    Returns list of trade result dicts.
    """
    trades = []

    for i, technical, df_slice in precomputed:
        try:
            setups_obj = strategy.generate_setups(technical, df_slice)
        except Exception:
            continue

        if setups_obj is None:
            continue

        for setup in setups_obj.setups:
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
                    cost_r=0.0,  # optimization uses gross PF for comparability
                )
            except Exception:
                continue
            trades.append({
                "bar_index": i,
                "bar_time":  df.index[i],
                "outcome":   tr.outcome,
                "pnl_r":     tr.pnl_r,
                "net_pnl_r": tr.net_pnl_r,
                # Regime features — used by run_strategy_learner.py
                "adx":             technical.get("adx_14"),
                "rsi":             technical.get("rsi_14"),
                "atr_pct":         technical.get("atr_pct"),
                "ema_alignment":   technical.get("ema_alignment"),
                "stoch_signal":    technical.get("stoch_signal"),
                "volume_trend":    technical.get("volume_trend"),
                "bb_position":     technical.get("bb_position"),
                "trend":           technical.get("trend"),
                "macd_signal":     technical.get("macd_signal"),
                "cmf_signal":      technical.get("cmf_signal"),
                "williams_r_sig":  technical.get("williams_r_sig"),
                "consec_bearish":  technical.get("consec_bearish"),
                "atr_ratio":       technical.get("atr_ratio"),
                "candle_close_pct":technical.get("candle_close_pct"),
            })

    return trades


def _compute_pf(trades: list[dict]) -> tuple[float, float, int]:
    """Return (gross_pf, win_rate, n_filled)."""
    filled = [t for t in trades if t["outcome"] != "EXPIRED"]
    if not filled:
        return 0.0, 0.0, 0
    wins   = [t for t in filled if t["outcome"] in ("WIN", "PARTIAL_WIN")]
    losses = [t for t in filled if t["outcome"] == "LOSS"]
    gp = sum(t["pnl_r"] for t in wins)
    gl = abs(sum(t["pnl_r"] for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / len(filled)
    return round(pf, 3), round(wr, 3), len(filled)


def _net_r(trades: list[dict]) -> float:
    filled = [t for t in trades if t["outcome"] != "EXPIRED"]
    return round(sum(t["net_pnl_r"] for t in filled), 2)


def split_trades(trades: list[dict], df: pd.DataFrame) -> tuple[list, list, list]:
    """Split trades into train/val/test windows (60/20/20% of df bars)."""
    n = len(df)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)

    train_ts = df.index[train_end - 1]
    val_ts   = df.index[val_end   - 1]

    train = [t for t in trades if t["bar_time"] <= train_ts]
    val   = [t for t in trades if train_ts < t["bar_time"] <= val_ts]
    test  = [t for t in trades if t["bar_time"] > val_ts]
    return train, val, test


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid(strategy_name: str, grid: dict, df: pd.DataFrame,
             precomputed: list, signal_indices: list[int]) -> list[dict]:
    """
    Run grid search using pre-computed technicals.
    Returns list of result dicts sorted by test PF descending.
    """
    keys   = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))

    n = len(df)
    train_end_dt = df.index[int(n * 0.60) - 1]
    val_end_dt   = df.index[int(n * 0.80) - 1]

    console.print(
        f"\n[bold]{strategy_name}[/bold] — "
        f"{len(combos)} combos | "
        f"train ≤{train_end_dt.date()}  val ≤{val_end_dt.date()}  "
        f"test >{val_end_dt.date()}"
    )

    results = []
    skipped = 0
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Enforce tp2_r > tp1_r and derive tp2_alloc
        if params.get("tp2_r", 999) <= params.get("tp1_r", 0):
            skipped += 1
            continue
        params["tp2_alloc"] = 100 - params["tp1_alloc"]

        # Build strategy
        if strategy_name == "ema_continuation":
            strategy = EMAContinuationStrategy(**params)
        else:
            strategy = EMAPullbackStrategy(**params)

        # Fast backtest using pre-computed technicals
        trades = run_backtest_inmem(strategy, precomputed, df)

        train_t, val_t, test_t = split_trades(trades, df)

        train_pf, train_wr, train_n = _compute_pf(train_t)
        val_pf,   val_wr,   val_n   = _compute_pf(val_t)
        test_pf,  test_wr,  test_n  = _compute_pf(test_t)

        # Combined val+test as the OOS window
        oos_t = val_t + test_t
        oos_pf, oos_wr, oos_n = _compute_pf(oos_t)
        full_n = train_n + val_n + test_n

        # Skip combos with too few OOS or total trades
        if test_n < MIN_TRADES_TEST or full_n < MIN_TRADES_FULL:
            continue

        results.append({
            "params":    params,
            "train_pf":  train_pf,
            "train_wr":  train_wr,
            "train_n":   train_n,
            "val_pf":    val_pf,
            "val_wr":    val_wr,
            "val_n":     val_n,
            "test_pf":   test_pf,
            "test_wr":   test_wr,
            "test_n":    test_n,
            "test_net_r": _net_r(test_t),
            "oos_pf":    oos_pf,
            "oos_wr":    oos_wr,
            "oos_n":     oos_n,
        })

        # Progress every 25 combos
        if (idx + 1) % 25 == 0 or idx + 1 == len(combos):
            console.print(
                f"  [dim]{idx+1}/{len(combos)} combos scanned, "
                f"{len(results)} valid so far[/dim]"
            )

    results.sort(key=lambda r: r["oos_pf"], reverse=True)
    return results


# ── Display ───────────────────────────────────────────────────────────────────

def display_results(strategy_name: str, results: list[dict], top_n: int = 10):
    if not results:
        console.print(f"[red]No valid combos found for {strategy_name}.[/red]")
        return

    console.print()
    def _color(v, thr=1.0):
        return "green" if v > thr else ("yellow" if v == thr else "red")

    t = Table(
        title=f"[bold]{strategy_name}[/bold] — Top {min(top_n, len(results))} by OOS Profit Factor (val+test)",
        box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1),
    )
    t.add_column("Rank",    justify="right", style="dim", width=5)
    t.add_column("tp1_r",   justify="right")
    t.add_column("tp2_r",   justify="right")
    t.add_column("tp1%",    justify="right")
    t.add_column("ADX",     justify="right")
    t.add_column("Train PF",justify="right")
    t.add_column("OOS PF",  justify="right", style="bold")
    t.add_column("OOS WR",  justify="right")
    t.add_column("OOS n",   justify="right")
    t.add_column("Test PF", justify="right")
    t.add_column("Test n",  justify="right")

    for rank, r in enumerate(results[:top_n], 1):
        p = r["params"]
        t.add_row(
            str(rank),
            f"{p['tp1_r']:.1f}",
            f"{p['tp2_r']:.1f}",
            f"{p['tp1_alloc']}",
            str(p["adx_threshold"]),
            f"[{_color(r['train_pf'])}]{r['train_pf']:.2f}[/{_color(r['train_pf'])}]",
            f"[{_color(r['oos_pf'])}]{r['oos_pf']:.2f}[/{_color(r['oos_pf'])}]",
            f"{r['oos_wr']:.1%}",
            str(r["oos_n"]),
            f"[{_color(r['test_pf'])}]{r['test_pf']:.2f}[/{_color(r['test_pf'])}]",
            str(r["test_n"]),
        )

    console.print(t)

    # Best combo summary
    best = results[0]
    p = best["params"]
    console.print(
        f"\n[bold green]Recommended params for {strategy_name}:[/bold green]\n"
        f"  adx_threshold: {p['adx_threshold']}\n"
        f"  tp1_r: {p['tp1_r']}  tp2_r: {p['tp2_r']}\n"
        f"  tp1_alloc: {p['tp1_alloc']}  tp2_alloc: {p['tp2_alloc']}\n"
        f"  OOS PF: [bold]{best['oos_pf']:.2f}[/bold]  "
        f"WR: {best['oos_wr']:.1%}  n: {best['oos_n']}\n"
        f"  [dim]Note: small sample — treat these as directional guidance, not definitive.[/dim]"
    )


# ── Cross-asset consensus ─────────────────────────────────────────────────────

def find_consensus(per_ticker_results: dict[str, list[dict]], min_tickers: int = 2) -> list[dict]:
    """
    Find param combos that perform well across multiple assets.
    Score = n_tickers × mean_oos_pf (rewards breadth + quality).
    """
    from collections import defaultdict
    combo_data: dict = defaultdict(dict)   # params_key -> {ticker: (oos_pf, oos_n)}

    for ticker, results in per_ticker_results.items():
        for r in results:
            key = tuple(sorted(r["params"].items()))
            combo_data[key][ticker] = (r["oos_pf"], r["oos_n"])

    consensus = []
    for key, ticker_data in combo_data.items():
        if len(ticker_data) < min_tickers:
            continue
        params   = dict(key)
        pfs      = [v[0] for v in ticker_data.values()]
        total_n  = sum(v[1] for v in ticker_data.values())
        n_assets = len(ticker_data)
        mean_pf  = sum(pfs) / n_assets
        min_pf   = min(pfs)
        consensus.append({
            "params":     params,
            "n_assets":   n_assets,
            "mean_pf":    round(mean_pf, 3),
            "min_pf":     round(min_pf, 3),
            "score":      round(n_assets * mean_pf, 3),
            "total_n":    total_n,
            "per_ticker": {t: v for t, v in ticker_data.items()},
        })

    consensus.sort(key=lambda x: (x["n_assets"], x["mean_pf"]), reverse=True)
    return consensus


def display_consensus(strategy_name: str, consensus: list[dict], top_n: int = 5):
    if not consensus:
        console.print(f"[yellow]No cross-asset consensus found for {strategy_name} "
                      f"(too few assets with valid combos).[/yellow]")
        return

    tickers = sorted({t for c in consensus for t in c["per_ticker"]})

    t = Table(
        title=f"[bold]{strategy_name}[/bold] — Cross-asset consensus (top {min(top_n, len(consensus))})",
        box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1),
    )
    t.add_column("Rank",   justify="right", style="dim", width=5)
    t.add_column("ADX",    justify="right")
    t.add_column("tp1_r",  justify="right")
    t.add_column("tp2_r",  justify="right")
    t.add_column("tp1%",   justify="right")
    t.add_column("Assets", justify="right")
    t.add_column("Mean PF",justify="right", style="bold")
    t.add_column("Min PF", justify="right")
    t.add_column("Total n",justify="right")
    for tk in tickers:
        t.add_column(tk, justify="right")

    def _c(v):
        return "green" if v > 1.0 else ("yellow" if v == 1.0 else "red")

    for rank, c in enumerate(consensus[:top_n], 1):
        p = c["params"]
        per_tk_cells = []
        for tk in tickers:
            if tk in c["per_ticker"]:
                pf, n = c["per_ticker"][tk]
                per_tk_cells.append(f"[{_c(pf)}]{pf:.2f}[/{_c(pf)}](n={n})")
            else:
                per_tk_cells.append("—")

        t.add_row(
            str(rank),
            str(p.get("adx_threshold", "—")),
            f"{p.get('tp1_r', '—')}",
            f"{p.get('tp2_r', '—')}",
            str(p.get("tp1_alloc", "—")),
            str(c["n_assets"]),
            f"[{_c(c['mean_pf'])}]{c['mean_pf']:.2f}[/{_c(c['mean_pf'])}]",
            f"[{_c(c['min_pf'])}]{c['min_pf']:.2f}[/{_c(c['min_pf'])}]",
            str(c["total_n"]),
            *per_tk_cells,
        )

    console.print(t)

    best = consensus[0]
    p    = best["params"]
    console.print(
        f"\n[bold green]Consensus params for {strategy_name}:[/bold green]\n"
        f"  adx_threshold: {p.get('adx_threshold')}  "
        f"tp1_r: {p.get('tp1_r')}  tp2_r: {p.get('tp2_r')}  "
        f"tp1_alloc: {p.get('tp1_alloc')}\n"
        f"  Mean OOS PF: [bold]{best['mean_pf']:.2f}[/bold]  "
        f"Min PF: {best['min_pf']:.2f}  "
        f"Assets: {best['n_assets']}  Total trades: {best['total_n']}\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMA strategy parameter optimizer")
    parser.add_argument("--strategy", choices=["ema_continuation", "ema_pullback", "both"],
                        default="both")
    # --tickers accepts comma-separated list; --ticker is legacy alias for single asset
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated tickers (e.g. GC=F,^GSPC,BTC-USD)")
    parser.add_argument("--ticker",  default=None,
                        help="Single ticker — legacy alias for --tickers")
    parser.add_argument("--interval", default="1d",  help="Candlestick interval (default: 1d)")
    parser.add_argument("--start",    default="2018-01-01")
    parser.add_argument("--end",      default="2026-04-19")
    parser.add_argument("--top",      type=int, default=10, help="Top N combos to display")
    args = parser.parse_args()

    tickers_str = args.tickers or args.ticker or "GC=F"
    tickers     = [t.strip() for t in tickers_str.split(",")]

    run_continuation = args.strategy in ("ema_continuation", "both")
    run_pullback      = args.strategy in ("ema_pullback",     "both")

    console.print(f"\n[bold]EMA Strategy Optimizer[/bold]")
    console.print(
        f"Assets: {', '.join(tickers)}  Interval: {args.interval}  "
        f"Window: {args.start} → {args.end}\n"
    )

    from p3_backtester.market_data import split_backtest_window

    all_cont_results: dict[str, list[dict]] = {}
    all_pull_results: dict[str, list[dict]] = {}

    for ticker in tickers:
        console.rule(f"[bold]{ticker}[/bold]")

        # ── Fetch data ────────────────────────────────────────────────────────
        console.print("  Fetching OHLCV data...")
        df = fetch_ohlcv(ticker, args.interval, args.start, args.end)
        console.print(f"  {len(df)} bars loaded")

        first_valid, _ = split_backtest_window(df, args.start)
        signal_indices = get_signal_indices(df, "session-open", first_valid, every_n_bars=10)
        console.print(f"  {len(signal_indices)} signal bars")

        n = len(df)
        console.print(
            f"  Walk-forward: "
            f"Train ≤{df.index[int(n*0.60)-1].date()}  "
            f"Val ≤{df.index[int(n*0.80)-1].date()}  "
            f"Test >{df.index[int(n*0.80)].date()}  ({n - int(n*0.80)} bars)"
        )

        # ── Pre-compute technicals once (shared by both strategies) ──────────
        console.print("  Pre-computing technical indicators...")
        precomputed = precompute_technicals(df, signal_indices, ticker, args.interval)
        console.print(f"  {len(precomputed)} bars with valid technicals")

        # ── EMA Continuation ─────────────────────────────────────────────────
        if run_continuation:
            results_cont = run_grid(
                "ema_continuation", GRID_CONTINUATION, df, precomputed, signal_indices,
            )
            all_cont_results[ticker] = results_cont
            display_results("ema_continuation", results_cont, args.top)

        # ── EMA Pullback ─────────────────────────────────────────────────────
        if run_pullback:
            results_pull = run_grid(
                "ema_pullback", GRID_PULLBACK, df, precomputed, signal_indices,
            )
            all_pull_results[ticker] = results_pull
            display_results("ema_pullback", results_pull, args.top)

    # ── Cross-asset consensus (only when >1 ticker) ───────────────────────────
    if len(tickers) > 1:
        console.rule("[bold]Cross-asset consensus[/bold]")
        if run_continuation and all_cont_results:
            consensus = find_consensus(all_cont_results)
            display_consensus("ema_continuation", consensus)
        if run_pullback and all_pull_results:
            consensus = find_consensus(all_pull_results)
            display_consensus("ema_pullback", consensus)

    console.print(
        "\n[dim]To apply results: update config/strategies.yaml with the recommended params.[/dim]\n"
        "[dim]To learn from trade patterns: python run_strategy_learner.py --strategy ema_continuation[/dim]\n"
    )


if __name__ == "__main__":
    main()
