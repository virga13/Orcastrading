"""
run_grid_search.py — Full grid backtest: all assets × strategies × timeframes.

Tests every viable combination, runs walk-forward validation, and outputs a
ranked CSV so you can see exactly which combos are worth keeping.

Usage:
    python run_grid_search.py
    python run_grid_search.py --csv outputs/grid_results.csv

What it tests:
  Assets    : spx500, us30, bitcoin, gold, silver, ger40  (all, even if disabled)
  Strategies: mtf_trend, ema_continuation, ema_pullback, momentum_breakout
  Timeframes: 1h (729d), 4h (729d), 1d (10yr via Stooq)
  ORB skipped — intraday session-based, needs separate session-level analysis.
"""
import argparse
import sys
import traceback
import csv
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ── Grid definition ───────────────────────────────────────────────────────────

ALL_ASSETS = ["spx500", "us30", "bitcoin", "gold", "silver", "ger40"]

# Strategies to test. ORB excluded — session-based, short data history.
ALL_STRATEGIES = ["mtf_trend", "ema_continuation", "ema_pullback", "momentum_breakout"]

# Timeframes to test per strategy
STRATEGY_TIMEFRAMES = {
    "mtf_trend":          ["1h", "4h", "1d"],
    "ema_continuation":   ["1h", "4h", "1d"],
    "ema_pullback":       ["1h", "4h", "1d"],
    "momentum_breakout":  ["1h", "4h", "1d"],
}

# Data windows — use maximum available history per timeframe
# 1d: Stooq provides 10+ years; 1h/4h: yfinance caps at ~729 days
DATE_WINDOWS = {
    "1h":  (date.today() - timedelta(days=728), date.today()),
    "4h":  (date.today() - timedelta(days=728), date.today()),
    "1d":  (date(2015, 1, 1), date.today()),
    "1wk": (date(2010, 1, 1), date.today()),
}

# Minimum fills required to report a result (below this → "insufficient data")
MIN_FILLS = 10

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_one(asset_id: str, strategy_id: str, tf: str) -> dict:
    from core.config import (
        get_ticker, get_strategy_lookback, get_asset_strategy_params,
        get_strategy_params, get_strategy_config,
    )
    from p3_backtester.strategies.registry import build_from_config
    from p3_backtester.schema import BacktestConfig
    from p3_backtester.market_data import fetch_ohlcv, split_backtest_window, MarketDataError
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats
    from p3_backtester.walk_forward import compute_walk_forward_dates, kelly_from_stats
    from p1_analysis_engine.utils.asset_classifier import classify_asset

    start_d, end_d = DATE_WINDOWS[tf]
    start = start_d.strftime("%Y-%m-%d")
    end   = end_d.strftime("%Y-%m-%d")

    base = {
        "asset": asset_id, "strategy": strategy_id, "tf": tf,
        "start": start, "end": end,
    }

    try:
        ticker = get_ticker(asset_id, source="yfinance")
    except KeyError as e:
        return {**base, "error": str(e)}

    base["ticker"] = ticker
    asset_class = classify_asset(ticker)

    # Merge default params with any per-asset overrides
    overrides = get_asset_strategy_params(asset_id, strategy_id)
    try:
        strategy = build_from_config(strategy_id, overrides)
    except Exception as e:
        return {**base, "error": f"strategy build: {e}"}

    config = BacktestConfig(
        ticker=ticker,
        interval=tf,
        start_date=start,
        end_date=end,
        signal_mode="session-open",
        entry_timeout_bars=20,
    )

    try:
        df = fetch_ohlcv(ticker, tf, start, end)
    except MarketDataError as e:
        return {**base, "error": f"data: {e}"}

    if len(df) < 50:
        return {**base, "error": f"too few bars ({len(df)})"}

    first_valid, _ = split_backtest_window(df, start)
    signal_indices = get_signal_indices(df, config.signal_mode, first_valid, config.every_n_bars)

    signals = run_pass1(config, df, signal_indices, use_claude=False, strategy=strategy)
    if not signals:
        return {**base, "ticker": ticker, "signals": 0, "trades": 0,
                "win_rate": 0, "net_pf": 0, "avg_net_r": 0, "max_dd": 0,
                "kelly": 0, "error": "no signals"}

    trades = run_pass2(config, df, signals, asset_class=asset_class)
    filled = [t for t in trades if t.outcome != "EXPIRED"]

    if not filled:
        return {**base, "ticker": ticker, "signals": len(signals), "trades": 0,
                "win_rate": 0, "net_pf": 0, "avg_net_r": 0, "max_dd": 0,
                "kelly": 0, "error": "no fills"}

    stats = compute_stats(config, signals, trades, strategy_name=strategy.name)

    result = {
        **base,
        "ticker":    ticker,
        "signals":   stats.total_signals,
        "trades":    stats.total_trades_filled,
        "win_rate":  round(stats.actual_win_rate, 4),
        "net_pf":    round(stats.actual_net_profit_factor, 3),
        "avg_net_r": round(stats.actual_avg_net_pnl_r, 4),
        "max_dd":    round(stats.max_drawdown_r, 2),
        "sharpe":    round(stats.sharpe_r, 3),
        "kelly":     round(stats.kelly_25pct, 4),
        "error":     None,
        # walk-forward fields (filled below)
        "oos_trades": None, "oos_win_rate": None,
        "oos_net_pf": None, "oos_avg_net_r": None,
    }

    # Walk-forward (train 60% / val 20% / test 20%)
    try:
        train_r, val_r, test_r = compute_walk_forward_dates(df, start, end)
        ts, te = test_r
        import pandas as _pd
        w_idx = [i for i in signal_indices
                 if _pd.Timestamp(ts) <= df.index[i] <= _pd.Timestamp(te)]
        if w_idx:
            wc = config.model_copy(update={"start_date": ts, "end_date": te})
            ws = run_pass1(wc, df, w_idx, use_claude=False, strategy=strategy)
            if ws:
                wt = run_pass2(wc, df, ws, asset_class=asset_class)
                wf = [t for t in wt if t.outcome != "EXPIRED"]
                if wf:
                    wst = compute_stats(wc, ws, wt, strategy_name=strategy.name)
                    result["oos_trades"]   = wst.total_trades_filled
                    result["oos_win_rate"] = round(wst.actual_win_rate, 4)
                    result["oos_net_pf"]   = round(wst.actual_net_profit_factor, 3)
                    result["oos_avg_net_r"]= round(wst.actual_avg_net_pnl_r, 4)
    except Exception:
        pass

    return result


def _verdict(r: dict) -> str:
    pf  = r.get("net_pf") or 0
    oos = r.get("oos_net_pf") or 0
    n   = r.get("trades") or 0
    if n < MIN_FILLS:
        return "THIN"
    if pf < 1.0:
        return "LOSS"
    if pf < 1.3:
        return "WEAK"
    if oos and oos < 1.0:
        return "OOS_FAIL"
    if pf >= 1.5 and (not oos or oos >= 1.2):
        return "GOOD"
    return "OK"


def _print_summary(results: list[dict]) -> None:
    ok  = [r for r in results if not r.get("error") or r.get("trades", 0) > 0]
    err = [r for r in results if r.get("error") and not r.get("trades")]

    def _pf_color(v):
        if v is None: return "dim"
        return "green" if v >= 1.5 else ("yellow" if v >= 1.0 else "red")

    def _wr_color(v):
        if v is None: return "dim"
        return "green" if v >= 0.40 else ("yellow" if v >= 0.30 else "red")

    console.print()
    t = Table(
        title="Full Grid — All Assets × Strategies × Timeframes",
        box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1),
    )
    t.add_column("Asset",     style="bold", width=10)
    t.add_column("Strategy",  width=20)
    t.add_column("TF",        width=4)
    t.add_column("Fills",     justify="right", width=6)
    t.add_column("WR%",       justify="right", width=7)
    t.add_column("Net PF",    justify="right", width=8)
    t.add_column("Avg R",     justify="right", width=8)
    t.add_column("MaxDD",     justify="right", width=7)
    t.add_column("Sharpe",    justify="right", width=7)
    t.add_column("Kelly%",    justify="right", width=7)
    t.add_column("OOS PF",    justify="right", width=7)
    t.add_column("Verdict",   width=9)

    ok_sorted = sorted(ok, key=lambda x: (x.get("oos_net_pf") or x.get("net_pf") or 0), reverse=True)

    for r in ok_sorted:
        pf  = r.get("net_pf") or 0
        wr  = r.get("win_rate") or 0
        oos = r.get("oos_net_pf")
        verdict = _verdict(r)
        verdict_color = {"GOOD": "green", "OK": "cyan", "THIN": "dim",
                         "WEAK": "yellow", "LOSS": "red", "OOS_FAIL": "red"}.get(verdict, "white")
        t.add_row(
            r["asset"],
            r["strategy"],
            r["tf"],
            str(r.get("trades") or 0),
            f"[{_wr_color(wr)}]{wr:.1%}[/{_wr_color(wr)}]",
            f"[{_pf_color(pf)}]{pf:.2f}[/{_pf_color(pf)}]",
            f"{r.get('avg_net_r') or 0:+.3f}R",
            f"-{r.get('max_dd') or 0:.1f}R",
            f"{r.get('sharpe') or 0:.2f}",
            f"{(r.get('kelly') or 0)*100:.1f}%",
            (f"[{_pf_color(oos)}]{oos:.2f}[/{_pf_color(oos)}]" if oos else "—"),
            f"[{verdict_color}]{verdict}[/{verdict_color}]",
        )

    console.print(t)

    if err:
        console.print()
        e = Table(title="Skipped / Errors", box=box.SIMPLE, padding=(0, 1))
        e.add_column("Asset",    width=10)
        e.add_column("Strategy", width=20)
        e.add_column("TF",       width=4)
        e.add_column("Reason",   style="dim")
        for r in err:
            e.add_row(r["asset"], r["strategy"], r["tf"], r.get("error") or "")
        console.print(e)

    good = [r for r in ok if _verdict(r) in ("GOOD", "OK")]
    console.print(
        f"\n  [dim]Total: {len(results)} combos — "
        f"[green]{len(good)} viable[/green], "
        f"[red]{len(results) - len(good) - len(err)} unprofitable[/red], "
        f"[yellow]{len(err)} errors[/yellow][/dim]\n"
    )

    # Recommended config changes
    console.print("[bold]Recommended combos to keep (PF >= 1.3, OOS not failing):[/bold]")
    for r in ok_sorted:
        if _verdict(r) in ("GOOD", "OK"):
            oos_str = f"OOS={r['oos_net_pf']:.2f}" if r.get("oos_net_pf") else "no OOS"
            console.print(
                f"  [green]✓[/green] {r['asset']:10} {r['strategy']:22} {r['tf']:4} "
                f"PF={r['net_pf']:.2f}  {oos_str}  n={r['trades']}"
            )


def _save_csv(results: list[dict], path: str) -> None:
    fields = ["asset", "strategy", "tf", "ticker", "signals", "trades",
              "win_rate", "net_pf", "avg_net_r", "max_dd", "sharpe", "kelly",
              "oos_trades", "oos_win_rate", "oos_net_pf", "oos_avg_net_r",
              "start", "end", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    console.print(f"  [dim]CSV saved → {path}[/dim]")


def main():
    parser = argparse.ArgumentParser(prog="run_grid_search")
    parser.add_argument("--asset",    default=None, help="Filter to one asset")
    parser.add_argument("--strategy", default=None, help="Filter to one strategy")
    parser.add_argument("--tf",       default=None, help="Filter to one timeframe")
    parser.add_argument("--csv",      default="outputs/grid_results.csv")
    args = parser.parse_args()

    assets     = [args.asset]    if args.asset    else ALL_ASSETS
    strategies = [args.strategy] if args.strategy else ALL_STRATEGIES

    # Build full combo list
    combos = []
    for asset in assets:
        for strategy in strategies:
            tfs = STRATEGY_TIMEFRAMES.get(strategy, ["1d"])
            if args.tf:
                tfs = [t for t in tfs if t == args.tf]
            for tf in tfs:
                combos.append((asset, strategy, tf))

    console.print()
    console.print(Panel(
        f"[bold gold1]ORCASTRADING[/bold gold1]  [dim]Full Grid Search[/dim]\n"
        f"[dim]{len(combos)} combinations  |  "
        f"1d: 2015-today (Stooq)  |  1h/4h: 729 days (yfinance)[/dim]",
        box=box.ROUNDED, padding=(1, 4),
    ))
    console.print()

    results = []
    for i, (asset_id, strategy_id, tf) in enumerate(combos, 1):
        console.print(
            f"  [{i:>2}/{len(combos)}] [bold]{asset_id:10}[/bold] "
            f"[cyan]{strategy_id:22}[/cyan] {tf}  … ",
            end="",
        )
        try:
            r = _run_one(asset_id, strategy_id, tf)
            if r.get("error") and not r.get("trades"):
                console.print(f"[yellow]skipped ({r['error']})[/yellow]")
            else:
                pf = r.get("net_pf") or 0
                n  = r.get("trades") or 0
                oos = r.get("oos_net_pf")
                color = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
                oos_str = f"  OOS={oos:.2f}" if oos else ""
                console.print(f"[{color}]{n} fills  PF={pf:.2f}[/{color}]{oos_str}")
            results.append(r)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            traceback.print_exc()
            results.append({"asset": asset_id, "strategy": strategy_id,
                             "tf": tf, "error": str(e)})

    _save_csv(results, args.csv)
    _print_summary(results)


if __name__ == "__main__":
    main()
