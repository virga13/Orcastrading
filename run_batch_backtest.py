"""
run_batch_backtest.py — run backtests for all enabled (asset, strategy, timeframe) triples.

Usage:
    python run_batch_backtest.py                    # all watchlist combos
    python run_batch_backtest.py --asset gold       # one asset, all its strategies
    python run_batch_backtest.py --strategy mtf_trend
    python run_batch_backtest.py --csv results.csv  # also save summary to CSV
    python run_batch_backtest.py --walk-forward     # add walk-forward per combo
"""
import argparse
import sys
import traceback
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


def _date_range(lookback_days: int, interval: str,
                start_override: str | None = None) -> tuple[str, str]:
    """
    Return (start, end) covering the strategy's lookback window.
    start_override sets a fixed start date (e.g. '2015-01-01') for max history.
    Intraday intervals are still hard-capped at what the data source can provide.
    """
    # Hard caps on intraday data (yfinance / Stooq limits)
    intraday_caps = {"1m": 5, "5m": 50, "15m": 50, "30m": 50, "1h": 729, "4h": 729}
    end = date.today().strftime("%Y-%m-%d")

    if start_override and interval not in intraday_caps:
        return start_override, end

    # Intraday: ignore start_override — data sources cap these regardless
    cap_days = intraday_caps.get(interval, lookback_days)
    days  = min(lookback_days, cap_days) if not start_override else cap_days
    start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    return start, end


def _run_one(asset_id: str, strategy_id: str, timeframe: str,
             walk_forward: bool, start_override: str | None = None) -> dict | None:
    """
    Run a single backtest combination.
    Returns a result dict, or None on hard failure.
    """
    from core.config import (
        get_ticker, get_strategy_lookback, get_asset_strategy_params,
        get_strategy_params,
    )
    from p3_backtester.strategies.registry import build_from_config
    from p3_backtester.schema import BacktestConfig
    from p3_backtester.market_data import fetch_ohlcv, split_backtest_window, MarketDataError
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats
    from p1_analysis_engine.utils.asset_classifier import classify_asset

    ticker = get_ticker(asset_id, source="yfinance")
    asset_class = classify_asset(ticker)
    lookback = get_strategy_lookback(strategy_id)
    start, end = _date_range(lookback, timeframe, start_override=start_override)

    # Merge default params with any per-asset overrides
    overrides = get_asset_strategy_params(asset_id, strategy_id)
    strategy = build_from_config(strategy_id, overrides)

    config = BacktestConfig(
        ticker=ticker,
        interval=timeframe,
        start_date=start,
        end_date=end,
        signal_mode="session-open",
        entry_timeout_bars=20,
    )

    try:
        df = fetch_ohlcv(ticker, timeframe, start, end)
    except MarketDataError as e:
        return {"asset": asset_id, "strategy": strategy_id, "tf": timeframe,
                "error": f"data: {e}", "ticker": ticker}

    if len(df) < 30:
        return {"asset": asset_id, "strategy": strategy_id, "tf": timeframe,
                "error": f"too few bars ({len(df)})", "ticker": ticker}

    first_valid, _ = split_backtest_window(df, start)
    signal_indices = get_signal_indices(df, config.signal_mode, first_valid, config.every_n_bars)

    signals = run_pass1(config, df, signal_indices, use_claude=False, strategy=strategy)
    if not signals:
        return {"asset": asset_id, "strategy": strategy_id, "tf": timeframe,
                "ticker": ticker, "trades": 0, "signals": 0,
                "win_rate": 0, "net_pf": 0, "avg_net_r": 0,
                "max_dd": 0, "kelly": 0, "error": "no signals"}

    trades = run_pass2(config, df, signals, asset_class=asset_class)
    filled = [t for t in trades if t.outcome != "EXPIRED"]

    if not filled:
        return {"asset": asset_id, "strategy": strategy_id, "tf": timeframe,
                "ticker": ticker, "trades": 0, "signals": len(signals),
                "win_rate": 0, "net_pf": 0, "avg_net_r": 0,
                "max_dd": 0, "kelly": 0, "error": "no fills"}

    stats = compute_stats(config, signals, trades, strategy_name=strategy.name)

    result = {
        "asset":      asset_id,
        "strategy":   strategy_id,
        "tf":         timeframe,
        "ticker":     ticker,
        "signals":    stats.total_signals,
        "trades":     stats.total_trades_filled,
        "win_rate":   stats.actual_win_rate,
        "net_pf":     stats.actual_net_profit_factor,
        "avg_net_r":  stats.actual_avg_net_pnl_r,
        "max_dd":     stats.max_drawdown_r,
        "kelly":      stats.kelly_25pct,
        "error":      None,
        "start":      start,
        "end":        end,
    }

    if walk_forward:
        from p3_backtester.walk_forward import compute_walk_forward_dates, kelly_from_stats
        from p3_backtester.schema import WalkForwardWindow
        try:
            train_r, val_r, test_r = compute_walk_forward_dates(df, start, end)
            test_start, test_end = test_r
            import pandas as _pd
            ts = _pd.Timestamp(test_start)
            te = _pd.Timestamp(test_end)
            w_idx = [i for i in signal_indices
                     if ts <= df.index[i] <= te]
            if w_idx:
                w_signals = run_pass1(config.model_copy(
                    update={"start_date": test_start, "end_date": test_end}),
                    df, w_idx, use_claude=False, strategy=strategy)
                if w_signals:
                    w_trades = run_pass2(
                        config.model_copy(
                            update={"start_date": test_start, "end_date": test_end}),
                        df, w_signals, asset_class=asset_class)
                    w_stats = compute_stats(
                        config.model_copy(
                            update={"start_date": test_start, "end_date": test_end}),
                        w_signals, w_trades, strategy_name=strategy.name)
                    result["oos_trades"]  = w_stats.total_trades_filled
                    result["oos_win_rate"] = w_stats.actual_win_rate
                    result["oos_net_pf"]   = w_stats.actual_net_profit_factor
                    result["oos_avg_net_r"] = w_stats.actual_avg_net_pnl_r
        except Exception:
            pass  # walk-forward optional — don't fail the whole run

    return result


def _print_summary(results: list[dict], walk_forward: bool) -> None:
    ok  = [r for r in results if not r.get("error") or r.get("trades", 0) > 0]
    err = [r for r in results if r.get("error") and r.get("trades", 0) == 0]

    console.print()
    t = Table(
        title="Batch Backtest — Summary",
        box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1),
    )
    t.add_column("Asset",    style="bold", width=12)
    t.add_column("Strategy", width=18)
    t.add_column("TF",       width=5)
    t.add_column("Fills",    justify="right", width=6)
    t.add_column("WR%",      justify="right", width=6)
    t.add_column("Net PF",   justify="right", width=7)
    t.add_column("Avg Net R",justify="right", width=9)
    t.add_column("Max DD",   justify="right", width=7)
    t.add_column("Kelly%",   justify="right", width=7)
    if walk_forward:
        t.add_column("OOS Fills", justify="right", width=9)
        t.add_column("OOS PF",    justify="right", width=7)

    def _pf_color(v):
        return "green" if v >= 1.5 else ("yellow" if v >= 1.0 else "red")

    def _wr_color(v):
        return "green" if v >= 0.45 else ("yellow" if v >= 0.35 else "red")

    for r in sorted(ok, key=lambda x: x.get("net_pf", 0), reverse=True):
        pf  = r.get("net_pf", 0)
        wr  = r.get("win_rate", 0)
        row = [
            r["asset"],
            r["strategy"],
            r["tf"],
            str(r.get("trades", 0)),
            f"[{_wr_color(wr)}]{wr:.1%}[/{_wr_color(wr)}]",
            f"[{_pf_color(pf)}]{pf:.2f}[/{_pf_color(pf)}]",
            f"{r.get('avg_net_r', 0):+.3f}R",
            f"-{r.get('max_dd', 0):.1f}R",
            f"{r.get('kelly', 0)*100:.1f}%",
        ]
        if walk_forward:
            oos_pf = r.get("oos_net_pf", 0) or 0
            row += [
                str(r.get("oos_trades", "-")),
                f"[{_pf_color(oos_pf)}]{oos_pf:.2f}[/{_pf_color(oos_pf)}]"
                if r.get("oos_net_pf") else "—",
            ]
        t.add_row(*row)

    console.print(t)

    if err:
        console.print()
        e_table = Table(title="Skipped / Errors", box=box.SIMPLE, padding=(0, 1))
        e_table.add_column("Asset",    width=12)
        e_table.add_column("Strategy", width=18)
        e_table.add_column("TF",       width=5)
        e_table.add_column("Reason",   style="dim")
        for r in err:
            e_table.add_row(r["asset"], r["strategy"], r["tf"], r.get("error", ""))
        console.print(e_table)

    console.print(
        f"\n  [dim]Total: {len(results)} combinations — "
        f"[green]{len(ok)} ran[/green], "
        f"[yellow]{len(err)} skipped[/yellow][/dim]\n"
    )


def _save_csv(results: list[dict], path: str) -> None:
    import csv
    fields = ["asset", "strategy", "tf", "ticker", "signals", "trades",
              "win_rate", "net_pf", "avg_net_r", "max_dd", "kelly",
              "oos_trades", "oos_win_rate", "oos_net_pf", "oos_avg_net_r",
              "start", "end", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    console.print(f"  [dim]CSV saved to {path}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        prog="run_batch_backtest",
        description="Run backtests for all enabled (asset, strategy, timeframe) combinations.",
    )
    parser.add_argument("--asset",        type=str, default=None,
                        help="Filter to a single asset_id (e.g. gold, spx500)")
    parser.add_argument("--strategy",     type=str, default=None,
                        help="Filter to a single strategy_id (e.g. mtf_trend, ema_continuation)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run OOS (test) window walk-forward for each combination")
    parser.add_argument("--start",        type=str, default=None,
                        help="Override start date for daily strategies (e.g. 2015-01-01). "
                             "Intraday timeframes are still capped by data-source limits.")
    parser.add_argument("--csv",          type=str, default=None,
                        help="Save results summary to a CSV file")
    args = parser.parse_args()

    from core.config import get_watchlist

    combos = get_watchlist()

    if args.asset:
        combos = [c for c in combos if c[0] == args.asset.lower()]
    if args.strategy:
        combos = [c for c in combos if c[1] == args.strategy.lower()]

    if not combos:
        console.print("[red]No matching combinations found.[/red]")
        sys.exit(1)

    console.print()
    console.print(Panel(
        f"[bold gold1]ORCASTRADING[/bold gold1]  [dim]Batch Backtest Runner[/dim]\n"
        f"[dim]{len(combos)} combination(s) queued[/dim]",
        box=box.ROUNDED, padding=(1, 4),
    ))
    console.print()

    results = []
    for i, (asset_id, strategy_id, tf) in enumerate(combos, 1):
        console.print(
            f"  [{i}/{len(combos)}] [bold]{asset_id}[/bold] / "
            f"[cyan]{strategy_id}[/cyan] / {tf}  … ",
            end="",
        )
        try:
            r = _run_one(asset_id, strategy_id, tf, args.walk_forward,
                         start_override=args.start)
            if r is None:
                console.print("[red]hard error[/red]")
                results.append({"asset": asset_id, "strategy": strategy_id,
                                 "tf": tf, "error": "exception (see above)"})
            elif r.get("error") and r.get("trades", 0) == 0:
                console.print(f"[yellow]skipped ({r['error']})[/yellow]")
                results.append(r)
            else:
                pf = r.get("net_pf", 0)
                color = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
                console.print(
                    f"[{color}]{r['trades']} fills, PF={pf:.2f}[/{color}]"
                )
                results.append(r)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            traceback.print_exc()
            results.append({"asset": asset_id, "strategy": strategy_id,
                             "tf": tf, "error": str(e)})

    _print_summary(results, args.walk_forward)

    if args.csv:
        _save_csv(results, args.csv)


if __name__ == "__main__":
    main()
