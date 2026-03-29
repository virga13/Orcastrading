import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="p3_backtester",
        description="Orcastrading P3 — Historical Strategy Backtester",
    )
    parser.add_argument("ticker",      type=str, nargs="?", default=None)
    parser.add_argument("--interval",  type=str, default="1d",
                        help="Candlestick interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk (default: 1d)")
    parser.add_argument("--start",     type=str, required=False, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",       type=str, required=False, help="End date YYYY-MM-DD")
    parser.add_argument("--signal-mode", type=str, default="session-open",
                        choices=["every-n-bars", "session-open", "key-level-touch"])
    parser.add_argument("--every-n",   type=int, default=10,
                        help="Bar spacing for every-n-bars mode (default: 10)")
    parser.add_argument("--entry-timeout", type=int, default=20,
                        help="Bars to wait for entry fill (default: 20)")
    parser.add_argument("--model",     type=str, default="claude-sonnet-4-6")
    parser.add_argument("--use-claude", action="store_true",
                        help="Use Claude API for signal generation instead of rule engine (costs ~$0.025-0.035/signal)")
    parser.add_argument("--simulate-only", action="store_true",
                        help="Skip Pass 1 — use existing cache only")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Re-run Pass 1 even if cache exists")
    parser.add_argument("--report",    action="store_true",
                        help="Generate HTML report and open in browser")
    parser.add_argument("--json",      action="store_true",
                        help="Print RunStats JSON to stdout")
    parser.add_argument("--list-runs", action="store_true",
                        help="List cached backtest runs and exit")
    return parser


def _interactive_prompt() -> tuple[str, str, str, str]:
    from rich.prompt import Prompt
    console.print()
    console.print(Panel(
        "[bold gold1]ORCASTRADING[/bold gold1]  [dim]Historical Strategy Backtester[/dim]",
        box=box.ROUNDED, padding=(1, 4),
    ))
    console.print()
    ticker = Prompt.ask("[bold]  Asset ticker[/bold]  [dim](e.g. AAPL, BTC, SILVER)[/dim]")
    interval = Prompt.ask("  Interval", choices=["1m","5m","15m","30m","1h","1d","1wk"], default="1d")
    start = Prompt.ask("  Start date [dim](YYYY-MM-DD)[/dim]")
    end   = Prompt.ask("  End date   [dim](YYYY-MM-DD)[/dim]")
    console.print()
    return ticker.upper().strip(), interval, start.strip(), end.strip()


def _list_runs():
    results_dir = Path(__file__).parent / "results"
    dbs = sorted(results_dir.glob("backtest_*.db"))
    if not dbs:
        console.print("[dim]No backtest runs found.[/dim]")
        return
    t = Table(box=box.SIMPLE, show_header=True)
    t.add_column("File", style="dim")
    t.add_column("Size")
    for db in dbs:
        t.add_row(db.name, f"{db.stat().st_size / 1024:.1f} KB")
    console.print(t)


def print_rich_stats(stats) -> None:
    console.print()
    console.print(Panel(
        f"[bold]{stats.ticker}[/bold]  [dim]{stats.interval}[/dim]  "
        f"{stats.start_date} -> {stats.end_date}  [dim]({stats.signal_mode})[/dim]",
        title="[bold]Orcastrading P3 — Backtest Results[/bold]",
        box=box.ROUNDED,
    ))

    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column(style="dim", width=22)
    summary.add_column()
    summary.add_column(style="dim", width=22)
    summary.add_column()
    summary.add_row("Signal Bars",       str(stats.total_signals),
                    "Setups Generated",  str(stats.total_setups_generated))
    summary.add_row("Trades Filled",     str(stats.total_trades_filled),
                    "Expired",           str(stats.total_trades_expired))
    console.print(summary)

    # Claude vs Actual comparison
    comp = Table(title="Claude Estimates vs Actual", box=box.SIMPLE_HEAVY,
                 show_header=True, padding=(0, 2))
    comp.add_column("Metric",        style="dim", width=20)
    comp.add_column("Claude",        justify="right")
    comp.add_column("Actual",        justify="right")
    comp.add_column("Delta",         justify="right")

    def delta_color(d: float) -> str:
        return "green" if d >= 0 else "red"

    wr_delta = stats.actual_win_rate - stats.avg_claude_win_rate
    ev_delta = stats.actual_avg_pnl_r - stats.avg_claude_ev

    comp.add_row("Win Rate",
                 f"{stats.avg_claude_win_rate:.1%}",
                 f"{stats.actual_win_rate:.1%}",
                 f"[{delta_color(wr_delta)}]{wr_delta:+.1%}[/{delta_color(wr_delta)}]")
    comp.add_row("Avg EV (R)",
                 f"{stats.avg_claude_ev:+.2f}",
                 f"{stats.actual_avg_pnl_r:+.2f}",
                 f"[{delta_color(ev_delta)}]{ev_delta:+.2f}[/{delta_color(ev_delta)}]")
    comp.add_row("Profit Factor",    "—", f"{stats.actual_profit_factor:.2f}", "")
    comp.add_row("Max Drawdown (R)", "—", f"-{stats.max_drawdown_r:.2f}R",    "")
    comp.add_row("Sharpe (R)",       "—", f"{stats.sharpe_r:.2f}",            "")
    console.print(comp)

    # By type
    by_type_t = Table(title="By Trade Type", box=box.SIMPLE, show_header=True, padding=(0, 2))
    by_type_t.add_column("Type",    style="dim")
    by_type_t.add_column("Trades",  justify="right")
    by_type_t.add_column("Win %",   justify="right")
    by_type_t.add_column("Avg R",   justify="right")
    by_type_t.add_column("PF",      justify="right")
    for tt, cs in stats.by_trade_type.items():
        if cs.n_trades == 0:
            continue
        by_type_t.add_row(tt, str(cs.n_trades),
                          f"{cs.win_rate:.1%}", f"{cs.avg_pnl_r:+.2f}", f"{cs.profit_factor:.2f}")
    console.print(by_type_t)
    console.print()


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_runs:
        _list_runs()
        return

    if args.ticker is None:
        ticker, interval, start, end = _interactive_prompt()
    else:
        ticker   = args.ticker.upper().strip()
        interval = args.interval
        start    = args.start
        end      = args.end
        if not start or not end:
            console.print("[bold red]Error:[/bold red] --start and --end are required")
            sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv(override=True)

    from p1_analysis_engine.utils.asset_classifier import resolve_ticker
    from p3_backtester.schema import BacktestConfig
    from p3_backtester.market_data import fetch_ohlcv, split_backtest_window, MarketDataError
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats, save_to_db

    ticker = resolve_ticker(ticker)

    config = BacktestConfig(
        ticker=ticker,
        interval=interval,
        start_date=start,
        end_date=end,
        signal_mode=args.signal_mode,
        every_n_bars=args.every_n,
        entry_timeout_bars=args.entry_timeout,
        model=args.model,
        force_regenerate=args.force_regenerate,
    )

    # ── Fetch OHLCV ──
    console.print(f"\n[bold]Fetching OHLCV:[/bold] {ticker}  {interval}  {start} -> {end}")
    try:
        df = fetch_ohlcv(ticker, interval, start, end)
    except MarketDataError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    console.print(f"  [dim]{len(df)} bars loaded[/dim]")

    first_valid, _ = split_backtest_window(df, start)
    signal_indices = get_signal_indices(df, config.signal_mode, first_valid, config.every_n_bars)
    console.print(f"  [dim]{len(signal_indices)} signal bars scheduled ({config.signal_mode})[/dim]\n")

    # ── Pass 1 ──
    use_claude = getattr(args, "use_claude", False)

    if args.simulate_only:
        from p3_backtester.pass1_signal_gen import _cache_path, CACHE_DIR
        import json as _json
        signals = []
        for i in signal_indices:
            cp = _cache_path(ticker, interval, df.index[i], use_claude)
            if cp.exists():
                raw = _json.loads(cp.read_text(encoding="utf-8"))
                from p3_backtester.schema import SignalRecord
                signals.append(SignalRecord(**raw["_signal_meta"]))
        console.print(f"  [dim]--simulate-only: loaded {len(signals)} cached signals[/dim]\n")
    else:
        if use_claude:
            console.print("  [bold cyan]Mode: Claude API[/bold cyan] — costs ~$0.025-0.035 per signal\n")
        else:
            console.print("  [bold green]Mode: Rule engine[/bold green] — free, deterministic\n")
        signals = run_pass1(config, df, signal_indices, use_claude=use_claude)

    if not signals:
        console.print("[bold red]No signals generated. Aborting.[/bold red]")
        sys.exit(1)

    # ── Pass 2 ──
    trades = run_pass2(config, df, signals)

    if not trades:
        console.print("[bold red]No trades simulated.[/bold red]")
        sys.exit(1)

    # ── Aggregate ──
    stats = compute_stats(config, signals, trades)
    db = save_to_db(config, signals, trades, stats)
    console.print(f"  [dim]Results saved to {db}[/dim]\n")

    if args.json:
        print(stats.model_dump_json(indent=2))
        return

    print_rich_stats(stats)

    if args.report:
        from p3_backtester.report import save_report
        import webbrowser
        report_path = save_report(stats, trades, db)
        console.print(f"[dim]Report saved to {report_path}[/dim]")
        webbrowser.open(f"file:///{report_path}")


if __name__ == "__main__":
    main()
