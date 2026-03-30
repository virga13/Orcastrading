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
    parser.add_argument("ticker",         type=str, nargs="?", default=None)
    parser.add_argument("--interval",     type=str, default="1d",
                        help="Candlestick interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk (default: 1d)")
    parser.add_argument("--start",        type=str, required=False, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",          type=str, required=False, help="End date YYYY-MM-DD")
    parser.add_argument("--signal-mode",  type=str, default="session-open",
                        choices=["every-n-bars", "session-open", "key-level-touch"])
    parser.add_argument("--every-n",      type=int, default=10,
                        help="Bar spacing for every-n-bars mode (default: 10)")
    parser.add_argument("--entry-timeout", type=int, default=20,
                        help="Bars to wait for entry fill (default: 20)")
    parser.add_argument("--model",        type=str, default="claude-sonnet-4-6")
    parser.add_argument("--strategy",     type=str, default=None,
                        help='Strategy name: "ema-continuation", "ema-pullback", "rule-engine"')
    parser.add_argument("--ema-period",   type=int, default=20,
                        help="EMA period for EMA-based strategies (default: 20)")
    parser.add_argument("--adx-threshold", type=int, default=20,
                        help="ADX threshold for trend filter (default: 20)")
    parser.add_argument("--top-adx-pct",  type=float, default=1.0,
                        help="Keep only top N%% signals by ADX strength, e.g. 0.30 (default: 1.0 = all)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation (train 60%% / val 20%% / test 20%%)")
    parser.add_argument("--query",        type=str, default=None,
                        help='Natural language query, e.g. "Backtest 25 EMA continuation for Gold on 1h"')
    parser.add_argument("--use-claude",   action="store_true",
                        help="Use Claude API for signal generation instead of rule engine")
    parser.add_argument("--simulate-only", action="store_true",
                        help="Skip Pass 1 — use existing cache only")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Re-run Pass 1 even if cache exists")
    parser.add_argument("--report",       action="store_true",
                        help="Generate HTML report and open in browser")
    parser.add_argument("--json",         action="store_true",
                        help="Print RunStats JSON to stdout")
    parser.add_argument("--list-runs",    action="store_true",
                        help="List cached backtest runs and exit")
    parser.add_argument("--list-strategies", action="store_true",
                        help="List available strategies and exit")
    return parser


def _interactive_prompt(query_hint: str | None = None) -> tuple[str, str, str, str, str | None]:
    from rich.prompt import Prompt
    console.print()
    console.print(Panel(
        "[bold gold1]ORCASTRADING[/bold gold1]  [dim]Historical Strategy Backtester[/dim]",
        box=box.ROUNDED, padding=(1, 4),
    ))
    console.print()

    if query_hint is None:
        console.print("  [dim]Type a natural language query or press Enter to configure manually.[/dim]")
        console.print("  [dim]Example: Backtest 25 EMA continuation for Gold on 1h[/dim]")
        console.print()
        query = Prompt.ask("  [bold]Query[/bold] [dim](or Enter to skip)[/dim]", default="").strip()
        if query:
            return None, None, None, None, query

    ticker   = Prompt.ask("[bold]  Asset ticker[/bold]  [dim](e.g. GOLD, BTC, NAS100)[/dim]")
    interval = Prompt.ask("  Interval", choices=["1m","5m","15m","30m","1h","1d","1wk"], default="1h")
    start    = Prompt.ask("  Start date [dim](YYYY-MM-DD)[/dim]")
    end      = Prompt.ask("  End date   [dim](YYYY-MM-DD)[/dim]")
    console.print()
    return ticker.upper().strip(), interval, start.strip(), end.strip(), None


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


def _list_strategies():
    from p3_backtester.strategies import list_strategies
    console.print()
    console.print("[bold]Available strategies:[/bold]")
    for s in list_strategies():
        console.print(f"  [green]{s}[/green]")
    console.print()
    console.print("[dim]Usage examples:[/dim]")
    console.print('  python -m p3_backtester --query "Backtest 25 EMA continuation for Gold on 1h"')
    console.print('  python -m p3_backtester --query "EMA pullback BTC 5m"')
    console.print('  python -m p3_backtester GOLD --interval 1h --strategy ema-continuation --ema-period 25')
    console.print()


def _resolve_strategy(args):
    from p3_backtester.strategies import build_strategy
    name = (args.strategy or "rule-engine").lower().replace(" ", "-")
    params = {"ema_period": args.ema_period, "adx_threshold": args.adx_threshold}
    if "mtf" in name or "multi" in name:
        return build_strategy("MTFTrend", params), "MTF Trend"
    if "continuation" in name:
        return build_strategy("EMAContinuation", params), "EMA Continuation"
    if "pullback" in name or "bounce" in name:
        return build_strategy("EMAPullback", params), "EMA Pullback"
    if "trend" in name:
        return build_strategy("MTFTrend", params), "MTF Trend"
    return build_strategy("RuleEngine", params), "Rule Engine"


def print_rich_stats(stats, show_walk_forward: bool = True) -> None:
    console.print()
    console.print(Panel(
        f"[bold]{stats.ticker}[/bold]  [dim]{stats.interval}[/dim]  "
        f"{stats.start_date} -> {stats.end_date}  "
        f"[dim]({stats.signal_mode} | {stats.strategy_name})[/dim]",
        title="[bold]Orcastrading P3 — Backtest Results[/bold]",
        box=box.ROUNDED,
    ))

    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column(style="dim", width=24)
    summary.add_column()
    summary.add_column(style="dim", width=24)
    summary.add_column()
    summary.add_row("Signal Bars",      str(stats.total_signals),
                    "Setups Generated", str(stats.total_setups_generated))
    summary.add_row("Trades Filled",    str(stats.total_trades_filled),
                    "Expired",          str(stats.total_trades_expired))
    console.print(summary)

    # Gross vs Net comparison
    comp = Table(title="Performance (Gross vs Net after Costs)", box=box.SIMPLE_HEAVY,
                 show_header=True, padding=(0, 2))
    comp.add_column("Metric",        style="dim", width=22)
    comp.add_column("Gross",         justify="right")
    comp.add_column("Net (after costs)", justify="right")

    def color(v, threshold=0):
        return "green" if v > threshold else ("yellow" if v == threshold else "red")

    comp.add_row("Win Rate",
                 f"{stats.actual_win_rate:.1%}", "—")
    comp.add_row("Avg EV (R)",
                 f"[{color(stats.actual_avg_pnl_r)}]{stats.actual_avg_pnl_r:+.3f}R[/{color(stats.actual_avg_pnl_r)}]",
                 f"[{color(stats.actual_avg_net_pnl_r)}]{stats.actual_avg_net_pnl_r:+.3f}R[/{color(stats.actual_avg_net_pnl_r)}]")
    comp.add_row("Profit Factor",
                 f"[{color(stats.actual_profit_factor, 1)}]{stats.actual_profit_factor:.2f}[/{color(stats.actual_profit_factor, 1)}]",
                 f"[{color(stats.actual_net_profit_factor, 1)}]{stats.actual_net_profit_factor:.2f}[/{color(stats.actual_net_profit_factor, 1)}]")
    comp.add_row("Max Drawdown (R)", f"-{stats.max_drawdown_r:.2f}R", "—")
    comp.add_row("Sharpe (R)",       f"{stats.sharpe_r:.2f}",         "—")
    console.print(comp)

    # Kelly sizing
    if stats.kelly_25pct > 0:
        console.print(f"\n  [bold]Position Sizing:[/bold] 25% Kelly = [green]{stats.kelly_25pct*100:.1f}% of account per trade[/green]")
        if stats.total_trades_filled < 50:
            console.print(f"  [yellow]Warning:[/yellow] only {stats.total_trades_filled} trades — use minimum size until n >= 100")

    # Walk-forward
    if show_walk_forward and stats.walk_forward:
        wf = Table(title="Walk-Forward Validation", box=box.SIMPLE_HEAVY,
                   show_header=True, padding=(0, 2))
        wf.add_column("Window",   style="dim")
        wf.add_column("Period")
        wf.add_column("Fills",    justify="right")
        wf.add_column("Win %",    justify="right")
        wf.add_column("Avg R",    justify="right")
        wf.add_column("Net Avg R",justify="right")
        wf.add_column("PF (net)", justify="right")

        for w in stats.walk_forward:
            is_test = w.name == "test"
            style = "bold" if is_test else ""
            label = f"[{style}]{w.name.upper()}[/{style}]" if style else w.name.upper()
            wf.add_row(
                label,
                f"{w.start_date} -> {w.end_date}",
                str(w.n_fills),
                f"{w.win_rate:.1%}",
                f"{w.avg_pnl_r:+.3f}R",
                f"[{color(w.avg_net_pnl_r)}]{w.avg_net_pnl_r:+.3f}R[/{color(w.avg_net_pnl_r)}]",
                f"[{color(w.net_profit_factor, 1)}]{w.net_profit_factor:.2f}[/{color(w.net_profit_factor, 1)}]",
            )
        console.print(wf)

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


def _run_walk_forward(config, df, signal_indices, strategy, asset_class, use_claude):
    """Run three separate backtests for train/val/test windows and attach to stats."""
    from p3_backtester.walk_forward import compute_walk_forward_dates, kelly_from_stats
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.schema import WalkForwardWindow

    try:
        train_r, val_r, test_r = compute_walk_forward_dates(
            df, config.start_date, config.end_date
        )
    except ValueError as e:
        console.print(f"  [yellow]Walk-forward skipped: {e}[/yellow]")
        return []

    windows = [("train", train_r), ("validation", val_r), ("test", test_r)]
    results = []

    for wname, (wstart, wend) in windows:
        from p3_backtester.market_data import split_backtest_window
        first_valid, _ = split_backtest_window(df, wstart)
        w_indices = get_signal_indices(df, config.signal_mode, first_valid, config.every_n_bars)
        # Restrict to within the window date range
        wstart_ts = __import__('pandas').Timestamp(wstart)
        wend_ts   = __import__('pandas').Timestamp(wend)
        w_indices = [i for i in w_indices if wstart_ts <= df.index[i] <= wend_ts]

        if not w_indices:
            continue

        import dataclasses
        w_config = config.model_copy(update={"start_date": wstart, "end_date": wend})
        w_signals = run_pass1(w_config, df, w_indices, use_claude=use_claude, strategy=strategy)
        if not w_signals:
            continue
        w_trades = run_pass2(w_config, df, w_signals, asset_class=asset_class)
        if not w_trades:
            continue
        w_stats = compute_stats(w_config, w_signals, w_trades, strategy_name=strategy.name)

        filled = [t for t in w_trades if t.outcome != "EXPIRED"]
        wins   = [t for t in filled if t.outcome in ("WIN", "PARTIAL_WIN")]
        losses = [t for t in filled if t.outcome == "LOSS"]
        net_gp = sum(t.net_pnl_r for t in wins)
        net_gl = abs(sum(t.net_pnl_r for t in losses))
        net_pf = net_gp / net_gl if net_gl > 0 else (999.0 if net_gp > 0 else 0.0)

        results.append(WalkForwardWindow(
            name=wname,
            start_date=wstart,
            end_date=wend,
            n_fills=len(filled),
            win_rate=w_stats.actual_win_rate,
            avg_pnl_r=w_stats.actual_avg_pnl_r,
            avg_net_pnl_r=w_stats.actual_avg_net_pnl_r,
            profit_factor=w_stats.actual_profit_factor,
            net_profit_factor=round(net_pf, 3),
            max_drawdown_r=w_stats.max_drawdown_r,
            sharpe_r=w_stats.sharpe_r,
            kelly_25pct=kelly_from_stats(w_stats),
        ))

    return results


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_runs:
        _list_runs()
        return

    if args.list_strategies:
        _list_strategies()
        return

    from dotenv import load_dotenv
    load_dotenv(override=True)

    from p1_analysis_engine.utils.asset_classifier import resolve_ticker, classify_asset
    from p3_backtester.schema import BacktestConfig
    from p3_backtester.market_data import fetch_ohlcv, split_backtest_window, MarketDataError
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats, save_to_db
    from p3_backtester.strategies import build_strategy, parse_query

    # ── Resolve inputs ────────────────────────────────────────────────────────
    query = args.query
    ticker = args.ticker
    interval = args.interval
    start = args.start
    end = args.end
    strategy_obj = None
    strategy_label = "Rule Engine"

    if query:
        # Natural language mode
        parsed = parse_query(query)
        console.print(f"\n  [dim]Query parsed:[/dim]")
        console.print(f"    Strategy : [bold]{parsed['strategy_name']}[/bold]  params={parsed['params']}")
        console.print(f"    Ticker   : [bold]{parsed['ticker'] or '(not detected)'}[/bold]")
        console.print(f"    Interval : [bold]{parsed['interval']}[/bold]")
        console.print()

        ticker   = parsed["ticker"] or ticker
        interval = parsed["interval"]
        strategy_obj, strategy_label = build_strategy(parsed["strategy_name"], parsed["params"]), parsed["strategy_name"]

        if not ticker:
            from rich.prompt import Prompt
            ticker = Prompt.ask("  Ticker not detected in query. Enter asset ticker").upper().strip()
        if not start or not end:
            start, end = _auto_date_range(interval)
            console.print(f"  [dim]Auto date range: {start} to {end}[/dim]")

    elif ticker is None:
        ticker_raw, interval, start, end, inline_query = _interactive_prompt()
        if inline_query:
            parsed = parse_query(inline_query)
            ticker   = parsed["ticker"]
            interval = parsed["interval"]
            strategy_obj, strategy_label = build_strategy(parsed["strategy_name"], parsed["params"]), parsed["strategy_name"]
            if not ticker:
                from rich.prompt import Prompt
                ticker = Prompt.ask("  Ticker not detected. Enter asset ticker").upper().strip()
            if not start or not end:
                start, end = _auto_date_range(interval)
        else:
            ticker = ticker_raw
    else:
        if not start or not end:
            start, end = _auto_date_range(interval)
            console.print(f"  [dim]Auto date range: {start} to {end}[/dim]")

    ticker = resolve_ticker(ticker)
    asset_class = classify_asset(ticker)

    if strategy_obj is None:
        strategy_obj, strategy_label = _resolve_strategy(args)

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
        top_adx_pct=args.top_adx_pct,
    )

    console.print(f"\n[bold]Strategy:[/bold] {strategy_obj}")
    console.print(f"[bold]Asset:[/bold]    {ticker}  ({asset_class})")
    console.print(f"[bold]Window:[/bold]   {interval}  {start} -> {end}\n")

    # ── Fetch OHLCV ───────────────────────────────────────────────────────────
    console.print(f"[bold]Fetching OHLCV ...[/bold]")
    try:
        df = fetch_ohlcv(ticker, interval, start, end)
    except MarketDataError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    console.print(f"  [dim]{len(df)} bars loaded[/dim]")

    first_valid, _ = split_backtest_window(df, start)
    signal_indices = get_signal_indices(df, config.signal_mode, first_valid, config.every_n_bars)
    console.print(f"  [dim]{len(signal_indices)} signal bars ({config.signal_mode})[/dim]\n")

    # ── Pass 1 ────────────────────────────────────────────────────────────────
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
        signals = run_pass1(config, df, signal_indices,
                            use_claude=use_claude, strategy=strategy_obj)

    if not signals:
        console.print("[bold red]No signals generated. Aborting.[/bold red]")
        sys.exit(1)

    # ── Pass 2 ────────────────────────────────────────────────────────────────
    trades = run_pass2(config, df, signals, asset_class=asset_class)

    if not trades:
        console.print("[bold red]No trades simulated.[/bold red]")
        sys.exit(1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    stats = compute_stats(config, signals, trades, strategy_name=strategy_label)

    # ── Walk-forward ──────────────────────────────────────────────────────────
    if args.walk_forward:
        console.print("\n[bold]Running walk-forward validation ...[/bold]")
        wf_windows = _run_walk_forward(config, df, signal_indices, strategy_obj, asset_class, use_claude)
        stats.walk_forward = wf_windows

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


def _auto_date_range(interval: str) -> tuple[str, str]:
    """Return a sensible start/end date within yfinance limits for the given interval."""
    from datetime import date, timedelta
    end = date.today().strftime("%Y-%m-%d")
    lookback = {
        "1m": 5, "5m": 50, "15m": 50, "30m": 50,
        "1h": 640, "1d": 3650, "1wk": 7300,   # 1d -> 10 years, 1wk -> 20 years
    }.get(interval, 365)
    start = (date.today() - timedelta(days=lookback)).strftime("%Y-%m-%d")
    return start, end


if __name__ == "__main__":
    main()
