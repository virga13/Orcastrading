"""
run_gold_scalper_backtest.py — Full backtest of the Gold Scalper strategy (GC=F, 5m).

Loads from data/XAUUSD_5m.parquet when available (built by scripts/download_gold_5m.py),
otherwise falls back to yfinance (~58 calendar days).

Signal mode : every-n-bars  N=3  (one evaluation every 15 min)
Entry timeout: 6 bars (30 min — tight zone, near-market entry)
Walk-forward : 60% train / 20% val / 20% test

Usage:
    python run_gold_scalper_backtest.py
    python run_gold_scalper_backtest.py --from 2023-01-01   # limit start date
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ── Config ─────────────────────────────────────────────────────────────────────
TICKER      = "GC=F"
LABEL       = "Gold (XAU/USD)"
INTERVAL    = "5m"
PARQUET     = Path(__file__).parent.parent.parent / "data" / "XAUUSD_5m.parquet"
YFINANCE_DAYS = 57   # yfinance 5m cap

EVERY_N_BARS      = 3    # signal every 3 × 5m = 15 min
ENTRY_TIMEOUT     = 6    # 30 min to fill the tight near-market zone


def _load_ohlcv(start_from: str | None) -> tuple[str, str, "pd.DataFrame"]:
    """Load 5m OHLCV: parquet cache first, yfinance fallback."""
    import pandas as pd
    from p3_backtester.market_data import fetch_ohlcv, MarketDataError

    if PARQUET.exists():
        console.print(f"[bold]Loading 5m OHLCV from cache ...[/bold]")
        df = pd.read_parquet(PARQUET)
        if start_from:
            df = df[df.index >= pd.Timestamp(start_from)]
        start = df.index[0].date().isoformat()
        end   = df.index[-1].date().isoformat()
        console.print(f"  {len(df):,} bars  ({start} to {end})  [dim](parquet)[/dim]")
        return start, end, df

    console.print("[bold]Fetching 5m OHLCV from yfinance ...[/bold]")
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=YFINANCE_DAYS)).isoformat()
    if start_from and start_from > start:
        start = start_from
    try:
        df = fetch_ohlcv(TICKER, INTERVAL, start, end, source="auto")
    except MarketDataError as e:
        console.print(f"[red]Data fetch failed: {e}[/red]")
        sys.exit(1)
    console.print(f"  {len(df):,} bars  ({df.index[0].date()} to {df.index[-1].date()})  [dim](yfinance)[/dim]")
    return start, end, df


def main():
    parser = argparse.ArgumentParser(description="Backtest Gold Scalper on 5m XAU/USD")
    parser.add_argument("--from", dest="start_from", default=None,
                        help="Limit start date YYYY-MM-DD (only applies when using parquet)")
    args = parser.parse_args()

    from p3_backtester.strategies.registry import build_from_config
    from p3_backtester.schema import BacktestConfig, WalkForwardWindow
    from p3_backtester.market_data import split_backtest_window
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats
    from p3_backtester.walk_forward import compute_walk_forward_dates, kelly_from_stats
    from p1_analysis_engine.utils.asset_classifier import classify_asset
    import pandas as pd

    strategy    = build_from_config("gold_scalper")
    asset_class = classify_asset(TICKER)

    START_DATE, END_DATE, df = _load_ohlcv(args.start_from)

    console.print()
    console.print(Panel(
        f"[bold gold1]GOLD SCALPER BACKTEST[/bold gold1]  "
        f"[dim]{LABEL}  |  {INTERVAL}  |  {START_DATE} to {END_DATE}[/dim]",
        box=box.ROUNDED, padding=(0, 4),
    ))
    console.print(f"  Strategy : [bold]{strategy}[/bold]")
    console.print(f"  Signal   : every {EVERY_N_BARS} bars ({EVERY_N_BARS * 5} min)")
    console.print(f"  Entry TO : {ENTRY_TIMEOUT} bars ({ENTRY_TIMEOUT * 5} min)")
    console.print()

    config = BacktestConfig(
        ticker            = TICKER,
        interval          = INTERVAL,
        start_date        = START_DATE,
        end_date          = END_DATE,
        signal_mode       = "every-n-bars",
        every_n_bars      = EVERY_N_BARS,
        entry_timeout_bars = ENTRY_TIMEOUT,
        model             = "claude-sonnet-4-6",
        force_regenerate  = True,
        top_adx_pct       = 1.0,
    )

    # Pre-compute HTF indicator series once to avoid O(n²) per-bar resampling
    console.print("[bold]Pre-computing M15/M30 indicator series ...[/bold]")
    strategy.prime(df)
    console.print("  Done.\n")

    first_valid, _ = split_backtest_window(df, START_DATE)
    signal_indices = get_signal_indices(df, "every-n-bars", first_valid, EVERY_N_BARS)
    console.print(f"  {len(signal_indices)} signal bars to evaluate\n")

    # ── Pass 1: signal generation ───────────────────────────────────────────────
    console.print("[bold]Pass 1 — generating signals ...[/bold]")
    signals = run_pass1(config, df, signal_indices, use_claude=False, strategy=strategy)
    if not signals:
        console.print("[red]No signals generated — check warmup requirements.[/red]")
        sys.exit(1)
    console.print(f"  {len(signals)} signal bars produced a setup\n")

    # ── Pass 2: trade simulation ────────────────────────────────────────────────
    console.print("[bold]Pass 2 — simulating trades ...[/bold]")
    trades = run_pass2(config, df, signals, asset_class=asset_class)
    if not trades:
        console.print("[red]No trades simulated.[/red]")
        sys.exit(1)

    filled  = [t for t in trades if t.outcome != "EXPIRED"]
    console.print(f"  {len(trades)} setups to{len(filled)} fills\n")

    # ── Full-period stats ───────────────────────────────────────────────────────
    stats = compute_stats(config, signals, trades, strategy_name=strategy.name)

    _print_stats(stats, filled)

    # ── Walk-forward ────────────────────────────────────────────────────────────
    console.print("\n[bold]Walk-forward validation (60% train / 20% val / 20% test) ...[/bold]")
    wf_windows = _walk_forward(
        config, df, asset_class, strategy,
        first_valid, pd.Timestamp(START_DATE), pd.Timestamp(END_DATE),
    )
    if wf_windows:
        _print_walk_forward(wf_windows)
    else:
        console.print("  [yellow]Walk-forward skipped (insufficient data for 3 windows)[/yellow]")

    console.print()


def _walk_forward(config, df, asset_class, strategy, first_valid, ts_start, ts_end):
    from p3_backtester.walk_forward import compute_walk_forward_dates, kelly_from_stats
    from p3_backtester.schema import WalkForwardWindow
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats
    from p3_backtester.market_data import split_backtest_window
    from p3_backtester.strategies.registry import build_from_config
    import pandas as pd

    try:
        train_r, val_r, test_r = compute_walk_forward_dates(df, config.start_date, config.end_date)
    except ValueError as e:
        console.print(f"  [yellow]{e}[/yellow]")
        return []

    results = []
    for wname, (wstart, wend) in [("train", train_r), ("val", val_r), ("test", test_r)]:
        first_w, _ = split_backtest_window(df, wstart)
        w_idx = get_signal_indices(df, "every-n-bars", first_w, EVERY_N_BARS)
        w_idx = [i for i in w_idx if pd.Timestamp(wstart) <= df.index[i] <= pd.Timestamp(wend)]
        if not w_idx:
            continue

        wc = config.model_copy(update={
            "start_date": wstart, "end_date": wend, "force_regenerate": True
        })
        wstrat = build_from_config("gold_scalper")
        wstrat.prime(df)
        ws = run_pass1(wc, df, w_idx, use_claude=False, strategy=wstrat)
        if not ws:
            continue
        wt = run_pass2(wc, df, ws, asset_class=asset_class)
        if not wt:
            continue
        wst = compute_stats(wc, ws, wt, strategy_name=strategy.name)

        filled  = [t for t in wt if t.outcome != "EXPIRED"]
        wins    = [t for t in filled if t.outcome in ("WIN", "PARTIAL_WIN")]
        losses  = [t for t in filled if t.outcome == "LOSS"]
        gp = sum(t.net_pnl_r for t in wins)
        gl = abs(sum(t.net_pnl_r for t in losses))
        npf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

        results.append(WalkForwardWindow(
            name=wname, start_date=wstart, end_date=wend,
            n_fills=len(filled), win_rate=wst.actual_win_rate,
            avg_pnl_r=wst.actual_avg_pnl_r,
            avg_net_pnl_r=wst.actual_avg_net_pnl_r,
            profit_factor=wst.actual_profit_factor,
            net_profit_factor=round(npf, 3),
            max_drawdown_r=wst.max_drawdown_r,
            sharpe_r=wst.sharpe_r,
            kelly_25pct=kelly_from_stats(wst),
        ))
    return results


def _print_stats(stats, filled):
    def c(v, thr=0): return "green" if v > thr else ("yellow" if v == thr else "red")

    tbl = Table(title="Full Period Results", box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 2))
    tbl.add_column("Metric",       style="dim", width=22)
    tbl.add_column("Gross",        justify="right")
    tbl.add_column("Net (costs)",  justify="right")

    tbl.add_row("Fills / Signals",
                f"{stats.total_trades_filled} / {stats.total_signals}", "—")
    tbl.add_row("Expired",         str(stats.total_trades_expired), "—")
    tbl.add_row("Win Rate",
                f"{stats.actual_win_rate:.1%}", "—")
    tbl.add_row("Avg EV (R)",
                f"[{c(stats.actual_avg_pnl_r)}]{stats.actual_avg_pnl_r:+.3f}R[/{c(stats.actual_avg_pnl_r)}]",
                f"[{c(stats.actual_avg_net_pnl_r)}]{stats.actual_avg_net_pnl_r:+.3f}R[/{c(stats.actual_avg_net_pnl_r)}]")
    tbl.add_row("Profit Factor",
                f"[{c(stats.actual_profit_factor, 1)}]{stats.actual_profit_factor:.2f}[/{c(stats.actual_profit_factor, 1)}]",
                f"[{c(stats.actual_net_profit_factor, 1)}]{stats.actual_net_profit_factor:.2f}[/{c(stats.actual_net_profit_factor, 1)}]")
    tbl.add_row("Max Drawdown",    f"-{stats.max_drawdown_r:.2f}R", "—")
    tbl.add_row("Sharpe (R)",      f"{stats.sharpe_r:.2f}", "—")
    tbl.add_row("Kelly 25%",       f"{stats.kelly_25pct * 100:.1f}%", "—")

    wins   = [t for t in filled if t.outcome in ("WIN", "PARTIAL_WIN")]
    losses = [t for t in filled if t.outcome == "LOSS"]
    tbl.add_row("Wins / Losses",   f"{len(wins)} / {len(losses)}", "—")

    console.print(tbl)

    # Direction breakdown
    longs  = [t for t in filled if t.direction == "long"]
    shorts = [t for t in filled if t.direction == "short"]
    if longs or shorts:
        dt = Table(title="By Direction", box=box.SIMPLE, show_header=True, padding=(0, 2))
        dt.add_column("Direction", style="dim")
        dt.add_column("Trades",    justify="right")
        dt.add_column("Win %",     justify="right")
        dt.add_column("Avg R",     justify="right")
        for label, grp in [("Long", longs), ("Short", shorts)]:
            if not grp:
                continue
            w   = [t for t in grp if t.outcome in ("WIN", "PARTIAL_WIN")]
            avg = sum(t.pnl_r for t in grp) / len(grp)
            wr  = len(w) / len(grp)
            rc  = "green" if avg > 0 else "red"
            wc  = "green" if wr >= 0.4 else "red"
            dt.add_row(label, str(len(grp)),
                       f"[{wc}]{wr:.1%}[/{wc}]",
                       f"[{rc}]{avg:+.3f}R[/{rc}]")
        console.print(dt)


def _print_walk_forward(wf_windows):
    wt = Table(title="Walk-Forward (60% train / 20% val / 20% test)",
               box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 2))
    wt.add_column("Window",    style="dim", width=7)
    wt.add_column("Period",    width=26)
    wt.add_column("Fills",     justify="right")
    wt.add_column("Win %",     justify="right")
    wt.add_column("Net Avg R", justify="right")
    wt.add_column("Net PF",    justify="right")
    wt.add_column("MaxDD",     justify="right")
    wt.add_column("Kelly 25%", justify="right")

    for w in wf_windows:
        bold = w.name == "test"
        s    = "bold" if bold else ""
        rc   = "green" if w.avg_net_pnl_r > 0 else "red"
        pc   = "green" if w.net_profit_factor >= 1.0 else "red"
        wt.add_row(
            f"[{s}]{w.name.upper()}[/{s}]" if s else w.name.upper(),
            f"{w.start_date} to{w.end_date}",
            str(w.n_fills),
            f"{w.win_rate:.1%}",
            f"[{rc}]{w.avg_net_pnl_r:+.3f}R[/{rc}]",
            f"[{pc}]{w.net_profit_factor:.2f}[/{pc}]",
            f"-{w.max_drawdown_r:.1f}R",
            f"{w.kelly_25pct * 100:.1f}%",
        )
    console.print(wt)


if __name__ == "__main__":
    main()
