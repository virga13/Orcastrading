"""
run_batch.py — focused cross-asset backtest: SPX500, NAS100, US30 long-only.

Results from the v1 run (2026-03-31) showed:
  - Short side has no edge in 2015-2026 bull market (10-15% WR on shorts)
  - Long side is genuinely strong: SPX500 38.2% WR / PF 2.25, NAS100 39.0% / PF 2.10
  - Gold / Brent / Bitcoin rejected — no structural edge with EMA trend-following

This run:
  - Long-only mode (shorts_enabled=False)
  - Walk-forward validation (60% train / 20% val / 20% test)
  - Regime filter active (SMA200 — no longs in confirmed bear)
  - Directional prior distance gate active (v1 bug fix)

Usage:
    python run_batch.py
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

ASSETS = ["^GSPC", "^NDX", "^DJI"]
ASSET_LABELS = {
    "^GSPC": "SPX500",
    "^NDX":  "NAS100",
    "^DJI":  "US30",
}

# 1D entry / 1W bias — 10-year window
START = "2015-01-01"
END   = "2026-03-28"


def _run_walk_forward(config, df, strategy, asset_class):
    """Run train/val/test sub-windows and return list[WalkForwardWindow]."""
    from p3_backtester.walk_forward import compute_walk_forward_dates, kelly_from_stats
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.market_data import split_backtest_window
    from p3_backtester.schema import WalkForwardWindow

    try:
        train_r, val_r, test_r = compute_walk_forward_dates(df, config.start_date, config.end_date)
    except ValueError as e:
        console.print(f"  [yellow]Walk-forward skipped: {e}[/yellow]")
        return []

    windows = [("train", train_r), ("validation", val_r), ("test", test_r)]
    results = []

    for wname, (wstart, wend) in windows:
        first_valid, _ = split_backtest_window(df, wstart)
        w_indices = get_signal_indices(df, config.signal_mode, first_valid, config.every_n_bars)
        wstart_ts = pd.Timestamp(wstart)
        wend_ts   = pd.Timestamp(wend)
        w_indices = [i for i in w_indices if wstart_ts <= df.index[i] <= wend_ts]

        if not w_indices:
            continue

        w_config  = config.model_copy(update={"start_date": wstart, "end_date": wend})
        w_signals = run_pass1(w_config, df, w_indices, use_claude=False, strategy=strategy)
        if not w_signals:
            continue
        w_trades  = run_pass2(w_config, df, w_signals, asset_class=asset_class)
        if not w_trades:
            continue
        w_stats   = compute_stats(w_config, w_signals, w_trades, strategy_name=strategy.name)

        filled    = [t for t in w_trades if t.outcome != "EXPIRED"]
        wins      = [t for t in filled if t.outcome in ("WIN", "PARTIAL_WIN")]
        losses    = [t for t in filled if t.outcome == "LOSS"]
        net_gp    = sum(t.net_pnl_r for t in wins)
        net_gl    = abs(sum(t.net_pnl_r for t in losses))
        net_pf    = net_gp / net_gl if net_gl > 0 else (999.0 if net_gp > 0 else 0.0)

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


def run_one(ticker: str):
    from p3_backtester.schema import BacktestConfig
    from p3_backtester.market_data import fetch_ohlcv, split_backtest_window, MarketDataError
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats, save_to_db
    from p3_backtester.strategies.mtf_trend import MTFTrendStrategy
    from p1_analysis_engine.utils.asset_classifier import classify_asset

    strategy = MTFTrendStrategy(
        ema_fast=20,
        ema_slow=50,
        adx_threshold=20,
        adx_threshold_short=15,
        pullback_max_atr=1.5,
        min_prior_distance_atr=1.5,
        prior_lookback=5,
        rsi_max=65,
        rsi_min=35,
        sl_buffer_atr=0.15,
        sl_lookback=10,
        tp1_r=2.5,
        tp2_r=4.0,
        tp1_alloc=70,
        tp2_alloc=30,
        win_rate=0.48,
        regime_filter=True,
        shorts_enabled=False,       # long-only: short side has no edge 2015-2026
    )

    config = BacktestConfig(
        ticker=ticker,
        interval="1d",
        start_date=START,
        end_date=END,
        signal_mode="session-open",
        every_n_bars=1,
        entry_timeout_bars=10,
        model="claude-sonnet-4-6",
        force_regenerate=True,
        top_adx_pct=1.0,
    )

    asset_class = classify_asset(ticker)

    try:
        df = fetch_ohlcv(ticker, "1d", START, END, source="auto")
        console.print(f"  [dim]{len(df)} bars loaded[/dim]")
    except MarketDataError as e:
        return None, str(e)

    first_valid, _ = split_backtest_window(df, START)
    signal_indices = get_signal_indices(df, config.signal_mode, first_valid, config.every_n_bars)

    if not signal_indices:
        return None, "No signal bars (insufficient warmup)"

    signals = run_pass1(config, df, signal_indices, use_claude=False, strategy=strategy)
    if not signals:
        return None, "No signals generated"

    trades = run_pass2(config, df, signals, asset_class=asset_class)
    if not trades:
        return None, "No trades simulated"

    stats = compute_stats(config, signals, trades, strategy_name=strategy.name)

    # Walk-forward
    console.print("  [dim]Running walk-forward validation ...[/dim]")
    wf = _run_walk_forward(config, df, strategy, asset_class)
    stats.walk_forward = wf

    save_to_db(config, signals, trades, stats)
    return stats, None


def main():
    console.print()
    console.print(Panel(
        "[bold gold1]ORCASTRADING[/bold gold1]  "
        "[dim]MTF Trend — Long-Only — SPX500 / NAS100 / US30 — 10Y Walk-Forward[/dim]",
        box=box.ROUNDED, padding=(1, 4),
    ))

    results = []

    for i, ticker in enumerate(ASSETS, 1):
        label = ASSET_LABELS[ticker]
        console.print(f"\n[bold][{i}/{len(ASSETS)}] {label}[/bold]  [dim]{START} to {END}[/dim]")
        stats, err = run_one(ticker)
        results.append((label, ticker, stats, err))
        if err:
            console.print(f"  [red]FAILED:[/red] {err}")

    # ── Summary table ────────────────────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]Overall Results (Long-Only)[/bold]", box=box.ROUNDED))

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 2))
    t.add_column("Asset",   style="bold", width=8)
    t.add_column("Fills",   justify="right")
    t.add_column("Win %",   justify="right")
    t.add_column("Avg R",   justify="right")
    t.add_column("Net PF",  justify="right")
    t.add_column("MaxDD",   justify="right")
    t.add_column("Sharpe",  justify="right")
    t.add_column("Calmar",  justify="right")
    t.add_column("Kelly",   justify="right")

    for label, ticker, stats, err in results:
        if err or stats is None:
            t.add_row(label, "—", "—", "—", "—", "—", "—", f"[red]{err or 'ERR'}[/red]", "—")
            continue
        wr_c  = "green" if stats.actual_win_rate >= 0.35 else ("yellow" if stats.actual_win_rate >= 0.28 else "red")
        pf_c  = "green" if stats.actual_net_profit_factor >= 1.5 else ("yellow" if stats.actual_net_profit_factor >= 1.0 else "red")
        cal_c = "green" if stats.calmar_ratio >= 2.0 else ("yellow" if stats.calmar_ratio >= 1.0 else "red")
        t.add_row(
            label,
            str(stats.total_trades_filled),
            f"[{wr_c}]{stats.actual_win_rate:.1%}[/{wr_c}]",
            f"{stats.actual_avg_net_pnl_r:+.3f}R",
            f"[{pf_c}]{stats.actual_net_profit_factor:.2f}[/{pf_c}]",
            f"-{stats.max_drawdown_r:.1f}R",
            f"{stats.sharpe_r:.2f}",
            f"[{cal_c}]{stats.calmar_ratio:.2f}[/{cal_c}]",
            f"{stats.kelly_25pct*100:.1f}%",
        )
    console.print(t)

    # ── Walk-forward per asset ────────────────────────────────────────────────
    for label, ticker, stats, err in results:
        if err or stats is None or not stats.walk_forward:
            continue

        console.print()
        console.print(f"[bold]{label}[/bold] — Walk-Forward (60% train / 20% val / 20% test)")
        wf_t = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        wf_t.add_column("Window",     style="dim", width=12)
        wf_t.add_column("Period",     width=26)
        wf_t.add_column("Fills",      justify="right")
        wf_t.add_column("Win %",      justify="right")
        wf_t.add_column("Net Avg R",  justify="right")
        wf_t.add_column("Net PF",     justify="right")
        wf_t.add_column("MaxDD",      justify="right")
        wf_t.add_column("Kelly 25%",  justify="right")

        for w in stats.walk_forward:
            is_test = w.name == "test"
            style   = "bold" if is_test else ""
            pf_c    = "green" if w.net_profit_factor >= 1.0 else "red"
            r_c     = "green" if w.avg_net_pnl_r > 0 else "red"
            label_w = f"[{style}]{w.name.upper()}[/{style}]" if style else w.name.upper()
            wf_t.add_row(
                label_w,
                f"{w.start_date} -> {w.end_date}",
                str(w.n_fills),
                f"{w.win_rate:.1%}",
                f"[{r_c}]{w.avg_net_pnl_r:+.3f}R[/{r_c}]",
                f"[{pf_c}]{w.net_profit_factor:.2f}[/{pf_c}]",
                f"-{w.max_drawdown_r:.1f}R",
                f"{w.kelly_25pct*100:.1f}%",
            )
        console.print(wf_t)

    console.print()
    console.print("[dim]Results saved to p3_backtester/results/[/dim]\n")


if __name__ == "__main__":
    main()
