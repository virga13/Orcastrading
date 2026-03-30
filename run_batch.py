"""
run_batch.py — cross-asset batch backtest using MTF Trend strategy.

Usage:
    python run_batch.py

Strategy: MTF Trend (Multi-Timeframe Trend Following)
  Entry:  1D bars — 10-year window (2015-2026)
  Bias:   1W (resampled from daily data inside strategy — zero lookahead)
  Signal: EMA20/50 aligned on weekly + price retesting EMA20 on daily
  RSI:    not overbought on entry
  Stop:   structural swing low/high (15-bar lookback)
  Target: TP1 2.5R (70%), TP2 4.0R (30%)

Data sources (auto-fallback chain):
  Gold, Silver, Brent, Indices -> Stooq (free, 10-20yr daily) -> yfinance
  Bitcoin -> CCXT/Binance (full history) -> yfinance

Expected fills: ~50-150 per asset over 10 years — statistically meaningful.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

ASSETS = ["GC=F", "SI=F", "BTC-USD", "^GSPC", "^GDAXI", "^DJI", "^NDX", "BZ=F"]
ASSET_LABELS = {
    "GC=F":    "Gold",
    "SI=F":    "Silver",
    "BTC-USD": "Bitcoin",
    "^GSPC":   "SPX500",
    "^GDAXI":  "GER40",
    "^DJI":    "US30",
    "^NDX":    "NAS100",
    "BZ=F":    "Brent",
}

# (interval, start, end, entry_timeout_bars, signal_mode, every_n)
# 1D bars — 10-year window; weekly bias resampled inside strategy
INTERVALS = [
    ("1d", "2015-01-01", "2026-03-28", 10, "session-open", 1),
]


def run_one(ticker: str, interval: str, start: str, end: str,
            entry_timeout: int = 20, signal_mode: str = "session-open", every_n: int = 10):
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
        adx_threshold=20,        # daily ADX is more stable; 20 is appropriate
        pullback_max_atr=1.5,    # daily candles are larger — allow slightly more slack
        min_prior_distance_atr=1.5,
        prior_lookback=5,
        rsi_max=65,
        rsi_min=35,
        sl_buffer_atr=0.15,
        sl_lookback=10,          # swing lookback in daily bars (~2 weeks)
        tp1_r=2.5,
        tp2_r=4.0,
        tp1_alloc=70,
        tp2_alloc=30,
        win_rate=0.48,
    )

    config = BacktestConfig(
        ticker=ticker,
        interval=interval,
        start_date=start,
        end_date=end,
        signal_mode=signal_mode,
        every_n_bars=every_n,
        entry_timeout_bars=entry_timeout,
        model="claude-sonnet-4-6",
        force_regenerate=True,
        top_adx_pct=1.0,        # no percentile filter — let all signals through
    )

    asset_class = classify_asset(ticker)

    try:
        df = fetch_ohlcv(ticker, interval, start, end, source="auto")
        console.print(f"  [dim]{len(df)} bars loaded[/dim]")
    except MarketDataError as e:
        return None, str(e)

    first_valid, _ = split_backtest_window(df, start)
    signal_indices = get_signal_indices(df, config.signal_mode, first_valid, config.every_n_bars)

    if not signal_indices:
        return None, "No signal bars available (insufficient warmup data)"

    signals = run_pass1(config, df, signal_indices, use_claude=False, strategy=strategy)
    if not signals:
        return None, "No signals generated"

    trades = run_pass2(config, df, signals, asset_class=asset_class)
    if not trades:
        return None, "No trades simulated"

    stats = compute_stats(config, signals, trades, strategy_name=strategy.name)
    save_to_db(config, signals, trades, stats)
    return stats, None


def main():
    console.print()
    console.print(Panel(
        "[bold gold1]ORCASTRADING[/bold gold1]  [dim]MTF Trend — 1D entry / 1W bias — 10-Year Cross-Asset Test[/dim]",
        box=box.ROUNDED, padding=(1, 4),
    ))

    results = []   # list of (label, interval, stats | None, error | None)

    total = len(ASSETS) * len(INTERVALS)
    done  = 0

    for interval, start, end, timeout, sig_mode, every_n in INTERVALS:
        for ticker in ASSETS:
            done += 1
            label = ASSET_LABELS[ticker]
            console.print(f"\n[bold][{done}/{total}] {label} - {interval}[/bold]  [dim]{start} to {end} ({sig_mode})[/dim]")
            stats, err = run_one(ticker, interval, start, end,
                                 entry_timeout=timeout, signal_mode=sig_mode, every_n=every_n)
            results.append((label, interval, stats, err))
            if err:
                console.print(f"  [red]FAILED:[/red] {err}")

    # ── Summary table ────────────────────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]Batch Results Summary[/bold]", box=box.ROUNDED))

    for interval, _, _, _, _, _ in INTERVALS:
        t = Table(
            title=f"Interval: {interval}",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            padding=(0, 2),
        )
        t.add_column("Asset",    style="bold", width=10)
        t.add_column("Signals",  justify="right")
        t.add_column("Fills",    justify="right")
        t.add_column("Win %",    justify="right")
        t.add_column("Avg R",    justify="right")
        t.add_column("PF",       justify="right")
        t.add_column("MaxDD",    justify="right")
        t.add_column("Sharpe",   justify="right")

        for label, iv, stats, err in results:
            if iv != interval:
                continue
            if err or stats is None:
                t.add_row(label, "—", "—", "—", "—", "—", "—", f"[red]{err or 'ERR'}[/red]")
                continue

            wr_color  = "green" if stats.actual_win_rate >= 0.50 else ("yellow" if stats.actual_win_rate >= 0.40 else "red")
            r_color   = "green" if stats.actual_avg_pnl_r > 0    else "red"
            pf_color  = "green" if stats.actual_profit_factor >= 1.5 else ("yellow" if stats.actual_profit_factor >= 1.0 else "red")

            t.add_row(
                label,
                str(stats.total_signals),
                str(stats.total_trades_filled),
                f"[{wr_color}]{stats.actual_win_rate:.0%}[/{wr_color}]",
                f"[{r_color}]{stats.actual_avg_pnl_r:+.2f}R[/{r_color}]",
                f"[{pf_color}]{stats.actual_profit_factor:.2f}[/{pf_color}]",
                f"-{stats.max_drawdown_r:.1f}R",
                f"{stats.sharpe_r:.2f}",
            )

        console.print(t)

    console.print()
    console.print("[dim]All results saved to p3_backtester/results/[/dim]\n")


if __name__ == "__main__":
    main()
