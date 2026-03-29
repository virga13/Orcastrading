"""
pass2_simulator.py — loads cached P1 signals and simulates trade execution
forward through the OHLCV data. No API calls — fast and free to re-run.
"""
import json
from pathlib import Path

import pandas as pd
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console

from p1_analysis_engine.schema import TradingSetup, SetupsOutput
from p3_backtester.schema import BacktestConfig, SignalRecord, TradeRecord
from p3_backtester.trade_simulator import simulate_trade

console = Console()
CACHE_DIR = Path(__file__).parent / "cache"


def run_pass2(
    config: BacktestConfig,
    df: pd.DataFrame,
    signals: list[SignalRecord],
) -> list[TradeRecord]:
    """
    For each signal:
    1. Load its JSON cache file
    2. Deserialize setups
    3. Simulate each setup forward through df
    4. Return all TradeRecords

    Pass 2 is deterministic — same cache + same df = same results.
    """
    all_trades: list[TradeRecord] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Pass 2[/bold] — simulating trades"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(signals))

        for signal in signals:
            cache_path = Path(__file__).parent / signal.cache_file

            if not cache_path.exists():
                console.print(f"[yellow]  Cache missing: {signal.cache_file}[/yellow]")
                progress.advance(task)
                continue

            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            try:
                setups_data = raw["setups"]
                setups_obj  = SetupsOutput(**setups_data)
            except Exception as e:
                console.print(f"[yellow]  Failed to deserialize setups at {signal.bar_time}: {e}[/yellow]")
                progress.advance(task)
                continue

            for setup in setups_obj.setups:
                try:
                    trade = simulate_trade(
                        setup=setup,
                        signal_bar_index=signal.bar_index,
                        signal_bar_time=signal.bar_time,
                        df=df,
                        entry_timeout_bars=config.entry_timeout_bars,
                        claude_confidence=signal.confidence_score,
                    )
                    all_trades.append(trade)
                except Exception as e:
                    console.print(f"[yellow]  Simulation error for {setup.name} at {signal.bar_time}: {e}[/yellow]")

            progress.advance(task)

    filled   = sum(1 for t in all_trades if t.outcome != "EXPIRED")
    expired  = sum(1 for t in all_trades if t.outcome == "EXPIRED")
    console.print(f"  [dim]Pass 2 complete: {len(all_trades)} trades ({filled} filled, {expired} expired)[/dim]")
    return all_trades
