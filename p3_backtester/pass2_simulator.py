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
from p3_backtester.costs import round_trip_cost_r

console = Console()
CACHE_DIR = Path(__file__).parent / "cache"


def run_pass2(
    config: BacktestConfig,
    df: pd.DataFrame,
    signals: list[SignalRecord],
    asset_class: str = "equity",
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

            # Skip signals at or past the last bar (no forward bars to simulate)
            if signal.bar_index >= len(df) - 1:
                progress.advance(task)
                continue

            for setup in setups_obj.setups:
                try:
                    # Compute ATR at signal bar for cost calculation
                    signal_bar_atr = float(df["Close"].iloc[signal.bar_index]) * 0.01  # fallback
                    try:
                        import ta as _ta
                        _end = min(signal.bar_index + 1, len(df))
                        _start = max(0, _end - 20)
                        _close = df["Close"].iloc[_start:_end]
                        _high  = df["High"].iloc[_start:_end]
                        _low   = df["Low"].iloc[_start:_end]
                        if len(_close) >= 2:
                            signal_bar_atr = float(_ta.volatility.AverageTrueRange(
                                _high, _low, _close, window=min(14, len(_close))
                            ).average_true_range().iloc[-1])
                    except Exception:
                        pass

                    cost_r = round_trip_cost_r(
                        asset_class=asset_class,
                        interval=config.interval,
                        atr=signal_bar_atr,
                        initial_risk=abs(
                            (setup.entry_high if setup.direction == "long" else setup.entry_low)
                            - setup.stop_loss
                        ),
                    )
                    trade = simulate_trade(
                        setup=setup,
                        signal_bar_index=signal.bar_index,
                        signal_bar_time=signal.bar_time,
                        df=df,
                        entry_timeout_bars=config.entry_timeout_bars,
                        claude_confidence=signal.confidence_score,
                        cost_r=cost_r,
                    )
                    all_trades.append(trade)
                except Exception as e:
                    console.print(f"[yellow]  Simulation error for {setup.name} at {signal.bar_time}: {e}[/yellow]")

            progress.advance(task)

    filled   = sum(1 for t in all_trades if t.outcome != "EXPIRED")
    expired  = sum(1 for t in all_trades if t.outcome == "EXPIRED")
    console.print(f"  [dim]Pass 2 complete: {len(all_trades)} trades ({filled} filled, {expired} expired)[/dim]")
    return all_trades
