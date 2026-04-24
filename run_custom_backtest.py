"""
run_custom_backtest.py — Drop-in single-file backtester.

HOW TO USE
----------
1. Edit the THREE sections marked with  ← EDIT THIS
2. Run:  python run_custom_backtest.py

That's it. No registry, no YAML, no schema knowledge needed.

Available tech keys inside entry_logic():
    current_price   — latest close
    atr_14          — 14-bar ATR
    rsi_14          — 14-bar RSI  (0–100)
    adx_14          — 14-bar ADX  (0–100)
    ema20           — 20-bar EMA
    ema50           — 50-bar EMA
    nearest_support    — closest support level below price
    nearest_resistance — closest resistance level above price
    atr_pct         — ATR as % of price

You also get the full `df` slice (OHLCV, zero lookahead) to compute
anything not in tech — e.g. ta.trend.EMAIndicator(df["Close"], 100)
"""
import sys
import argparse
import json as _json
import importlib.util
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import ta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ── CLI argument handling (used when called from UI via subprocess) ────────────
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("--logic-file",  default=None)   # path to .py file with entry_logic()
_cli.add_argument("--asset",       default=None)    # single ticker override, e.g. "GC=F"
_cli.add_argument("--label",       default=None)    # human label for the asset
_cli.add_argument("--interval",    default=None)
_cli.add_argument("--start",       default=None)
_cli.add_argument("--end",         default=None)
_cli.add_argument("--tp1-alloc",   default=None, type=int)
_cli.add_argument("--tp2-alloc",   default=None, type=int)
_cli.add_argument("--no-wf",       action="store_true")
_cli.add_argument("--json",        action="store_true")   # print JSON to stdout instead of rich
_ARGS, _ = _cli.parse_known_args()

# If a logic file was supplied via CLI, load it and override entry_logic
if _ARGS.logic_file:
    _spec = importlib.util.spec_from_file_location("_custom_logic", _ARGS.logic_file)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)          # type: ignore[union-attr]
    _external_entry_logic = getattr(_mod, "entry_logic", None)
else:
    _external_entry_logic = None


# ═══════════════════════════════════════════════════════════════════════════════
# ← EDIT THIS  1/3 — what to test
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_NAME = "My Strategy"   # just a label for the output

ASSETS = {
    # "^GSPC":  "SPX500",
    # "^DJI":   "US30",
    "GC=F":   "Gold",
    # "BTC-USD": "Bitcoin",
}

INTERVAL   = "1d"       # 1m 5m 15m 30m 1h 1d 1wk
START_DATE = "2018-01-01"
END_DATE   = "2026-01-01"

TP1_ALLOC = 100   # % of position closed at TP1
TP2_ALLOC = 0     # % of position closed at TP2

# CLI overrides (set by UI; don't edit these)
if _ARGS.asset:    ASSETS    = {_ARGS.asset: _ARGS.label or _ARGS.asset}
if _ARGS.interval: INTERVAL   = _ARGS.interval
if _ARGS.start:    START_DATE = _ARGS.start
if _ARGS.end:      END_DATE   = _ARGS.end
if _ARGS.tp1_alloc is not None: TP1_ALLOC = _ARGS.tp1_alloc
if _ARGS.tp2_alloc is not None: TP2_ALLOC = _ARGS.tp2_alloc


# ═══════════════════════════════════════════════════════════════════════════════
# ← EDIT THIS  2/3 — your entry conditions
# ═══════════════════════════════════════════════════════════════════════════════

def entry_logic(df: pd.DataFrame, tech: dict):
    """
    Return (direction, entry_low, entry_high, stop_loss, tp1, tp2)
    or None to skip this bar.

    direction: "long" or "short"
    """
    price = tech["current_price"]
    atr   = tech["atr_14"]
    rsi   = tech["rsi_14"]
    adx   = tech["adx_14"]
    ema20 = tech["ema20"]
    ema50 = tech["ema50"]

    # ── Example: EMA20/50 trend + RSI pullback ──────────────────────────────
    # Replace everything below with your own conditions.

    # Trend filter
    if ema20 <= ema50:
        return None

    # RSI: not overbought
    if rsi > 65:
        return None

    # Price near EMA20 (pullback arrived)
    if abs(price - ema20) > 1.5 * atr:
        return None

    # ADX: trend must have some strength
    if adx < 20:
        return None

    # Entry zone around EMA20
    entry_low  = round(ema20 - 0.3 * atr, 4)
    entry_high = round(ema20 + 0.2 * atr, 4)

    # Stop-loss below recent swing (simplified: 1.5 ATR below EMA)
    stop_loss  = round(ema20 - 1.5 * atr, 4)

    risk = entry_high - stop_loss
    if risk <= 0:
        return None

    tp1 = round(entry_high + 2.5 * risk, 4)
    tp2 = round(entry_high + 4.0 * risk, 4)

    return "long", entry_low, entry_high, stop_loss, tp1, tp2


# ═══════════════════════════════════════════════════════════════════════════════
# ← EDIT THIS  3/3 — advanced settings (leave as-is to start)
# ═══════════════════════════════════════════════════════════════════════════════

ENTRY_TIMEOUT_BARS = 10   # bars to wait for price to enter entry zone
WALK_FORWARD       = True  # 60/20/20 train/val/test split


# ═══════════════════════════════════════════════════════════════════════════════
# Engine — don't edit below this line
# ═══════════════════════════════════════════════════════════════════════════════

def _build_strategy():
    """Wrap entry_logic into the StrategyBase interface automatically."""
    from p3_backtester.strategies.base import StrategyBase
    from p3_backtester.bar_slicer import recompute_technical
    from p1_analysis_engine.schema import (
        SetupsOutput, TradingSetup, SetupTarget,
        InvalidationScenario, DecisionTreeEntry,
    )

    class _CustomStrategy(StrategyBase):
        name        = STRATEGY_NAME
        description = "Custom strategy from run_custom_backtest.py"

        @property
        def params(self) -> dict:
            return {}

        def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
            # Prefer external logic loaded via --logic-file (from UI), else use inline function
            _logic = _external_entry_logic if _external_entry_logic is not None else entry_logic
            result = _logic(df, tech)
            if result is None:
                return None

            direction, entry_low, entry_high, sl, tp1, tp2 = result
            fill_ref = entry_high if direction == "long" else entry_low
            risk     = abs(fill_ref - sl)
            if risk <= 0:
                return None

            avg_r = (abs(tp1 - fill_ref) / risk * TP1_ALLOC / 100
                     + abs(tp2 - fill_ref) / risk * TP2_ALLOC / 100)

            setup = TradingSetup(
                name="Setup A",
                label=f"{direction.upper()} — {STRATEGY_NAME}",
                direction=direction,
                trade_type="swing",
                status="ACTIVE",
                priority="primary",
                rationale=f"Custom entry: entry {entry_low}–{entry_high} SL {sl}",
                trigger=f"Price enters {entry_low}–{entry_high}",
                entry_low=entry_low,
                entry_high=entry_high,
                stop_loss=sl,
                trailing_sl_to_breakeven=tp1,
                targets=[
                    SetupTarget(price=tp1, label=f"TP1 ({TP1_ALLOC}%)", allocation_pct=TP1_ALLOC),
                    SetupTarget(price=tp2, label=f"TP2 ({TP2_ALLOC}%)", allocation_pct=TP2_ALLOC),
                ],
                rr_ratio=round(avg_r, 2),
                win_rate_estimate=0.45,
                trade_duration="varies",
                ev=round(0.45 * avg_r - 0.55 * 1.0, 4),
                profit_factor=round(0.45 * avg_r / 0.55, 2),
                confidence=0.72,
                confidence_note="custom",
            )

            return SetupsOutput(
                asset=tech.get("ticker", ""),
                current_price=tech["current_price"],
                setups=[setup],
                invalidation=InvalidationScenario(
                    condition=f"Close below {sl}" if direction == "long" else f"Close above {sl}",
                    description="SL broken",
                    price_trigger=sl,
                    action="Stand aside",
                ),
                decision_tree=[
                    DecisionTreeEntry(
                        scenario="Price enters entry zone",
                        outcome=f"ENTER {direction.upper()}",
                        direction=direction,
                        setup_name="Setup A",
                        entry_price=entry_high if direction == "long" else entry_low,
                    ),
                ],
                position_sizing_note=f"ATR={tech['atr_14']:.4f}",
            )

    return _CustomStrategy()


def _run_one(ticker: str, label: str) -> tuple:
    from p3_backtester.schema import BacktestConfig
    from p3_backtester.market_data import fetch_ohlcv, split_backtest_window, MarketDataError
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.pass1_signal_gen import run_pass1
    from p3_backtester.pass2_simulator import run_pass2
    from p3_backtester.aggregator import compute_stats
    from p3_backtester.walk_forward import compute_walk_forward_dates, kelly_from_stats
    from p3_backtester.schema import WalkForwardWindow
    from p1_analysis_engine.utils.asset_classifier import classify_asset

    strategy   = _build_strategy()
    asset_class = classify_asset(ticker)

    config = BacktestConfig(
        ticker=ticker,
        interval=INTERVAL,
        start_date=START_DATE,
        end_date=END_DATE,
        signal_mode="session-open",
        every_n_bars=1,
        entry_timeout_bars=ENTRY_TIMEOUT_BARS,
        model="claude-sonnet-4-6",
        force_regenerate=True,
        top_adx_pct=1.0,
    )

    try:
        df = fetch_ohlcv(ticker, INTERVAL, START_DATE, END_DATE, source="auto")
        console.print(f"  [dim]{len(df)} bars loaded[/dim]")
    except MarketDataError as e:
        return None, str(e), []

    first_valid, _ = split_backtest_window(df, START_DATE)
    signal_indices  = get_signal_indices(df, config.signal_mode, first_valid, config.every_n_bars)

    if not signal_indices:
        return None, "No signal bars (insufficient warmup)", []

    signals = run_pass1(config, df, signal_indices, use_claude=False, strategy=strategy)
    if not signals:
        return None, "No signals generated", []

    trades = run_pass2(config, df, signals, asset_class=asset_class)
    if not trades:
        return None, "No trades simulated", []

    stats = compute_stats(config, signals, trades, strategy_name=strategy.name)

    # Walk-forward
    wf = []
    if WALK_FORWARD:
        console.print("  [dim]Walk-forward ...[/dim]")
        try:
            train_r, val_r, test_r = compute_walk_forward_dates(df, START_DATE, END_DATE)
            for wname, (wstart, wend) in [("train", train_r), ("val", val_r), ("test", test_r)]:
                from p3_backtester.market_data import split_backtest_window as _spw
                first_w, _ = _spw(df, wstart)
                w_idx = get_signal_indices(df, config.signal_mode, first_w, config.every_n_bars)
                wts   = pd.Timestamp(wstart); wte = pd.Timestamp(wend)
                w_idx = [i for i in w_idx if wts <= df.index[i] <= wte]
                if not w_idx:
                    continue
                wc    = config.model_copy(update={"start_date": wstart, "end_date": wend,
                                                   "force_regenerate": True})
                ws    = run_pass1(wc, df, w_idx, use_claude=False, strategy=_build_strategy())
                if not ws:
                    continue
                wt    = run_pass2(wc, df, ws, asset_class=asset_class)
                if not wt:
                    continue
                wst   = compute_stats(wc, ws, wt, strategy_name=strategy.name)
                filled  = [t for t in wt if t.outcome != "EXPIRED"]
                wins    = [t for t in filled if t.outcome in ("WIN", "PARTIAL_WIN")]
                losses  = [t for t in filled if t.outcome == "LOSS"]
                gp = sum(t.net_pnl_r for t in wins)
                gl = abs(sum(t.net_pnl_r for t in losses))
                npf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
                wf.append(WalkForwardWindow(
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
        except Exception as e:
            console.print(f"  [yellow]Walk-forward failed: {e}[/yellow]")

    return stats, None, wf


def main():
    # ── JSON mode (called from UI via subprocess) ─────────────────────────────
    if _ARGS.json:
        results = []
        for ticker, label in ASSETS.items():
            stats, err, wf = _run_one(ticker, label)
            if err or stats is None:
                print(_json.dumps({"error": err or "no stats", "ticker": ticker, "label": label}))
                return
            out = stats.model_dump()
            out["label"]        = label
            out["ticker"]       = ticker
            out["walk_forward"] = [w.model_dump() for w in wf]
            print(_json.dumps(out, default=str))
        return

    # ── Rich console mode (normal CLI run) ────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold gold1]ORCASTRADING[/bold gold1]  "
        f"[dim]{STRATEGY_NAME}  |  {INTERVAL}  |  {START_DATE} → {END_DATE}[/dim]",
        box=box.ROUNDED, padding=(0, 4),
    ))

    results = []
    for ticker, label in ASSETS.items():
        console.print(f"\n[bold]{label} ({ticker})[/bold]")
        stats, err, wf = _run_one(ticker, label)
        results.append((label, ticker, stats, err, wf))
        if err:
            console.print(f"  [red]FAILED:[/red] {err}")

    # ── Summary table ─────────────────────────────────────────────────────────
    console.print()
    t = Table(title=f"{STRATEGY_NAME} — Summary", box=box.SIMPLE_HEAVY,
              show_header=True, padding=(0, 2))
    t.add_column("Asset",   style="bold", width=8)
    t.add_column("Fills",   justify="right")
    t.add_column("Win %",   justify="right")
    t.add_column("Avg R",   justify="right")
    t.add_column("Net PF",  justify="right")
    t.add_column("MaxDD",   justify="right")
    t.add_column("Sharpe",  justify="right")
    t.add_column("Kelly",   justify="right")

    for label, ticker, stats, err, wf in results:
        if err or stats is None:
            t.add_row(label, "—", "—", "—", f"[red]{err or 'ERR'}[/red]", "—", "—", "—")
            continue
        wr_c = "green" if stats.actual_win_rate >= 0.35 else ("yellow" if stats.actual_win_rate >= 0.28 else "red")
        pf_c = "green" if stats.actual_net_profit_factor >= 1.5 else ("yellow" if stats.actual_net_profit_factor >= 1.0 else "red")
        t.add_row(
            label,
            str(stats.total_trades_filled),
            f"[{wr_c}]{stats.actual_win_rate:.1%}[/{wr_c}]",
            f"{stats.actual_avg_net_pnl_r:+.3f}R",
            f"[{pf_c}]{stats.actual_net_profit_factor:.2f}[/{pf_c}]",
            f"-{stats.max_drawdown_r:.1f}R",
            f"{stats.sharpe_r:.2f}",
            f"{stats.kelly_25pct * 100:.1f}%",
        )
    console.print(t)

    # ── Walk-forward per asset ────────────────────────────────────────────────
    for label, ticker, stats, err, wf in results:
        if err or not wf:
            continue
        console.print(f"\n[bold]{label}[/bold] — Walk-Forward (60% train / 20% val / 20% test)")
        wt = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        wt.add_column("Window",    style="dim", width=7)
        wt.add_column("Period",    width=26)
        wt.add_column("Fills",     justify="right")
        wt.add_column("Win %",     justify="right")
        wt.add_column("Net Avg R", justify="right")
        wt.add_column("Net PF",    justify="right")
        wt.add_column("MaxDD",     justify="right")
        wt.add_column("Kelly 25%", justify="right")
        for w in wf:
            bold = w.name == "test"
            s    = "bold" if bold else ""
            rc   = "green" if w.avg_net_pnl_r > 0 else "red"
            pc   = "green" if w.net_profit_factor >= 1.0 else "red"
            wt.add_row(
                f"[{s}]{w.name.upper()}[/{s}]" if s else w.name.upper(),
                f"{w.start_date} → {w.end_date}",
                str(w.n_fills),
                f"{w.win_rate:.1%}",
                f"[{rc}]{w.avg_net_pnl_r:+.3f}R[/{rc}]",
                f"[{pc}]{w.net_profit_factor:.2f}[/{pc}]",
                f"-{w.max_drawdown_r:.1f}R",
                f"{w.kelly_25pct * 100:.1f}%",
            )
        console.print(wt)

    console.print()


if __name__ == "__main__":
    main()
