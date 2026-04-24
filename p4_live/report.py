"""
p4_live/report.py — Forward-test performance report.

Compares live trading results against the validated backtest baseline.
The baseline numbers are locked from the walk-forward TEST window.
"""
import math
from datetime import date
from p4_live.journal import get_all_trades, get_closed_trades
from core.config import get_asset_map, get_all_assets


def _traffic_light(actual: float, target: float, tolerance: float = 0.10) -> str:
    """Return a colored status string."""
    if actual >= target * (1 - tolerance):
        return "green"
    elif actual >= target * (1 - tolerance * 3):
        return "yellow"
    return "red"


def compute_forward_stats(
    trades: list[dict] | None = None,
    ticker: str | None = None,
    strategy: str | None = None,
) -> dict:
    if trades is None:
        trades = get_all_trades()
    if ticker:
        trades = [t for t in trades if t.get("ticker") == ticker]
    if strategy:
        trades = [t for t in trades if t.get("strategy") == strategy]

    all_    = trades
    closed  = [t for t in all_ if t["status"] not in ("pending", "filled", "expired")]
    filled  = [t for t in all_ if t["status"] not in ("pending", "expired")]
    pending = [t for t in all_ if t["status"] == "pending"]
    expired = [t for t in all_ if t["status"] == "expired"]
    open_   = [t for t in all_ if t["status"] == "filled"]

    wins    = [t for t in closed if t["status"] in ("win", "partial_win")]
    losses  = [t for t in closed if t["status"] == "loss"]

    n_signals = len(all_)
    n_fills   = len(filled)
    n_closed  = len(closed)
    n_wins    = len(wins)
    n_losses  = len(losses)

    win_rate = n_wins / n_closed if n_closed else 0.0

    pnl_rs   = [t["pnl_r"] for t in closed if t["pnl_r"] is not None]
    avg_r    = sum(pnl_rs) / len(pnl_rs) if pnl_rs else 0.0
    total_r  = sum(pnl_rs)

    gp = sum(r for r in pnl_rs if r > 0)
    gl = abs(sum(r for r in pnl_rs if r < 0))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    # Max drawdown on closed trades
    equity = 0.0; peak = 0.0; max_dd = 0.0
    for r in pnl_rs:
        equity += r
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    calmar = round(total_r / max_dd, 2) if max_dd > 0 else 0.0

    # ── Trade duration (fill_date → exit_date) ────────────────────────────────
    durations = []
    for t in closed:
        if t.get("fill_date") and t.get("exit_date"):
            try:
                d1 = date.fromisoformat(t["fill_date"])
                d2 = date.fromisoformat(t["exit_date"])
                durations.append((d2 - d1).days)
            except (ValueError, TypeError):
                pass
    avg_duration = round(sum(durations) / len(durations), 1) if durations else None

    # Sharpe (annualized) — scale by sqrt(trades per year) based on avg hold duration.
    # Using per-trade R as returns and converting to annual frequency.
    if len(pnl_rs) > 1:
        mean = avg_r
        var  = sum((r - mean) ** 2 for r in pnl_rs) / (len(pnl_rs) - 1)
        std  = math.sqrt(var) if var > 0 else 0.0
        if std > 0:
            days_per_trade  = avg_duration if avg_duration and avg_duration > 0 else 5
            trades_per_year = 252 / days_per_trade
            sharpe = round((mean / std) * math.sqrt(trades_per_year), 2)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # ── Actual R per win / loss (vs theoretical TP/SL) ───────────────────────
    win_pnl_r  = [t["pnl_r"] for t in wins   if t["pnl_r"] is not None]
    loss_pnl_r = [t["pnl_r"] for t in losses if t["pnl_r"] is not None]
    avg_actual_win_r  = round(sum(win_pnl_r)  / len(win_pnl_r),  3) if win_pnl_r  else None
    avg_actual_loss_r = round(sum(loss_pnl_r) / len(loss_pnl_r), 3) if loss_pnl_r else None
    actual_rr = (
        round(abs(avg_actual_win_r / avg_actual_loss_r), 2)
        if avg_actual_win_r and avg_actual_loss_r
        else None
    )

    return {
        "n_signals": n_signals,
        "n_fills":   n_fills,
        "n_closed":  n_closed,
        "n_open":    len(open_),
        "n_pending": len(pending),
        "n_expired": len(expired),
        "n_wins":    n_wins,
        "n_losses":  n_losses,
        "win_rate":  round(win_rate, 4),
        "avg_r":     round(avg_r, 4),
        "total_r":   round(total_r, 4),
        "profit_factor": round(pf, 3),
        "max_drawdown_r": round(max_dd, 2),
        "sharpe":    round(sharpe, 2),
        "calmar":    calmar,
        "avg_duration_days":  avg_duration,
        "avg_actual_win_r":   avg_actual_win_r,
        "avg_actual_loss_r":  avg_actual_loss_r,
        "actual_rr":          actual_rr,
        "trades":    all_,
    }


def print_report(console, ticker: str | None = None) -> None:
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    _asset_map = get_asset_map()
    _assets_with_yf = [a for a in get_all_assets() if a.get("tickers", {}).get("yfinance")]
    asset = _asset_map.get(ticker) if ticker else (_assets_with_yf[0] if _assets_with_yf else None)
    if asset is None:
        console.print(f"[red]Unknown ticker '{ticker}'[/red]")
        return

    asset_ticker = asset.get("tickers", {}).get("yfinance", ticker)
    stats = compute_forward_stats(ticker=asset_ticker)
    base  = asset.get("baseline") or {}
    label = asset.get("label", asset_ticker)

    console.print()
    console.print(Panel(
        "[bold gold1]ORCASTRADING[/bold gold1]  "
        f"[dim]P4 — Forward-Test Report  |  {label} Long-Only  |  1D/1W ADX>=15[/dim]",
        box=box.ROUNDED, padding=(0, 4),
    ))

    # Sample size warning
    if stats["n_closed"] < 10:
        console.print(
            f"\n  [yellow]!! Only {stats['n_closed']} closed trades — "
            "results are not yet statistically meaningful (need >=30).[/yellow]"
        )

    # Overview counts
    counts = Table(box=box.SIMPLE, show_header=False, padding=(0, 3))
    counts.add_column(style="dim", width=14)
    counts.add_column()
    counts.add_column(style="dim", width=14)
    counts.add_column()
    counts.add_row(
        "Signals fired", str(stats["n_signals"]),
        "Closed",        str(stats["n_closed"]),
    )
    counts.add_row(
        "Fills",         str(stats["n_fills"]),
        "Open",          str(stats["n_open"]),
    )
    counts.add_row(
        "Expired",       str(stats["n_expired"]),
        "Pending",       str(stats["n_pending"]),
    )
    console.print(counts)

    # Performance vs baseline
    if stats["n_closed"] == 0:
        console.print("\n  [dim]No closed trades yet.[/dim]\n")
        return

    perf = Table(
        title="Live vs Backtest Baseline",
        box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 3),
    )
    perf.add_column("Metric",     style="dim", width=18)
    perf.add_column("Live",       justify="right", width=10)
    perf.add_column("Baseline",   justify="right", width=10, style="dim")
    perf.add_column("Status",     justify="center", width=10)

    def _row(label, live_val, base_val, fmt, higher_is_better=True):
        live_s = fmt.format(live_val)
        base_s = fmt.format(base_val)
        ratio  = live_val / base_val if base_val else 0
        if higher_is_better:
            color = "green" if ratio >= 0.90 else ("yellow" if ratio >= 0.70 else "red")
            icon  = "[OK]" if ratio >= 0.90 else ("~" if ratio >= 0.70 else "[x]")
        else:
            color = "green" if ratio <= 1.10 else ("yellow" if ratio <= 1.30 else "red")
            icon  = "[OK]" if ratio <= 1.10 else ("~" if ratio <= 1.30 else "[x]")
        perf.add_row(label, f"[bold]{live_s}[/bold]", base_s, f"[{color}]{icon}[/{color}]")

    _row("Win Rate",      stats["win_rate"],       base["win_rate"],     "{:.1%}")
    _row("Avg R/trade",   stats["avg_r"],          base["avg_net_pnl_r"],"{:+.3f}R")
    _row("Profit Factor", stats["profit_factor"],  base["net_pf"],       "{:.2f}")
    _row("Max Drawdown",  stats["max_drawdown_r"], base["max_drawdown_r"],"-{:.1f}R",
         higher_is_better=False)

    console.print(perf)

    # ── Actual R/R and duration block ─────────────────────────────────────────
    if stats["n_closed"] > 0:
        actual = []
        if stats["avg_actual_win_r"] is not None:
            actual.append(f"  Avg win  : {stats['avg_actual_win_r']:+.2f}R")
        if stats["avg_actual_loss_r"] is not None:
            actual.append(f"  Avg loss : {stats['avg_actual_loss_r']:+.2f}R")
        if stats["actual_rr"] is not None:
            actual.append(f"  Actual R/R: {stats['actual_rr']:.2f}×")
        if stats["avg_duration_days"] is not None:
            actual.append(f"  Avg hold  : {stats['avg_duration_days']:.1f} days")
        if actual:
            console.print("\n  [dim]Trade quality:[/dim]")
            for line in actual:
                console.print(f"  {line}")

    # Verdict
    if stats["n_closed"] >= 30:
        on_track = (
            stats["win_rate"]     >= base["win_rate"]     * 0.85 and
            stats["profit_factor"] >= 1.0
        )
        if on_track:
            verdict = "[green]ON TRACK[/green] — performance consistent with backtest expectations."
        else:
            verdict = "[red]UNDERPERFORMING[/red] — review open trades and check market regime."
    else:
        verdict = f"[dim]INSUFFICIENT DATA[/dim] — need {30 - stats['n_closed']} more closed trades for verdict."

    console.print(f"\n  Verdict: {verdict}\n")

    # Trade log
    if stats["trades"]:
        log = Table(title="Trade Log", box=box.SIMPLE, show_header=True, padding=(0, 2))
        log.add_column("Date",      width=11)
        log.add_column("Dir",       width=5)
        log.add_column("Entry",     justify="right", width=8)
        log.add_column("SL",        justify="right", width=8)
        log.add_column("TP1",       justify="right", width=8)
        log.add_column("Fill",      justify="right", width=8)
        log.add_column("Status",    width=12)
        log.add_column("P&L (R)",   justify="right", width=8)

        status_colors = {
            "win": "green", "partial_win": "cyan",
            "loss": "red", "breakeven": "yellow",
            "filled": "blue", "pending": "dim", "expired": "dim",
        }
        for t in sorted(stats["trades"], key=lambda x: x["signal_date"], reverse=True):
            sc    = status_colors.get(t["status"], "white")
            pnl_s = f"{t['pnl_r']:+.2f}R" if t["pnl_r"] is not None else "—"
            pnl_c = "green" if (t["pnl_r"] or 0) > 0 else ("red" if (t["pnl_r"] or 0) < 0 else "dim")
            log.add_row(
                t["signal_date"],
                t["direction"].upper(),
                f"{t['entry_low']:,.0f}",
                f"{t['stop_loss']:,.0f}",
                f"{t['tp1']:,.0f}",
                f"{t['fill_price']:,.0f}" if t["fill_price"] else "—",
                f"[{sc}]{t['status'].upper()}[/{sc}]",
                f"[{pnl_c}]{pnl_s}[/{pnl_c}]",
            )
        console.print(log)
