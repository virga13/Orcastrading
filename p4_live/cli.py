"""
p4_live/cli.py — Command-line interface for the P4 live forward-test module.

Usage:
    python -m p4_live scan                          # Run today's signal check
    python -m p4_live fill DATE PRICE               # Record a fill
    python -m p4_live close DATE OUTCOME EXIT       # Record a close with auto pnl_r
    python -m p4_live expire DATE                   # Mark signal as expired (price never entered)
    python -m p4_live check                         # Check open trades vs latest prices
    python -m p4_live note DATE "text"              # Attach a note to a trade
    python -m p4_live report                        # Forward-test performance report
    python -m p4_live log [--open|--closed]         # Display trade journal
"""
import sys
import argparse
from datetime import date

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console(legacy_windows=False)


# ── scan ──────────────────────────────────────────────────────────────────────

def _display_result(result: dict, no_record: bool) -> None:
    """Display one scan result and optionally record it to the journal."""
    from p4_live.journal import record_signal, get_active_strategy_set

    r        = result
    label    = r["label"]
    strategy = r.get("strategy", "mtf_trend")
    strat_s  = strategy.upper().replace("_", " ")
    title    = f"Signal Scan — {label}  [{strat_s}]"

    if not r["fired"]:
        price_s = (
            f"Price [bold]{r['price']:,.2f}[/bold]  "
            f"ATR [bold]{r['atr']:.1f}[/bold]  RSI [bold]{r['rsi']:.0f}[/bold]"
        ) if r["price"] is not None else ""
        console.print(Panel(
            f"[yellow]NO SIGNAL[/yellow]  {r['signal_date']}\n\n"
            f"[dim]{r.get('reason', '')}[/dim]"
            + (f"\n\n{price_s}" if price_s else ""),
            title=title, box=box.ROUNDED, padding=(0, 3),
        ))
        return

    tp2_s = f"  TP2 ({r['tp2_alloc']}%): [bold green]{r['tp2']:,.2f}[/bold green]" \
            if r.get("tp2") else ""
    console.print(Panel(
        f"[green bold]SIGNAL FIRED[/green bold]  {r['signal_date']}\n\n"
        f"Direction : [bold]{r['direction'].upper()}[/bold]\n"
        f"Entry zone: [bold]{r['entry_low']:,.2f} – {r['entry_high']:,.2f}[/bold]\n"
        f"Stop loss : [bold red]{r['stop_loss']:,.2f}[/bold red]  "
        f"({r['risk_pts']:,.1f} pts risk)\n"
        f"TP1 ({r['tp1_alloc']}%): [bold green]{r['tp1']:,.2f}[/bold green]{tp2_s}\n"
        f"R/R ratio : {r['rr']:.2f}  |  Confidence: {r['confidence']:.0%}\n\n"
        f"[dim]{r.get('rationale', '')}[/dim]",
        title=title, box=box.ROUNDED, padding=(0, 3),
    ))

    if not no_record:
        active = get_active_strategy_set()
        if (r["ticker"], strategy) in active:
            console.print(f"  [dim]Active {strat_s} trade for {label} — signal skipped.[/dim]")
        else:
            recorded = record_signal(result)
            if recorded:
                console.print(f"  [green][OK] Recorded to journal.[/green]")
                from p4_live.alerts import alert_signal
                alert_signal(result)
            else:
                console.print(f"  [dim]Already in journal — skipped.[/dim]")
    console.print()


def cmd_scan(args) -> None:
    from p4_live.scanner import scan_strategy
    from core.config import get_all_assets, get_enabled_strategies, get_ticker
    from p4_live.alerts import configured_channels

    as_of = date.fromisoformat(args.date) if args.date else None

    # Build list of (asset_id, strategy_id) pairs to scan
    if args.ticker:
        # Find asset by ticker
        from core.config import get_asset_by_ticker as _gat
        asset = _gat(args.ticker)
        if not asset:
            console.print(f"[red]Unknown ticker '{args.ticker}'[/red]"); return
        pairs = [(asset["id"], s) for s in get_enabled_strategies(asset["id"])]
        if args.strategy:
            pairs = [(aid, sid) for aid, sid in pairs if sid == args.strategy]
    elif args.strategy:
        pairs = [(a["id"], args.strategy) for a in get_all_assets()
                 if args.strategy in get_enabled_strategies(a["id"])]
    else:
        from core.config import get_watchlist
        pairs = get_watchlist()  # list of (asset_id, strategy_id, timeframe)

    ch = configured_channels()
    ch_str = ", ".join(ch) if ch else "none — run alert-test to configure"
    console.print(
        f"\n[dim]Scanning {len(pairs)} asset/strategy pair(s) "
        f"as of {as_of or date.today()}  |  alerts: {ch_str}[/dim]\n"
    )

    for entry in pairs:
        asset_id, strategy_id = entry[0], entry[1]
        timeframe = entry[2] if len(entry) > 2 else None
        try:
            result = scan_strategy(asset_id, strategy_id, timeframe=timeframe, as_of=as_of)
        except Exception as e:
            console.print(f"[red]Error ({asset_id}/{strategy_id}):[/red] {e}\n")
            continue
        _display_result(result, no_record=args.no_record)


# ── fill ──────────────────────────────────────────────────────────────────────

def cmd_fill(args) -> None:
    from p4_live.journal import record_fill, get_all_trades

    signal_date = args.signal_date
    fill_price  = float(args.price)
    fill_date   = args.fill_date or signal_date

    record_fill(signal_date, fill_price, fill_date)
    console.print(
        f"[green][OK][/green] Recorded fill for {signal_date}: "
        f"price={fill_price:,.2f}  fill_date={fill_date}"
    )


# ── close ─────────────────────────────────────────────────────────────────────

def cmd_close(args) -> None:
    from p4_live.journal import record_close

    valid_outcomes = ("win", "partial_win", "loss", "breakeven")
    if args.outcome not in valid_outcomes:
        console.print(
            f"[red]Invalid outcome '{args.outcome}'. "
            f"Choose from: {', '.join(valid_outcomes)}[/red]"
        )
        sys.exit(1)

    exit_price = float(args.exit_price)
    exit_date  = args.exit_date or date.today().isoformat()

    record_close(args.signal_date, args.outcome, exit_price, exit_date)

    # Look up computed pnl_r to display
    from p4_live.journal import get_all_trades
    trades = get_all_trades()
    trade  = next((t for t in trades if t["signal_date"] == args.signal_date), None)
    pnl_s  = f"{trade['pnl_r']:+.3f}R" if trade and trade["pnl_r"] is not None else "?"
    color  = "green" if (trade and (trade["pnl_r"] or 0) > 0) else "red"
    console.print(
        f"[green][OK][/green] Closed {args.signal_date}: "
        f"outcome=[bold]{args.outcome}[/bold]  exit={exit_price:,.2f}  "
        f"pnl=[{color}]{pnl_s}[/{color}]"
    )


# ── expire ────────────────────────────────────────────────────────────────────

def cmd_expire(args) -> None:
    from p4_live.journal import record_expired
    record_expired(args.signal_date)
    console.print(f"[green][OK][/green] Marked {args.signal_date} as expired.")


# ── note ──────────────────────────────────────────────────────────────────────

def cmd_note(args) -> None:
    from p4_live.journal import add_note
    add_note(args.signal_date, args.text)
    console.print(f"[green][OK][/green] Note added to {args.signal_date}.")


# ── check ─────────────────────────────────────────────────────────────────────

def _fetch_latest_bar(ticker: str) -> dict | None:
    from p4_live.scanner import _fetch_latest_bar as _scan_fetch
    return _scan_fetch(ticker)


def cmd_check(args) -> None:
    """Fetch latest prices per asset and flag open trades where SL or TP has been breached."""
    from p4_live.journal import get_open_trades

    open_trades = get_open_trades()
    if not open_trades:
        console.print("\n[dim]No open trades to check.[/dim]\n")
        return

    # Fetch one price bar per unique ticker
    tickers = list({t["ticker"] for t in open_trades})
    bars: dict[str, dict] = {}
    for ticker in tickers:
        bar = _fetch_latest_bar(ticker)
        if bar:
            bars[ticker] = bar
            console.print(f"  [dim]{ticker:10} {bar['date']}  C={bar['close']:,.2f}  H={bar['high']:,.2f}  L={bar['low']:,.2f}[/dim]")
        else:
            console.print(f"  [red]Failed to fetch price for {ticker}[/red]")
    console.print()

    tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 3))
    tbl.add_column("Date",      width=12)
    tbl.add_column("Asset",     width=8)
    tbl.add_column("Status",    width=10)
    tbl.add_column("Dir",       width=5)
    tbl.add_column("Entry zone",width=20)
    tbl.add_column("SL",        justify="right", width=9)
    tbl.add_column("TP1",       justify="right", width=9)
    tbl.add_column("Fill",      justify="right", width=9)
    tbl.add_column("Flag",      width=30)

    from p4_live.alerts import alert_price

    for t in open_trades:
        bar        = bars.get(t["ticker"])
        flag       = ""
        alert_evt  = None
        alert_det  = ""

        if bar is None:
            flag = "[dim]price unavailable[/dim]"
        else:
            price = bar["close"]
            high  = bar["high"]
            low   = bar["low"]

            if t["status"] == "pending":
                zone_touched = (
                    (t["direction"] == "long"  and low <= t["entry_high"] and high >= t["entry_low"]) or
                    (t["direction"] == "short" and high >= t["entry_low"] and low <= t["entry_high"])
                )
                if zone_touched:
                    flag      = "[yellow]! Entry zone touched[/yellow]"
                    alert_evt = "entry"
                    alert_det = f"Bar {low:,.2f}–{high:,.2f} overlaps entry {t['entry_low']:,.2f}–{t['entry_high']:,.2f}"
            elif t["status"] == "filled":
                sl  = t["stop_loss"]
                tp1 = t["tp1"]
                tp2 = t.get("tp2")
                if t["direction"] == "long":
                    if low <= sl:
                        flag      = "[red][x] SL breached — record close[/red]"
                        alert_evt = "sl"
                        alert_det = f"Bar low {low:,.2f} <= SL {sl:,.2f}"
                    elif tp2 and high >= tp2:
                        flag      = "[green][OK] TP2 hit — close full position[/green]"
                        alert_evt = "tp"
                        alert_det = f"Bar high {high:,.2f} >= TP2 {tp2:,.2f}"
                    elif high >= tp1:
                        flag      = "[green][OK] TP1 hit — close 70%[/green]"
                        alert_evt = "tp"
                        alert_det = f"Bar high {high:,.2f} >= TP1 {tp1:,.2f}"
                    else:
                        pct  = (price - sl) / (tp1 - sl) * 100 if tp1 != sl else 0
                        flag = f"[dim]{pct:.0f}% to TP1[/dim]"
                else:
                    if high >= sl:
                        flag      = "[red][x] SL breached — record close[/red]"
                        alert_evt = "sl"
                        alert_det = f"Bar high {high:,.2f} >= SL {sl:,.2f}"
                    elif tp2 and low <= tp2:
                        flag      = "[green][OK] TP2 hit — close full position[/green]"
                        alert_evt = "tp"
                        alert_det = f"Bar low {low:,.2f} <= TP2 {tp2:,.2f}"
                    elif low <= tp1:
                        flag      = "[green][OK] TP1 hit — close 70%[/green]"
                        alert_evt = "tp"
                        alert_det = f"Bar low {low:,.2f} <= TP1 {tp1:,.2f}"
                    else:
                        pct  = (sl - price) / (sl - tp1) * 100 if sl != tp1 else 0
                        flag = f"[dim]{pct:.0f}% to TP1[/dim]"

            if alert_evt:
                alert_price(alert_evt, t, price, alert_det)

        fill_s = f"{t['fill_price']:,.0f}" if t["fill_price"] else "—"
        tbl.add_row(
            t["signal_date"],
            t.get("label") or t.get("ticker", ""),
            t["status"].upper(),
            t["direction"].upper(),
            f"{t['entry_low']:,.0f} – {t['entry_high']:,.0f}",
            f"{t['stop_loss']:,.0f}",
            f"{t['tp1']:,.0f}",
            fill_s,
            flag,
        )

    console.print(tbl)


# ── daily ─────────────────────────────────────────────────────────────────────

def cmd_daily(args) -> None:
    """Full end-of-day routine: scan all asset/strategy pairs, check open trades, report."""
    console.print(f"\n[bold]Orcastrading — Daily Run  {date.today()}[/bold]\n")

    from p4_live.scanner import scan_strategy
    from p4_live.alerts import configured_channels, alert_daily_summary
    from core.config import get_watchlist

    ch = configured_channels()
    ch_str = ", ".join(ch) if ch else "none"
    console.print(f"[dim]Alerts: {ch_str}[/dim]\n")

    # 1. Scan all asset/strategy pairs
    console.rule("[dim]Signal Scan[/dim]")
    pairs   = get_watchlist()
    results = []
    for asset_id, strategy_id, timeframe in pairs:
        try:
            result = scan_strategy(asset_id, strategy_id, timeframe=timeframe)
        except Exception as e:
            console.print(f"[red]Error ({asset_id}/{strategy_id}/{timeframe}):[/red] {e}\n")
            continue
        results.append(result)
        _display_result(result, no_record=False)

    # 2. Check open trades
    console.rule("[dim]Open Trade Check[/dim]")
    from p4_live.journal import get_open_trades
    if not get_open_trades():
        console.print("\n[dim]No open trades.[/dim]\n")
    else:
        class _FakeArgs: pass
        cmd_check(_FakeArgs())

    # 3. CLI performance summary (MTF Trend per asset)
    console.rule("[dim]Performance Summary[/dim]")
    from p4_live.report import print_report
    from p4_live.scanner import ASSETS
    for asset in ASSETS:
        print_report(console, ticker=asset["ticker"])

    # 4. Daily summary alert (fires even if no signals)
    if ch:
        alert_daily_summary(results)

    # 5. Optionally generate HTML
    if args.html:
        from p4_live.html_report import generate_html
        path = generate_html()
        console.print(f"\n[green][OK][/green] HTML report: [bold]{path}[/bold]\n")


# ── alert-test ────────────────────────────────────────────────────────────────

def cmd_alert_test(args) -> None:
    from p4_live.alerts import alert_test, configured_channels, is_configured

    channels = configured_channels()
    if not channels:
        console.print(
            "\n[yellow]No alert channels configured.[/yellow]\n\n"
            "Add to your .env file:\n\n"
            "  [bold]Telegram[/bold] (recommended):\n"
            "    TELEGRAM_BOT_TOKEN=<token from @BotFather>\n"
            "    TELEGRAM_CHAT_ID=<your chat or group ID>\n\n"
            "  [bold]Email[/bold] (Gmail example):\n"
            "    ALERT_EMAIL_FROM=you@gmail.com\n"
            "    ALERT_EMAIL_TO=you@gmail.com\n"
            "    ALERT_EMAIL_PASSWORD=<Gmail app password>\n"
            "    ALERT_SMTP_HOST=smtp.gmail.com\n"
            "    ALERT_SMTP_PORT=587\n\n"
            "Then run [bold]python -m p4_live alert-test[/bold] again.\n"
        )
        return

    console.print(f"\n[dim]Configured channels: {', '.join(channels)}[/dim]")
    console.print("[dim]Sending test alert...[/dim]\n")
    results = alert_test()
    for ch, ok in results.items():
        if ok:
            console.print(f"  [green][OK][/green] {ch.capitalize()} — delivered")
        elif ch == "telegram" and "Telegram" in channels:
            console.print(f"  [red][x][/red] {ch.capitalize()} — failed (check token/chat_id)")
        elif ch == "email" and "Email" in channels:
            console.print(f"  [red][x][/red] {ch.capitalize()} — failed (check credentials)")
    console.print()


# ── backfill ──────────────────────────────────────────────────────────────────

def cmd_backfill(args) -> None:
    from p4_live.backfill import backfill
    backfill(
        start_date=args.start,
        end_date=args.end or date.today().isoformat(),
        tickers=[args.ticker] if args.ticker else None,
        dry_run=args.dry_run,
    )


# ── report ────────────────────────────────────────────────────────────────────

def cmd_report(args) -> None:
    if args.html:
        from p4_live.html_report import generate_html
        path = generate_html()
        console.print(f"[green][OK][/green] HTML report saved: [bold]{path}[/bold]")
        import webbrowser
        webbrowser.open(f"file:///{path}")
        return
    from p4_live.report import print_report
    from p4_live.scanner import ASSETS
    tickers = [args.ticker] if args.ticker else [a["ticker"] for a in ASSETS]
    for ticker in tickers:
        print_report(console, ticker=ticker)


# ── log ───────────────────────────────────────────────────────────────────────

def cmd_log(args) -> None:
    from p4_live.journal import get_all_trades, get_open_trades, get_closed_trades

    if args.open:
        trades = get_open_trades()
        title  = "Open Trades"
    elif args.closed:
        trades = get_closed_trades()
        title  = "Closed Trades"
    else:
        trades = get_all_trades()
        title  = "All Trades"

    if args.ticker:
        trades = [t for t in trades if t["ticker"] == args.ticker]
        title  += f" — {args.ticker}"

    if not trades:
        console.print(f"\n[dim]No trades in journal.[/dim]\n")
        return

    tbl = Table(title=title, box=box.SIMPLE, show_header=True, padding=(0, 2))
    tbl.add_column("Date",     width=11)
    tbl.add_column("Asset",    width=7)
    tbl.add_column("Dir",      width=5)
    tbl.add_column("Entry",    justify="right", width=9)
    tbl.add_column("SL",       justify="right", width=9)
    tbl.add_column("TP1",      justify="right", width=9)
    tbl.add_column("Fill",     justify="right", width=9)
    tbl.add_column("Exit",     justify="right", width=9)
    tbl.add_column("Status",   width=13)
    tbl.add_column("P&L (R)",  justify="right", width=9)
    tbl.add_column("Notes",    width=25)

    status_colors = {
        "win": "green", "partial_win": "cyan",
        "loss": "red", "breakeven": "yellow",
        "filled": "blue", "pending": "dim", "expired": "dim",
    }

    for t in sorted(trades, key=lambda x: x["signal_date"], reverse=True):
        sc    = status_colors.get(t["status"], "white")
        pnl_s = f"{t['pnl_r']:+.2f}R" if t["pnl_r"] is not None else "—"
        pnl_c = "green" if (t["pnl_r"] or 0) > 0 else ("red" if (t["pnl_r"] or 0) < 0 else "dim")
        tbl.add_row(
            t["signal_date"],
            t.get("label") or t.get("ticker", ""),
            t["direction"].upper(),
            f"{t['entry_low']:,.0f}",
            f"{t['stop_loss']:,.0f}",
            f"{t['tp1']:,.0f}",
            f"{t['fill_price']:,.0f}" if t["fill_price"] else "—",
            f"{t['exit_price']:,.0f}" if t["exit_price"] else "—",
            f"[{sc}]{t['status'].upper()}[/{sc}]",
            f"[{pnl_c}]{pnl_s}[/{pnl_c}]",
            (t["notes"] or "")[:23],
        )

    console.print(tbl)


# ── parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m p4_live",
        description="Orcastrading P4 — Live Forward-Test CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # scan
    sc = sub.add_parser("scan", help="Run today's signal check (all assets by default)")
    sc.add_argument("--date",     default=None, metavar="YYYY-MM-DD",
                    help="Scan as of a specific date (default: today)")
    sc.add_argument("--ticker",   default=None, metavar="TICKER",
                    help="Scan one asset only, e.g. ^GSPC")
    sc.add_argument("--strategy", default=None, metavar="STRATEGY",
                    help="Scan one strategy only, e.g. orb or mtf_trend")
    sc.add_argument("--no-record", action="store_true",
                    help="Display signal but do not write to journal")
    sc.set_defaults(func=cmd_scan)

    # fill
    fi = sub.add_parser("fill", help="Record a trade fill")
    fi.add_argument("signal_date", metavar="DATE", help="Signal date (YYYY-MM-DD)")
    fi.add_argument("price",       metavar="PRICE", help="Fill price")
    fi.add_argument("--fill-date", default=None, metavar="YYYY-MM-DD",
                    help="Date price was filled (default: signal_date)")
    fi.set_defaults(func=cmd_fill)

    # close
    cl = sub.add_parser("close", help="Record a trade close")
    cl.add_argument("signal_date", metavar="DATE",    help="Signal date (YYYY-MM-DD)")
    cl.add_argument("outcome",     metavar="OUTCOME",
                    help="win | partial_win | loss | breakeven")
    cl.add_argument("exit_price",  metavar="EXIT",    help="Exit price")
    cl.add_argument("--exit-date", default=None, metavar="YYYY-MM-DD",
                    help="Date the trade was closed (default: today)")
    cl.set_defaults(func=cmd_close)

    # expire
    ex = sub.add_parser("expire", help="Mark pending signal as expired (never filled)")
    ex.add_argument("signal_date", metavar="DATE", help="Signal date (YYYY-MM-DD)")
    ex.set_defaults(func=cmd_expire)

    # note
    nt = sub.add_parser("note", help="Attach a note to a trade")
    nt.add_argument("signal_date", metavar="DATE", help="Signal date (YYYY-MM-DD)")
    nt.add_argument("text",        metavar="TEXT", help="Note text")
    nt.set_defaults(func=cmd_note)

    # check
    ch = sub.add_parser("check", help="Fetch latest prices and flag open trades")
    ch.set_defaults(func=cmd_check)

    # daily
    dy = sub.add_parser("daily", help="Full end-of-day routine: scan + check + report")
    dy.add_argument("--html", action="store_true",
                    help="Also generate HTML report after the run")
    dy.set_defaults(func=cmd_daily)

    # alert-test
    at = sub.add_parser("alert-test", help="Send a test alert on all configured channels")
    at.set_defaults(func=cmd_alert_test)

    # backfill
    bf = sub.add_parser("backfill", help="Replay scanner historically and simulate outcomes")
    bf.add_argument("start", metavar="START", help="Start date YYYY-MM-DD")
    bf.add_argument("--end",     default=None, metavar="YYYY-MM-DD",
                    help="End date (default: today)")
    bf.add_argument("--ticker",  default=None, metavar="TICKER",
                    help="Backfill one asset only (default: all)")
    bf.add_argument("--dry-run", action="store_true",
                    help="Show signals without writing to journal")
    bf.set_defaults(func=cmd_backfill)

    # report
    rp = sub.add_parser("report", help="Display forward-test performance report")
    rp.add_argument("--ticker", default=None, metavar="TICKER",
                    help="Report for one asset only (default: all)")
    rp.add_argument("--html", action="store_true",
                    help="Generate HTML report and open in browser")
    rp.set_defaults(func=cmd_report)

    # log
    lg = sub.add_parser("log", help="Display trade journal")
    lg.add_argument("--open",   action="store_true", help="Show only open trades")
    lg.add_argument("--closed", action="store_true", help="Show only closed trades")
    lg.add_argument("--ticker", default=None, metavar="TICKER",
                    help="Filter by asset ticker, e.g. ^GSPC")
    lg.set_defaults(func=cmd_log)

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
