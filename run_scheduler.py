"""
run_scheduler.py — Orcastrading continuous market scanner.

Runs 24/7, scanning for setups across all configured assets and strategies.
Sends Telegram alerts immediately when a signal fires.

Schedule (all times are Romania local — EEST = UTC+3 in summer, EET = UTC+2 in winter):
  Every 15 min, 15:30–18:30 local  → 15m intraday scan  (US session open, 2h)
  Every 4 hours around the clock   → 4h / 1h scan
  Daily at 22:00 local             → 1d scan + HTML report + daily summary
  At startup                       → All three scans run once immediately

Usage:
  python run_scheduler.py

To run at Windows startup (run once as Administrator):
  schtasks /create /tn "Orcastrading Scheduler" ^
    /tr "\"C:\\Users\\admin\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe\" \"C:\\Users\\admin\\Desktop\\Orcastrading\\run_scheduler.py\"" ^
    /sc onstart /ru SYSTEM /f

Logs: logs/scheduler.log
"""
import sys
import io
import os
import json
import logging
import schedule
import threading
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

# ── UTF-8 stdout (avoid Windows CP1252 issues) ────────────────────────────────
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Load .env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orcastrading")

# ── Last-run tracking (missed-job catch-up) ───────────────────────────────────
_LAST_RUN_PATH = LOG_DIR / "scheduler_last_run.json"


def _read_last_run() -> dict:
    try:
        if _LAST_RUN_PATH.exists():
            return json.loads(_LAST_RUN_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_last_run(job: str) -> None:
    data = _read_last_run()
    data[job] = datetime.now().isoformat()
    try:
        _LAST_RUN_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── US session intraday window (Romania local time) ───────────────────────────
# Romania is EEST (UTC+3) from late March to late October.
# 15:30–18:30 local = 12:30–15:30 UTC = 08:30–11:30 ET (pre-open + first 2h)
# Adjust if your system clock does not match Romania local time.
INTRADAY_START = dtime(15, 30)
INTRADAY_END   = dtime(18, 30)


def _within_intraday_window() -> bool:
    now = datetime.now().time()
    return INTRADAY_START <= now <= INTRADAY_END


def _at_bar_boundary(bar_minutes: int = 15, window_secs: int = 90) -> bool:
    """True when within window_secs seconds after a completed N-minute bar close."""
    now = datetime.now()
    elapsed_secs = (now.minute % bar_minutes) * 60 + now.second
    return elapsed_secs <= window_secs


# ── MT5: execute a signal and store tickets in journal ────────────────────────

_MT5_CONFIDENCE_THRESHOLD = 0.70


def _execute_mt5(result: dict) -> None:
    """Place MT5 pending orders for a fired signal and record the tickets."""
    confidence = result.get("confidence") or 0.0
    if confidence < _MT5_CONFIDENCE_THRESHOLD:
        log.info(
            f"  MT5 skipped: {result.get('label', result.get('ticker', ''))} "
            f"[{result.get('strategy', '')}] confidence {confidence:.0%} "
            f"< {_MT5_CONFIDENCE_THRESHOLD:.0%} threshold"
        )
        return
    try:
        from p4_live import mt5_broker
        if not mt5_broker.is_enabled():
            return
        tickets = mt5_broker.execute_signal(result)
        if tickets:
            from p4_live.journal import record_mt5_ticket
            record_mt5_ticket(
                result["signal_date"],
                result["ticker"],
                result.get("strategy", "mtf_trend"),
                result.get("timeframe", "1d"),
                tickets,
            )
            log.info(f"  MT5 orders placed: tickets={tickets}")
        else:
            log.warning(
                f"  MT5 execution failed for {result['label']} — "
                f"trade is journal-only (check MT5_ENABLED / tickers.mt5)"
            )
    except Exception as e:
        log.error(f"MT5 execute failed: {e}", exc_info=True)


# ── MT5: sync open positions back into journal ────────────────────────────────

def _retry_pending_mt5_orders() -> None:
    """
    Re-attempt MT5 order placement for pending journal trades that have no ticket.
    This handles signals that fired while the market was closed (retcode=10018).
    Called on every MT5 monitor cycle (every 60 s) until the market reopens.
    """
    try:
        from p4_live import mt5_broker
        from p4_live.journal import get_pending_without_ticket, record_mt5_ticket

        pending = get_pending_without_ticket()
        if not pending:
            return

        for trade in pending:
            signal = {
                "ticker":     trade["ticker"],
                "strategy":   trade["strategy"],
                "timeframe":  trade.get("timeframe", "1d"),
                "direction":  trade["direction"],
                "entry_low":  trade["entry_low"],
                "entry_high": trade["entry_high"],
                "stop_loss":  trade["stop_loss"],
                "tp1":        trade["tp1"],
                "tp2":        trade.get("tp2"),
                "tp1_alloc":  trade.get("tp1_alloc", 70),
                "tp2_alloc":  trade.get("tp2_alloc", 30),
                "risk_pts":   trade.get("risk_pts"),
                "signal_date": trade["signal_date"],
            }
            tickets = mt5_broker.execute_signal(signal)
            if tickets:
                record_mt5_ticket(
                    trade["signal_date"], trade["ticker"],
                    trade["strategy"], trade.get("timeframe", "1d"),
                    tickets,
                )
                log.info(
                    f"  MT5 retry OK: {trade.get('label', trade['ticker'])} "
                    f"[{trade['strategy']}@{trade.get('timeframe','1d')}] "
                    f"tickets={tickets}"
                )
    except Exception as e:
        log.error(f"MT5 order retry failed: {e}", exc_info=True)


def _run_mt5_monitor() -> None:
    """Poll MT5 terminal for fills/closes and write them to the journal."""
    try:
        from p4_live import mt5_broker
        if not mt5_broker.is_enabled():
            return
        from p4_live.mt5_monitor import sync_mt5_positions, cancel_pending_for_expired
        updates = sync_mt5_positions()
        if updates:
            fills   = sum(1 for u in updates if u.get("status") == "filled")
            closed  = sum(1 for u in updates if u.get("status") in
                         ("win", "loss", "partial_win", "breakeven"))
            expired = sum(1 for u in updates if u.get("status") == "expired")
            log.info(f"MT5 monitor: {fills} filled, {closed} closed, {expired} expired")
        # Cancel any MT5 limit orders left open after the journal expired the signal.
        # Prevents ghost fills on trades the journal considers dead.
        cancelled = cancel_pending_for_expired()
        if cancelled:
            log.info(f"MT5 monitor: cancelled {len(cancelled)} order(s) for expired signals")
        _retry_pending_mt5_orders()
    except Exception as e:
        log.error(f"MT5 monitor error: {e}", exc_info=True)


# ── Shared: run simulate_pending after any scan ───────────────────────────────

def _run_simulate() -> None:
    """Resolve pending → filled → closed for all open trades."""
    try:
        from p4_live.simulate import simulate_pending
        results = simulate_pending()
        fills   = sum(1 for r in results if r.get("status") == "filled")
        active  = sum(1 for r in results if r.get("status") == "active")
        closed  = sum(1 for r in results if r.get("outcome") in ("win", "loss", "partial_win"))
        expired = sum(1 for r in results if r.get("status") == "expired")
        if any([fills, active, closed, expired]):
            log.info(f"  Simulate: {fills} filled, {active} active, {closed} closed, {expired} expired")
    except Exception as e:
        log.error(f"Simulate failed: {e}", exc_info=True)


# ── Shared: rule-based scan for specific timeframes ───────────────────────────

def _load_risk_gates() -> dict:
    """
    Load account-level risk gates from .env.
    Returns dict with: max_open (int), daily_limit_r (float), weekly_limit_r (float).
    Zero means the gate is disabled.
    """
    return {
        "max_open":       int(float(os.getenv("MAX_OPEN_TRADES", "5"))),
        "daily_limit_r":  float(os.getenv("DAILY_LOSS_LIMIT_R", "0")),
        "weekly_limit_r": float(os.getenv("WEEKLY_DD_LIMIT_R", "0")),
    }


def _check_risk_gates(gates: dict, n_open: int, daily_pnl: float, weekly_pnl: float) -> str | None:
    """
    Return a human-readable reason string if a new signal should be suppressed,
    or None if all gates pass.
    """
    if n_open >= gates["max_open"]:
        return f"max open trades reached ({n_open}/{gates['max_open']})"
    if gates["daily_limit_r"] > 0 and daily_pnl <= -gates["daily_limit_r"]:
        return f"daily loss limit hit ({daily_pnl:+.2f}R / -{gates['daily_limit_r']:.1f}R)"
    if gates["weekly_limit_r"] > 0 and weekly_pnl <= -gates["weekly_limit_r"]:
        return f"weekly DD limit hit ({weekly_pnl:+.2f}R / -{gates['weekly_limit_r']:.1f}R)"
    return None


def _run_scan(label: str, timeframe_filter: list[str] | None = None) -> tuple[list[dict], int]:
    """
    Scan every (asset, strategy, timeframe) triple in the watchlist,
    optionally filtered to specific timeframes.
    Records new signals, fires Telegram alerts, then resolves pending trades.
    Returns (fired_signals, total_pairs_checked).

    Risk gates (read from .env before each scan):
      MAX_OPEN_TRADES     — suppress new signals when open/pending count >= this
      DAILY_LOSS_LIMIT_R  — suppress new signals when today's closed P&L <= -limit
      WEEKLY_DD_LIMIT_R   — suppress new signals when this week's closed P&L <= -limit
      Zero (default) means the gate is disabled.
    """
    from datetime import date as _date, timedelta as _td
    log.info(f"{label} scan starting...")
    fired_signals: list[dict] = []
    try:
        from core.config import get_watchlist
        from p4_live.scanner import scan_strategy
        from p4_live.journal import (record_signal, get_active_strategy_set,
                                     get_active_ticker_set, get_open_trades,
                                     get_closed_trades)
        from p4_live.alerts import alert_signal

        watchlist = get_watchlist()
        if timeframe_filter:
            watchlist = [(a, s, tf) for a, s, tf in watchlist if tf in timeframe_filter]

        if not watchlist:
            log.warning(f"{label}: no pairs found for timeframes {timeframe_filter}")
            return fired_signals, 0

        active       = get_active_strategy_set()
        open_tickers = list(get_active_ticker_set())
        checked      = 0

        # ── Risk gate state (computed once, updated as signals fire) ──────────
        gates = _load_risk_gates()
        n_open = len(get_open_trades())

        today_str  = _date.today().isoformat()
        week_start = (_date.today() - _td(days=_date.today().weekday())).isoformat()
        closed_all = get_closed_trades()

        daily_pnl = sum(
            t["pnl_r"] for t in closed_all
            if t.get("exit_date") == today_str and t.get("pnl_r") is not None
        ) if gates["daily_limit_r"] > 0 else 0.0

        weekly_pnl = sum(
            t["pnl_r"] for t in closed_all
            if (t.get("exit_date") or "") >= week_start and t.get("pnl_r") is not None
        ) if gates["weekly_limit_r"] > 0 else 0.0

        for asset_id, strategy_id, timeframe in watchlist:
            checked += 1
            try:
                result = scan_strategy(asset_id, strategy_id, timeframe=timeframe)
            except Exception as e:
                log.warning(f"  {asset_id}/{strategy_id}/{timeframe}: {e}")
                continue

            if not result.get("fired"):
                log.debug(f"  {result.get('label', asset_id):8} [{strategy_id}@{timeframe}] no signal")
                continue

            ticker = result["ticker"]
            if (ticker, strategy_id) in active:
                log.info(
                    f"  {result['label']:8} [{strategy_id}@{timeframe}] "
                    f"signal skipped — trade already open"
                )
                continue

            # ── Account-level risk gates ──────────────────────────────────────
            block_reason = _check_risk_gates(gates, n_open, daily_pnl, weekly_pnl)
            if block_reason:
                log.warning(
                    f"  {result.get('label', asset_id):8} [{strategy_id}@{timeframe}] "
                    f"signal blocked — {block_reason}"
                )
                continue

            recorded = record_signal(result)
            if recorded:
                log.info(
                    f"  {result['label']:8} [{strategy_id}@{timeframe}] SIGNAL FIRED "
                    f"{result['direction'].upper()} "
                    f"entry {result['entry_low']:,.2f}–{result['entry_high']:,.2f}"
                )
                alert_signal(result, open_positions=open_tickers)
                _execute_mt5(result)
                active.add((ticker, strategy_id))   # block same strategy on other TFs this scan
                if ticker not in open_tickers:
                    open_tickers.append(ticker)
                n_open += 1   # keep the gate counter in sync for remaining signals
                fired_signals.append(result)
            else:
                log.info(f"  {result['label']:8} [{strategy_id}@{timeframe}] already in journal")

        log.info(f"{label} scan done — {len(fired_signals)} new signal(s) across {checked} pair(s)")

    except Exception as e:
        log.error(f"{label} scan failed: {e}", exc_info=True)

    # Always attempt to resolve pending/filled trades after a scan
    _run_simulate()
    return fired_signals, checked


# ── Missed-signal replay helpers ─────────────────────────────────────────────

def _is_signal_stale(result: dict, current_bar: dict | None) -> tuple[bool, str]:
    """
    True when a replayed signal is no longer actionable at the current price.
    The only unambiguous invalidation is an SL breach: if price has already
    traded through the stop level, the setup is structurally broken regardless
    of how it was generated.  Signals that are merely "not yet triggered" (entry
    zone not yet touched) are left as pending and will expire via expire_stale_signals().
    """
    if current_bar is None:
        return False, ""
    price = current_bar["close"]
    if result["direction"] == "long":
        if price <= result["stop_loss"]:
            return True, f"price {price:,.2f} already at/below SL {result['stop_loss']:,.2f}"
    else:
        if price >= result["stop_loss"]:
            return True, f"price {price:,.2f} already at/above SL {result['stop_loss']:,.2f}"
    return False, ""


def _run_replay_scan(label: str, replay_date, timeframe_filter: list[str]) -> list[dict]:
    """
    Re-run the scanner as-of a past date, then drop any signals whose stop-loss
    has already been breached by the current price.  Valid signals are recorded
    with source='replay' and an explanatory note, then alerted normally.

    Only 1d timeframes are replayed — intraday bars expire too fast to be worth it.
    """
    from datetime import date as _date
    from p4_live.scanner import scan_strategy, _fetch_latest_bar
    from p4_live.journal import record_signal, get_active_strategy_set
    from p4_live.alerts import alert_signal
    from core.config import get_watchlist

    log.info(f"{label} — replaying bar {replay_date}...")
    fired: list[dict] = []

    watchlist = get_watchlist()
    if timeframe_filter:
        watchlist = [(a, s, tf) for a, s, tf in watchlist if tf in timeframe_filter]

    active       = get_active_strategy_set()
    open_tickers: list[str] = []

    # Cache current bars so we only fetch each ticker once
    current_bars: dict[str, dict | None] = {}

    for asset_id, strategy_id, timeframe in watchlist:
        try:
            result = scan_strategy(
                asset_id, strategy_id, timeframe=timeframe, as_of=replay_date
            )
        except Exception as e:
            log.warning(f"  Replay {asset_id}/{strategy_id}/{timeframe} @ {replay_date}: {e}")
            continue

        if not result.get("fired"):
            continue

        ticker = result["ticker"]
        if (ticker, strategy_id) in active:
            log.info(
                f"  {result['label']:8} [{strategy_id}@{timeframe}] "
                f"replay skipped — trade already open"
            )
            continue

        if ticker not in current_bars:
            current_bars[ticker] = _fetch_latest_bar(ticker)

        stale, reason = _is_signal_stale(result, current_bars[ticker])
        if stale:
            log.info(
                f"  {result['label']:8} [{strategy_id}@{timeframe}] "
                f"REPLAY STALE (bar={replay_date}) — {reason}"
            )
            continue

        result["notes"]  = f"replayed from {replay_date} (scheduler was offline)"
        result["source"] = "replay"
        result["rationale"] = f"[Replayed {replay_date}] " + (result.get("rationale") or "")

        recorded = record_signal(result)
        if recorded:
            log.info(
                f"  {result['label']:8} [{strategy_id}@{timeframe}] REPLAY SIGNAL "
                f"{result['direction'].upper()} "
                f"entry {result['entry_low']:,.2f}–{result['entry_high']:,.2f} "
                f"(original bar={replay_date})"
            )
            alert_signal(result, open_positions=open_tickers)
            _execute_mt5(result)
            active.add((ticker, strategy_id))
            if ticker not in open_tickers:
                open_tickers.append(ticker)
            fired.append(result)
        else:
            log.info(
                f"  {result['label']:8} [{strategy_id}@{timeframe}] "
                f"replay already in journal"
            )

    log.info(f"{label} — {len(fired)} replayed signal(s) still valid")
    _run_simulate()
    return fired


# ── 5m scan: gold scalper and other 5m strategies ────────────────────────────

def run_5m_scan():
    """5m bar scan — fires every 5 min but only during the intraday window.
    Covers gold_scalper and any other 5m strategies in the watchlist."""
    if not _within_intraday_window():
        return
    if not _at_bar_boundary(bar_minutes=5, window_secs=45):
        log.debug("5m scan skipped — not at a 5m bar boundary")
        return
    _run_scan("Intraday (5m)", timeframe_filter=["5m"])
    _write_last_run("5m")


# ── Intraday scan: 15m bars during US session open ────────────────────────────

def run_intraday_scan():
    """15m bar scan — fires every 15 min but only executes inside the intraday window
    and within 90 seconds of a completed 15-minute bar close."""
    if not _within_intraday_window():
        return
    if not _at_bar_boundary():
        log.debug("15m scan skipped — not at a bar boundary")
        return
    _run_scan("Intraday (15m)", timeframe_filter=["15m"])
    _write_last_run("15m")


# ── 4h/1h scan: runs every 4 hours around the clock ──────────────────────────

def run_4h_scan():
    """4h and 1h bar scan — runs every 4 hours regardless of session."""
    _run_scan("Multi-TF (4h/1h)", timeframe_filter=["4h", "1h"])
    _write_last_run("4h")


# ── Morning scan: catch the US daily bar that closed after last night's 22:00 ──

def run_morning_scan():
    """
    09:00 local scan for 1d strategies.

    Why this exists: the daily 22:00 scan runs BEFORE the US market closes
    (US close = 23:00 Romania EEST). The bar that closed at 23:00 is therefore
    not visible until this morning scan, which fires 10 hours later.
    Without this scan, today's US daily bar would wait 23 hours (until tomorrow's
    22:00 scan) before any signals are evaluated.

    European markets (GER40) close at ~19:30 Romania — already captured by the
    22:00 scan — but they're re-evaluated here harmlessly (dedup prevents dups).
    """
    _run_scan("Morning (1d)", timeframe_filter=["1d"])
    _write_last_run("morning")


# ── Daily scan: 1d bars + maintenance ────────────────────────────────────────

def run_daily_scan():
    """
    1d bar scan at end of day.
    Also runs strategy degradation checks, auto-expires stale signals,
    generates the HTML forward-test report, and sends a daily summary.
    """
    fired_signals, _checked = _run_scan("Daily (1d)", timeframe_filter=["1d"])

    # ── Auto-expire stale pending signals (all active strategies) ────────────
    try:
        from core.config import get_watchlist, get_strategy_config
        from p4_live.journal import expire_stale_signals
        expiry_map = {}
        for _, sid, _ in get_watchlist():
            if sid not in expiry_map:
                try:
                    cfg = get_strategy_config(sid)
                    expiry_map[sid] = cfg.get("expiry_days", 5)
                except Exception:
                    pass
        expired_n = expire_stale_signals(expiry_map)
        if expired_n:
            log.info(f"Auto-expired {expired_n} stale pending signal(s)")
    except Exception as e:
        log.warning(f"Auto-expire failed: {e}")

    # ── Strategy degradation checks ───────────────────────────────────────────
    try:
        from core.config import get_all_assets
        from p4_live.alerts import alert_degradation
        from p4_live.risk import check_degradation

        for asset in get_all_assets():
            ticker = asset.get("tickers", {}).get("yfinance", "")
            for entry in asset.get("strategies", []):
                strategy_id = entry if isinstance(entry, str) else entry.get("id")
                if not strategy_id:
                    continue
                stats = check_degradation(ticker, strategy_id)
                if stats.get("degraded"):
                    log.warning(
                        f"  DEGRADATION: {asset['label']} [{strategy_id}] "
                        f"live WR {stats['live_wr']:.1%} vs baseline {stats['baseline_wr']:.1%} "
                        f"(p={stats['p_value']:.3f})"
                    )
                    alert_degradation(ticker, strategy_id, asset["label"], stats)
    except Exception as e:
        log.warning(f"Degradation checks failed: {e}")

    # ── HTML forward-test report ──────────────────────────────────────────────
    try:
        from p4_live.html_report import generate_html
        path = generate_html()
        log.info(f"HTML report: {path}")
    except Exception as e:
        log.warning(f"HTML report failed: {e}")

    # ── Daily summary alert ───────────────────────────────────────────────────
    try:
        from p4_live.alerts import alert_daily_summary
        alert_daily_summary(fired_signals, total_pairs=_checked)
    except Exception as e:
        log.warning(f"Daily summary alert failed: {e}")

    _write_last_run("daily")


# ── Open trade price check (shared by daily scan and _check_open_trades) ─────

def _check_open_trades(open_trades: list[dict]) -> None:
    """Check each open trade against latest price and fire alerts as needed."""
    from p4_live.scanner import _fetch_latest_bar
    from p4_live.alerts import alert_price

    tickers = list({t["ticker"] for t in open_trades})
    bars = {}
    for ticker in tickers:
        bar = _fetch_latest_bar(ticker)
        if bar:
            bars[ticker] = bar

    for t in open_trades:
        bar = bars.get(t["ticker"])
        if not bar:
            continue

        price = bar["close"]
        high  = bar["high"]
        low   = bar["low"]

        if t["status"] == "pending":
            zone_touched = (
                t["direction"] == "long"  and low  <= t["entry_high"] and high >= t["entry_low"] or
                t["direction"] == "short" and high >= t["entry_low"]  and low  <= t["entry_high"]
            )
            if zone_touched:
                log.info(f"  {t['label']} [{t.get('strategy','')}] entry zone touched")
                alert_price("entry", t, price,
                            f"Bar {low:,.2f}–{high:,.2f} overlaps entry zone")

        elif t["status"] == "filled":
            if t["direction"] == "long":
                if low <= t["stop_loss"]:
                    log.info(f"  {t['label']} [{t.get('strategy','')}] SL breached")
                    alert_price("sl", t, price, f"Low {low:,.2f} <= SL {t['stop_loss']:,.2f}")
                elif high >= t["tp1"]:
                    log.info(f"  {t['label']} [{t.get('strategy','')}] TP1 hit")
                    alert_price("tp", t, price, f"High {high:,.2f} >= TP1 {t['tp1']:,.2f}")
            else:
                if high >= t["stop_loss"]:
                    log.info(f"  {t['label']} [{t.get('strategy','')}] SL breached")
                    alert_price("sl", t, price, f"High {high:,.2f} >= SL {t['stop_loss']:,.2f}")
                elif low <= t["tp1"]:
                    log.info(f"  {t['label']} [{t.get('strategy','')}] TP1 hit")
                    alert_price("tp", t, price, f"Low {low:,.2f} <= TP1 {t['tp1']:,.2f}")


# ── Missed-job catch-up ───────────────────────────────────────────────────────

def _run_catchup() -> None:
    """
    Run any scheduled jobs that were missed while the scheduler was offline
    (e.g. machine sleep, reboot). Called once at startup before the loop begins.

    Catch-up thresholds:
      daily — re-runs if last run was > 26h ago (catches one missed 22:00 window)
      4h    — re-runs if last run was > 4.5h ago
      15m   — skipped; handled by the normal scheduled tick as soon as we re-enter
               the intraday window and reach a bar boundary
    """
    last = _read_last_run()
    now  = datetime.now()

    daily_last   = datetime.fromisoformat(last["daily"])   if "daily"   in last else None
    morning_last = datetime.fromisoformat(last["morning"]) if "morning" in last else None
    h4_last      = datetime.fromisoformat(last["4h"])      if "4h"      in last else None

    if daily_last is None or (now - daily_last) > timedelta(hours=26):
        log.info("Catch-up: daily scan not run in > 26h — running now")
        run_daily_scan()
    else:
        log.info(f"Catch-up: daily OK (last ran {daily_last:%Y-%m-%d %H:%M})")

    if morning_last is None or (now - morning_last) > timedelta(hours=26):
        log.info("Catch-up: morning scan not run today — running now")
        run_morning_scan()
    else:
        log.info(f"Catch-up: morning OK (last ran {morning_last:%Y-%m-%d %H:%M})")

    if h4_last is None or (now - h4_last) > timedelta(hours=4, minutes=30):
        log.info("Catch-up: 4h/1h scan not run in > 4.5h — running now")
        run_4h_scan()
    else:
        log.info(f"Catch-up: 4h/1h OK (last ran {h4_last:%Y-%m-%d %H:%M})")

    if _within_intraday_window():
        log.info("Catch-up: inside intraday window — running 5m + 15m scans once")
        _run_scan("Intraday (5m) [catch-up]", timeframe_filter=["5m"])
        _write_last_run("5m")
        _run_scan("Intraday (15m) [catch-up]", timeframe_filter=["15m"])
        _write_last_run("15m")

    # ── Replay missed 1d bar closes ───────────────────────────────────────────
    # If the scheduler was offline for N days, the daily closes that occurred
    # during that window were never evaluated.  Replay each one (oldest first),
    # drop any signal whose stop-loss has since been breached, and record the
    # rest so they can still fill normally.  Capped at 7 days to avoid flooding
    # after a long outage; 15m/5m bars are skipped (intraday entries expire fast).
    if daily_last is not None:
        days_missed = max(0, (now.date() - daily_last.date()).days - 1)
        days_missed = min(days_missed, 7)
        if days_missed > 0:
            log.info(f"Catch-up: {days_missed} missed daily close(s) — replaying...")
            for d in range(days_missed, 0, -1):   # oldest → newest
                missed_date = now.date() - timedelta(days=d)
                _run_replay_scan(f"Replay 1d [{missed_date}]", missed_date, ["1d"])
        else:
            log.info("Catch-up: no missed daily closes to replay")


# ── US market hours guard (for momentum scanner) ─────────────────────────────

def _within_us_market_hours() -> bool:
    """True during the regular US equity session (9:25–16:05 ET, Mon–Fri)."""
    try:
        from zoneinfo import ZoneInfo
        from datetime import time as _t
        now_et = datetime.now(ZoneInfo("America/New_York"))
        return (
            now_et.weekday() < 5
            and _t(9, 25) <= now_et.time() <= _t(16, 5)
        )
    except Exception:
        return False


# ── Momentum scan: Russell 2000 price/volume burst detection ──────────────────

def run_momentum_scan() -> None:
    """
    Scan the full Russell 2000 for sudden price/volume moves.
    Runs every 5 minutes but only executes during US market hours (9:30–16:00 ET).
    Fires Telegram alerts when thresholds are breached.
    """
    if not _within_us_market_hours():
        return
    try:
        from p5_insider.momentum_scanner import scan_momentum, send_momentum_alerts
        signals = scan_momentum()
        if signals:
            n = send_momentum_alerts(signals)
            log.info(f"Momentum: {len(signals)} signal(s), {n} alert(s) sent")
    except Exception as e:
        log.error(f"Momentum scan failed: {e}", exc_info=True)
    _write_last_run("momentum")


# ── Insider data refresh ──────────────────────────────────────────────────────

def run_insider_refresh() -> None:
    """
    Fetch latest congressional + Form 4 insider trades, store new records,
    and send alerts for any that match the user's portfolio (score >= 5).
    """
    try:
        from p5_insider.scrapers.congress import refresh as congress_refresh
        from p5_insider.scrapers.sec_form4 import refresh as form4_refresh
        from p5_insider.alerts import send_new_alerts

        c = congress_refresh()
        f = form4_refresh()
        n_alerts = send_new_alerts(since_days=7)
        log.info(
            f"Insider refresh done — "
            f"congress: +{c['house']}H +{c['senate']}S, "
            f"form4: +{f}, alerts: {n_alerts}"
        )
    except Exception as e:
        log.error(f"Insider refresh failed: {e}", exc_info=True)


# ── Trump Truth Social monitor (background thread) ───────────────────────────

def _trump_monitor_thread() -> None:
    """
    Runs trump_monitor.run() in a daemon thread.
    Restarts automatically if it crashes, with a 30-second back-off.
    """
    import trump_monitor
    interval   = float(os.getenv("TRUMP_POLL_INTERVAL", "2"))
    no_telegram = os.getenv("TRUMP_NO_TELEGRAM", "").lower() in ("1", "true", "yes")
    while True:
        try:
            trump_monitor.run(interval=interval, use_telegram=not no_telegram)
        except Exception as e:
            log.error(f"Trump monitor crashed: {e} — restarting in 30s", exc_info=True)
            time.sleep(30)


def start_trump_monitor() -> None:
    t = threading.Thread(target=_trump_monitor_thread, name="trump-monitor", daemon=True)
    t.start()
    log.info("Trump monitor started (background thread, daemon)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Orcastrading scheduler starting")
    log.info("=" * 60)

    # Startup Telegram alert
    try:
        from core.config import get_all_assets, get_watchlist
        from p4_live.alerts import alert_startup, is_configured

        assets     = [a["label"] for a in get_all_assets()]
        strategies = list({sid for _, sid, _ in get_watchlist()})

        if is_configured():
            alert_startup(strategies, assets)
            log.info("Startup alert sent")
        else:
            log.warning("No Telegram token configured — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env")
    except Exception as e:
        log.error(f"Startup alert failed: {e}")

    # ── MT5 connection ────────────────────────────────────────────────────────
    try:
        from p4_live import mt5_broker
        if mt5_broker.is_enabled():
            if mt5_broker.connect():
                info = mt5_broker.get_account_info()
                log.info(
                    f"MT5 demo execution active — "
                    f"account={info['login']}  balance={info['balance']:.2f} {info['currency']}"
                )
            else:
                log.error("MT5 connection failed — execution disabled for this session")
        else:
            log.info("MT5 execution disabled (set MT5_ENABLED=true in .env to enable)")
    except Exception as e:
        log.error(f"MT5 startup error: {e}")

    # ── Schedule ──────────────────────────────────────────────────────────────
    schedule.every(5).minutes.do(run_5m_scan)
    log.info(f"Scheduled: 5m scan every 5 min (active {INTRADAY_START:%H:%M}–{INTRADAY_END:%H:%M} local, covers gold_scalper)")

    schedule.every(15).minutes.do(run_intraday_scan)
    log.info(f"Scheduled: 15m intraday scan every 15 min (active {INTRADAY_START:%H:%M}–{INTRADAY_END:%H:%M} local)")

    schedule.every(4).hours.do(run_4h_scan)
    log.info("Scheduled: 4h/1h scan every 4 hours")

    schedule.every().day.at("09:00").do(run_morning_scan)
    log.info("Scheduled: morning 1d scan at 09:00 local (catches US bar closed after 22:00)")

    schedule.every().day.at("22:00").do(run_daily_scan)
    log.info("Scheduled: daily 1d scan at 22:00 local")

    schedule.every(5).minutes.do(run_momentum_scan)
    log.info("Scheduled: momentum scan every 5 min (active during US market hours 09:30–16:00 ET)")

    schedule.every(6).hours.do(run_insider_refresh)
    log.info("Scheduled: insider data refresh every 6 hours")

    schedule.every(1).minutes.do(_run_mt5_monitor)
    log.info("Scheduled: MT5 position monitor every 60 s")

    start_trump_monitor()

    log.info("Scheduler running. Press Ctrl+C to stop.")
    log.info("-" * 60)

    # ── Catch-up: run any jobs missed while the scheduler was offline ─────────
    log.info("Running catch-up checks...")
    _run_catchup()

    _hb_path = LOG_DIR / "scheduler_heartbeat"
    while True:
        try:
            schedule.run_pending()
            _hb_path.touch()          # keep mtime fresh so UI shows "Scheduler ON"
            time.sleep(30)
        except KeyboardInterrupt:
            log.info("Scheduler stopped by user.")
            break
        except Exception as e:
            log.error(f"Scheduler loop error: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
