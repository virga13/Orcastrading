"""
run_scheduler.py — Orcastrading continuous market scanner.

Runs 24/7, scanning for setups across all configured assets and strategies.
Sends Telegram/email alerts immediately when a signal fires.

Schedule:
  Every 15 min, 16:30–18:30 local (US session open) → ORB scan
  Every day at 22:00 local                           → Daily scan (MTF Trend, Momentum)
  At startup                                         → Telegram "scheduler started" alert

Usage:
  python run_scheduler.py

To run at Windows startup:
  schtasks /create /tn "Orcastrading Scheduler" ^
    /tr "\"C:\\Users\\admin\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe\" \"C:\\Users\\admin\\Desktop\\Orcastrading\\run_scheduler.py\"" ^
    /sc onstart /ru SYSTEM /f

Logs: logs/scheduler.log
"""
import sys
import io
import os
import logging
import schedule
import time
from datetime import datetime, time as dtime
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

# ── US session window for intraday scans (local time) ─────────────────────────
# Adjust these if your system clock is not GMT+2 (Romanian time).
# 16:30–18:30 local = 09:30–11:30 ET = first 2 hours of NYSE session.
INTRADAY_START = dtime(16, 30)
INTRADAY_END   = dtime(18, 30)


def _within_intraday_window() -> bool:
    now = datetime.now().time()
    return INTRADAY_START <= now <= INTRADAY_END


# ── Intraday scan (ORB + any 15m strategy) ────────────────────────────────────

def run_intraday_scan():
    if not _within_intraday_window():
        return   # outside trading window — job fires every 15 min but skips silently

    log.info("Intraday scan starting...")
    try:
        from core.config import get_watchlist, get_strategy_timeframe
        from p4_live.scanner import scan_strategy
        from p4_live.journal import record_signal, get_active_strategy_set
        from p4_live.alerts import alert_signal

        intraday_pairs = [
            (aid, sid) for aid, sid in get_watchlist()
            if get_strategy_timeframe(sid) != "1d"
        ]

        active = get_active_strategy_set()
        fired  = 0

        for asset_id, strategy_id in intraday_pairs:
            try:
                result = scan_strategy(asset_id, strategy_id)
            except Exception as e:
                log.warning(f"  {asset_id}/{strategy_id}: {e}")
                continue

            if not result.get("fired"):
                log.debug(f"  {result['label']:8} [{strategy_id}] no signal")
                continue

            ticker = result["ticker"]
            if (ticker, strategy_id) in active:
                log.info(f"  {result['label']:8} [{strategy_id}] signal skipped — trade already open")
                continue

            recorded = record_signal(result)
            if recorded:
                log.info(
                    f"  {result['label']:8} [{strategy_id}] SIGNAL FIRED "
                    f"{result['direction'].upper()} "
                    f"entry {result['entry_low']:,.2f}–{result['entry_high']:,.2f}"
                )
                alert_signal(result)
                fired += 1
            else:
                log.info(f"  {result['label']:8} [{strategy_id}] already in journal")

        log.info(f"Intraday scan done — {fired} new signal(s) across {len(intraday_pairs)} pairs")

    except Exception as e:
        log.error(f"Intraday scan failed: {e}", exc_info=True)


# ── Daily scan (P1 AI strategy scanner + open-trade check) ───────────────────

def run_daily_scan():
    log.info("Daily scan starting...")
    try:
        from core.config import get_all_assets, get_strategy_config
        from p1_analysis_engine.strategy_scanner import scan_asset_for_strategies
        from p1_analysis_engine.fetchers.calendar import get_event_today
        from p1_analysis_engine.fetchers.macro import fetch_macro
        from p4_live.journal import (record_signal, get_active_strategy_set,
                                     get_open_trades, expire_stale_signals,
                                     get_active_ticker_set)
        from p4_live.alerts import (alert_signal, alert_daily_summary,
                                    alert_circuit_breaker, alert_degradation)
        from p4_live.risk import position_sizing, check_circuit_breaker, check_degradation

        # ── Auto-expire stale pending signals ─────────────────────────────────
        expiry_map = {}
        for sid in ["mtf_trend", "momentum_breakout", "orb"]:
            cfg = get_strategy_config(sid)
            expiry_map[sid] = cfg.get("expiry_days", 5)
        expired_n = expire_stale_signals(expiry_map)
        if expired_n:
            log.info(f"Auto-expired {expired_n} stale pending signal(s)")

        # ── Open position tickers (for correlation cap) ───────────────────────
        open_tickers = list(get_active_ticker_set())

        # ── Economic calendar check ───────────────────────────────────────────
        event = get_event_today()
        if event:
            log.warning(
                f"High-impact event: {event['name']} on {event['date']} "
                f"({event['days_away']} day(s) away) — signals will be flagged"
            )

        # ── Fetch macro once — shared across all assets ───────────────────────
        macro_context: dict = {}
        vix_level: float | None = None
        try:
            macro_context = fetch_macro("equity")
            vix_level = macro_context.get("vix_level")
            vix_reg = macro_context.get("vix_regime", "")
            log.info(f"Macro: VIX={vix_level} ({vix_reg})  macro_bias={macro_context.get('macro_bias','?')}")
        except Exception as e:
            log.warning(f"Macro fetch failed: {e} — scanning without macro context")

        all_assets = get_all_assets()
        active     = get_active_strategy_set()
        fired_signals: list[dict] = []
        fired = 0

        for asset in all_assets:
            asset_id = asset["id"]
            try:
                scan = scan_asset_for_strategies(
                    asset_id, macro_context=macro_context
                )
            except Exception as e:
                log.warning(f"  {asset_id}: P1 scan failed — {e}")
                continue

            for signal in scan.signals:
                if not signal.fired:
                    log.debug(f"  {scan.label:8} [{signal.strategy_id}] no signal")
                    continue

                # ── Circuit breaker ───────────────────────────────────────────
                breached, live_dd, max_dd = check_circuit_breaker(
                    scan.ticker, signal.strategy_id
                )
                if breached:
                    log.warning(
                        f"  {scan.label} [{signal.strategy_id}] CIRCUIT BREAKER "
                        f"live_dd={live_dd}R >= max={max_dd}R — signal skipped"
                    )
                    alert_circuit_breaker(
                        scan.ticker, signal.strategy_id, scan.label, live_dd, max_dd
                    )
                    continue

                # ── Confidence gate via position sizing ───────────────────────
                risk_pts = None
                if signal.entry_high is not None and signal.stop_loss is not None:
                    risk_pts = round(signal.entry_high - signal.stop_loss, 4)

                signal_dict = {
                    "signal_date": scan.analysis_date,
                    "ticker":      scan.ticker,
                    "strategy":    signal.strategy_id,
                    "label":       scan.label,
                    "direction":   signal.direction,
                    "entry_low":   signal.entry_low,
                    "entry_high":  signal.entry_high,
                    "stop_loss":   signal.stop_loss,
                    "tp1":         signal.tp1,
                    "tp2":         signal.tp2,
                    "tp1_alloc":   70,
                    "tp2_alloc":   30,
                    "risk_pts":    risk_pts,
                    "rr":          signal.rr,
                    "confidence":  signal.confidence,
                    "rationale":   signal.rationale,
                    "price":       scan.current_price,
                    "atr":         scan.atr,
                    "rsi":         scan.rsi,
                    "regime":      scan.market_regime,
                    "fired":       True,
                }

                sizing = position_sizing(
                    signal_dict,
                    open_positions=open_tickers,
                    vix_level=vix_level,
                )
                if sizing["skip"]:
                    log.info(
                        f"  {scan.label} [{signal.strategy_id}] skipped — "
                        f"{sizing['sizing_note']}"
                    )
                    continue

                if (scan.ticker, signal.strategy_id) in active:
                    log.info(f"  {scan.label} [{signal.strategy_id}] skipped — trade already open")
                    continue

                # ── Economic calendar flag ────────────────────────────────────
                if event:
                    signal_dict["rationale"] = (
                        f"[{event['name']} in {event['days_away']}d — verify before entry] "
                        + signal_dict["rationale"]
                    )

                recorded = record_signal(signal_dict)
                if recorded:
                    log.info(
                        f"  {scan.label} [{signal.strategy_id}] SIGNAL FIRED "
                        f"{signal.direction.upper()} "
                        f"entry {signal.entry_low:,.2f}–{signal.entry_high:,.2f}  "
                        f"suggest {sizing['suggested_pct']:.1%} risk"
                    )
                    alert_signal(signal_dict)
                    fired_signals.append(signal_dict)
                    fired += 1
                    # Update local open_tickers so correlation cap catches same-scan co-firing
                    if scan.ticker not in open_tickers:
                        open_tickers.append(scan.ticker)
                else:
                    log.info(f"  {scan.label} [{signal.strategy_id}] already in journal")

        # ── Strategy degradation checks ───────────────────────────────────────
        for asset in all_assets:
            ticker = asset.get("tickers", {}).get("yfinance", "")
            for strategy_id in asset.get("strategies", []):
                stats = check_degradation(ticker, strategy_id)
                if stats.get("degraded"):
                    log.warning(
                        f"  DEGRADATION: {asset['label']} [{strategy_id}] "
                        f"live WR {stats['live_wr']:.1%} vs baseline {stats['baseline_wr']:.1%} "
                        f"(p={stats['p_value']:.3f})"
                    )
                    alert_degradation(ticker, strategy_id, asset["label"], stats)

        # ── Open trade check ──────────────────────────────────────────────────
        open_trades = get_open_trades()
        if open_trades:
            log.info(f"Open trades: {len(open_trades)}")
            _check_open_trades(open_trades)

        # ── Daily summary + HTML report ───────────────────────────────────────
        alert_daily_summary(fired_signals)

        try:
            from p4_live.html_report import generate_html
            path = generate_html()
            log.info(f"HTML report: {path}")
        except Exception as e:
            log.warning(f"HTML report failed: {e}")

        log.info(f"Daily scan done — {fired} new signal(s) across {len(all_assets)} assets")

    except Exception as e:
        log.error(f"Daily scan failed: {e}", exc_info=True)


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Orcastrading scheduler starting")
    log.info("=" * 60)

    # Startup alert
    try:
        from core.config import get_all_assets, get_watchlist, get_strategy_timeframe
        from p4_live.alerts import alert_startup, is_configured

        assets     = [a["label"] for a in get_all_assets()]
        strategies = list({sid for _, sid in get_watchlist()})

        if is_configured():
            alert_startup(strategies, assets)
            log.info("Startup alert sent")
        else:
            log.warning("No alert channels configured — set TELEGRAM_BOT_TOKEN in .env")
    except Exception as e:
        log.error(f"Startup alert failed: {e}")

    # Schedule intraday scan every 15 minutes
    schedule.every(15).minutes.do(run_intraday_scan)
    log.info("Scheduled: intraday scan every 15 min (active 16:30–18:30 local)")

    # Schedule daily scan at 22:00
    schedule.every().day.at("22:00").do(run_daily_scan)
    log.info("Scheduled: daily scan at 22:00")

    log.info("Scheduler running. Press Ctrl+C to stop.")
    log.info("-" * 60)

    # Run daily scan immediately on startup (so you get a first report right away)
    log.info("Running initial daily scan...")
    run_daily_scan()

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)   # check every 30 seconds
        except KeyboardInterrupt:
            log.info("Scheduler stopped by user.")
            break
        except Exception as e:
            log.error(f"Scheduler loop error: {e}", exc_info=True)
            time.sleep(60)   # wait a minute before retrying


if __name__ == "__main__":
    main()
