"""
p4_live/mt5_monitor.py — MT5 position monitor.

Polls the MT5 terminal for order/position state changes and writes the
results back into the journal.  Only processes trades that have an mt5_ticket
set — trades without one continue to be handled by simulate.py.

Called from run_scheduler.py every 60 seconds.

Two-phase logic (mirrors simulate.py's phases):
  Phase 1 — pending orders:
    Still in open orders?       → leave as pending
    Filled (in history)?        → record_fill_direct()
    Cancelled/expired?          → record_expired_direct()

  Phase 2 — open positions:
    Still in positions_get()?   → leave as filled
    Closed (closing deal found) → record_close_direct(), compute pnl_r from legs
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

log = logging.getLogger("orcastrading.mt5_monitor")


# ── Public entry point ────────────────────────────────────────────────────────

def sync_mt5_positions(dry_run: bool = False) -> list[dict]:
    """
    Sync MT5 order/position states into the journal.

    dry_run=True: compute outcomes but do not write to journal.
    Returns list of update dicts (one per action taken).
    """
    from p4_live import mt5_broker as broker        # noqa: PLC0415
    from p4_live.journal import (                   # noqa: PLC0415
        get_mt5_open_trades,
        record_fill_direct,
        record_expired_direct,
        record_close_direct,
    )

    if not broker.is_enabled():
        return []

    updates: list[dict] = []
    trades  = get_mt5_open_trades()

    if not trades:
        return updates

    # ── Phase 1: pending orders ───────────────────────────────────────────────
    for t in (t for t in trades if t["status"] == "pending"):
        tickets = _parse_tickets(t.get("mt5_ticket") or "")
        if not tickets:
            continue

        sig_date  = t["signal_date"]
        ticker    = t["ticker"]
        strategy  = t["strategy"]
        timeframe = t.get("timeframe", "1d")

        filled_info = None   # first filled leg we find
        all_done    = True   # becomes False if any ticket is still undecided

        for ticket in tickets:
            # Check open orders first (still pending)
            if broker.get_order_state(ticket) is not None:
                all_done = False
                continue

            # Look up in history
            hist = broker.get_historical_order(ticket)
            if hist is None:
                # Not in history yet — could be a very recent order
                all_done = False
                continue

            if hist["state"] == "filled" and filled_info is None:
                filled_info = hist
            # "cancelled" / "expired" → this ticket is done, continue checking others

        if filled_info:
            fill_price = filled_info.get("price") or float(t["entry_high"])
            fill_date  = _today_str()
            if not dry_run:
                record_fill_direct(
                    sig_date, ticker, strategy, timeframe,
                    fill_price, fill_date,
                )
            log.info(
                f"MT5 filled: {t['label']} [{strategy}@{timeframe}] "
                f"@ {fill_price:.4f}"
            )
            updates.append({
                "ticker":      ticker,
                "strategy":    strategy,
                "timeframe":   timeframe,
                "signal_date": sig_date,
                "status":      "filled",
                "fill_price":  fill_price,
                "fill_date":   fill_date,
            })

        elif all_done:
            # Every ticket is cancelled or expired — signal never entered
            if not dry_run:
                record_expired_direct(sig_date, ticker, strategy, timeframe)
            log.info(
                f"MT5 expired: {t['label']} [{strategy}@{timeframe}] "
                f"(all {len(tickets)} order(s) cancelled/expired)"
            )
            updates.append({
                "ticker":      ticker,
                "strategy":    strategy,
                "timeframe":   timeframe,
                "signal_date": sig_date,
                "status":      "expired",
            })

    # ── Phase 2: filled positions ─────────────────────────────────────────────
    # Re-read after Phase 1 so trades just promoted to 'filled' are included.
    if not dry_run:
        trades = get_mt5_open_trades()
    filled_trades = [t for t in trades if t["status"] == "filled"]

    for t in filled_trades:
        tickets = _parse_tickets(t.get("mt5_ticket") or "")
        if not tickets:
            continue

        sig_date   = t["signal_date"]
        ticker     = t["ticker"]
        strategy   = t["strategy"]
        timeframe  = t.get("timeframe", "1d")
        direction  = t["direction"]
        fill_price = float(t.get("fill_price") or t["entry_high"])
        sl         = float(t["stop_loss"])
        risk       = abs(fill_price - sl)
        tp1_alloc  = (t.get("tp1_alloc") or 70) / 100
        tp2_alloc  = (t.get("tp2_alloc") or 30) / 100
        alloc_per_leg = [tp1_alloc, tp2_alloc] if len(tickets) > 1 else [1.0]
        sign = 1.0 if direction == "long" else -1.0

        if risk < 1e-8:
            continue

        # ── Gather close info keyed by leg index (preserves alloc mapping) ────
        # close_by_idx[i] = closing deal dict for tickets[i]
        # still_open      = set of indices not yet closed
        close_by_idx: dict[int, dict] = {}
        still_open:   set[int]        = set()

        for idx, ticket in enumerate(tickets):
            if broker.get_position_info(ticket) is not None:
                still_open.add(idx)
                continue

            hist = broker.get_historical_order(ticket)
            if hist is None or hist["state"] != "filled":
                still_open.add(idx)   # pending cancel or not yet in history
                continue

            position_id = hist.get("position_id")
            if position_id is None:
                still_open.add(idx)
                continue

            closed = broker.get_closed_position(position_id)
            if closed:
                close_by_idx[idx] = closed
            else:
                still_open.add(idx)   # position open, not yet closed

        if not close_by_idx:
            continue   # no leg has closed yet — nothing to do

        # ── All legs closed → finalise the trade ─────────────────────────────
        if not still_open:
            total_pnl_r = round(sum(
                sign * (close_by_idx[i]["close_price"] - fill_price) / risk * alloc_per_leg[i]
                for i in close_by_idx
            ), 3)

            if total_pnl_r >= 0.9:
                outcome = "win"
            elif total_pnl_r > 0.05:
                outcome = "partial_win"
            elif total_pnl_r >= -0.05:
                outcome = "breakeven"
            else:
                outcome = "loss"

            latest     = max(close_by_idx.values(), key=lambda x: x["close_time"])
            exit_price = latest["close_price"]
            exit_date  = latest["close_time"][:10]

            if not dry_run:
                record_close_direct(
                    signal_date = sig_date,
                    ticker      = ticker,
                    strategy    = strategy,
                    timeframe   = timeframe,
                    outcome     = outcome,
                    exit_price  = exit_price,
                    exit_date   = exit_date,
                    pnl_r       = total_pnl_r,
                )

            log.info(
                f"MT5 closed: {t['label']} [{strategy}@{timeframe}]  "
                f"outcome={outcome}  pnl_r={total_pnl_r:+.2f}R  "
                f"exit={exit_price:.4f}"
            )
            updates.append({
                "ticker":      ticker,
                "strategy":    strategy,
                "timeframe":   timeframe,
                "signal_date": sig_date,
                "status":      outcome,
                "exit_price":  exit_price,
                "exit_date":   exit_date,
                "pnl_r":       total_pnl_r,
            })

        # ── Some legs closed, runner(s) still open → record partial P&L ──────
        else:
            partial_pnl_r = round(sum(
                sign * (close_by_idx[i]["close_price"] - fill_price) / risk * alloc_per_leg[i]
                for i in close_by_idx
            ), 3)

            prev_partial = t.get("partial_close_pnl_r")
            if prev_partial is None or abs(prev_partial - partial_pnl_r) > 1e-4:
                if not dry_run:
                    from p4_live.journal import update_partial_close_pnl  # noqa: PLC0415
                    update_partial_close_pnl(
                        sig_date, ticker, strategy, timeframe, partial_pnl_r
                    )
                closed_legs = sorted(close_by_idx.keys())
                log.info(
                    f"MT5 partial close: {t['label']} [{strategy}@{timeframe}]  "
                    f"leg(s) {closed_legs} closed  pnl_so_far={partial_pnl_r:+.2f}R  "
                    f"{len(still_open)} runner(s) still open"
                )
                updates.append({
                    "ticker":            ticker,
                    "strategy":          strategy,
                    "timeframe":         timeframe,
                    "signal_date":       sig_date,
                    "status":            "partial_close",
                    "partial_close_pnl_r": partial_pnl_r,
                })

    return updates


# ── Cancel MT5 orders for journal-expired trades ──────────────────────────────

def cancel_pending_for_expired(lookback_days: int = 14) -> list[dict]:
    """
    Cancel any live MT5 pending orders whose journal trade was marked 'expired'.

    The monitor's sync_mt5_positions() only processes status IN ('pending','filled'),
    so when expire_stale_signals() flips a trade to 'expired', the MT5 limit orders
    become invisible to the monitor and can fill silently, creating ghost positions.

    This function plugs that gap: it queries recently expired trades with MT5 tickets,
    checks whether each ticket is still a pending order in MT5, and cancels it.

    dry_run is not needed here — cancelling a non-existent order is a no-op in MT5.

    Returns list of action dicts (one per cancel attempted).
    """
    from p4_live import mt5_broker as broker
    from p4_live.journal import get_expired_with_tickets

    if not broker.is_enabled():
        return []

    updates: list[dict] = []
    expired_trades = get_expired_with_tickets(lookback_days=lookback_days)

    for t in expired_trades:
        tickets = _parse_tickets(t.get("mt5_ticket") or "")
        for ticket in tickets:
            state = broker.get_order_state(ticket)
            if state is None:
                continue   # already cancelled/filled/expired by MT5 itself
            ok = broker.cancel_order(ticket)
            log.info(
                f"Cancel expired MT5 order: ticket={ticket}  "
                f"{t['ticker']} {t['direction']} [{t['strategy']}]  "
                f"signal={t['signal_date']}  {'OK' if ok else 'FAILED'}"
            )
            updates.append({
                "ticker":      t["ticker"],
                "strategy":    t["strategy"],
                "timeframe":   t.get("timeframe", "1d"),
                "signal_date": t["signal_date"],
                "ticket":      ticket,
                "cancelled":   ok,
            })

    return updates


# ── Helpers ───────────────────────────────────────────────────────────────────

def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _parse_tickets(raw: str) -> list[int]:
    """
    Parse the mt5_ticket journal field (JSON array or single int string)
    into a list of integer ticket numbers.
    """
    if not raw:
        return []
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
        return [int(parsed)]
    except (json.JSONDecodeError, ValueError):
        try:
            return [int(raw)]
        except ValueError:
            return []
