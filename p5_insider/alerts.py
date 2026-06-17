"""
p5_insider/alerts.py — Format and send insider alerts as a single digest message.

All qualifying trades are batched into one Telegram message grouped by score tier,
instead of firing one message per trade.
"""
from __future__ import annotations
import json
import logging
import os
import urllib.request
from datetime import date as _date

from p5_insider.db import was_alerted, mark_alerted
from p5_insider.correlator import get_scored_trades

log = logging.getLogger("orcastrading.insider.alerts")

ALERT_THRESHOLD = 5.0

_TIERS = [
    (10, "Critical"),
    (9,  "High"),
    (8,  "Elevated"),
    (7,  "Medium"),
    (6,  "Lower priority"),
    (5,  "Watch"),
]

_TX_LABEL: dict[str, str] = {
    "P": "BUY",
    "S": "SALE",
    "A": "AWARD",
    "F": "TAX",
    "M": "EXERCISE",
    "G": "GIFT",
    "purchase":     "BUY",
    "sale":         "SALE",
    "sale_partial": "SALE",
    "exchange":     "SWAP",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_date(iso: str | None) -> str:
    """'2026-04-21' → 'Apr 21'"""
    if not iso:
        return "?"
    try:
        d = _date.fromisoformat(str(iso)[:10])
        return f"{d.strftime('%b')} {d.day}"
    except Exception:
        return str(iso)[:7]


def _trade_name_title(trade: dict) -> tuple[str, str]:
    if trade["type"] == "form4":
        return (
            trade.get("insider_name", "Unknown"),
            trade.get("insider_title", "Insider"),
        )
    pol   = trade.get("politician", "Unknown")
    party = trade.get("party", "?")
    state = trade.get("state", "?")
    return pol, f"{party} · {state}"


def _trade_action(trade: dict) -> str:
    code = (
        trade.get("transaction_code", "?")
        if trade["type"] == "form4"
        else trade.get("transaction_type", "?")
    )
    return _TX_LABEL.get(code, code.upper())


def _trade_shares(trade: dict) -> str:
    if trade["type"] == "form4":
        s = trade.get("shares")
        return f"{s:,.0f}" if s else "—"
    return "—"


def _trade_value(trade: dict) -> str:
    if trade["type"] == "form4":
        v = trade.get("total_value")
        return f"${v:,.0f}" if v else "—"
    return trade.get("amount_str", "—")


def _fmt_entry(trade: dict) -> str:
    """Two-line entry for one trade."""
    ticker     = trade.get("ticker", "?")
    name, title = _trade_name_title(trade)
    action     = _trade_action(trade)
    shares     = _trade_shares(trade)
    value      = _trade_value(trade)
    txn_date   = _fmt_date(
        trade.get("transaction_date") or trade.get("txn_date") or trade.get("disclosure_date")
    )
    score_s    = f"{trade['score']:.0f}/10"

    line1 = f"<b>{ticker}</b>   {name}"
    line2 = f"   {title}   {action}   {shares}   {value}   {txn_date}   {score_s}"
    return f"{line1}\n{line2}"


# ── Digest builder ────────────────────────────────────────────────────────────

def _build_digest(trades: list[dict]) -> str:
    today_str  = _date.today().strftime("%B %-d, %Y") if os.name != "nt" else _date.today().strftime("%B %d, %Y").replace(" 0", " ")
    directions = {t.get("p_direction", "long").upper() for t in trades}
    dir_str    = (
        f"all positions: {next(iter(directions))}"
        if len(directions) == 1
        else "positions: " + " / ".join(sorted(directions))
    )

    lines = [
        "<b>ORCA — insider alerts</b>",
        f"{today_str} · {dir_str}",
        "",
        "<b>Ticker   Insider   Action   Shares   Value   Date   Score</b>",
    ]

    by_score: dict[int, list[dict]] = {}
    for t in trades:
        by_score.setdefault(int(t["score"]), []).append(t)

    for min_score, tier_name in _TIERS:
        group = by_score.get(min_score, [])
        if not group:
            continue
        lines.append("")
        lines.append(f"<b>{tier_name} — score {min_score}/10</b>")
        for t in group:
            lines.append(_fmt_entry(t))

    return "\n".join(lines)


# ── Telegram send (HTML) ──────────────────────────────────────────────────────

_MAX_CHARS = 4000   # Telegram hard limit is 4096; leave headroom


def _send_telegram_html(text: str) -> bool:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = (
        os.getenv("TELEGRAM_INSIDER_CHAT_ID", "").strip()
        or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    )
    if not token or not chat_id:
        return False

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n\n<i>… (truncated — too many trades to show)</i>"

    try:
        payload = json.dumps({
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data    = payload,
            headers = {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                log.warning(f"Telegram insider digest failed: {result.get('description')}")
            return bool(result.get("ok"))
    except Exception as e:
        log.warning(f"Telegram insider digest error: {e}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def send_new_alerts(since_days: int = 7) -> int:
    """
    Collect all unalerted scored trades, send as one digest message, and mark
    each trade as alerted.  Returns the number of trades included in the digest.
    """
    trades = get_scored_trades(since_days=since_days)

    pending = [
        t for t in trades
        if t["score"] >= ALERT_THRESHOLD and not was_alerted(t["type"], t["id"])
    ]

    if not pending:
        return 0

    pending.sort(key=lambda t: (-t["score"], t.get("ticker", "")))

    text = _build_digest(pending)
    ok   = _send_telegram_html(text)

    sent = 0
    if ok:
        for t in pending:
            if mark_alerted(t["type"], t["id"], t.get("ticker", "")):
                sent += 1
        tickers = len({t.get("ticker") for t in pending})
        log.info(f"Insider digest sent — {sent} trade(s) across {tickers} ticker(s)")
    else:
        log.warning("Insider digest failed to send — trades not marked alerted, will retry next run")

    return sent
