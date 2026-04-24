"""
p4_live/alerts.py — Multi-channel alert system for live trade events.

Supported channels (configured via environment variables):
  Telegram  — TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  Email     — ALERT_EMAIL_FROM + ALERT_EMAIL_TO + ALERT_EMAIL_PASSWORD
               + ALERT_SMTP_HOST (default smtp.gmail.com) + ALERT_SMTP_PORT (default 587)

All channels are optional and independent — if one fails the others still fire.
Alerts are non-blocking: failures are printed but never raise.

Event types and when they fire:
  signal    — scanner found a new entry setup
  entry     — pending trade's entry zone was touched (check command)
  sl        — open trade's stop loss was breached (check command)
  tp        — open trade hit TP1 or TP2 (check command)
  test      — manual test from `python -m p4_live alert-test`
"""
from __future__ import annotations
import os
import textwrap
from datetime import datetime, timezone


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_signal(
    result: dict,
    open_positions: list[str] | None = None,
    vix_level: float | None = None,
) -> tuple[str, str]:
    """Return (subject, body) for a signal-fired alert."""
    from p4_live.risk import position_sizing
    r        = result
    strategy = r.get("strategy", "mtf_trend").upper().replace("_", " ")
    subject  = f"[ORCA] {strategy} — {r['label']} {r['direction'].upper()} {r['signal_date']}"

    # Position sizing block — pass open positions and VIX so correlation/VIX adjustments apply
    sizing   = position_sizing(r, open_positions=open_positions, vix_level=vix_level)
    pt_risk  = f"{sizing['point_risk']:,.2f} pts" if sizing["point_risk"] else "N/A"
    sz_lines = [f"Point risk : {pt_risk}  (entry_high - stop_loss)"]
    sz_lines.append(f"Kelly (25%): {sizing['kelly_pct']:.1%}  |  Suggested: {sizing['suggested_pct']:.1%}")
    sz_lines.append(f"Note: {sizing['sizing_note'].split('|')[-1].strip()}")
    sz_lines.append("")
    sz_lines.append("Account  Risk $    Position size")
    sz_lines.append("-" * 36)
    for ex in sizing["sizing_examples"]:
        sz_lines.append(
            f"${ex['account']:>6,}  ${ex['dollar_risk']:>6.2f}   "
            f"{ex['position']:.4f} units"
        )
    sz_lines.append("")
    sz_lines.append("R = your chosen risk per trade.")
    sz_lines.append("+2.5R means you made 2.5x what you risked.")
    sz_lines.append("-1R means you were stopped out (lost 1x risk).")
    sizing_block = "\n".join(sz_lines)

    pr    = sizing["point_risk"]
    tp2   = r.get("tp2")
    tp1_r = f"+{(r['tp1'] - r['entry_high']) / pr:.1f}R" if pr else "N/A"
    tp2_r = f"+{(tp2 - r['entry_high']) / pr:.1f}R" if (pr and tp2 is not None) else "N/A"
    tp2_s = f"{tp2:,.2f}" if tp2 is not None else "N/A"
    conf  = f"{r['confidence']:.0%}" if r.get("confidence") is not None else "N/A"

    body = textwrap.dedent(f"""
        {strategy} setup on {r['label']} ({r['signal_date']})

        Direction : {r['direction'].upper()}
        Entry zone: {r['entry_low']:,.2f} - {r['entry_high']:,.2f}
        Stop loss : {r['stop_loss']:,.2f}
        TP1 ({r['tp1_alloc']}%) : {r['tp1']:,.2f}  ({tp1_r})
        TP2 ({r['tp2_alloc']}%) : {tp2_s}  ({tp2_r})
        R/R ratio : {r['rr']:.2f}  |  Confidence: {conf}

        {r.get('rationale', '')}

        --- POSITION SIZING ---
        {sizing_block}
    """).strip()
    return subject, body


def alert_circuit_breaker(ticker: str, strategy: str, label: str,
                          live_dd: float, max_dd: float) -> None:
    """Alert when live drawdown breaches the backtest maximum."""
    subject = f"[ORCA] CIRCUIT BREAKER — {label} {strategy.upper().replace('_',' ')} paused"
    body = textwrap.dedent(f"""
        CIRCUIT BREAKER TRIGGERED — new signals paused.

        Strategy : {strategy}
        Asset    : {label} ({ticker})

        Live drawdown : {live_dd:.1f}R
        Max allowed   : {max_dd:.1f}R  (backtest validated maximum)

        The live drawdown has reached or exceeded the historical maximum.
        Review live performance before resuming this strategy.

        To resume: investigate why the strategy is underperforming.
        Check if market regime has changed or strategy conditions need recalibration.
    """).strip()
    send(subject, body)


def alert_degradation(ticker: str, strategy: str, label: str, stats: dict) -> None:
    """Alert when live win rate has statistically degraded below backtest."""
    subject = f"[ORCA] DEGRADATION WARNING — {label} {strategy.upper().replace('_',' ')}"
    body = textwrap.dedent(f"""
        STRATEGY DEGRADATION DETECTED

        Strategy   : {strategy}
        Asset      : {label} ({ticker})

        Live win rate  : {stats['live_wr']:.1%}  ({stats['n']} trades)
        Backtest WR    : {stats['baseline_wr']:.1%}
        Z-score        : {stats['z_score']:.2f}  (p = {stats['p_value']:.3f})

        The live win rate is statistically significantly below the backtest
        baseline (p < 5%). This is an early warning — the strategy may be
        losing its edge or market conditions have shifted.

        Recommended: reduce position size by 50% and monitor for 10 more trades.
    """).strip()
    send(subject, body)


def _fmt_price_alert(event: str, trade: dict, price: float, detail: str) -> tuple[str, str]:
    """Return (subject, body) for a price-level alert."""
    label      = trade.get("label") or trade.get("ticker", "")
    sig_date   = trade["signal_date"]
    icons      = {"entry": "ENTRY ZONE", "sl": "STOP LOSS HIT", "tp": "TARGET HIT"}
    tag        = icons.get(event, event.upper())
    subject    = f"[ORCA] {tag} - {label} {sig_date}"
    fill_price = trade.get("fill_price")
    fill_s     = f"{fill_price:,.2f}" if fill_price else "N/A"
    body = textwrap.dedent(f"""
        {tag} on {label} (signal {sig_date})

        Current price : {price:,.2f}
        {detail}

        Entry zone : {trade['entry_low']:,.2f} - {trade['entry_high']:,.2f}
        Stop loss  : {trade['stop_loss']:,.2f}
        TP1        : {trade['tp1']:,.2f}
        Fill price : {fill_s}
    """).strip()
    return subject, body


def _fmt_test() -> tuple[str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = f"[ORCA] Alert test — {ts}"
    body    = f"Alert system is working correctly.\nSent at {ts}"
    return subject, body


# ── Telegram ──────────────────────────────────────────────────────────────────

def _send_telegram(subject: str, body: str) -> bool:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False

    try:
        import urllib.request, urllib.parse, json
        text = f"*{subject}*\n\n{body}"
        payload = json.dumps({
            "chat_id": chat_id,
            "text":    text,
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return bool(result.get("ok"))
    except Exception as e:
        print(f"  [alert] Telegram failed: {e}")
        return False


# ── Email ─────────────────────────────────────────────────────────────────────

def _send_email(subject: str, body: str) -> bool:
    from_addr = os.getenv("ALERT_EMAIL_FROM", "").strip()
    to_addr   = os.getenv("ALERT_EMAIL_TO", "").strip()
    password  = os.getenv("ALERT_EMAIL_PASSWORD", "").strip()
    if not from_addr or not to_addr or not password:
        return False

    smtp_host = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))

    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = from_addr
        msg["To"]      = to_addr
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(from_addr, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        return True
    except Exception as e:
        print(f"  [alert] Email failed: {e}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def send(subject: str, body: str) -> dict[str, bool]:
    """
    Fire an alert on all configured channels.
    Returns dict of {channel: success}.
    Never raises.
    """
    results = {}
    results["telegram"] = _send_telegram(subject, body)
    results["email"]    = _send_email(subject, body)
    return results


def alert_signal(
    result: dict,
    open_positions: list[str] | None = None,
    vix_level: float | None = None,
) -> None:
    """Send a signal-fired alert. Pass open_positions and vix_level for accurate sizing."""
    subject, body = _fmt_signal(result, open_positions=open_positions, vix_level=vix_level)
    sent = send(subject, body)
    _log_sent(sent, result["label"])


def alert_price(event: str, trade: dict, price: float, detail: str) -> None:
    """Send a price-level alert (entry/sl/tp)."""
    subject, body = _fmt_price_alert(event, trade, price, detail)
    sent = send(subject, body)
    _log_sent(sent, trade.get("label", trade.get("ticker", "")))


def alert_test() -> dict[str, bool]:
    """Send a test alert on all channels and return results."""
    subject, body = _fmt_test()
    return send(subject, body)


def alert_startup(strategies: list[str], assets: list[str]) -> None:
    """Send a startup notification when the scheduler begins."""
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = f"[ORCA] Scheduler started — {ts}"
    body    = textwrap.dedent(f"""
        Orcastrading scheduler is running.

        Monitoring {len(assets)} assets: {', '.join(assets)}
        Active strategies: {', '.join(strategies)}

        Intraday scan (ORB): every 15 min during market hours
        Morning scan (1d): 09:00 — catches US bar closed after 22:00 scan
        Daily scan (MTF Trend, Momentum): 22:00 — maintenance + report

        Started at {ts}
    """).strip()
    send(subject, body)


def alert_daily_summary(results: list[dict], total_pairs: int | None = None) -> None:
    """Send end-of-day summary even when no signals fire."""
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fired   = [r for r in results if r.get("fired")]
    subject = f"[ORCA] Daily scan — {len(fired)} signal(s) — {ts}"
    if fired:
        lines = []
        for r in fired:
            strategy = r.get("strategy", "").upper().replace("_", " ")
            lines.append(
                f"  {r['label']:8} {strategy:20} {r['direction'].upper()} "
                f"entry {r['entry_low']:,.2f}–{r['entry_high']:,.2f}"
            )
        signals_text = "\n".join(lines)
    else:
        signals_text = "  No signals across all assets and strategies."

    pairs_str = str(total_pairs) if total_pairs is not None else "?"
    body = textwrap.dedent(f"""
        Daily scan complete — {ts}

        Signals fired ({len(fired)}):
{signals_text}

        Scanned {pairs_str} asset/strategy pairs.
    """).strip()
    send(subject, body)


def _log_sent(results: dict[str, bool], label: str) -> None:
    fired  = [ch for ch, ok in results.items() if ok]
    silent = [ch for ch, ok in results.items() if not ok]
    if fired:
        print(f"  [alert] Sent via {', '.join(fired)} for {label}")
    if silent and not fired:
        print(f"  [alert] No channels configured (set TELEGRAM_BOT_TOKEN or ALERT_EMAIL_FROM)")


def is_configured() -> bool:
    """Return True if at least one alert channel is configured."""
    has_telegram = bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))
    has_email    = bool(os.getenv("ALERT_EMAIL_FROM") and os.getenv("ALERT_EMAIL_TO")
                        and os.getenv("ALERT_EMAIL_PASSWORD"))
    return has_telegram or has_email


def configured_channels() -> list[str]:
    channels = []
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        channels.append("Telegram")
    if (os.getenv("ALERT_EMAIL_FROM") and os.getenv("ALERT_EMAIL_TO")
            and os.getenv("ALERT_EMAIL_PASSWORD")):
        channels.append("Email")
    return channels
