"""
p4_live/journal.py — Append-only trade journal backed by SQLite.

All signals are written on scan. Outcomes are updated on fill/close.
No signal can be deleted or modified retroactively — integrity is paramount.

Primary key: (signal_date, ticker) — supports multiple assets firing same day.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

JOURNAL_DIR = Path(__file__).parent / "journal_data"
JOURNAL_DB  = JOURNAL_DIR / "live_trades.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS manual_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_date      TEXT NOT NULL,
    asset           TEXT,
    direction       TEXT,
    entry_price     REAL,
    stop_loss       REAL,
    take_profit     REAL,
    exit_price      REAL,
    pnl_r           REAL,
    quality_score   INTEGER,
    notes           TEXT,
    ai_analysis     TEXT,
    screenshot_path TEXT,
    tags            TEXT,
    recorded_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    signal_date     TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    strategy        TEXT NOT NULL DEFAULT 'mtf_trend',
    timeframe       TEXT NOT NULL DEFAULT '1d',
    label           TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_low       REAL NOT NULL,
    entry_high      REAL NOT NULL,
    stop_loss       REAL NOT NULL,
    tp1             REAL NOT NULL,
    tp2             REAL,
    tp1_alloc       INTEGER,
    tp2_alloc       INTEGER,
    risk_pts        REAL,
    rr              REAL,
    confidence      REAL,
    rationale       TEXT,
    price_at_signal REAL,
    atr_at_signal   REAL,
    rsi_at_signal   REAL,
    regime_note     TEXT,
    status          TEXT DEFAULT 'pending',
    fill_price      REAL,
    fill_date       TEXT,
    exit_price      REAL,
    exit_date       TEXT,
    pnl_r           REAL,
    notes           TEXT,
    recorded_at     TEXT NOT NULL,
    PRIMARY KEY (signal_date, ticker, strategy, timeframe)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date TEXT NOT NULL,
    ticker      TEXT NOT NULL DEFAULT '',
    action      TEXT NOT NULL,
    detail      TEXT,
    ts          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS psychology (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_ref_type    TEXT NOT NULL,
    trade_ref_id      TEXT NOT NULL,
    pre_state         TEXT,
    pre_confidence    INTEGER,
    pre_notes         TEXT,
    post_state        TEXT,
    post_notes        TEXT,
    execution_quality INTEGER,
    mistakes          TEXT,
    lesson            TEXT,
    recorded_at       TEXT NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    JOURNAL_DIR.mkdir(exist_ok=True)
    c = sqlite3.connect(str(JOURNAL_DB))
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    # Migrations — safe to run on every connect (no-op if column already exists)
    for migration in (
        "ALTER TABLE trades ADD COLUMN source TEXT DEFAULT 'scanner'",
        "ALTER TABLE trades ADD COLUMN signal_bar_ts TEXT DEFAULT NULL",
        "ALTER TABLE trades ADD COLUMN fill_bar_ts TEXT DEFAULT NULL",
        "ALTER TABLE trades ADD COLUMN chart_screenshot_path TEXT DEFAULT NULL",
        "ALTER TABLE trades ADD COLUMN mt5_ticket TEXT DEFAULT NULL",
        "ALTER TABLE trades ADD COLUMN partial_close_pnl_r REAL DEFAULT NULL",
    ):
        try:
            c.execute(migration)
            c.commit()
        except sqlite3.OperationalError:
            pass
    return c


def _log(conn, signal_date: str, ticker: str, action: str, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO audit_log (signal_date, ticker, action, detail, ts) VALUES (?,?,?,?,?)",
        (signal_date, ticker, action, detail, datetime.now(timezone.utc).isoformat()),
    )


def _resolve(
    conn, signal_date: str, ticker: str | None, strategy: str | None = None
) -> sqlite3.Row:
    """
    Look up a trade by (signal_date, ticker, strategy).
    If ticker/strategy are None and exactly one trade exists for that date, auto-selects it.
    Raises ValueError otherwise.
    """
    if ticker and strategy:
        row = conn.execute(
            "SELECT * FROM trades WHERE signal_date=? AND ticker=? AND strategy=?",
            (signal_date, ticker, strategy),
        ).fetchone()
        if not row:
            raise ValueError(f"No trade found for {signal_date} / {ticker} / {strategy}")
        return row

    if ticker:
        rows = conn.execute(
            "SELECT * FROM trades WHERE signal_date=? AND ticker=?",
            (signal_date, ticker),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades WHERE signal_date=?", (signal_date,)
        ).fetchall()

    if len(rows) == 0:
        raise ValueError(f"No trade found for {signal_date}")
    if len(rows) > 1:
        ids = [f"{r['ticker']}/{r['strategy']}" for r in rows]
        raise ValueError(
            f"Multiple trades on {signal_date} ({', '.join(ids)}). "
            f"Specify --ticker and --strategy."
        )
    return rows[0]


# ── Write operations ──────────────────────────────────────────────────────────

def record_signal(signal: dict) -> bool:
    """
    Persist a fired signal. Returns False if already recorded or blocked by conflict resolution.

    Conflict resolution (same ticker, opposite direction):
      - Filled opposing trade exists → block new pending (cannot auto-close a live position).
      - Pending opposing trade exists → higher confidence wins; lower-confidence pending is
        auto-expired and the new signal is recorded. Tie goes to the incumbent (no change).
    """
    conn = _conn()
    tf      = signal.get("timeframe", "1d")
    ticker  = signal["ticker"]
    new_dir = signal["direction"]
    new_conf = signal.get("confidence") or 0.0

    # ── Duplicate guard ───────────────────────────────────────────────────────
    existing = conn.execute(
        "SELECT 1 FROM trades WHERE signal_date=? AND ticker=? AND strategy=? AND timeframe=?",
        (signal["signal_date"], ticker, signal.get("strategy", "mtf_trend"), tf),
    ).fetchone()
    if existing:
        conn.close()
        return False

    # ── Conflict resolution — opposing direction already open on this ticker ──
    opposing_dir = "short" if new_dir == "long" else "long"
    conflicts = conn.execute(
        "SELECT signal_date, ticker, strategy, timeframe, status, confidence "
        "FROM trades "
        "WHERE ticker=? AND direction=? AND status IN ('pending','filled')",
        (ticker, opposing_dir),
    ).fetchall()

    for c in conflicts:
        if c["status"] == "filled":
            # A live position is already running in the opposite direction.
            # Cannot auto-close — block the new signal.
            _log(conn, signal["signal_date"], ticker, "signal_blocked",
                 f"filled {opposing_dir} trade exists "
                 f"({c['strategy']} {c['signal_date']}) — new {new_dir} blocked")
            conn.commit()
            conn.close()
            return False

        # Pending vs pending — confidence decides
        incumbent_conf = c["confidence"] or 0.0
        if new_conf > incumbent_conf:
            # New signal is better — expire the incumbent
            conn.execute(
                "UPDATE trades SET status='expired' "
                "WHERE signal_date=? AND ticker=? AND strategy=? AND timeframe=? AND status='pending'",
                (c["signal_date"], c["ticker"], c["strategy"], c["timeframe"]),
            )
            _log(conn, c["signal_date"], c["ticker"], "conflict_expired",
                 f"replaced by {new_dir} {signal.get('strategy')} "
                 f"(conf {incumbent_conf:.2f} < {new_conf:.2f})")
        else:
            # Incumbent has equal or higher confidence — block the new signal
            _log(conn, signal["signal_date"], ticker, "signal_blocked",
                 f"pending {opposing_dir} trade ({c['strategy']}) conf={incumbent_conf:.2f} "
                 f">= new conf={new_conf:.2f} — new {new_dir} blocked")
            conn.commit()
            conn.close()
            return False

    conn.execute("""
        INSERT INTO trades (
            signal_date, ticker, strategy, timeframe, label, direction,
            entry_low, entry_high, stop_loss, tp1, tp2,
            tp1_alloc, tp2_alloc, risk_pts, rr, confidence, rationale,
            price_at_signal, atr_at_signal, rsi_at_signal, regime_note,
            signal_bar_ts, notes, source, status, recorded_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?)
    """, (
        signal["signal_date"], ticker, signal.get("strategy", "mtf_trend"), tf,
        signal["label"], new_dir,
        signal["entry_low"], signal["entry_high"], signal["stop_loss"],
        signal["tp1"], signal.get("tp2"),
        signal.get("tp1_alloc", 70), signal.get("tp2_alloc", 30),
        signal.get("risk_pts"), signal.get("rr"), new_conf,
        signal.get("rationale"), signal["price"], signal["atr"], signal["rsi"],
        signal.get("regime"),
        signal.get("signal_bar_ts"),
        signal.get("notes"),
        signal.get("source", "scanner"),
        datetime.now(timezone.utc).isoformat(),
    ))
    _log(conn, signal["signal_date"], ticker, "signal_recorded",
         f"entry={signal['entry_low']}-{signal['entry_high']} sl={signal['stop_loss']}")
    conn.commit()
    conn.close()
    return True


def record_fill(
    signal_date: str,
    fill_price: float,
    fill_date: str | None = None,
    ticker: str | None = None,
    strategy: str | None = None,
) -> None:
    fd = fill_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = _conn()
    row = _resolve(conn, signal_date, ticker, strategy)
    conn.execute(
        "UPDATE trades SET status='filled', fill_price=?, fill_date=? "
        "WHERE signal_date=? AND ticker=? AND strategy=?",
        (fill_price, fd, row["signal_date"], row["ticker"], row["strategy"]),
    )
    _log(conn, row["signal_date"], row["ticker"], "filled",
         f"fill_price={fill_price} fill_date={fd}")
    conn.commit()
    conn.close()


def record_close(
    signal_date: str,
    outcome: str,
    exit_price: float,
    exit_date: str | None = None,
    ticker: str | None = None,
    strategy: str | None = None,
) -> None:
    """
    outcome: win | partial_win | loss | breakeven
    pnl_r computed from fill_price and stop_loss stored in the trade.
    """
    ed = exit_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = _conn()
    row = _resolve(conn, signal_date, ticker, strategy)

    if row["fill_price"] is None:
        conn.close()
        raise ValueError(f"Trade {signal_date}/{row['ticker']} has no fill — mark it filled first")

    fill  = row["fill_price"]
    sl    = row["stop_loss"]
    risk  = abs(fill - sl)
    tp1   = row["tp1"]
    tp2   = row["tp2"]
    alloc1 = (row["tp1_alloc"] or 70) / 100
    alloc2 = (row["tp2_alloc"] or 30) / 100
    direction = row["direction"]

    if outcome == "win":
        pnl1 = (tp1 - fill) * alloc1 / risk if direction == "long" else (fill - tp1) * alloc1 / risk
        pnl2 = ((tp2 - fill) * alloc2 / risk if tp2 else 0.0) if direction == "long" \
               else ((fill - tp2) * alloc2 / risk if tp2 else 0.0)
        pnl_r = round(pnl1 + pnl2, 3)
    elif outcome == "partial_win":
        pnl1  = (tp1 - fill) * alloc1 / risk if direction == "long" else (fill - tp1) * alloc1 / risk
        pnl2  = (exit_price - fill) * alloc2 / risk if direction == "long" \
                else (fill - exit_price) * alloc2 / risk
        pnl_r = round(pnl1 + pnl2, 3)
    elif outcome == "loss":
        pnl_r = round(-1.0, 3)
    elif outcome == "breakeven":
        pnl_r = 0.0
    else:
        sign  = 1 if direction == "long" else -1
        pnl_r = round(sign * (exit_price - fill) / risk, 3)

    conn.execute("""
        UPDATE trades SET status=?, exit_price=?, exit_date=?, pnl_r=?
        WHERE signal_date=? AND ticker=? AND strategy=?
    """, (outcome, exit_price, ed, pnl_r, row["signal_date"], row["ticker"], row["strategy"]))
    _log(conn, row["signal_date"], row["ticker"], "closed",
         f"outcome={outcome} exit={exit_price} pnl_r={pnl_r}")
    conn.commit()
    conn.close()


def record_fill_close_direct(
    signal_date: str,
    ticker: str,
    fill_price: float,
    fill_date: str,
    outcome: str,
    exit_price: float,
    exit_date: str,
    pnl_r: float,
    strategy: str = "mtf_trend",
    timeframe: str = "1d",
) -> None:
    """
    Atomically record fill + close with a pre-computed pnl_r.
    Used by the backfill script where the simulator already computed pnl_r.
    strategy + timeframe must be included to avoid corrupting a different strategy's record
    for the same ticker on the same date.
    """
    conn = _conn()
    conn.execute("""
        UPDATE trades
        SET status=?, fill_price=?, fill_date=?, exit_price=?, exit_date=?, pnl_r=?
        WHERE signal_date=? AND ticker=? AND strategy=? AND timeframe=?
    """, (outcome, fill_price, fill_date, exit_price, exit_date, pnl_r,
          signal_date, ticker, strategy, timeframe))
    _log(conn, signal_date, ticker, "backfill_closed",
         f"strategy={strategy} tf={timeframe} fill={fill_price} exit={exit_price} pnl_r={pnl_r} outcome={outcome}")
    conn.commit()
    conn.close()


def record_expired(signal_date: str, ticker: str | None = None, strategy: str | None = None) -> None:
    conn = _conn()
    row = _resolve(conn, signal_date, ticker, strategy)
    conn.execute(
        "UPDATE trades SET status='expired' "
        "WHERE signal_date=? AND ticker=? AND strategy=? AND status='pending'",
        (row["signal_date"], row["ticker"], row["strategy"]),
    )
    _log(conn, row["signal_date"], row["ticker"], "expired", "")
    conn.commit()
    conn.close()


def record_expired_direct(signal_date: str, ticker: str, strategy: str, timeframe: str = "1d") -> None:
    """Expire a specific (signal_date, ticker, strategy, timeframe) record. No ambiguity."""
    conn = _conn()
    conn.execute(
        "UPDATE trades SET status='expired' "
        "WHERE signal_date=? AND ticker=? AND strategy=? AND timeframe=? AND status='pending'",
        (signal_date, ticker, strategy, timeframe),
    )
    _log(conn, signal_date, ticker, "expired", f"tf={timeframe}")
    conn.commit()
    conn.close()


def record_fill_direct(
    signal_date: str,
    ticker: str,
    strategy: str,
    timeframe: str,
    fill_price: float,
    fill_date: str,
    fill_bar_ts: str | None = None,
) -> None:
    """Mark a pending trade as filled. Does not close it — outcome resolved later."""
    conn = _conn()
    conn.execute(
        "UPDATE trades SET status='filled', fill_price=?, fill_date=?, fill_bar_ts=? "
        "WHERE signal_date=? AND ticker=? AND strategy=? AND timeframe=? AND status='pending'",
        (fill_price, fill_date, fill_bar_ts, signal_date, ticker, strategy, timeframe),
    )
    _log(conn, signal_date, ticker, "filled",
         f"tf={timeframe} fill={fill_price} date={fill_date}")
    conn.commit()
    conn.close()


def record_close_direct(
    signal_date: str,
    ticker: str,
    strategy: str,
    timeframe: str,
    outcome: str,
    exit_price: float,
    exit_date: str,
    pnl_r: float,
) -> None:
    """Close a filled trade with its final outcome and pre-computed pnl_r."""
    conn = _conn()
    conn.execute(
        "UPDATE trades SET status=?, exit_price=?, exit_date=?, pnl_r=? "
        "WHERE signal_date=? AND ticker=? AND strategy=? AND timeframe=? AND status='filled'",
        (outcome, exit_price, exit_date, pnl_r,
         signal_date, ticker, strategy, timeframe),
    )
    _log(conn, signal_date, ticker, "closed",
         f"tf={timeframe} outcome={outcome} exit={exit_price} pnl_r={pnl_r}")
    conn.commit()
    conn.close()


def add_note(signal_date: str, note: str, ticker: str | None = None, strategy: str | None = None) -> None:
    conn = _conn()
    row = _resolve(conn, signal_date, ticker, strategy)
    conn.execute(
        "UPDATE trades SET notes=? WHERE signal_date=? AND ticker=? AND strategy=?",
        (note, row["signal_date"], row["ticker"], row["strategy"]),
    )
    conn.commit()
    conn.close()


def save_trade_chart_screenshot(
    signal_date: str, ticker: str, strategy: str, timeframe: str, path: str
) -> None:
    """Attach a chart screenshot path to a live trade."""
    conn = _conn()
    conn.execute(
        "UPDATE trades SET chart_screenshot_path=? "
        "WHERE signal_date=? AND ticker=? AND strategy=? AND timeframe=?",
        (path, signal_date, ticker, strategy, timeframe),
    )
    conn.commit()
    conn.close()


# ── MT5 execution tracking ───────────────────────────────────────────────────

def record_mt5_ticket(
    signal_date: str,
    ticker: str,
    strategy: str,
    timeframe: str,
    tickets: list[int],
) -> None:
    """Store MT5 order tickets (JSON array) for a trade after execution."""
    import json as _json
    conn = _conn()
    conn.execute(
        "UPDATE trades SET mt5_ticket=? "
        "WHERE signal_date=? AND ticker=? AND strategy=? AND timeframe=?",
        (_json.dumps(tickets), signal_date, ticker, strategy, timeframe),
    )
    _log(conn, signal_date, ticker, "mt5_orders_placed", f"tickets={tickets}")
    conn.commit()
    conn.close()


def update_partial_close_pnl(
    signal_date: str,
    ticker: str,
    strategy: str,
    timeframe: str,
    partial_pnl_r: float,
) -> None:
    """
    Record the P&L of legs that have already closed while at least one runner is still open.
    Does NOT change the trade status — the trade stays 'filled' until all legs close.
    Idempotent: safe to call on every monitor cycle.
    """
    conn = _conn()
    conn.execute(
        "UPDATE trades SET partial_close_pnl_r=? "
        "WHERE signal_date=? AND ticker=? AND strategy=? AND timeframe=? AND status='filled'",
        (round(partial_pnl_r, 3), signal_date, ticker, strategy, timeframe),
    )
    _log(conn, signal_date, ticker, "mt5_partial_close",
         f"pnl_so_far={partial_pnl_r:+.3f}R strategy={strategy} tf={timeframe}")
    conn.commit()
    conn.close()


def get_mt5_open_trades() -> list[dict]:
    """Return pending/filled trades that have an MT5 ticket (managed by MT5 terminal)."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM trades "
        "WHERE status IN ('pending','filled') AND mt5_ticket IS NOT NULL "
        "ORDER BY signal_date, ticker"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_expired_with_tickets(lookback_days: int = 14) -> list[dict]:
    """
    Return recently expired trades that still have MT5 tickets.
    Used by the monitor to cancel any MT5 limit orders that were left open
    after the journal expired the signal (the monitor only processes pending/filled,
    so expired rows with live MT5 orders would otherwise be silently ignored).
    """
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM trades "
        "WHERE status='expired' AND mt5_ticket IS NOT NULL AND signal_date >= ? "
        "ORDER BY signal_date, ticker",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pending_without_ticket() -> list[dict]:
    """Return pending trades with no MT5 ticket — order placement failed (e.g. market closed)."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM trades "
        "WHERE status='pending' AND mt5_ticket IS NULL "
        "ORDER BY signal_date, ticker"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Expiry ───────────────────────────────────────────────────────────────────

def expire_stale_signals(expiry_map: dict[str, int]) -> int:
    """
    Mark pending signals as 'expired' if they are older than the strategy's expiry window.

    expiry_map: {strategy_id: days_before_expiry}
      e.g. {"mtf_trend": 5, "momentum_breakout": 3, "orb": 1}

    Returns the number of signals expired.
    A signal is expired when: today - signal_date >= expiry_days (pending only).
    """
    from datetime import date, timedelta

    today = date.today()
    conn = _conn()
    expired_count = 0

    for strategy_id, days in expiry_map.items():
        cutoff = (today - timedelta(days=days)).isoformat()
        rows = conn.execute(
            "SELECT signal_date, ticker, strategy FROM trades "
            "WHERE strategy=? AND status='pending' AND signal_date <= ?",
            (strategy_id, cutoff),
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE trades SET status='expired' "
                "WHERE signal_date=? AND ticker=? AND strategy=? AND status='pending'",
                (row["signal_date"], row["ticker"], row["strategy"]),
            )
            _log(conn, row["signal_date"], row["ticker"], "auto_expired",
                 f"strategy={strategy_id} expiry_days={days}")
            expired_count += 1

    conn.commit()
    conn.close()
    return expired_count


# ── Read operations ───────────────────────────────────────────────────────────

def get_all_trades() -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM trades ORDER BY signal_date, ticker"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_open_trades() -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE status IN ('pending','filled') "
        "ORDER BY signal_date, ticker"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_active_ticker_set() -> set[str]:
    """Return tickers that have at least one pending or filled trade (any strategy)."""
    conn = _conn()
    rows = conn.execute(
        "SELECT DISTINCT ticker FROM trades WHERE status IN ('pending', 'filled')"
    ).fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def get_active_strategy_set() -> set[tuple[str, str]]:
    """
    Return (ticker, strategy) pairs with a pending or filled trade.
    Used to enforce one-trade-at-a-time per asset/strategy pair.
    """
    conn = _conn()
    rows = conn.execute(
        "SELECT DISTINCT ticker, strategy FROM trades WHERE status IN ('pending', 'filled')"
    ).fetchall()
    conn.close()
    return {(r["ticker"], r["strategy"]) for r in rows}


def get_closed_trades() -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE status NOT IN ('pending','filled','expired') "
        "ORDER BY signal_date, ticker"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Manual journal entries (screenshot-based) ─────────────────────────────────

def save_manual_entry(data: dict) -> int:
    """Save a manually journaled trade (with optional AI analysis). Returns row id."""
    from datetime import date as _date
    conn = _conn()
    cursor = conn.execute("""
        INSERT INTO manual_entries (
            entry_date, asset, direction, entry_price, stop_loss, original_sl,
            take_profit, exit_price, pnl_r, pnl_dollars, quality_score, notes,
            ai_analysis, screenshot_path, chart_screenshot_path, tags, recorded_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("entry_date", _date.today().isoformat()),
        data.get("asset"), data.get("direction"),
        data.get("entry_price"), data.get("stop_loss"), data.get("original_sl"),
        data.get("take_profit"),
        data.get("exit_price"), data.get("pnl_r"), data.get("pnl_dollars"),
        data.get("quality_score"),
        data.get("notes"), data.get("ai_analysis"),
        data.get("screenshot_path"), data.get("chart_screenshot_path"),
        data.get("tags"),
        datetime.now(timezone.utc).isoformat(),
    ))
    entry_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return entry_id


def update_manual_entry(entry_id: int, **fields) -> None:
    """Update specific fields on an existing manual entry (e.g. exit_price, pnl_r)."""
    if not fields:
        return
    allowed = {"exit_price", "pnl_r", "pnl_dollars", "stop_loss", "original_sl",
               "take_profit", "notes", "tags", "quality_score",
               "chart_screenshot_path", "screenshot_path", "direction"}
    valid = {k: v for k, v in fields.items() if k in allowed}
    if not valid:
        return
    set_clause = ", ".join(f"{k} = ?" for k in valid)
    conn = _conn()
    conn.execute(f"UPDATE manual_entries SET {set_clause} WHERE id = ?",
                 (*valid.values(), entry_id))
    conn.commit()
    conn.close()


def delete_manual_entry(entry_id: int) -> None:
    """Permanently delete a manual journal entry by id."""
    conn = _conn()
    conn.execute("DELETE FROM manual_entries WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()


def get_manual_entries(limit: int = 200) -> list[dict]:
    """Return all manual journal entries, newest first."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM manual_entries ORDER BY entry_date DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Manual signals (Trade Planner → Journal) ──────────────────────────────────

def save_manual_signal(signal: dict) -> bool:
    """
    Save a UI-originated trade signal (from Trade Planner) into the trades table
    with source='manual'. Returns False if the PK already exists.
    """
    from datetime import date as _date
    conn = _conn()
    sig_date = signal.get("signal_date", _date.today().isoformat())
    ticker   = signal.get("ticker", "MANUAL")
    strategy = signal.get("strategy", "manual")

    existing = conn.execute(
        "SELECT 1 FROM trades WHERE signal_date=? AND ticker=? AND strategy=?",
        (sig_date, ticker, strategy),
    ).fetchone()
    if existing:
        conn.close()
        return False

    risk_pts = abs((signal.get("entry_high") or 0) - (signal.get("stop_loss") or 0)) or None
    rr = None
    if risk_pts and signal.get("tp1"):
        rr = round(abs((signal["tp1"] - (signal.get("entry_high") or 0)) / risk_pts), 2)

    conn.execute("""
        INSERT INTO trades (
            signal_date, ticker, strategy, label, direction,
            entry_low, entry_high, stop_loss, tp1, tp2,
            tp1_alloc, tp2_alloc, risk_pts, rr, confidence, rationale,
            price_at_signal, atr_at_signal, rsi_at_signal, regime_note,
            status, source, recorded_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'manual','pending','manual',?)
    """, (
        sig_date, ticker, strategy,
        signal.get("label", signal.get("ticker", "MANUAL")),
        signal.get("direction", "long"),
        signal.get("entry_low", 0), signal.get("entry_high", 0),
        signal.get("stop_loss", 0), signal.get("tp1", 0),
        signal.get("tp2"),
        signal.get("tp1_alloc", 70), signal.get("tp2_alloc", 30),
        risk_pts, rr,
        signal.get("confidence"), signal.get("rationale", ""),
        signal.get("entry_high", 0), 0.0, 0.0,
        datetime.now(timezone.utc).isoformat(),
    ))
    _log(conn, sig_date, ticker, "manual_signal_logged",
         f"entry={signal.get('entry_low')}-{signal.get('entry_high')} sl={signal.get('stop_loss')}")
    conn.commit()
    conn.close()
    return True


# ── Psychology journal ────────────────────────────────────────────────────────

def save_psychology(data: dict) -> int:
    """Save a psychology entry linked to a trade or manual entry."""
    import json
    conn = _conn()
    cursor = conn.execute("""
        INSERT INTO psychology (
            trade_ref_type, trade_ref_id,
            pre_state, pre_confidence, pre_notes,
            post_state, post_notes, execution_quality,
            mistakes, lesson, recorded_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("trade_ref_type", "live"),
        data.get("trade_ref_id", ""),
        data.get("pre_state"), data.get("pre_confidence"),
        data.get("pre_notes"), data.get("post_state"),
        data.get("post_notes"), data.get("execution_quality"),
        json.dumps(data.get("mistakes", [])),
        data.get("lesson"),
        datetime.now(timezone.utc).isoformat(),
    ))
    entry_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return entry_id


def get_psychology(trade_ref_id: str) -> dict | None:
    """Return psychology entry for a given trade ref, or None."""
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM psychology WHERE trade_ref_id=? ORDER BY id DESC LIMIT 1",
        (trade_ref_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_psychology() -> list[dict]:
    """Return all psychology entries, newest first."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM psychology ORDER BY recorded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
