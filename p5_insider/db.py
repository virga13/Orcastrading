"""
p5_insider/db.py — SQLite storage for insider trade data and user portfolio.

Tables:
  congress_trades — House + Senate STOCK Act disclosures
  form4_trades    — SEC Form 4 insider transactions (for portfolio tickers)
  portfolio       — User's holdings for correlation (separate from trading signals)
  alert_log       — Tracks which trades already triggered an alert
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_DIR  = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "insider.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS congress_trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source            TEXT NOT NULL,
    disclosure_date   TEXT NOT NULL,
    transaction_date  TEXT,
    politician        TEXT NOT NULL,
    party             TEXT,
    state             TEXT,
    ticker            TEXT,
    asset_description TEXT,
    transaction_type  TEXT,
    amount_str        TEXT,
    amount_low        REAL,
    amount_high       REAL,
    comment           TEXT,
    fetched_at        TEXT NOT NULL,
    UNIQUE(source, disclosure_date, politician, ticker, transaction_date, transaction_type)
);

CREATE TABLE IF NOT EXISTS form4_trades (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_date        TEXT NOT NULL,
    transaction_date   TEXT,
    company            TEXT,
    ticker             TEXT,
    cik                TEXT,
    insider_name       TEXT,
    insider_title      TEXT,
    transaction_code   TEXT,
    shares             REAL,
    price_per_share    REAL,
    total_value        REAL,
    shares_owned_after REAL,
    accession_number   TEXT UNIQUE,
    fetched_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker    TEXT UNIQUE NOT NULL,
    label     TEXT,
    direction TEXT NOT NULL DEFAULT 'long',
    notes     TEXT,
    added_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alert_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_type TEXT NOT NULL,
    trade_id   INTEGER NOT NULL,
    ticker     TEXT,
    sent_at    TEXT NOT NULL,
    UNIQUE(trade_type, trade_id)
);
"""


def _conn() -> sqlite3.Connection:
    DB_DIR.mkdir(exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    return c


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Congress trades ───────────────────────────────────────────────────────────

def insert_congress_trade(trade: dict) -> bool:
    """Insert one congressional trade. Returns True if new, False if duplicate."""
    conn = _conn()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO congress_trades
            (source, disclosure_date, transaction_date, politician, party, state,
             ticker, asset_description, transaction_type, amount_str,
             amount_low, amount_high, comment, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trade["source"], trade["disclosure_date"], trade.get("transaction_date"),
            trade["politician"], trade.get("party"), trade.get("state"),
            trade.get("ticker"), trade.get("asset_description"), trade.get("transaction_type"),
            trade.get("amount_str"), trade.get("amount_low"), trade.get("amount_high"),
            trade.get("comment"), _now(),
        ))
        new = conn.execute("SELECT changes()").fetchone()[0] > 0
        conn.commit()
        return new
    finally:
        conn.close()


def get_congress_trades(
    since_date: str | None = None,
    ticker: str | None = None,
    limit: int = 500,
) -> list[dict]:
    conn = _conn()
    q = "SELECT * FROM congress_trades WHERE 1=1"
    params: list = []
    if since_date:
        q += " AND disclosure_date >= ?"
        params.append(since_date)
    if ticker:
        q += " AND ticker = ?"
        params.append(ticker.upper())
    q += " ORDER BY disclosure_date DESC, id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Form 4 trades ─────────────────────────────────────────────────────────────

def insert_form4_trade(trade: dict) -> bool:
    conn = _conn()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO form4_trades
            (filing_date, transaction_date, company, ticker, cik,
             insider_name, insider_title, transaction_code,
             shares, price_per_share, total_value, shares_owned_after,
             accession_number, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trade["filing_date"], trade.get("transaction_date"),
            trade.get("company"), trade.get("ticker"), trade.get("cik"),
            trade.get("insider_name"), trade.get("insider_title"),
            trade.get("transaction_code"),
            trade.get("shares"), trade.get("price_per_share"),
            trade.get("total_value"), trade.get("shares_owned_after"),
            trade.get("accession_number"), _now(),
        ))
        new = conn.execute("SELECT changes()").fetchone()[0] > 0
        conn.commit()
        return new
    finally:
        conn.close()


def get_form4_trades(
    since_date: str | None = None,
    ticker: str | None = None,
    limit: int = 500,
) -> list[dict]:
    conn = _conn()
    q = "SELECT * FROM form4_trades WHERE 1=1"
    params: list = []
    if since_date:
        q += " AND filing_date >= ?"
        params.append(since_date)
    if ticker:
        q += " AND ticker = ?"
        params.append(ticker.upper())
    q += " ORDER BY filing_date DESC, id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_known_form4_accessions() -> set[str]:
    conn = _conn()
    rows = conn.execute("SELECT accession_number FROM form4_trades").fetchall()
    conn.close()
    return {r[0] for r in rows if r[0]}


# ── Portfolio ─────────────────────────────────────────────────────────────────

def get_portfolio() -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM portfolio ORDER BY ticker"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_portfolio_item(ticker: str, label: str = "", direction: str = "long", notes: str = "") -> bool:
    conn = _conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO portfolio (ticker, label, direction, notes, added_at)
            VALUES (?,?,?,?,?)
        """, (ticker.upper(), label or ticker.upper(), direction, notes, _now()))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def remove_portfolio_item(ticker: str) -> bool:
    conn = _conn()
    conn.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker.upper(),))
    changed = conn.execute("SELECT changes()").fetchone()[0] > 0
    conn.commit()
    conn.close()
    return changed


def get_portfolio_tickers() -> set[str]:
    return {r["ticker"] for r in get_portfolio()}


# ── Alert log ─────────────────────────────────────────────────────────────────

def mark_alerted(trade_type: str, trade_id: int, ticker: str) -> bool:
    """Return True if newly marked (not previously alerted)."""
    conn = _conn()
    conn.execute(
        "INSERT OR IGNORE INTO alert_log (trade_type, trade_id, ticker, sent_at) VALUES (?,?,?,?)",
        (trade_type, trade_id, ticker, _now()),
    )
    new = conn.execute("SELECT changes()").fetchone()[0] > 0
    conn.commit()
    conn.close()
    return new


def was_alerted(trade_type: str, trade_id: int) -> bool:
    conn = _conn()
    row = conn.execute(
        "SELECT 1 FROM alert_log WHERE trade_type=? AND trade_id=?",
        (trade_type, trade_id),
    ).fetchone()
    conn.close()
    return row is not None


def get_alert_history(limit: int = 100) -> list[dict]:
    """Return recent alert log entries joined with trade details."""
    conn = _conn()
    rows = conn.execute("""
        SELECT
            al.trade_type,
            al.trade_id,
            al.ticker,
            al.sent_at,
            CASE al.trade_type
                WHEN 'congress' THEN ct.politician
                ELSE f4.insider_name
            END AS who,
            CASE al.trade_type
                WHEN 'congress' THEN ct.transaction_type
                ELSE f4.transaction_code
            END AS action,
            CASE al.trade_type
                WHEN 'congress' THEN ct.amount_str
                ELSE CAST(ROUND(f4.total_value) AS TEXT)
            END AS amount,
            CASE al.trade_type
                WHEN 'congress' THEN ct.disclosure_date
                ELSE f4.filing_date
            END AS trade_date
        FROM alert_log al
        LEFT JOIN congress_trades ct ON al.trade_type='congress' AND al.trade_id=ct.id
        LEFT JOIN form4_trades    f4 ON al.trade_type='form4'    AND al.trade_id=f4.id
        ORDER BY al.sent_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
