"""
core/local_client.py — SQLite-backed drop-in for supabase.Client (dev / local mode).

Used automatically when ORCA_DEV_MODE=1 and SUPABASE_URL is not set.
Implements the Supabase PostgREST fluent query API so all journal_supabase.py
calls work with zero modifications to app.py or journal_supabase.py.

Data is stored in the same SQLite file the scheduler writes to:
  p4_live/journal_data/live_trades.db

Extra tables added here (absent from the scheduler schema):
  user_settings  — singleton settings row (id=1), seeded from .env defaults
  user_watchlist — asset/strategy watchlist, seeded from assets.yaml on first run
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from p4_live.journal import _conn as _journal_conn


# ── Extra migrations for UI-only tables ──────────────────────────────────────

_UI_MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS user_settings (
        id                  INTEGER PRIMARY KEY CHECK (id = 1),
        telegram_bot_token  TEXT    DEFAULT '',
        telegram_chat_id    TEXT    DEFAULT '',
        max_open_trades     INTEGER DEFAULT 5,
        daily_loss_limit_r  REAL    DEFAULT 0.0,
        weekly_dd_limit_r   REAL    DEFAULT 0.0,
        account_balance     REAL    DEFAULT 0.0,
        risk_per_trade_pct  REAL    DEFAULT 1.0,
        updated_at          TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS user_watchlist (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id    TEXT NOT NULL,
        strategy_id TEXT NOT NULL,
        timeframe   TEXT NOT NULL,
        params      TEXT DEFAULT '{}',
        enabled     INTEGER DEFAULT 1,
        UNIQUE (asset_id, strategy_id, timeframe)
    )""",
    # Columns the Supabase manual_entries table has that the SQLite one may lack
    "ALTER TABLE manual_entries ADD COLUMN original_sl           REAL DEFAULT NULL",
    "ALTER TABLE manual_entries ADD COLUMN pnl_dollars           REAL DEFAULT NULL",
    "ALTER TABLE manual_entries ADD COLUMN chart_screenshot_path TEXT DEFAULT NULL",
]


def _get_conn() -> sqlite3.Connection:
    """Open the shared journal DB and apply UI-specific migrations."""
    conn = _journal_conn()
    for sql in _UI_MIGRATIONS:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column/table already exists — safe to ignore
    return conn


# ── Response shim ─────────────────────────────────────────────────────────────

class _Response:
    """Mimics supabase.PostgrestAPIResponse (.data attribute)."""
    __slots__ = ("data",)

    def __init__(self, data: list[dict]):
        self.data = data or []


# ── Fluent query builder ──────────────────────────────────────────────────────

class _QueryBuilder:
    """
    Mirrors the Supabase PostgREST fluent API and executes against SQLite.

    Supported chain:
      .select() .order() .limit()
      .eq() .neq() .lte() .in_() .not_.in_()
      .update() .insert() .upsert() .delete()
      .execute()  →  _Response
    """

    def __init__(self, table: str):
        self._table   = table
        self._op      = "select"
        self._payload: Any = None
        self._filters: list[tuple] = []   # (col, op, val, negate)
        self._order   : str | None = None
        self._desc    : bool       = False
        self._limit   : int | None = None
        self._negate  : bool       = False

    # ── Selector modifiers ────────────────────────────────────────────────────

    def select(self, cols: str = "*"):
        self._op = "select"
        # We always SELECT * from SQLite; column filtering isn't needed
        return self

    def order(self, col: str, desc: bool = False):
        self._order = col
        self._desc  = desc
        return self

    def limit(self, n: int):
        self._limit = n
        return self

    # ── Filters ──────────────────────────────────────────────────────────────

    def eq(self, col: str, val):
        self._filters.append((col, "=", val, self._negate))
        self._negate = False
        return self

    def neq(self, col: str, val):
        self._filters.append((col, "!=", val, False))
        return self

    def lte(self, col: str, val):
        self._filters.append((col, "<=", val, False))
        return self

    def in_(self, col: str, vals: list):
        self._filters.append((col, "IN", vals, self._negate))
        self._negate = False
        return self

    @property
    def not_(self):
        """Enable NOT modifier for the next filter: .not_.in_(...)"""
        self._negate = True
        return self

    # ── Mutations ─────────────────────────────────────────────────────────────

    def update(self, payload: dict):
        self._op      = "update"
        self._payload = payload
        return self

    def insert(self, payload):
        self._op      = "insert"
        self._payload = payload   # dict or list[dict]
        return self

    def upsert(self, payload: dict, on_conflict: str | None = None):
        self._op      = "upsert"
        self._payload = payload
        return self

    def delete(self):
        self._op = "delete"
        return self

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self) -> _Response:
        conn = _get_conn()
        try:
            data = self._run(conn)
        finally:
            conn.close()
        return _Response(data)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _where(self) -> tuple[str, list]:
        if not self._filters:
            return "", []
        parts: list[str] = []
        params: list     = []
        for col, op, val, negate in self._filters:
            if op == "=":
                parts.append(f"{col} {'!=' if negate else '='} ?")
                params.append(val)
            elif op == "!=":
                parts.append(f"{col} != ?")
                params.append(val)
            elif op == "<=":
                parts.append(f"{col} <= ?")
                params.append(val)
            elif op == "IN":
                ph = ",".join("?" * len(val))
                parts.append(f"{col} {'NOT ' if negate else ''}IN ({ph})")
                params.extend(val)
        return "WHERE " + " AND ".join(parts), params

    def _run(self, conn: sqlite3.Connection) -> list[dict]:
        table        = self._table
        where, params = self._where()

        if self._op == "select":
            sql = f"SELECT * FROM {table} {where}".strip()
            if self._order:
                sql += f" ORDER BY {self._order} {'DESC' if self._desc else 'ASC'}"
            if self._limit:
                sql += f" LIMIT {int(self._limit)}"
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_dict(r, table) for r in rows]

        if self._op == "insert":
            if isinstance(self._payload, list):
                result: list[dict] = []
                for item in self._payload:
                    result.extend(_do_insert(conn, table, item))
                return result
            return _do_insert(conn, table, self._payload)

        if self._op == "update":
            p          = dict(self._payload)
            set_clause = ", ".join(f"{k} = ?" for k in p)
            conn.execute(
                f"UPDATE {table} SET {set_clause} {where}".strip(),
                list(p.values()) + params,
            )
            conn.commit()
            return []

        if self._op == "delete":
            conn.execute(f"DELETE FROM {table} {where}".strip(), params)
            conn.commit()
            return []

        if self._op == "upsert":
            return _do_upsert(conn, table, self._payload)

        return []


# ── Shared SQL helpers ────────────────────────────────────────────────────────

def _prep(table: str, raw: dict) -> dict:
    """Coerce payload for SQLite: booleans→int, dicts→JSON, inject required fields."""
    out: dict = {}
    for k, v in raw.items():
        if isinstance(v, bool):
            out[k] = int(v)
        elif k == "params" and table == "user_watchlist" and isinstance(v, dict):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    # Auto-inject recorded_at for tables that declare it NOT NULL
    if table in ("trades", "manual_entries", "psychology") and "recorded_at" not in out:
        out["recorded_at"] = datetime.now(timezone.utc).isoformat()
    # user_settings is a singleton — always enforce id=1
    if table == "user_settings":
        out["id"] = 1
    return out


def _do_insert(conn: sqlite3.Connection, table: str, payload: dict) -> list[dict]:
    p    = _prep(table, payload)
    cols = ", ".join(p.keys())
    ph   = ", ".join("?" * len(p))
    cur  = conn.execute(
        f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({ph})",
        list(p.values()),
    )
    conn.commit()
    if cur.lastrowid:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE rowid = ?", (cur.lastrowid,)
        ).fetchone()
        if row:
            return [_row_to_dict(row, table)]
    return [{"id": cur.lastrowid}]


def _do_upsert(conn: sqlite3.Connection, table: str, payload: dict) -> list[dict]:
    p    = _prep(table, payload)
    cols = ", ".join(p.keys())
    ph   = ", ".join("?" * len(p))
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({ph})",
        list(p.values()),
    )
    conn.commit()
    return []


def _row_to_dict(row: sqlite3.Row, table: str) -> dict:
    d = dict(row)
    if table == "user_watchlist" and "params" in d:
        try:
            d["params"] = json.loads(d["params"]) if d["params"] else {}
        except (json.JSONDecodeError, TypeError):
            d["params"] = {}
    return d


# ── Public LocalClient ────────────────────────────────────────────────────────

class LocalClient:
    """
    Drop-in replacement for supabase.Client used in dev / local mode.

    Injected into st.session_state["supabase_client"] by core.auth when
    ORCA_DEV_MODE=1 and SUPABASE_URL is not set.  All journal_supabase.py
    functions receive this object as `client` and work unchanged.
    """

    class _AuthStub:
        """Stub for client.auth.sign_out() — no-op locally."""
        def sign_out(self):
            pass

    auth = _AuthStub()

    def __init__(self):
        self._seed_watchlist()

    def table(self, name: str) -> _QueryBuilder:
        return _QueryBuilder(name)

    def _seed_watchlist(self) -> None:
        """Populate user_watchlist from assets.yaml on first run (when empty)."""
        conn = _get_conn()
        try:
            count = conn.execute("SELECT COUNT(*) FROM user_watchlist").fetchone()[0]
            if count > 0:
                return
            from core.config import get_watchlist
            entries = [
                {
                    "asset_id":    a,
                    "strategy_id": s,
                    "timeframe":   tf,
                    "params":      "{}",
                    "enabled":     1,
                }
                for a, s, tf in get_watchlist()
            ]
            if entries:
                conn.executemany(
                    "INSERT OR IGNORE INTO user_watchlist "
                    "(asset_id, strategy_id, timeframe, params, enabled) "
                    "VALUES (:asset_id, :strategy_id, :timeframe, :params, :enabled)",
                    entries,
                )
                conn.commit()
        finally:
            conn.close()
