"""
tests/test_journal.py — sanity checks for the live journal database.
"""
import pytest
from p4_live.journal import (
    JOURNAL_DB, _conn,
    get_open_trades, get_all_trades, get_closed_trades,
    get_active_strategy_set, get_active_ticker_set,
)


class TestJournalDB:
    def test_db_file_exists(self):
        assert JOURNAL_DB.exists(), f"Journal DB not found at {JOURNAL_DB}"

    def test_required_tables_exist(self):
        c = _conn()
        tables = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        c.close()
        for t in ("trades", "manual_entries", "psychology", "audit_log"):
            assert t in tables, f"Table '{t}' missing from journal DB"

    def test_trades_schema(self):
        c = _conn()
        cols = {r[1] for r in c.execute("PRAGMA table_info(trades)").fetchall()}
        c.close()
        required = {
            "ticker", "strategy", "direction", "status",
            "signal_date", "entry_low", "entry_high",
            "stop_loss", "tp1", "tp2", "pnl_r", "mt5_ticket",
        }
        missing = required - cols
        assert not missing, f"trades table missing columns: {missing}"

    def test_no_ghost_trades(self):
        """No trade should be open/filled without an MT5 ticket."""
        c = _conn()
        ghosts = c.execute(
            "SELECT rowid, ticker, strategy, status, signal_date "
            "FROM trades "
            "WHERE status IN ('pending','filled') "
            "AND (mt5_ticket IS NULL OR mt5_ticket = '')"
        ).fetchall()
        c.close()
        assert len(ghosts) == 0, (
            f"{len(ghosts)} ghost trade(s) found (open/filled, no MT5 ticket): "
            + str([(g[1], g[2], g[3], g[4]) for g in ghosts])
        )

    def test_valid_status_values(self):
        valid = {"pending", "filled", "win", "partial_win", "loss", "breakeven", "expired"}
        c = _conn()
        rows = c.execute("SELECT DISTINCT status FROM trades").fetchall()
        c.close()
        for (status,) in rows:
            assert status in valid, f"Unexpected status value: {status!r}"

    def test_direction_values(self):
        c = _conn()
        rows = c.execute("SELECT DISTINCT direction FROM trades").fetchall()
        c.close()
        for (d,) in rows:
            assert d in ("long", "short"), f"Unexpected direction: {d!r}"

    def test_closed_trades_have_pnl(self):
        closed = get_closed_trades()
        no_pnl = [t for t in closed if t.get("pnl_r") is None]
        assert len(no_pnl) == 0, f"{len(no_pnl)} closed trade(s) missing pnl_r"

    def test_entry_levels_ordered(self):
        """entry_low must always be <= entry_high."""
        c = _conn()
        bad = c.execute(
            "SELECT rowid, ticker, strategy, entry_low, entry_high "
            "FROM trades WHERE entry_low > entry_high"
        ).fetchall()
        c.close()
        assert len(bad) == 0, (
            f"{len(bad)} trade(s) have entry_low > entry_high: {bad[:5]}"
        )

    def test_get_open_trades_returns_list(self):
        result = get_open_trades()
        assert isinstance(result, list)

    def test_get_all_trades_count(self):
        all_t = get_all_trades()
        assert isinstance(all_t, list)
        assert len(all_t) > 0, "Journal has no trades at all"

    def test_get_closed_trades_subset_of_all(self):
        all_t = get_all_trades()
        closed = get_closed_trades()
        assert len(closed) <= len(all_t)

    def test_active_strategy_set_type(self):
        result = get_active_strategy_set()
        assert isinstance(result, (list, set, frozenset))

    def test_active_ticker_set_type(self):
        result = get_active_ticker_set()
        assert isinstance(result, (list, set, frozenset))
