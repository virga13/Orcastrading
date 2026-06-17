"""
p4_live/journal_supabase.py — Supabase-backed journal for the Streamlit UI.

Mirrors the API of p4_live/journal.py but reads/writes Supabase instead of
SQLite. All functions take a `client` (supabase.Client) as the first argument.
RLS on the Supabase side enforces user isolation automatically via auth.uid().

The SQLite journal.py is kept unchanged — the scheduler continues using it.
This module is only imported by ui/app.py (the multi-user web interface).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone, date, timedelta
from supabase import Client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(response) -> list[dict]:
    """Extract .data list from a Supabase response, defaulting to []."""
    return response.data or []


def _row(response) -> dict | None:
    data = response.data
    return data[0] if data else None


# ── Trades — read operations ───────────────────────────────────────────────────

def get_all_trades(client: Client) -> list[dict]:
    return _rows(client.table("trades").select("*").order("signal_date", desc=False).execute())


def get_open_trades(client: Client) -> list[dict]:
    return _rows(
        client.table("trades").select("*")
        .in_("status", ["pending", "filled"])
        .order("signal_date").execute()
    )


def get_closed_trades(client: Client) -> list[dict]:
    return _rows(
        client.table("trades").select("*")
        .not_.in_("status", ["pending", "filled", "expired"])
        .order("signal_date").execute()
    )


def get_active_ticker_set(client: Client) -> set[str]:
    rows = _rows(
        client.table("trades").select("ticker")
        .in_("status", ["pending", "filled"]).execute()
    )
    return {r["ticker"] for r in rows}


def get_active_strategy_set(client: Client) -> set[tuple[str, str]]:
    rows = _rows(
        client.table("trades").select("ticker,strategy")
        .in_("status", ["pending", "filled"]).execute()
    )
    return {(r["ticker"], r["strategy"]) for r in rows}


# ── Trades — write operations ─────────────────────────────────────────────────

def record_signal(client: Client, signal: dict) -> bool:
    """
    Persist a fired signal. Returns False if already recorded or blocked.

    Conflict rules (same as SQLite journal):
      - Duplicate PK (same date/ticker/strategy/timeframe) → False
      - Filled opposing trade exists → block new signal
      - Pending opposing trade → higher confidence wins; tie keeps incumbent
    """
    tf       = signal.get("timeframe", "1d")
    ticker   = signal["ticker"]
    strategy = signal.get("strategy", "mtf_trend")
    sig_date = signal["signal_date"]
    new_dir  = signal["direction"]
    new_conf = signal.get("confidence") or 0.0

    # Duplicate guard
    existing = _rows(
        client.table("trades").select("id")
        .eq("signal_date", sig_date).eq("ticker", ticker)
        .eq("strategy", strategy).eq("timeframe", tf).execute()
    )
    if existing:
        return False

    # Conflict resolution — opposing direction already open
    opp_dir = "short" if new_dir == "long" else "long"
    conflicts = _rows(
        client.table("trades").select("signal_date,ticker,strategy,timeframe,status,confidence")
        .eq("ticker", ticker).eq("direction", opp_dir)
        .in_("status", ["pending", "filled"]).execute()
    )
    for c in conflicts:
        if c["status"] == "filled":
            return False   # live opposing position — cannot auto-close
        inc_conf = c.get("confidence") or 0.0
        if new_conf > inc_conf:
            # New signal stronger — expire incumbent pending
            client.table("trades").update({"status": "expired"}) \
                .eq("signal_date", c["signal_date"]).eq("ticker", c["ticker"]) \
                .eq("strategy", c["strategy"]).eq("timeframe", c["timeframe"]) \
                .eq("status", "pending").execute()
        else:
            return False   # incumbent equal or stronger — block new signal

    client.table("trades").insert({
        "signal_date":      sig_date,
        "ticker":           ticker,
        "strategy":         strategy,
        "timeframe":        tf,
        "label":            signal["label"],
        "direction":        new_dir,
        "entry_low":        signal["entry_low"],
        "entry_high":       signal["entry_high"],
        "stop_loss":        signal["stop_loss"],
        "tp1":              signal["tp1"],
        "tp2":              signal.get("tp2"),
        "tp1_alloc":        signal.get("tp1_alloc", 70),
        "tp2_alloc":        signal.get("tp2_alloc", 30),
        "risk_pts":         signal.get("risk_pts"),
        "rr":               signal.get("rr"),
        "confidence":       new_conf,
        "rationale":        signal.get("rationale"),
        "price_at_signal":  signal.get("price"),
        "atr_at_signal":    signal.get("atr"),
        "rsi_at_signal":    signal.get("rsi"),
        "regime_note":      signal.get("regime"),
        "signal_bar_ts":    signal.get("signal_bar_ts"),
        "notes":            signal.get("notes"),
        "source":           signal.get("source", "scanner"),
        "status":           "pending",
    }).execute()
    return True


def record_fill_direct(
    client: Client,
    signal_date: str,
    ticker: str,
    strategy: str,
    timeframe: str,
    fill_price: float,
    fill_date: str,
    fill_bar_ts: str | None = None,
) -> None:
    client.table("trades").update({
        "status":     "filled",
        "fill_price": fill_price,
        "fill_date":  fill_date,
        "fill_bar_ts": fill_bar_ts,
    }).eq("signal_date", signal_date).eq("ticker", ticker) \
      .eq("strategy", strategy).eq("timeframe", timeframe) \
      .eq("status", "pending").execute()


def record_close_direct(
    client: Client,
    signal_date: str,
    ticker: str,
    strategy: str,
    timeframe: str,
    outcome: str,
    exit_price: float,
    exit_date: str,
    pnl_r: float,
) -> None:
    client.table("trades").update({
        "status":     outcome,
        "exit_price": exit_price,
        "exit_date":  exit_date,
        "pnl_r":      round(pnl_r, 3),
    }).eq("signal_date", signal_date).eq("ticker", ticker) \
      .eq("strategy", strategy).eq("timeframe", timeframe) \
      .eq("status", "filled").execute()


def record_expired_direct(
    client: Client,
    signal_date: str,
    ticker: str,
    strategy: str,
    timeframe: str = "1d",
) -> None:
    client.table("trades").update({"status": "expired"}) \
        .eq("signal_date", signal_date).eq("ticker", ticker) \
        .eq("strategy", strategy).eq("timeframe", timeframe) \
        .eq("status", "pending").execute()


def add_note(client: Client, signal_date: str, ticker: str, strategy: str,
             timeframe: str, note: str) -> None:
    client.table("trades").update({"notes": note}) \
        .eq("signal_date", signal_date).eq("ticker", ticker) \
        .eq("strategy", strategy).eq("timeframe", timeframe).execute()


def save_trade_chart_screenshot(
    client: Client,
    signal_date: str, ticker: str, strategy: str, timeframe: str, path: str,
) -> None:
    client.table("trades").update({"chart_screenshot_path": path}) \
        .eq("signal_date", signal_date).eq("ticker", ticker) \
        .eq("strategy", strategy).eq("timeframe", timeframe).execute()


def expire_stale_signals(client: Client, expiry_map: dict[str, int]) -> int:
    today   = date.today()
    expired = 0
    for strategy_id, days in expiry_map.items():
        cutoff = (today - timedelta(days=days)).isoformat()
        rows = _rows(
            client.table("trades").select("signal_date,ticker,strategy,timeframe")
            .eq("strategy", strategy_id).eq("status", "pending")
            .lte("signal_date", cutoff).execute()
        )
        for r in rows:
            client.table("trades").update({"status": "expired"}) \
                .eq("signal_date", r["signal_date"]).eq("ticker", r["ticker"]) \
                .eq("strategy", r["strategy"]).eq("timeframe", r["timeframe"]) \
                .eq("status", "pending").execute()
            expired += 1
    return expired


# ── Manual entries ────────────────────────────────────────────────────────────

def save_manual_entry(client: Client, data: dict) -> str:
    """Insert a manual journal entry. Returns the new row id (UUID string)."""
    today = date.today().isoformat()
    row = client.table("manual_entries").insert({
        "entry_date":            data.get("entry_date", today),
        "asset":                 data.get("asset"),
        "direction":             data.get("direction"),
        "entry_price":           data.get("entry_price"),
        "stop_loss":             data.get("stop_loss"),
        "original_sl":           data.get("original_sl"),
        "take_profit":           data.get("take_profit"),
        "exit_price":            data.get("exit_price"),
        "pnl_r":                 data.get("pnl_r"),
        "pnl_dollars":           data.get("pnl_dollars"),
        "quality_score":         data.get("quality_score"),
        "notes":                 data.get("notes"),
        "ai_analysis":           data.get("ai_analysis"),
        "screenshot_path":       data.get("screenshot_path"),
        "chart_screenshot_path": data.get("chart_screenshot_path"),
        "tags":                  data.get("tags"),
    }).execute()
    return row.data[0]["id"]


_MANUAL_ALLOWED = {
    "exit_price", "pnl_r", "pnl_dollars", "stop_loss", "original_sl",
    "take_profit", "notes", "tags", "quality_score",
    "chart_screenshot_path", "screenshot_path", "direction",
}


def update_manual_entry(client: Client, entry_id: str, **fields) -> None:
    valid = {k: v for k, v in fields.items() if k in _MANUAL_ALLOWED}
    if not valid:
        return
    client.table("manual_entries").update(valid).eq("id", entry_id).execute()


def delete_manual_entry(client: Client, entry_id: str) -> None:
    client.table("manual_entries").delete().eq("id", entry_id).execute()


def get_manual_entries(client: Client, limit: int = 200) -> list[dict]:
    return _rows(
        client.table("manual_entries").select("*")
        .order("entry_date", desc=True).limit(limit).execute()
    )


# ── Manual signals (Trade Planner) ────────────────────────────────────────────

def save_manual_signal(client: Client, signal: dict) -> bool:
    sig_date = signal.get("signal_date", date.today().isoformat())
    ticker   = signal.get("ticker", "MANUAL")
    strategy = signal.get("strategy", "manual")

    existing = _rows(
        client.table("trades").select("id")
        .eq("signal_date", sig_date).eq("ticker", ticker)
        .eq("strategy", strategy).execute()
    )
    if existing:
        return False

    risk_pts = abs((signal.get("entry_high") or 0) - (signal.get("stop_loss") or 0)) or None
    rr = None
    if risk_pts and signal.get("tp1"):
        rr = round(abs((signal["tp1"] - (signal.get("entry_high") or 0)) / risk_pts), 2)

    client.table("trades").insert({
        "signal_date":  sig_date,
        "ticker":       ticker,
        "strategy":     strategy,
        "timeframe":    signal.get("timeframe", "1d"),
        "label":        signal.get("label", ticker),
        "direction":    signal.get("direction", "long"),
        "entry_low":    signal.get("entry_low", 0),
        "entry_high":   signal.get("entry_high", 0),
        "stop_loss":    signal.get("stop_loss", 0),
        "tp1":          signal.get("tp1", 0),
        "tp2":          signal.get("tp2"),
        "tp1_alloc":    signal.get("tp1_alloc", 70),
        "tp2_alloc":    signal.get("tp2_alloc", 30),
        "risk_pts":     risk_pts,
        "rr":           rr,
        "confidence":   signal.get("confidence"),
        "rationale":    signal.get("rationale", ""),
        "price_at_signal": signal.get("entry_high", 0),
        "source":       "manual",
        "status":       "pending",
    }).execute()
    return True


# ── Psychology ────────────────────────────────────────────────────────────────

def save_psychology(client: Client, data: dict) -> str:
    row = client.table("psychology").insert({
        "trade_ref_type":    data.get("trade_ref_type", "live"),
        "trade_ref_id":      data.get("trade_ref_id", ""),
        "pre_state":         data.get("pre_state"),
        "pre_confidence":    data.get("pre_confidence"),
        "pre_notes":         data.get("pre_notes"),
        "post_state":        data.get("post_state"),
        "post_notes":        data.get("post_notes"),
        "execution_quality": data.get("execution_quality"),
        "mistakes":          json.dumps(data.get("mistakes", [])),
        "lesson":            data.get("lesson"),
    }).execute()
    return row.data[0]["id"]


def get_psychology(client: Client, trade_ref_id: str) -> dict | None:
    rows = _rows(
        client.table("psychology").select("*")
        .eq("trade_ref_id", trade_ref_id)
        .order("recorded_at", desc=True).limit(1).execute()
    )
    return rows[0] if rows else None


def get_all_psychology(client: Client) -> list[dict]:
    return _rows(
        client.table("psychology").select("*")
        .order("recorded_at", desc=True).execute()
    )


# ── User watchlist ────────────────────────────────────────────────────────────

def get_user_watchlist(client: Client) -> list[tuple[str, str, str]]:
    """
    Return the user's enabled watchlist as (asset_id, strategy_id, timeframe) triples.
    Same format as core.config.get_watchlist() so the scanner can consume it directly.
    """
    rows = _rows(
        client.table("user_watchlist").select("asset_id,strategy_id,timeframe")
        .eq("enabled", True).execute()
    )
    return [(r["asset_id"], r["strategy_id"], r["timeframe"]) for r in rows]


def get_user_watchlist_full(client: Client) -> list[dict]:
    """Return full watchlist rows (including params and enabled flag) for the Settings UI."""
    return _rows(
        client.table("user_watchlist").select("*")
        .order("asset_id").execute()
    )


def save_watchlist_entry(
    client: Client,
    asset_id: str,
    strategy_id: str,
    timeframe: str,
    params: dict | None = None,
    enabled: bool = True,
) -> None:
    """Upsert one watchlist entry (insert or update on conflict)."""
    client.table("user_watchlist").upsert({
        "asset_id":    asset_id,
        "strategy_id": strategy_id,
        "timeframe":   timeframe,
        "params":      params or {},
        "enabled":     enabled,
    }, on_conflict="user_id,asset_id,strategy_id,timeframe").execute()


def delete_watchlist_entry(
    client: Client,
    asset_id: str,
    strategy_id: str,
    timeframe: str,
) -> None:
    client.table("user_watchlist").delete() \
        .eq("asset_id", asset_id).eq("strategy_id", strategy_id) \
        .eq("timeframe", timeframe).execute()


def replace_user_watchlist(client: Client, entries: list[dict]) -> None:
    """
    Replace the user's entire watchlist atomically.
    Each entry dict: {asset_id, strategy_id, timeframe, params, enabled}
    """
    # Delete all existing rows for this user (RLS scopes the delete automatically)
    client.table("user_watchlist").delete().neq("asset_id", "").execute()
    if entries:
        client.table("user_watchlist").insert(entries).execute()


# ── User settings ─────────────────────────────────────────────────────────────

def get_user_settings(client: Client) -> dict:
    """Return the user's settings dict, with sensible defaults if not yet saved."""
    rows = _rows(client.table("user_settings").select("*").execute())
    if rows:
        return rows[0]
    return {
        "telegram_bot_token":  "",
        "telegram_chat_id":    "",
        "max_open_trades":     5,
        "daily_loss_limit_r":  0.0,
        "weekly_dd_limit_r":   0.0,
        "account_balance":     0.0,
        "risk_per_trade_pct":  1.0,
    }


def save_user_settings(client: Client, settings: dict) -> None:
    """Upsert user settings (insert on first save, update thereafter)."""
    allowed = {
        "telegram_bot_token", "telegram_chat_id",
        "max_open_trades", "daily_loss_limit_r", "weekly_dd_limit_r",
        "account_balance", "risk_per_trade_pct",
    }
    payload = {k: v for k, v in settings.items() if k in allowed}
    payload["updated_at"] = _now()
    client.table("user_settings").upsert(
        payload, on_conflict="user_id"
    ).execute()
