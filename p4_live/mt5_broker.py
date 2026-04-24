"""
p4_live/mt5_broker.py — MetaTrader 5 broker adapter.

Wraps the MetaTrader5 Python package. All public functions return
None/False/[] on failure — they never raise, so callers just log and move on.

Required .env keys (when MT5_ENABLED=true):
  MT5_ACCOUNT        — integer account number
  MT5_PASSWORD       — account password
  MT5_SERVER         — broker server name  (e.g. "ICMarkets-Demo01")

Optional .env keys:
  MT5_TERMINAL_PATH  — full path to terminal64.exe (auto-detected if omitted)
  MT5_LOT_SIZE       — fixed lot size per order leg (default 0.01)
  MT5_RISK_PCT       — fraction of equity to risk per trade; overrides MT5_LOT_SIZE
                       (e.g. "0.01" = 1 % of equity per signal)
  MT5_ENABLED        — must be exactly "true" to enable live execution

Each asset in assets.yaml must have tickers.mt5 set to the broker-specific symbol
(e.g. "XAUUSD", "US500", "US30").  Symbol names vary by broker — check your
broker's Market Watch window.

Order model
-----------
For each signal two pending limit orders are placed (one per TP leg):
  Long  → ORDER_TYPE_BUY_LIMIT  at entry_high
  Short → ORDER_TYPE_SELL_LIMIT at entry_low

Leg 1: tp1_alloc% of computed volume, TP = tp1
Leg 2: tp2_alloc% of computed volume, TP = tp2  (skipped when tp2 is None)

Both legs share the same SL.  The MT5 terminal handles SL/TP execution — no
polling of price levels is needed in the monitor.

Tickets are stored in the journal as a JSON array via record_mt5_ticket().
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger("orcastrading.mt5")

_MAGIC = 20260001        # Orcastrading magic number — identifies all orca orders in MT5
_mt5   = None            # MetaTrader5 module, imported lazily
_initialized = False     # track whether mt5.initialize() succeeded


# ── Module-level helpers ──────────────────────────────────────────────────────

def _lib():
    """Return the MetaTrader5 module, importing it on first call."""
    global _mt5
    if _mt5 is None:
        import MetaTrader5 as _m   # noqa: PLC0415
        _mt5 = _m
    return _mt5


def is_enabled() -> bool:
    """Return True only when MT5_ENABLED=true is set in the environment."""
    return os.getenv("MT5_ENABLED", "").strip().lower() == "true"


# ── Connection ────────────────────────────────────────────────────────────────

def connect() -> bool:
    """
    Initialize the MT5 terminal connection.  Safe to call repeatedly —
    mt5.initialize() is a no-op when already connected.
    Returns True on success, False on failure.
    """
    global _initialized
    if not is_enabled():
        return False

    mt5 = _lib()

    # Fast path: already up
    if _initialized:
        try:
            if mt5.terminal_info() is not None:
                return True
        except Exception:
            pass
        _initialized = False

    account  = int(os.getenv("MT5_ACCOUNT",  "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server   = os.getenv("MT5_SERVER",   "")
    path     = os.getenv("MT5_TERMINAL_PATH") or None

    kwargs: dict[str, Any] = {
        "login":    account,
        "password": password,
        "server":   server,
    }
    if path:
        kwargs["path"] = path

    if not mt5.initialize(**kwargs):
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        log.error("MT5 connected but account_info() returned None")
        return False

    _initialized = True
    log.info(
        f"MT5 connected — account={info.login}  server={info.server}  "
        f"balance={info.balance:.2f}  equity={info.equity:.2f}  "
        f"currency={info.currency}"
    )
    return True


def disconnect() -> None:
    """Shut down the MT5 terminal connection."""
    global _initialized
    _initialized = False
    if _mt5 is not None:
        try:
            _mt5.shutdown()
        except Exception:
            pass


def get_account_info() -> dict | None:
    """Return account balance/equity/margin as a plain dict.  None on failure."""
    if not is_enabled() or not connect():
        return None
    mt5  = _lib()
    info = mt5.account_info()
    if info is None:
        return None
    return {
        "login":       info.login,
        "server":      info.server,
        "balance":     info.balance,
        "equity":      info.equity,
        "margin":      info.margin,
        "free_margin": info.margin_free,
        "leverage":    info.leverage,
        "currency":    info.currency,
    }


# ── Volume calculation ────────────────────────────────────────────────────────

def _compute_lot(
    symbol:     str,
    risk_pts:   float,
    alloc_pct:  float = 1.0,
    kelly_pct:  float | None = None,
) -> float:
    """
    Compute lot size for one leg.

    alloc_pct — fraction of total trade risk allocated to this leg
                (0.7 for TP1 leg, 0.3 for TP2).
    kelly_pct — per-asset 25%-Kelly fraction (from assets.yaml baseline).
                Used when MT5_KELLY_SIZING=true; overrides MT5_RISK_PCT.

    Priority:
      1. kelly_pct (when MT5_KELLY_SIZING=true and kelly_pct provided)
      2. MT5_RISK_PCT — flat equity-based sizing
      3. MT5_LOT_SIZE — fixed lot size (fallback)
    """
    mt5         = _lib()
    default_lot = float(os.getenv("MT5_LOT_SIZE", "0.01"))
    leg_default = max(0.01, round(default_lot * alloc_pct, 8))

    use_kelly = (
        os.getenv("MT5_KELLY_SIZING", "").strip().lower() == "true"
        and kelly_pct is not None
        and kelly_pct > 0
    )
    risk_env = os.getenv("MT5_RISK_PCT", "").strip()

    effective_pct: float | None = None
    if use_kelly:
        effective_pct = kelly_pct
        log.debug(f"Kelly sizing: {kelly_pct:.3%} for {symbol}")
    elif risk_env:
        try:
            effective_pct = float(risk_env)
        except ValueError:
            pass

    if effective_pct is None or risk_pts <= 0:
        return leg_default

    try:
        info = mt5.account_info()
        sym  = mt5.symbol_info(symbol)
        if info is None or sym is None:
            return leg_default

        tick_value = sym.trade_tick_value
        tick_size  = sym.trade_tick_size
        if tick_size <= 0 or tick_value <= 0:
            return leg_default

        dollar_risk   = info.equity * effective_pct * alloc_pct
        value_per_lot = (risk_pts / tick_size) * tick_value
        if value_per_lot <= 0:
            return leg_default

        raw_lot = dollar_risk / value_per_lot
        step    = sym.volume_step if sym.volume_step > 0 else 0.01
        lot     = round(round(raw_lot / step) * step, 8)
        lot     = max(sym.volume_min, min(sym.volume_max, lot))
        return lot

    except Exception as e:
        log.warning(f"_compute_lot failed for {symbol}: {e} — using {leg_default}")
        return leg_default


# ── Filling mode ──────────────────────────────────────────────────────────────

def _filling_mode(symbol: str) -> int:
    """Return the best filling mode supported by the symbol (broker-dependent)."""
    mt5  = _lib()
    sym  = mt5.symbol_info(symbol)
    if sym is None:
        return getattr(mt5, "ORDER_FILLING_RETURN", 2)

    RETURN = getattr(mt5, "ORDER_FILLING_RETURN", 2)
    IOC    = getattr(mt5, "ORDER_FILLING_IOC",    1)
    FOK    = getattr(mt5, "ORDER_FILLING_FOK",    0)

    mode = sym.filling_mode
    if mode & RETURN:
        return RETURN
    if mode & IOC:
        return IOC
    return FOK


# ── Order placement ───────────────────────────────────────────────────────────

def _place_limit(
    symbol:    str,
    direction: str,
    price:     float,
    sl:        float,
    tp:        float,
    lot:       float,
    comment:   str,
    expiry_dt: datetime,
) -> int | None:
    """
    Place a single BUY_LIMIT or SELL_LIMIT pending order.
    Returns the MT5 ticket on success, None on failure.
    """
    mt5      = _lib()

    # Ensure the symbol is visible in Market Watch so prices flow through.
    # symbol_select() may take a tick cycle to propagate — poll up to 3 s.
    import time as _time
    mt5.symbol_select(symbol, True)
    for _ in range(6):
        _tick = mt5.symbol_info_tick(symbol)
        if _tick and (_tick.bid > 0 or _tick.ask > 0):
            break
        _time.sleep(0.5)

    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        log.error(
            f"MT5: symbol '{symbol}' not found. "
            f"Check tickers.mt5 in assets.yaml and your broker's Market Watch."
        )
        return None

    digits = sym_info.digits
    price  = round(price, digits)
    sl     = round(sl,    digits)
    tp     = round(tp,    digits)

    order_type = (
        mt5.ORDER_TYPE_BUY_LIMIT
        if direction == "long"
        else mt5.ORDER_TYPE_SELL_LIMIT
    )

    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "expiration":   int(expiry_dt.timestamp()),
        "type_time":    mt5.ORDER_TIME_SPECIFIED,
        "type_filling": _filling_mode(symbol),
        "comment":      comment[:31],    # MT5 hard limit is 31 chars
        "magic":        _MAGIC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode  = result.retcode if result else "None"
        rcomment = result.comment if result else ""
        log.error(
            f"MT5 order_send failed: retcode={retcode} '{rcomment}'  "
            f"{symbol} {direction} {lot}lot @ {price}  SL={sl}  TP={tp}"
        )
        return None

    log.info(
        f"MT5 order placed: ticket={result.order}  {symbol} {direction.upper()}  "
        f"{lot}lot @ {price}  SL={sl}  TP={tp}  "
        f"expires={expiry_dt.strftime('%Y-%m-%d')}"
    )
    return result.order


def place_orders(
    signal:      dict,
    mt5_symbol:  str,
    expiry_days: int = 5,
    kelly_pct:   float | None = None,
) -> list[int]:
    """
    Place two pending limit orders for a signal (one leg per TP level).
    kelly_pct: per-asset 25%-Kelly fraction; used when MT5_KELLY_SIZING=true.
    Returns list of tickets placed (0, 1, or 2 entries).
    """
    if not is_enabled() or not connect():
        return []

    direction  = signal["direction"]
    entry_high = float(signal["entry_high"])
    entry_low  = float(signal["entry_low"])
    sl         = float(signal["stop_loss"])
    tp1        = float(signal["tp1"])
    tp2        = signal.get("tp2")
    tp1_alloc  = (signal.get("tp1_alloc") or 70) / 100
    tp2_alloc  = (signal.get("tp2_alloc") or 30) / 100
    risk_pts   = float(signal.get("risk_pts") or abs(entry_high - sl))
    strategy   = (signal.get("strategy") or "orca")[:8]
    sig_date   = signal.get("signal_date", "")[:10]

    entry_price = entry_high if direction == "long" else entry_low
    expiry_dt   = datetime.now(timezone.utc) + timedelta(days=expiry_days)
    tickets: list[int] = []

    # ── Validate entry price vs current market (avoid retcode=10015) ──────────
    # A BUY_LIMIT must be below current ask; SELL_LIMIT above current bid.
    # If the market has entered the zone we fall back to the far edge of the
    # zone; if the zone is fully breached we skip rather than chase.
    _mt5 = _lib()
    _mt5.symbol_select(mt5_symbol, True)
    _tick     = _mt5.symbol_info_tick(mt5_symbol)
    _sym_info = _mt5.symbol_info(mt5_symbol)
    if _tick and _sym_info and _tick.ask > 0:
        _stops = max(
            _sym_info.trade_stops_level * _sym_info.point,
            _sym_info.point * 10,  # minimum sensible buffer
        )
        if direction == "long":
            if entry_price >= _tick.ask - _stops:
                if entry_low < _tick.ask - _stops:
                    log.info(
                        f"MT5: {mt5_symbol} BUY_LIMIT adjusted "
                        f"entry_high={entry_price} → entry_low={entry_low} "
                        f"(ask={_tick.ask:.5f} already in zone)"
                    )
                    entry_price = entry_low
                else:
                    log.warning(
                        f"MT5: {mt5_symbol} BUY_LIMIT skipped — "
                        f"ask={_tick.ask:.5f} already at/below zone [{entry_low}–{entry_high}]"
                    )
                    return []
        else:  # short
            if entry_price <= _tick.bid + _stops:
                if entry_high > _tick.bid + _stops:
                    log.info(
                        f"MT5: {mt5_symbol} SELL_LIMIT adjusted "
                        f"entry_low={entry_price} → entry_high={entry_high} "
                        f"(bid={_tick.bid:.5f} already in zone)"
                    )
                    entry_price = entry_high
                else:
                    log.warning(
                        f"MT5: {mt5_symbol} SELL_LIMIT skipped — "
                        f"bid={_tick.bid:.5f} already at/above zone [{entry_low}–{entry_high}]"
                    )
                    return []

    # ── Leg 1: TP1 ────────────────────────────────────────────────────────────
    lot1 = _compute_lot(mt5_symbol, risk_pts, alloc_pct=tp1_alloc, kelly_pct=kelly_pct)
    t1   = _place_limit(
        symbol    = mt5_symbol,
        direction = direction,
        price     = entry_price,
        sl        = sl,
        tp        = tp1,
        lot       = lot1,
        comment   = f"o:{strategy}:{sig_date}:tp1",
        expiry_dt = expiry_dt,
    )
    if t1:
        tickets.append(t1)

    # ── Leg 2: TP2 (optional) ─────────────────────────────────────────────────
    if tp2 is not None and tp2_alloc > 0:
        lot2 = _compute_lot(mt5_symbol, risk_pts, alloc_pct=tp2_alloc, kelly_pct=kelly_pct)
        t2   = _place_limit(
            symbol    = mt5_symbol,
            direction = direction,
            price     = entry_price,
            sl        = sl,
            tp        = float(tp2),
            lot       = lot2,
            comment   = f"o:{strategy}:{sig_date}:tp2",
            expiry_dt = expiry_dt,
        )
        if t2:
            tickets.append(t2)

    return tickets


def cancel_order(ticket: int) -> bool:
    """Cancel a pending order by ticket number."""
    if not is_enabled() or not connect():
        return False
    mt5    = _lib()
    result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
    ok     = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    if not ok:
        retcode = result.retcode if result else "None"
        log.warning(f"MT5 cancel_order failed: ticket={ticket} retcode={retcode}")
    return ok


# ── Status queries ────────────────────────────────────────────────────────────

def _hist_window() -> tuple[datetime, datetime]:
    """Return (from_dt, to_dt) spanning the last 365 days."""
    now = datetime.now(timezone.utc)
    return now - timedelta(days=365), now + timedelta(hours=1)


def get_order_state(ticket: int) -> dict | None:
    """
    Return state dict if the order is still pending in the terminal.
    Returns None if the order is no longer in the open-orders list
    (filled, cancelled, or expired).
    """
    if not is_enabled() or not connect():
        return None
    mt5    = _lib()
    orders = mt5.orders_get(ticket=ticket)
    if not orders:
        return None
    o = orders[0]
    return {"ticket": o.ticket, "state": "pending", "symbol": o.symbol}


def get_position_info(ticket: int) -> dict | None:
    """
    Return open position dict for the given ticket.
    In MT5, position ticket == opening order ticket for orders filled from pending.
    Returns None when position is not open.
    """
    if not is_enabled() or not connect():
        return None
    mt5 = _lib()
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return None
    p         = pos[0]
    direction = "long" if p.type == mt5.POSITION_TYPE_BUY else "short"
    return {
        "ticket":     p.ticket,
        "symbol":     p.symbol,
        "direction":  direction,
        "volume":     p.volume,
        "open_price": p.price_open,
        "sl":         p.sl,
        "tp":         p.tp,
        "profit":     p.profit,
        "open_time":  datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
    }


def get_historical_order(ticket: int) -> dict | None:
    """
    Look up a filled/cancelled/expired order in MT5 history.
    Returns dict with keys: state, position_id (if filled), price.
    Returns None if not found in history yet.
    """
    if not is_enabled() or not connect():
        return None
    mt5         = _lib()
    from_dt, to_dt = _hist_window()
    orders      = mt5.history_orders_get(from_dt, to_dt, ticket=ticket)
    if not orders:
        return None
    o = orders[0]

    STATE_FILLED   = getattr(mt5, "ORDER_STATE_FILLED",   1)
    STATE_CANCELED = getattr(mt5, "ORDER_STATE_CANCELED", 3)
    STATE_EXPIRED  = getattr(mt5, "ORDER_STATE_EXPIRED",  5)

    if o.state == STATE_FILLED:
        return {
            "ticket":      o.ticket,
            "state":       "filled",
            "position_id": o.position_id,
            "price":       o.price_current,
        }
    if o.state == STATE_CANCELED:
        return {"ticket": o.ticket, "state": "cancelled"}
    if o.state == STATE_EXPIRED:
        return {"ticket": o.ticket, "state": "expired"}
    return {"ticket": o.ticket, "state": "other"}


def get_closed_position(position_id: int) -> dict | None:
    """
    Check whether a position has been closed by SL or TP.
    Returns the closing deal dict, or None if still open.
    """
    if not is_enabled() or not connect():
        return None
    mt5         = _lib()
    from_dt, to_dt = _hist_window()
    deals       = mt5.history_deals_get(from_dt, to_dt, position=position_id)
    if not deals:
        return None

    DEAL_ENTRY_OUT = getattr(mt5, "DEAL_ENTRY_OUT", 1)
    REASON_SL      = getattr(mt5, "DEAL_REASON_SL", 3)
    REASON_TP      = getattr(mt5, "DEAL_REASON_TP", 4)

    for d in deals:
        if d.entry == DEAL_ENTRY_OUT:
            if d.reason == REASON_SL:
                close_reason = "sl"
            elif d.reason == REASON_TP:
                close_reason = "tp"
            else:
                close_reason = "manual"
            return {
                "close_price":  d.price,
                "close_time":   datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
                "profit":       d.profit,
                "close_reason": close_reason,
            }
    return None


# ── High-level: execute a signal ─────────────────────────────────────────────

def execute_signal(signal: dict) -> list[int]:
    """
    Place MT5 orders for a fired signal.  Resolves the broker symbol from
    assets.yaml tickers.mt5 and respects the strategy's expiry_days.

    Returns list of MT5 tickets (empty list = not executed or disabled).
    """
    if not is_enabled():
        return []

    from core.config import get_asset_by_ticker, get_strategy_config  # noqa: PLC0415

    ticker = signal.get("ticker", "")
    asset  = get_asset_by_ticker(ticker)
    if asset is None:
        log.warning(f"execute_signal: no asset found for ticker '{ticker}'")
        return []

    mt5_symbol = asset.get("tickers", {}).get("mt5")
    if not mt5_symbol:
        log.warning(
            f"execute_signal: tickers.mt5 not set for asset '{asset['id']}'. "
            f"Add it to config/assets.yaml (check your broker's Market Watch for the exact name)."
        )
        return []

    strategy_id = signal.get("strategy", "mtf_trend")
    try:
        expiry_days = int(get_strategy_config(strategy_id).get("expiry_days", 5))
    except KeyError:
        expiry_days = 5

    # Read per-asset Kelly fraction for position sizing
    kelly_pct: float | None = None
    try:
        kelly_pct = float(asset.get("baseline", {}).get("kelly_25pct", 0)) or None
    except (TypeError, ValueError):
        pass

    if kelly_pct:
        log.info(
            f"execute_signal: Kelly sizing {'ENABLED' if os.getenv('MT5_KELLY_SIZING','').lower()=='true' else 'available (set MT5_KELLY_SIZING=true)'} "
            f"— {asset['id']} kelly_25pct={kelly_pct:.3%}"
        )

    tickets = place_orders(signal, mt5_symbol, expiry_days=expiry_days, kelly_pct=kelly_pct)
    return tickets
