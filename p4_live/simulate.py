"""
p4_live/simulate.py — Simulate trade outcomes for pending signals + Gold historical backfill.

Two public entry points:
  simulate_pending(dry_run)  — resolve pending journal entries using real price data
  gold_backfill(start, end)  — replay all Gold strategies historically, writing to journal
"""
from __future__ import annotations

import os
import pandas as pd
from datetime import date, timedelta

from p3_backtester.market_data import fetch_ohlcv
from p4_live.journal import (
    get_open_trades,
    record_signal,
    record_fill_close_direct,
    record_expired_direct,
    record_fill_direct,
    record_close_direct,
)

# Gold's yfinance ticker — reliable historical data
_GOLD_YF = "GC=F"
_GOLD_LABEL = "Gold"

# Live-ticker → yfinance ticker (for simulation data fetching)
_TO_YF = {
    "XAU/USD": "GC=F",
    "XAG/USD": "SI=F",
    "BTC/USD": "BTC-USD",
    "BTC/USDT": "BTC-USD",
}

_ENTRY_TIMEOUT = 20   # bars to wait for fill before expiring


def _to_yf(ticker: str) -> str:
    return _TO_YF.get(ticker, ticker)


def _get_slippage_pct() -> float:
    """Read SLIPPAGE_PCT from env (default 0). Set e.g. SLIPPAGE_PCT=0.05 for 0.05%."""
    return float(os.getenv("SLIPPAGE_PCT", "0.0"))


# ── Core simulation logic ─────────────────────────────────────────────────────

def _find_fill(
    df: pd.DataFrame,
    direction: str,
    entry_low: float,
    entry_high: float,
    slippage_pct: float = 0.0,
) -> tuple[float, int] | None:
    """
    Scan df[1 .. _ENTRY_TIMEOUT] looking for price entering the entry zone.
    df[0] is the signal bar; df[1] is the first actionable bar.
    Returns (fill_price, bar_index_in_df) or None if not filled within timeout.
    slippage_pct worsens the fill: longs fill higher, shorts fill lower.
    """
    is_long = direction == "long"
    slip = 1.0 + slippage_pct / 100 if is_long else 1.0 - slippage_pct / 100
    for i in range(1, min(1 + _ENTRY_TIMEOUT, len(df))):
        hi = float(df["High"].iloc[i])
        lo = float(df["Low"].iloc[i])
        if is_long and lo <= entry_high and hi >= entry_low:
            return (min(entry_high, hi) * slip, i)
        elif not is_long and hi >= entry_low and lo <= entry_high:
            return (max(entry_low, lo) * slip, i)
    return None


def _find_exit(
    df: pd.DataFrame,
    direction: str,
    fill_price: float,
    stop_loss: float,
    tp1: float | None,
    tp2: float | None,
    tp1_alloc: int,
    tp2_alloc: int,
) -> dict | None:
    """
    Walk df[1..] looking for SL or TP hits.
    df[0] is the fill bar (not checked for SL/TP); df[1] is the first bar at risk.
    Returns a result dict when the trade is closed, or None when data is exhausted
    (trade still open — no SL/TP hit yet).
    """
    is_long = direction == "long"
    risk = abs(fill_price - stop_loss)
    if risk < 1e-8:
        return None

    targets: list[tuple[float, int]] = []
    if tp1 is not None:
        targets.append((tp1, int(tp1_alloc or 70)))
    if tp2 is not None and int(tp2_alloc or 0) > 0:
        targets.append((tp2, int(tp2_alloc)))
    if not targets:
        if tp1 is not None:
            targets = [(tp1, 100)]
        else:
            return None
    targets.sort(key=lambda x: x[0] if is_long else -x[0])

    eff_sl    = stop_loss
    rem_alloc = 100
    realized  = 0.0
    outcome   = exit_price = None
    exit_bar  = 0
    hit       = [False] * len(targets)

    for i in range(1, len(df)):
        hi = float(df["High"].iloc[i])
        lo = float(df["Low"].iloc[i])

        sl_hit = (is_long and lo <= eff_sl) or (not is_long and hi >= eff_sl)
        if sl_hit:
            pnl = (eff_sl - fill_price) * (rem_alloc / 100)
            if not is_long:
                pnl = -pnl
            realized += pnl
            exit_price = eff_sl
            exit_bar   = i
            pnl_r      = realized / risk
            outcome    = "partial_win" if pnl_r > 0.05 else ("breakeven" if abs(pnl_r) < 0.05 else "loss")
            break

        for ti, (tp, alloc) in enumerate(targets):
            if hit[ti]:
                continue
            tp_hit = (is_long and hi >= tp) or (not is_long and lo <= tp)
            if tp_hit:
                pnl = (tp - fill_price) * (alloc / 100)
                if not is_long:
                    pnl = -pnl
                hit[ti]    = True
                realized  += pnl
                rem_alloc -= alloc
                exit_price = tp

        if rem_alloc <= 0:
            exit_bar = i
            outcome  = "win"
            break

    if outcome is None:
        return None   # data exhausted — trade still open

    pnl_r = realized / risk
    return {
        "exit_price": round(float(exit_price), 4),
        "exit_date":  df.index[exit_bar].strftime("%Y-%m-%d"),
        "outcome":    outcome,
        "pnl_r":      round(pnl_r, 3),
    }


def _simulate_trade(
    direction: str,
    entry_low: float,
    entry_high: float,
    stop_loss: float,
    tp1: float | None,
    tp2: float | None,
    tp1_alloc: int,
    tp2_alloc: int,
    df: pd.DataFrame,
    slippage_pct: float = 0.0,
) -> dict | None:
    """
    Walk df bars (index 0 = signal bar, 1.. = future bars) to simulate fill → outcome.

    Returns:
      None                       — expired (no fill within _ENTRY_TIMEOUT bars)
      {"data_exhausted": True, "fill_price": ..., "fill_date": ...}
                                 — filled but SL/TP not yet hit; trade still open
      full dict with exit fields — trade closed at SL or TP

    Uses _find_fill() + _find_exit() internally — no duplicated logic.
    """
    fill = _find_fill(df, direction, entry_low, entry_high, slippage_pct=slippage_pct)
    if fill is None:
        return None  # expired — never entered

    fill_price, fill_idx = fill
    fill_date    = df.index[fill_idx].strftime("%Y-%m-%d")
    df_from_fill = df.iloc[fill_idx:]

    exit_result = _find_exit(
        df=df_from_fill,
        direction=direction,
        fill_price=fill_price,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp1_alloc=tp1_alloc,
        tp2_alloc=tp2_alloc,
    )

    result: dict = {
        "fill_price":     round(float(fill_price), 4),
        "fill_date":      fill_date,
        "data_exhausted": exit_result is None,
    }
    if exit_result is not None:
        result.update(exit_result)
    return result


# ── Pending signal resolver ───────────────────────────────────────────────────

def _fetch_sim_df(ticker_yf: str, tf: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV using the same source chain as the live scanner:
      1. core.data.fetch (TwelveData / DATA_PROVIDER) if the ticker maps to a known asset
      2. Falls back to fetch_ohlcv (yfinance/stooq) for unrecognised tickers

    This ensures signals and simulations are evaluated against identical price data.
    """
    from core.config import get_asset_by_ticker
    from core.data import fetch as _core_fetch

    asset = get_asset_by_ticker(ticker_yf)
    if asset:
        return _core_fetch(asset["id"], tf, start, end)
    return fetch_ohlcv(ticker_yf, tf, start, end, source="auto")


def _fetch_pending_df(ticker_yf: str, tf: str, sig_date: str) -> pd.DataFrame | None:
    """Fetch OHLCV from 2 days before signal_date to today. Returns None on error."""
    start = (date.fromisoformat(sig_date) - timedelta(days=2)).isoformat()
    end   = date.today().isoformat()
    df    = _fetch_sim_df(ticker_yf, tf, start, end)
    return df if not df.empty else None


def _fetch_filled_df(ticker_yf: str, tf: str, fill_date: str) -> pd.DataFrame | None:
    """Fetch OHLCV from 1 day before fill_date to today. Returns None on error."""
    start = (date.fromisoformat(fill_date) - timedelta(days=1)).isoformat()
    end   = date.today().isoformat()
    df    = _fetch_sim_df(ticker_yf, tf, start, end)
    return df if not df.empty else None


def simulate_pending(dry_run: bool = False) -> list[dict]:
    """
    Two-phase resolver for live journal signals.

    Phase 1 — pending → filled or expired:
      Walks bars from the signal date looking for a fill within _ENTRY_TIMEOUT bars.
      If filled: writes status='filled' with fill_price/fill_date.
      If timed out with no fill: writes status='expired'.
      If not enough bars yet: leaves as pending (returns 'in_progress').

    Phase 2 — filled → closed:
      For every filled trade (including those just promoted in Phase 1), walks bars
      from the fill date looking for SL or TP. If hit: writes final outcome.
      If no hit yet (data exhausted): leaves as filled (returns 'active').

    dry_run=True: compute outcomes but do not write to journal.
    Returns list of result dicts (one per trade processed).
    """
    results: list[dict] = []
    open_trades = get_open_trades()

    # Skip trades managed by the MT5 terminal — their status is synced by mt5_monitor.
    open_trades = [t for t in open_trades if not t.get("mt5_ticket")]

    # ── Phase 1: pending → filled or expired ──────────────────────────────────
    for t in (t for t in open_trades if t["status"] == "pending"):
        ticker_yf = _to_yf(t["ticker"])
        tf        = t.get("timeframe", "1d")
        sig_date  = t["signal_date"]
        strategy  = t.get("strategy", "mtf_trend")

        info = {
            "ticker":      t["ticker"],
            "signal_date": sig_date,
            "strategy":    strategy,
            "timeframe":   tf,
            "direction":   t["direction"],
        }

        try:
            df = _fetch_pending_df(ticker_yf, tf, sig_date)
        except Exception as e:
            results.append({**info, "status": "error", "reason": str(e)})
            continue

        if df is None:
            results.append({**info, "status": "no_data"})
            continue

        bar_dates  = df.index.normalize()
        sig_dt     = date.fromisoformat(sig_date)
        candidates = [i for i, d in enumerate(bar_dates) if d.date() >= sig_dt]
        if not candidates:
            results.append({**info, "status": "no_data"})
            continue

        sig_bar_i  = candidates[0]
        df_forward = df.iloc[sig_bar_i:]
        future_bars = len(df_forward) - 1

        if future_bars < 1:
            results.append({**info, "status": "in_progress"})
            continue

        fill = _find_fill(df_forward, t["direction"],
                          float(t["entry_low"]), float(t["entry_high"]),
                          slippage_pct=_get_slippage_pct())

        # Not enough history to declare "expired" — keep as in_progress
        if fill is None and future_bars < _ENTRY_TIMEOUT:
            results.append({**info, "status": "in_progress"})
            continue

        if fill is None:
            # Timed out — price never entered the zone
            if not dry_run:
                record_expired_direct(sig_date, t["ticker"], strategy, tf)
            results.append({**info, "status": "expired"})
            continue

        # Price entered zone → fill
        fill_price, fill_bar_idx = fill
        fill_bar    = df_forward.index[fill_bar_idx]
        fill_date   = fill_bar.strftime("%Y-%m-%d")
        fill_bar_ts = fill_bar.isoformat()
        if not dry_run:
            record_fill_direct(sig_date, t["ticker"], strategy, tf,
                               fill_price, fill_date, fill_bar_ts)
        results.append({**info, "status": "filled",
                        "fill_price": fill_price, "fill_date": fill_date,
                        "fill_bar_ts": fill_bar_ts})

    # ── Phase 2: filled → closed ───────────────────────────────────────────────
    # Re-read so trades just promoted from pending→filled in Phase 1 are included.
    if dry_run:
        filled_trades = [t for t in open_trades if t["status"] == "filled"]
    else:
        filled_trades = [t for t in get_open_trades() if t["status"] == "filled"]

    for t in filled_trades:
        ticker_yf  = _to_yf(t["ticker"])
        tf         = t.get("timeframe", "1d")
        sig_date   = t["signal_date"]
        strategy   = t.get("strategy", "mtf_trend")
        fill_date  = t["fill_date"]
        fill_price = float(t["fill_price"])

        info = {
            "ticker":      t["ticker"],
            "signal_date": sig_date,
            "strategy":    strategy,
            "timeframe":   tf,
            "direction":   t["direction"],
            "fill_price":  fill_price,
            "fill_date":   fill_date,
        }

        try:
            df = _fetch_filled_df(ticker_yf, tf, fill_date)
        except Exception as e:
            results.append({**info, "status": "active", "reason": str(e)})
            continue

        if df is None:
            results.append({**info, "status": "active"})
            continue

        bar_dates  = df.index.normalize()
        fill_dt    = date.fromisoformat(fill_date)
        candidates = [i for i, d in enumerate(bar_dates) if d.date() >= fill_dt]
        if not candidates or len(df.iloc[candidates[0]:]) < 2:
            results.append({**info, "status": "active"})
            continue

        df_from_fill = df.iloc[candidates[0]:]
        exit_result  = _find_exit(
            df=df_from_fill,
            direction=t["direction"],
            fill_price=fill_price,
            stop_loss=float(t["stop_loss"]),
            tp1=float(t["tp1"]) if t.get("tp1") else None,
            tp2=float(t["tp2"]) if t.get("tp2") else None,
            tp1_alloc=int(t.get("tp1_alloc") or 70),
            tp2_alloc=int(t.get("tp2_alloc") or 30),
        )

        if exit_result is None:
            # SL/TP not yet hit — trade still running
            results.append({**info, "status": "active"})
        else:
            if not dry_run:
                record_close_direct(
                    signal_date=sig_date,
                    ticker=t["ticker"],
                    strategy=strategy,
                    timeframe=tf,
                    outcome=exit_result["outcome"],
                    exit_price=exit_result["exit_price"],
                    exit_date=exit_result["exit_date"],
                    pnl_r=exit_result["pnl_r"],
                )
            results.append({**info, **exit_result})

    return results


# ── Gold historical backfill ──────────────────────────────────────────────────

def _fetch_backfill_df(ticker: str, tf: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV for backfill.
    - 1d / 4h / 1h  → yfinance/Stooq (years of history)
    - 5m / 15m / 30m → TwelveData via core.data.fetch (only source with intraday Gold)
      Requests from far in the past so TwelveData returns its maximum available window.
    """
    _SHORT_INTRADAY = {"5m", "15m", "30m"}
    if tf in _SHORT_INTRADAY:
        from core.data import fetch as _core_fetch
        # TwelveData ignores start_date when it predates available history;
        # passing "2000-01-01" ensures we get the full available window.
        return _core_fetch("gold", tf, "2000-01-01", end_date)
    else:
        return fetch_ohlcv(ticker, tf, start_date, end_date, source="auto")


def gold_backfill(
    start_date: str,
    end_date: str,
    strategies: list[str] | None = None,
    timeframes: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Replay all Gold strategies bar-by-bar historically.
    Records signals and simulates fill+close outcomes in the journal.

    strategies : list of strategy IDs to run (default: all non-ORB)
    timeframes : list of timeframes (default: ['1d', '4h'])
                 Include '30m' / '15m' / '5m' to use TwelveData intraday history.
    dry_run    : compute but do not write to journal
    verbose    : print progress to stdout
    Returns    : stats dict {signals, fills, expired, skipped, errors}
    """
    from p3_backtester.strategies.registry import build_from_config
    from p4_live.scanner import _compute_tech

    _STRATS = strategies or [
        "mtf_trend",
        "ema_continuation",
        "ema_pullback",
        "momentum_breakout",
    ]
    _TFS = timeframes or ["1d", "4h"]

    ticker = _GOLD_YF
    label  = _GOLD_LABEL
    stats  = {"signals": 0, "fills": 0, "expired": 0, "skipped": 0, "errors": 0}

    total_combos = len(_TFS) * len(_STRATS)
    done = 0

    for tf in _TFS:
        for strategy_id in _STRATS:
            done += 1
            if verbose:
                print(f"[{done}/{total_combos}] Gold {strategy_id} @ {tf} ...")

            try:
                df_full = _fetch_backfill_df(ticker, tf, start_date, end_date)
            except Exception as e:
                if verbose:
                    print(f"  ERROR fetching data: {e}")
                stats["errors"] += 1
                continue

            # Bar indices within the requested window.
            # Require at least _MIN_WARMUP bars before each signal bar so
            # indicators (EMA50, ATR14, etc.) are properly initialised.
            # For short intraday TFs the full df IS the available history,
            # so we use all bars (in_window = True for everything) and rely
            # solely on the _MIN_WARMUP guard.
            _MIN_WARMUP = 220
            _SHORT_INTRADAY = {"5m", "15m", "30m"}
            bar_dates = df_full.index.normalize()
            end_dt    = pd.Timestamp(end_date)

            if tf in _SHORT_INTRADAY:
                # Use everything up to end_date; first 220 bars are warmup
                in_window  = bar_dates <= end_dt
                window_idx = [i for i, m in enumerate(in_window) if m and i >= _MIN_WARMUP]
                if verbose and window_idx:
                    actual_start = df_full.index[window_idx[0]].strftime("%Y-%m-%d")
                    print(f"  TwelveData: {len(df_full)} bars available, "
                          f"window starts {actual_start} (after {_MIN_WARMUP}-bar warmup)")
            else:
                start_dt  = pd.Timestamp(start_date)
                in_window = (bar_dates >= start_dt) & (bar_dates <= end_dt)
                window_idx = [i for i, m in enumerate(in_window) if m and i >= _MIN_WARMUP]

            if not window_idx:
                if verbose:
                    print("  No bars in range.")
                continue

            strategy     = build_from_config(strategy_id)
            slippage_pct = _get_slippage_pct()
            exit_cutoff  = -1   # bar index — skip until previous trade closes

            sig_n = fill_n = exp_n = skip_n = 0

            for bar_i in window_idx:
                if bar_i <= exit_cutoff:
                    skip_n += 1
                    continue

                df_slice  = df_full.iloc[: bar_i + 1].copy()
                signal_dt = df_full.index[bar_i].strftime("%Y-%m-%d")

                tech = _compute_tech(df_slice, ticker, interval=tf)

                try:
                    result = strategy.generate_setups(tech, df_slice)
                except Exception:
                    continue

                if result is None or not result.setups:
                    continue

                setup = result.setups[0]
                risk  = abs(
                    (setup.entry_high if setup.direction == "long" else setup.entry_low)
                    - setup.stop_loss
                )
                if risk < 1e-8:
                    continue

                signal = {
                    "fired":         True,
                    "signal_date":   signal_dt,
                    "signal_bar_ts": df_full.index[bar_i].isoformat(),
                    "ticker":        ticker,
                    "label":         label,
                    "strategy":      strategy_id,
                    "timeframe":     tf,
                    "direction":   setup.direction,
                    "entry_low":   setup.entry_low,
                    "entry_high":  setup.entry_high,
                    "stop_loss":   setup.stop_loss,
                    "tp1":         setup.targets[0].price if setup.targets else None,
                    "tp2":         setup.targets[1].price if len(setup.targets) > 1 else None,
                    "tp1_alloc":   setup.targets[0].allocation_pct if setup.targets else 70,
                    "tp2_alloc":   setup.targets[1].allocation_pct if len(setup.targets) > 1 else 30,
                    "risk_pts":    round(risk, 4),
                    "rr":          setup.rr_ratio,
                    "confidence":  setup.confidence,
                    "rationale":   setup.rationale,
                    "price":       tech["current_price"],
                    "atr":         tech["atr_14"],
                    "rsi":         tech["rsi_14"],
                    "regime":      getattr(setup, "confidence_note", ""),
                }

                sig_n += 1

                if dry_run:
                    continue

                if not record_signal(signal):
                    continue  # already in journal

                # Simulate outcome using bars after the signal bar
                df_forward = df_full.iloc[bar_i:]
                sim = _simulate_trade(
                    direction=setup.direction,
                    entry_low=setup.entry_low,
                    entry_high=setup.entry_high,
                    stop_loss=setup.stop_loss,
                    tp1=setup.targets[0].price if setup.targets else None,
                    tp2=setup.targets[1].price if len(setup.targets) > 1 else None,
                    tp1_alloc=int(setup.targets[0].allocation_pct) if setup.targets else 70,
                    tp2_alloc=int(setup.targets[1].allocation_pct) if len(setup.targets) > 1 else 30,
                    df=df_forward,
                    slippage_pct=slippage_pct,
                )

                if sim is None:
                    # Never filled within timeout — expired
                    record_expired_direct(signal_dt, ticker, strategy_id, tf)
                    exp_n += 1
                    exit_cutoff = bar_i + _ENTRY_TIMEOUT
                elif sim.get("data_exhausted"):
                    # Filled but trade still running — record fill only, resolve later
                    record_fill_direct(signal_dt, ticker, strategy_id, tf,
                                       sim["fill_price"], sim["fill_date"])
                    fill_n += 1
                    exit_cutoff = bar_i + _ENTRY_TIMEOUT
                else:
                    record_fill_close_direct(
                        signal_date=signal_dt,
                        ticker=ticker,
                        fill_price=sim["fill_price"],
                        fill_date=sim["fill_date"],
                        outcome=sim["outcome"],
                        exit_price=sim["exit_price"],
                        exit_date=sim["exit_date"],
                        pnl_r=sim["pnl_r"],
                        strategy=strategy_id,
                        timeframe=tf,
                    )
                    fill_n += 1
                    # Lock out bars for the duration of this trade
                    fill_bar_abs = bar_i + next(
                        (j for j in range(1, len(df_forward))
                         if df_forward.index[j].strftime("%Y-%m-%d") >= sim["exit_date"]),
                        _ENTRY_TIMEOUT,
                    )
                    exit_cutoff = fill_bar_abs

            if verbose:
                print(f"  {sig_n} signals -> {fill_n} filled, {exp_n} expired, {skip_n} zone-skipped")

            stats["signals"] += sig_n
            stats["fills"]   += fill_n
            stats["expired"] += exp_n
            stats["skipped"] += skip_n

    return stats
