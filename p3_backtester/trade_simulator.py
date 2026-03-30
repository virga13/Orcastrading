"""
trade_simulator.py — simulates a single TradingSetup forward from the signal bar.

Rules:
- Pessimistic fill: entry at worst end of zone (entry_high for long, entry_low for short)
- On same-candle SL+TP conflict: SL wins (conservative)
- Trailing SL to breakeven: once trailing_sl_to_breakeven price is touched, SL moves to fill_price
- Remaining allocation at SL hit is closed at SL price
- cost_r: round-trip transaction cost subtracted from pnl_r to give net_pnl_r
"""
import pandas as pd
from p1_analysis_engine.schema import TradingSetup
from p3_backtester.schema import TradeRecord, TargetResult


def simulate_trade(
    setup: TradingSetup,
    signal_bar_index: int,
    signal_bar_time: str,
    df: pd.DataFrame,
    entry_timeout_bars: int,
    claude_confidence: float,
    cost_r: float = 0.0,
) -> TradeRecord:
    """
    Simulate a single setup forward from signal_bar_index.
    Only reads df rows with index > signal_bar_index.
    Returns a fully populated TradeRecord.
    """
    is_long = setup.direction == "long"

    # ── Pass 1: Wait for fill ─────────────────────────────────────────────────
    fill_price: float | None = None
    fill_bar_index: int | None = None
    scan_end = min(signal_bar_index + 1 + entry_timeout_bars, len(df))

    for i in range(signal_bar_index + 1, scan_end):
        bar_high = float(df["High"].iloc[i])
        bar_low  = float(df["Low"].iloc[i])

        if is_long:
            # Long: bar must overlap entry zone (handles both pullback and breakout entries)
            if bar_high >= setup.entry_low and bar_low <= setup.entry_high:
                fill_price = min(setup.entry_high, bar_high)  # pessimistic: top of zone (capped at bar high)
                fill_bar_index = i
                break
        else:
            # Short: bar must overlap entry zone (handles both pullback and breakout entries)
            if bar_low <= setup.entry_high and bar_high >= setup.entry_low:
                fill_price = max(setup.entry_low, bar_low)    # pessimistic: bottom of zone (floored at bar low)
                fill_bar_index = i
                break

    if fill_price is None:
        return TradeRecord(
            signal_bar_time=signal_bar_time,
            signal_bar_index=signal_bar_index,
            setup_name=setup.name,
            direction=setup.direction,
            trade_type=setup.trade_type,
            priority=setup.priority,
            entry_low=setup.entry_low,
            entry_high=setup.entry_high,
            stop_loss=setup.stop_loss,
            trailing_sl_to_breakeven=setup.trailing_sl_to_breakeven,
            effective_sl=setup.stop_loss,
            outcome="EXPIRED",
            pnl_pts=0.0,
            pnl_r=0.0,
            cost_r=0.0,
            net_pnl_r=0.0,
            claude_win_rate_est=setup.win_rate_estimate,
            claude_ev_est=setup.ev,
            claude_rr=setup.rr_ratio,
            claude_confidence=claude_confidence,
        )

    # ── Pass 2: Simulate forward from fill bar ────────────────────────────────
    initial_risk = abs(fill_price - setup.stop_loss)
    effective_sl = setup.stop_loss
    breakeven_activated = False

    # Track remaining allocation across targets
    targets_sorted = sorted(setup.targets, key=lambda t: t.price if is_long else -t.price)
    target_results: list[TargetResult] = [
        TargetResult(price=t.price, allocation_pct=t.allocation_pct, hit=False, pnl_pts=0.0)
        for t in targets_sorted
    ]
    remaining_alloc = 100
    realized_pnl_pts = 0.0

    outcome: str | None = None
    close_bar = fill_bar_index

    for i in range(fill_bar_index + 1, len(df)):
        bar_high = float(df["High"].iloc[i])
        bar_low  = float(df["Low"].iloc[i])

        # ── Check trailing SL to breakeven activation ──
        if not breakeven_activated:
            if is_long and bar_high >= setup.trailing_sl_to_breakeven:
                effective_sl = fill_price
                breakeven_activated = True
            elif not is_long and bar_low <= setup.trailing_sl_to_breakeven:
                effective_sl = fill_price
                breakeven_activated = True

        # ── Check SL first (conservative: SL beats TP on same bar) ──
        sl_hit = (is_long and bar_low <= effective_sl) or \
                 (not is_long and bar_high >= effective_sl)

        if sl_hit:
            # Close remaining allocation at SL
            sl_pnl = (effective_sl - fill_price) * (remaining_alloc / 100)
            if not is_long:
                sl_pnl = -sl_pnl
            realized_pnl_pts += sl_pnl
            close_bar = i

            if realized_pnl_pts > 0:
                outcome = "PARTIAL_WIN"
            elif abs(realized_pnl_pts) < 1e-8:
                outcome = "BREAKEVEN"
            else:
                outcome = "LOSS"
            break

        # ── Check targets (in order, only unhit ones) ──
        for tr in target_results:
            if tr.hit:
                continue
            tp_hit = (is_long and bar_high >= tr.price) or \
                     (not is_long and bar_low <= tr.price)
            if tp_hit:
                alloc_frac = tr.allocation_pct / 100
                pnl = (tr.price - fill_price) * alloc_frac
                if not is_long:
                    pnl = -pnl
                tr.hit = True
                tr.pnl_pts = pnl
                realized_pnl_pts += pnl
                remaining_alloc -= tr.allocation_pct

        if remaining_alloc <= 0:
            close_bar = i
            outcome = "WIN"
            break

    # If we reached end of data without closing
    if outcome is None:
        # Close at last available close price
        last_close = float(df["Close"].iloc[-1])
        final_pnl = (last_close - fill_price) * (remaining_alloc / 100)
        if not is_long:
            final_pnl = -final_pnl
        realized_pnl_pts += final_pnl
        close_bar = len(df) - 1
        outcome = "WIN" if realized_pnl_pts > 0 else ("BREAKEVEN" if abs(realized_pnl_pts) < 1e-8 else "LOSS")

    pnl_r     = realized_pnl_pts / initial_risk if initial_risk > 1e-10 else 0.0
    net_pnl_r = round(pnl_r - cost_r, 4)

    return TradeRecord(
        signal_bar_time=signal_bar_time,
        signal_bar_index=signal_bar_index,
        setup_name=setup.name,
        direction=setup.direction,
        trade_type=setup.trade_type,
        priority=setup.priority,
        entry_low=setup.entry_low,
        entry_high=setup.entry_high,
        fill_price=fill_price,
        fill_bar_index=fill_bar_index,
        stop_loss=setup.stop_loss,
        trailing_sl_to_breakeven=setup.trailing_sl_to_breakeven,
        breakeven_activated=breakeven_activated,
        effective_sl=effective_sl,
        outcome=outcome,
        target_results=target_results,
        pnl_pts=round(realized_pnl_pts, 6),
        pnl_r=round(pnl_r, 4),
        cost_r=round(cost_r, 4),
        net_pnl_r=net_pnl_r,
        bars_held=close_bar - fill_bar_index if fill_bar_index is not None else None,
        claude_win_rate_est=setup.win_rate_estimate,
        claude_ev_est=setup.ev,
        claude_rr=setup.rr_ratio,
        claude_confidence=claude_confidence,
    )
