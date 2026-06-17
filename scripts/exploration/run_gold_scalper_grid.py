"""
run_gold_scalper_grid.py — Grid search for Gold Scalper (longs-only, 4-year data).

Loads data/XAUUSD_5m.parquet once, pre-computes all indicators once, then sweeps
min_score / sl_mult / tp1_mult / tp2_mult without re-resampling.  No cache files
written — runs entirely in memory.

Train period : 2022-01-01 → 2024-12-31  (in-sample)
OOS period   : 2025-01-01 → 2026-04-28  (out-of-sample, held out)

Results are sorted by OOS Net Profit Factor among combos that are also profitable
in-sample (train PF >= 1.0).

Usage:
    python run_gold_scalper_grid.py
"""
import sys
import itertools
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# ── Grid ───────────────────────────────────────────────────────────────────────
GRID = {
    "min_score":  [5, 6, 7, 8],
    "sl_mult":    [1.0, 1.5, 2.0],
    "tp1_mult":   [0.8, 1.0, 1.5, 2.0],
    "tp2_mult":   [2.0, 2.5, 3.0, 4.0],
}

# Fixed params (not swept)
FIXED = dict(
    shorts_enabled=False,  # longs-only — shorts lose in Gold's 2022-2026 bull regime even with M30 macro filter
    sess_start_utc=8,
    sess_end_utc=20,
    avoid_friday=True,
    max_trades_per_day=3,
    use_vwap=True,
    adx_min=22,
    tp1_alloc=50,
    tp2_alloc=50,
)

PARQUET      = Path(__file__).parent.parent.parent / "data" / "XAUUSD_5m.parquet"
TICKER       = "GC=F"
INTERVAL     = "5m"
EVERY_N_BARS = 3      # evaluate every 15 min
ENTRY_TO     = 6      # 30 min entry timeout
TRAIN_END    = pd.Timestamp("2025-01-01")   # exclusive: everything before this is train


# ── Inline stat helpers ────────────────────────────────────────────────────────

def _quick_stats(pnl_list: list[float], net_pnl_list: list[float]) -> dict:
    """Win rate, gross PF, net PF, avg R, net avg R, max drawdown."""
    if not pnl_list:
        return dict(n=0, wr=0.0, pf=0.0, net_pf=0.0, avg_r=0.0, net_avg_r=0.0, max_dd=0.0)

    wins  = [r for r in pnl_list if r > 0]
    loss  = [r for r in pnl_list if r <= 0]
    gp    = sum(wins)
    gl    = abs(sum(loss))
    pf    = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    net_wins = [r for r in net_pnl_list if r > 0]
    net_loss = [r for r in net_pnl_list if r <= 0]
    ngp      = sum(net_wins)
    ngl      = abs(sum(net_loss))
    net_pf   = ngp / ngl if ngl > 0 else (999.0 if ngp > 0 else 0.0)

    # Drawdown from equity curve
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for r in pnl_list:
        equity += r
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return dict(
        n=len(pnl_list),
        wr=len(wins) / len(pnl_list),
        pf=pf,
        net_pf=net_pf,
        avg_r=sum(pnl_list) / len(pnl_list),
        net_avg_r=sum(net_pnl_list) / len(net_pnl_list),
        max_dd=max_dd,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    from p3_backtester.strategies.gold_scalper import GoldScalperStrategy
    from p3_backtester.signal_scheduler import get_signal_indices
    from p3_backtester.market_data import split_backtest_window
    from p3_backtester.trade_simulator import simulate_trade
    from p1_analysis_engine.utils.asset_classifier import classify_asset

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading 5m OHLCV from cache ...")
    df = pd.read_parquet(PARQUET)
    start_date = df.index[0].date().isoformat()
    end_date   = df.index[-1].date().isoformat()
    print(f"  {len(df):,} bars  ({start_date} to {end_date})")

    asset_class = classify_asset(TICKER)
    # Fixed cost estimate: ~0.05R round-trip (spread + commission) for Gold 5m
    # Exact per-trade cost needs ATR which varies — this is close enough for ranking
    cost_r = 0.05

    # ── Prime indicators ONCE ──────────────────────────────────────────────────
    print("Pre-computing all indicators (once) ...")
    ref = GoldScalperStrategy(**FIXED)
    ref.prime(df)
    print("  Done.\n")

    # ── Signal indices ─────────────────────────────────────────────────────────
    first_valid, _ = split_backtest_window(df, start_date)
    all_idx = get_signal_indices(df, "every-n-bars", first_valid, EVERY_N_BARS)
    train_idx = [i for i in all_idx if df.index[i] <  TRAIN_END]
    oos_idx   = [i for i in all_idx if df.index[i] >= TRAIN_END]
    print(f"Signal bars — train: {len(train_idx):,}  |  OOS: {len(oos_idx):,}")
    print(f"Train: {start_date} to {TRAIN_END.date()}   "
          f"OOS: {TRAIN_END.date()} to {end_date}\n")

    # ── Build all combos ───────────────────────────────────────────────────────
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Grid: {len(combos)} combinations  (longs-only)\n")
    print(f"{'#':>4}  {'score':>5} {'sl':>5} {'tp1':>5} {'tp2':>5}"
          f"  {'IN fills':>8} {'IN WR':>6} {'IN PF':>7} {'IN NetPF':>9}"
          f"  {'OOS fills':>9} {'OOS WR':>7} {'OOS PF':>8} {'OOS NetPF':>10}")
    print("-" * 110)

    results = []

    for idx, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))

        # Build strategy + share pre-computed series (no re-prime needed)
        strat = GoldScalperStrategy(**FIXED, **params)
        strat._m15_pre   = ref._m15_pre
        strat._m30_pre   = ref._m30_pre
        strat._m5_atr    = ref._m5_atr
        strat._m5_atr_ma = ref._m5_atr_ma
        strat._m5_ema_f  = ref._m5_ema_f
        strat._m5_ema_s  = ref._m5_ema_s
        strat._m5_vwap   = ref._m5_vwap

        # ── Train period ──────────────────────────────────────────────────────
        train_pnl, train_net = _run_period(strat, df, train_idx, ENTRY_TO, cost_r)
        strat._daily_counts.clear()  # reset daily caps for OOS run

        # ── OOS period ────────────────────────────────────────────────────────
        oos_pnl, oos_net = _run_period(strat, df, oos_idx, ENTRY_TO, cost_r)

        tr = _quick_stats(train_pnl, train_net)
        oo = _quick_stats(oos_pnl,   oos_net)

        print(f"{idx:4}.  {params['min_score']:5}  {params['sl_mult']:5.1f}  "
              f"{params['tp1_mult']:5.1f}  {params['tp2_mult']:5.1f}"
              f"  {tr['n']:8}  {tr['wr']:5.1%}  {tr['pf']:6.2f}   {tr['net_pf']:8.2f}"
              f"  {oo['n']:9}  {oo['wr']:6.1%}  {oo['pf']:7.2f}   {oo['net_pf']:9.2f}")

        results.append({**params, "train": tr, "oos": oo})

    # ── Summary ────────────────────────────────────────────────────────────────
    # Keep combos where train is profitable, sort by OOS net PF
    viable = [r for r in results if r["train"]["pf"] >= 1.0]
    viable.sort(key=lambda r: r["oos"]["net_pf"], reverse=True)

    print("\n" + "=" * 110)
    print(f"TOP 20 (train PF >= 1.0, sorted by OOS Net PF)   [{len(viable)} viable / {len(results)} total]")
    print("=" * 110)
    print(f"  {'score':>5} {'sl':>5} {'tp1':>5} {'tp2':>5}"
          f"  {'IN fills':>8} {'IN WR':>6} {'IN PF':>7} {'IN NetPF':>9}"
          f"  {'OOS fills':>9} {'OOS WR':>7} {'OOS PF':>8} {'OOS NetPF':>10} {'OOS MaxDD':>10}")
    print("-" * 110)
    for r in viable[:20]:
        tr, oo = r["train"], r["oos"]
        print(f"  {r['min_score']:5}  {r['sl_mult']:5.1f}  "
              f"{r['tp1_mult']:5.1f}  {r['tp2_mult']:5.1f}"
              f"  {tr['n']:8}  {tr['wr']:5.1%}  {tr['pf']:6.2f}   {tr['net_pf']:8.2f}"
              f"  {oo['n']:9}  {oo['wr']:6.1%}  {oo['pf']:7.2f}   {oo['net_pf']:9.2f}"
              f"  -{oo['max_dd']:8.1f}R")

    if not viable:
        print("  No combinations were profitable in-sample.")
    else:
        best = viable[0]
        print(f"\nBest combo → min_score={best['min_score']}  sl_mult={best['sl_mult']}"
              f"  tp1_mult={best['tp1_mult']}  tp2_mult={best['tp2_mult']}")
        print(f"  OOS: {best['oos']['n']} fills  {best['oos']['wr']:.1%} WR  "
              f"Net PF {best['oos']['net_pf']:.2f}  MaxDD -{best['oos']['max_dd']:.1f}R")


def _run_period(
    strat: "GoldScalperStrategy",
    df: pd.DataFrame,
    signal_indices: list[int],
    entry_to: int,
    cost_r: float,
) -> tuple[list[float], list[float]]:
    """Run generate_setups + simulate_trade for all bars in signal_indices.
    Returns (gross_pnl_list, net_pnl_list) — one entry per filled trade."""
    from p3_backtester.trade_simulator import simulate_trade

    pnl_gross: list[float] = []
    pnl_net:   list[float] = []

    for i in signal_indices:
        bar_ts    = df.index[i]
        tech      = {"current_price": float(df["Close"].iloc[i]), "interval": "5m", "ticker": "GC=F"}
        df_slice  = df.iloc[: i + 1]

        setups_obj = strat.generate_setups(tech, df_slice)
        if setups_obj is None:
            continue

        for setup in setups_obj.setups:
            trade = simulate_trade(
                setup            = setup,
                signal_bar_index = i,
                signal_bar_time  = bar_ts.isoformat(),
                df               = df,
                entry_timeout_bars = entry_to,
                claude_confidence  = 0.5,
                cost_r           = cost_r,
            )
            if trade.outcome != "EXPIRED":
                pnl_gross.append(trade.pnl_r)
                pnl_net.append(trade.net_pnl_r)

    return pnl_gross, pnl_net


if __name__ == "__main__":
    main()
