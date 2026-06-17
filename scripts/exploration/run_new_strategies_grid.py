"""
run_new_strategies_grid.py — Grid search for supertrend and macd_crossover.

Sweeps key parameters over Gold / SPX / BTC / Nasdaq daily (2019-now).
Train: 2019-2023  |  OOS: 2024-now

Results sorted by mean OOS Net PF across assets (only combos profitable in-sample).

Usage:
    python run_new_strategies_grid.py
    python run_new_strategies_grid.py --strategy supertrend
    python run_new_strategies_grid.py --strategy macd_crossover
"""
import sys
import argparse
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yfinance as yf

# ── Config ───────────────────────────────────────────────────────────────────

ASSETS = {
    "Gold":   "GC=F",
    "SPX":    "^GSPC",
    "BTC":    "BTC-USD",
    "Nasdaq": "^NDX",
}

INTERVAL  = "1d"
START     = "2019-01-01"
OOS_START = pd.Timestamp("2024-01-01")
ENTRY_TO  = 3
COST_R    = 0.03

GRIDS = {
    "supertrend": {
        "atr_period":  [7, 10, 14],
        "multiplier":  [2.0, 3.0, 4.0],
        "tp1_r":       [1.5, 2.0, 3.0],
        "tp2_r":       [3.0, 5.0, 7.0],
    },
    "macd_crossover": {
        "macd_fast":    [8, 12],
        "macd_slow":    [21, 26],
        "rsi_max_long": [60, 65, 70],
        "tp1_r":        [1.5, 2.0, 3.0],
        "tp2_r":        [3.0, 5.0],
    },
}

FIXED = {
    "supertrend": dict(
        sl_buffer_atr=0.05, htf_filter=True,
        tp1_alloc=60, tp2_alloc=40,
        shorts_enabled=True, win_rate=0.45,
    ),
    "macd_crossover": dict(
        macd_signal=9, rsi_period=14, rsi_min_short=35,
        require_above_zero=False, volume_mult=1.0,
        htf_filter=True, sl_lookback=10, sl_buffer_atr=0.1,
        tp1_alloc=60, tp2_alloc=40,
        shorts_enabled=True, win_rate=0.45,
    ),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stats(pnl: list, net: list) -> dict:
    if not pnl:
        return dict(n=0, wr=0.0, pf=0.0, net_pf=0.0, max_dd=0.0)
    wins = [r for r in pnl if r > 0];  loss = [r for r in pnl if r <= 0]
    gp = sum(wins);  gl = abs(sum(loss))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    nw = [r for r in net if r > 0];  nl = [r for r in net if r <= 0]
    npf = sum(nw) / abs(sum(nl)) if nl else (999.0 if nw else 0.0)
    eq = peak = dd = 0.0
    for r in pnl:
        eq += r; peak = max(peak, eq); dd = max(dd, peak - eq)
    return dict(n=len(pnl), wr=len(wins)/len(pnl), pf=pf, net_pf=npf, max_dd=dd)


def _load(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=START, interval=INTERVAL,
                         auto_adjust=True, progress=False)
        df.columns = df.columns.get_level_values(0)
        return df if len(df) >= 100 else None
    except Exception:
        return None


def _run(strat, df: pd.DataFrame, ticker: str):
    from p3_backtester.trade_simulator import simulate_trade
    tr_p, tr_n, oo_p, oo_n = [], [], [], []
    for i in range(60, len(df) - ENTRY_TO - 1):
        bar_ts = df.index[i]
        tech   = {"current_price": float(df["Close"].iloc[i]),
                  "interval": INTERVAL, "ticker": ticker,
                  "atr_14": 0.0, "adx_14": 25.0, "atr_pct": 0.0}
        out = strat.generate_setups(tech, df.iloc[:i + 1])
        if out is None:
            continue
        for setup in out.setups:
            t = simulate_trade(setup, i, bar_ts.isoformat(), df, ENTRY_TO, 0.5, COST_R)
            if t.outcome == "EXPIRED":
                continue
            (oo_p if bar_ts >= OOS_START else tr_p).append(t.pnl_r)
            (oo_n if bar_ts >= OOS_START else tr_n).append(t.net_pnl_r)
    return _stats(tr_p, tr_n), _stats(oo_p, oo_n)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_grid(strat_id: str):
    from p3_backtester.strategies.registry import build_from_config

    grid  = GRIDS[strat_id]
    fixed = FIXED[strat_id]
    keys  = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n{'='*70}")
    print(f"  {strat_id.upper()} — {len(combos)} combos x {len(ASSETS)} assets")
    print(f"  Train 2019-2023 | OOS 2024-now | {INTERVAL} data")
    print(f"{'='*70}")

    # Pre-load all data
    print("Loading data ...")
    data = {}
    for name, ticker in ASSETS.items():
        df = _load(ticker)
        if df is not None:
            data[name] = (ticker, df)
            print(f"  {name}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    print()

    results = []

    for ci, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        strat  = build_from_config(strat_id, {**fixed, **params})

        per_asset = {}
        for name, (ticker, df) in data.items():
            try:
                tr, oo = _run(strat, df, ticker)
                per_asset[name] = (tr, oo)
            except Exception:
                per_asset[name] = (None, None)

        # Summary across assets
        oos_net_pfs = [oo["net_pf"] for _, oo in per_asset.values()
                       if oo and oo["n"] >= 3]
        train_pfs   = [tr["pf"] for tr, _ in per_asset.values()
                       if tr and tr["n"] >= 3]
        if not oos_net_pfs:
            continue

        mean_oos_npf   = sum(oos_net_pfs) / len(oos_net_pfs)
        mean_train_pf  = sum(train_pfs) / len(train_pfs) if train_pfs else 0.0
        min_oos_npf    = min(oos_net_pfs)

        results.append({
            "params": params,
            "per_asset": per_asset,
            "mean_oos_npf": mean_oos_npf,
            "min_oos_npf": min_oos_npf,
            "mean_train_pf": mean_train_pf,
        })

        param_str = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"#{ci:3}  {param_str:<55}  "
              f"train_PF={mean_train_pf:.2f}  mean_OOS={mean_oos_npf:.2f}  min_OOS={min_oos_npf:.2f}")

    # Sort: train PF >= 1.0, best mean OOS
    viable = [r for r in results if r["mean_train_pf"] >= 1.0]
    viable.sort(key=lambda r: r["mean_oos_npf"], reverse=True)

    print(f"\nTOP 10 (train PF >= 1.0, by mean OOS Net PF)  [{len(viable)} viable / {len(results)} total]")
    print("-" * 110)
    for r in viable[:10]:
        p  = r["params"]
        param_str = "  ".join(f"{k}={v}" for k, v in p.items())
        print(f"  {param_str}")
        for name, (tr, oo) in r["per_asset"].items():
            if tr is None:
                continue
            print(f"    {name:<8}  train: {tr['n']:3} fills  {tr['wr']:.0%} WR  PF {tr['pf']:.2f}"
                  f"  |  OOS: {oo['n']:3} fills  {oo['wr']:.0%} WR  Net PF {oo['net_pf']:.2f}"
                  f"  MaxDD -{oo['max_dd']:.1f}R")
        print(f"    --> mean OOS Net PF: {r['mean_oos_npf']:.2f}  "
              f"min OOS Net PF: {r['min_oos_npf']:.2f}  "
              f"mean train PF: {r['mean_train_pf']:.2f}")
        print()

    if viable:
        best = viable[0]
        print(f"Best params: {best['params']}")

    return viable[0] if viable else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["supertrend", "macd_crossover", "both"],
                        default="both")
    args = parser.parse_args()

    targets = (["supertrend", "macd_crossover"] if args.strategy == "both"
                else [args.strategy])
    for strat_id in targets:
        run_grid(strat_id)


if __name__ == "__main__":
    main()
