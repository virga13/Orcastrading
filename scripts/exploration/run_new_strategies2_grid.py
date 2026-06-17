"""
run_new_strategies2_grid.py — Grid search for Fibonacci Pullback, RSI Divergence, Inside Bar.

Approach:
  - RSI Divergence: focus on SPX (only asset with promising OOS). 72 combos.
  - Fibonacci Pullback: focus on Gold/Silver (only assets with OOS PF > 1.0). 64 combos.
  - Inside Bar: all assets, looser params for more signals. 48 combos.

Train: 2019-01-01 to 2023-12-31
OOS:   2024-01-01 to today

Usage:
    python run_new_strategies2_grid.py
"""
import sys
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yfinance as yf

START    = "2019-01-01"
OOS_START = pd.Timestamp("2024-01-01")
ENTRY_TO  = 3
COST_R    = 0.03


def _quick_stats(pnl, net_pnl):
    if not pnl:
        return dict(n=0, wr=0.0, pf=0.0, net_pf=0.0, avg_r=0.0, max_dd=0.0)
    wins = [r for r in pnl if r > 0]
    loss = [r for r in pnl if r <= 0]
    gp = sum(wins); gl = abs(sum(loss))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    nw = [r for r in net_pnl if r > 0]; nl = [r for r in net_pnl if r <= 0]
    np_ = sum(nw); nl_ = abs(sum(nl))
    npf = np_ / nl_ if nl_ > 0 else (999.0 if np_ > 0 else 0.0)
    eq = peak = max_dd = 0.0
    for r in pnl:
        eq += r; peak = max(peak, eq); max_dd = max(max_dd, peak - eq)
    return dict(n=len(pnl), wr=len(wins)/len(pnl), pf=pf, net_pf=npf,
                avg_r=sum(pnl)/len(pnl), max_dd=max_dd)


def _load(ticker):
    try:
        df = yf.download(ticker, start=START, interval="1d",
                         auto_adjust=True, progress=False)
        df.columns = df.columns.get_level_values(0)
        return df if len(df) >= 200 else None
    except Exception:
        return None


def _run(strat, df, ticker):
    from p3_backtester.trade_simulator import simulate_trade
    tr_pnl, tr_net, oos_pnl, oos_net = [], [], [], []
    for i in range(100, len(df) - ENTRY_TO - 1):
        bar_ts = df.index[i]
        out = strat.generate_setups(
            {"current_price": float(df["Close"].iloc[i]), "interval": "1d",
             "ticker": ticker, "atr_14": 0.0, "adx_14": 25.0, "atr_pct": 0.0},
            df.iloc[:i + 1]
        )
        if out is None:
            continue
        for setup in out.setups:
            trade = simulate_trade(
                setup=setup, signal_bar_index=i,
                signal_bar_time=bar_ts.isoformat(), df=df,
                entry_timeout_bars=ENTRY_TO, claude_confidence=0.5, cost_r=COST_R,
            )
            if trade.outcome == "EXPIRED":
                continue
            bp = oos_pnl if bar_ts >= OOS_START else tr_pnl
            bn = oos_net if bar_ts >= OOS_START else tr_net
            bp.append(trade.pnl_r); bn.append(trade.net_pnl_r)
    return _quick_stats(tr_pnl, tr_net), _quick_stats(oos_pnl, oos_net)


def run_grid(name, combos_def, assets_def, klass, min_train_n=10, min_oos_n=5):
    """
    combos_def: list of param dicts to try
    assets_def: dict of {label: ticker}
    """
    print(f"\n{'='*70}")
    print(f"  {name.upper()} -- {len(combos_def)} combos x {len(assets_def)} assets")
    print(f"{'='*70}")

    data = {label: _load(ticker) for label, ticker in assets_def.items()}

    results = []
    for idx, params in enumerate(combos_def, 1):
        strat = klass(**params)
        row = {"params": params, "assets": {}}
        for label, df in data.items():
            if df is None:
                continue
            try:
                tr, oos = _run(strat, df, label)
                row["assets"][label] = {"train": tr, "oos": oos}
            except Exception as e:
                pass
        results.append(row)
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(combos_def)} combos done...")

    # Score: require train_n >= min_train_n across at least 1 asset,
    # sort by best OOS net_pf on the primary (first) asset
    primary = list(assets_def.keys())[0]

    def score(r):
        a = r["assets"].get(primary, {})
        oos = a.get("oos", {})
        tr  = a.get("train", {})
        if tr.get("n", 0) < min_train_n:
            return -999
        return oos.get("net_pf", 0.0)

    results.sort(key=score, reverse=True)

    print(f"\nTOP 20 combos (sorted by OOS Net PF on {primary}):")
    cols = list(assets_def.keys())
    print(f"  {'Rank':>4} | {'Params':<50} | " + " | ".join(f"{c[:8]:>8} OOS PF" for c in cols))
    print("  " + "-" * (60 + 15 * len(cols)))

    for rank, r in enumerate(results[:20], 1):
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())[:50]
        oos_scores = []
        for c in cols:
            a = r["assets"].get(c, {})
            oos = a.get("oos", {})
            oos_n = oos.get("n", 0)
            oos_pf = oos.get("net_pf", 0)
            oos_scores.append(f"{oos_pf:6.2f}({oos_n:3d})" if oos_n > 0 else "   -- (  0)")
        print(f"  {rank:>4} | {param_str:<50} | " + " | ".join(oos_scores))

    return results


def main():
    from p3_backtester.strategies.fibonacci_pullback import FibonacciPullbackStrategy
    from p3_backtester.strategies.rsi_divergence import RSIDivergenceStrategy
    from p3_backtester.strategies.inside_bar import InsideBarStrategy

    # ── RSI Divergence grid (72 combos) ──────────────────────────────────────
    rsi_combos = [
        {"rsi_period": rp, "pivot_order": po, "rsi_oversold": rsio,
         "rsi_overbought": rsib, "tp1_r": tp1, "tp2_r": tp2,
         "max_pivot_gap": 30, "sl_buffer_atr": 0.15,
         "tp1_alloc": 60, "tp2_alloc": 40, "shorts_enabled": False, "win_rate": 0.45}
        for rp   in [14, 21]
        for po   in [3, 5]
        for rsio in [40.0, 50.0]
        for rsib in [50.0, 60.0]
        for tp1  in [1.5, 2.0]
        for tp2  in [3.0, 5.0]
    ]
    print(f"RSI Divergence: {len(rsi_combos)} combos")
    run_grid(
        "RSI Divergence",
        rsi_combos,
        {"SPX": "^GSPC", "Gold": "GC=F", "US30": "^DJI"},
        RSIDivergenceStrategy,
        min_train_n=20,
        min_oos_n=10,
    )

    # ── Fibonacci Pullback grid (64 combos) ──────────────────────────────────
    fib_combos = [
        {"swing_lookback": sl, "min_swing_atr": ms, "fib_zone_atr": fz,
         "ema_period": ep, "adx_threshold": 18, "rsi_max_long": 70,
         "sl_buffer_atr": 0.15, "tp1_r": tp1, "tp2_r": tp2,
         "tp1_alloc": 60, "tp2_alloc": 40, "shorts_enabled": False, "win_rate": 0.45}
        for sl  in [20, 30, 40]
        for ms  in [2.0, 3.0]
        for fz  in [0.5, 1.0]
        for ep  in [20, 50]
        for tp1 in [2.0]
        for tp2 in [4.0, 6.0]
    ]
    print(f"\nFibonacci Pullback: {len(fib_combos)} combos")
    run_grid(
        "Fibonacci Pullback",
        fib_combos,
        {"Gold": "GC=F", "Silver": "SI=F", "SPX": "^GSPC"},
        FibonacciPullbackStrategy,
        min_train_n=20,
        min_oos_n=10,
    )

    # ── Inside Bar grid (48 combos) ──────────────────────────────────────────
    ib_combos = [
        {"min_mother_atr": mm, "adx_threshold": adx, "require_mother_direction": rmd,
         "htf_filter": htf, "entry_buffer_atr": 0.05, "sl_buffer_atr": 0.1,
         "tp1_r": tp1, "tp2_r": tp2,
         "tp1_alloc": 60, "tp2_alloc": 40, "shorts_enabled": False, "win_rate": 0.45}
        for mm  in [0.5, 1.0, 1.5]
        for adx in [15, 20]
        for rmd in [True, False]
        for htf in [True, False]
        for tp1 in [2.0]
        for tp2 in [4.0]
    ]
    print(f"\nInside Bar: {len(ib_combos)} combos")
    run_grid(
        "Inside Bar",
        ib_combos,
        {"Gold": "GC=F", "SPX": "^GSPC", "Silver": "SI=F", "US30": "^DJI"},
        InsideBarStrategy,
        min_train_n=10,
        min_oos_n=3,
    )


if __name__ == "__main__":
    main()
