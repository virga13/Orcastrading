"""
run_new_strategies7_grid.py — Grid search for ROC Momentum, Keltner Breakout, Consecutive Highs.

Run after reviewing batch 7 baseline results to find optimal params
for the strategies that show promise.

Primary asset: SPX for ROC and Keltner, Gold for Consecutive Highs.

Usage:
    python run_new_strategies7_grid.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yfinance as yf

START     = "2015-01-01"
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


def _load(ticker, start=None):
    try:
        df = yf.download(ticker, start=start or START, interval="1d",
                         auto_adjust=True, progress=False)
        df.columns = df.columns.get_level_values(0)
        return df if len(df) >= 200 else None
    except Exception:
        return None


def _run(strat, df, ticker, train_start=None):
    from p3_backtester.trade_simulator import simulate_trade
    ts = train_start or pd.Timestamp("2019-01-01")
    tr_pnl, tr_net, oos_pnl, oos_net = [], [], [], []
    for i in range(150, len(df) - ENTRY_TO - 1):
        bar_ts = df.index[i]
        if bar_ts < ts:
            continue
        out = strat.generate_setups(
            {"current_price": float(df["Close"].iloc[i]), "interval": "1d",
             "ticker": ticker, "atr_14": 0.0, "atr_pct": 0.0},
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


def run_grid(name, combos, assets_def, klass, min_train_n=20, primary=None):
    primary = primary or list(assets_def.keys())[0]
    print(f"\n{'='*72}")
    print(f"  {name.upper()} -- {len(combos)} combos x {len(assets_def)} assets  (primary: {primary})")
    print(f"{'='*72}")

    data = {label: _load(ticker) for label, ticker in assets_def.items()}

    results = []
    for idx, params in enumerate(combos, 1):
        strat = klass(**params)
        row = {"params": params, "assets": {}}
        for label, df in data.items():
            if df is None:
                continue
            try:
                tr, oos = _run(strat, df, label)
                row["assets"][label] = {"train": tr, "oos": oos}
            except Exception:
                pass
        results.append(row)
        if idx % 20 == 0:
            print(f"  {idx}/{len(combos)} combos done...")

    def score(r):
        a = r["assets"].get(primary, {})
        tr  = a.get("train", {})
        oos = a.get("oos", {})
        if tr.get("n", 0) < min_train_n:
            return -999
        return oos.get("net_pf", 0.0)

    results.sort(key=score, reverse=True)

    cols = list(assets_def.keys())
    top_n = min(20, len(results))
    print(f"\n  TOP {top_n} by OOS Net PF on {primary}:")
    print(f"  {'Rank':>4} | {'Params':<55} | " + " | ".join(f"{c[:7]:>12}" for c in cols))
    print("  " + "-" * (65 + 15 * len(cols)))

    for rank, r in enumerate(results[:top_n], 1):
        p = r["params"]
        skip = {"tp1_alloc", "tp2_alloc", "shorts_enabled", "win_rate",
                "sl_buffer_atr", "entry_buffer_atr", "rsi_max_long", "sl_lookback",
                "sl_lookback_extra", "require_fresh_cross"}
        parts = [f"{k}={v}" for k, v in p.items() if k not in skip]
        param_str = " ".join(parts)[:55]
        cols_str = []
        for c in cols:
            a = r["assets"].get(c, {})
            oos = a.get("oos", {}); tr = a.get("train", {})
            n = oos.get("n", 0); pf = oos.get("net_pf", 0); tn = tr.get("n", 0)
            cols_str.append(f"{pf:5.2f}({n:3d}|{tn:3d})" if n > 0 else "  -- (  0|  0)")
        print(f"  {rank:>4} | {param_str:<55} | " + " | ".join(cols_str))

    if results:
        best = results[0]
        bp = best["assets"].get(primary, {})
        bo = bp.get("oos", {}); bt = bp.get("train", {})
        wr_tr = bt.get("wr", 0); wr_oos = bo.get("wr", 0)
        print(f"\n  BEST ({primary}): Train n={bt.get('n',0)} WR={wr_tr:.1%} PF={bt.get('net_pf',0):.2f} "
              f"| OOS n={bo.get('n',0)} WR={wr_oos:.1%} PF={bo.get('net_pf',0):.2f}")
        print(f"  Best params: {best['params']}")

    return results


def main():
    from p3_backtester.strategies.roc_momentum import ROCMomentumStrategy
    from p3_backtester.strategies.keltner_breakout import KeltnerBreakoutStrategy
    from p3_backtester.strategies.consecutive_highs import ConsecutiveHighsStrategy

    # ── ROC Momentum grid (60 combos) ─────────────────────────────────────────
    roc_combos = [
        {"roc_period": rp, "min_roc_pct": mp, "ema_period": ep, "adx_threshold": adx,
         "rsi_max_long": 70, "sl_lookback": 5, "entry_buffer_atr": 0.05,
         "sl_buffer_atr": 0.1, "tp1_r": tp1, "tp2_r": tp2,
         "tp1_alloc": 60, "tp2_alloc": 40, "shorts_enabled": False, "win_rate": 0.45}
        for rp  in [5, 10, 14, 20]
        for mp  in [0.0, 0.3]
        for ep  in [20, 50]
        for adx in [15, 20]
        for tp1 in [2.0]
        for tp2 in [4.0, 6.0]
    ]
    print(f"ROC Momentum: {len(roc_combos)} combos")
    run_grid(
        "ROC Momentum",
        roc_combos,
        {"SPX": "^GSPC", "Gold": "GC=F", "US30": "^DJI", "Silver": "SI=F", "BTC": "BTC-USD"},
        ROCMomentumStrategy,
        min_train_n=20,
        primary="SPX",
    )

    # ── Keltner Breakout grid (48 combos) ─────────────────────────────────────
    kc_combos = [
        {"ema_period": ep, "atr_period": 14, "atr_mult": mult, "require_fresh_cross": rfc,
         "adx_threshold": adx, "rsi_max_long": 75, "entry_buffer_atr": 0.0,
         "sl_buffer_atr": 0.2, "tp1_r": tp1, "tp2_r": tp2,
         "tp1_alloc": 60, "tp2_alloc": 40, "shorts_enabled": False, "win_rate": 0.45}
        for ep   in [20, 50]
        for mult in [1.0, 1.5, 2.0]
        for rfc  in [True, False]
        for adx  in [15, 20]
        for tp1  in [2.0]
        for tp2  in [4.0, 6.0]
    ]
    print(f"\nKeltner Breakout: {len(kc_combos)} combos")
    run_grid(
        "Keltner Breakout",
        kc_combos,
        {"SPX": "^GSPC", "Gold": "GC=F", "US30": "^DJI", "Silver": "SI=F", "BTC": "BTC-USD"},
        KeltnerBreakoutStrategy,
        min_train_n=20,
        primary="SPX",
    )

    # ── Consecutive Highs grid (48 combos) ────────────────────────────────────
    ch_combos = [
        {"n_bars": n, "ema_period": ep, "adx_threshold": adx,
         "rsi_max_long": 70, "min_gain_pct": mg, "sl_lookback_extra": 2,
         "entry_buffer_atr": 0.05, "sl_buffer_atr": 0.1,
         "tp1_r": tp1, "tp2_r": tp2,
         "tp1_alloc": 60, "tp2_alloc": 40, "shorts_enabled": False, "win_rate": 0.45}
        for n   in [2, 3, 4, 5]
        for ep  in [20, 50]
        for adx in [15, 20]
        for mg  in [0.0, 0.3]
        for tp1 in [2.0]
        for tp2 in [4.0, 6.0]
    ]
    print(f"\nConsecutive Highs: {len(ch_combos)} combos")
    run_grid(
        "Consecutive Highs",
        ch_combos,
        {"SPX": "^GSPC", "Gold": "GC=F", "US30": "^DJI", "Silver": "SI=F", "BTC": "BTC-USD"},
        ConsecutiveHighsStrategy,
        min_train_n=20,
        primary="SPX",
    )


if __name__ == "__main__":
    main()
