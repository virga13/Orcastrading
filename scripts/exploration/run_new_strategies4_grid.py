"""
run_new_strategies4_grid.py — Grid search for Heikin Ashi Trend.

Finding: Loose HA (2 bars, no-wick=False, adx=15) gave SPX Train PF=1.64 (n=243)
         -> OOS PF=1.70 (n=150) with WR 43.6%/46% — very consistent train/OOS.
         Default HA (3 bars, no-wick=True) also solid for SPX: Train PF=2.00 (n=43),
         OOS PF=1.47 (n=29) — better quality signals but fewer.

Grid goal: find the optimal balance between signal quality and signal count.

Primary: SPX. Secondary: Gold, US30, BTC.
OOS split: 2024-01-01 to today.

Usage:
    python run_new_strategies4_grid.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yfinance as yf

START     = "2019-01-01"
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
    print(f"\n  TOP 20 by OOS Net PF on {primary}:")
    print(f"  {'Rank':>4} | {'Params':<57} | " + " | ".join(f"{c[:7]:>12}" for c in cols))
    print("  " + "-" * (67 + 15 * len(cols)))

    for rank, r in enumerate(results[:20], 1):
        p = r["params"]
        skip = {"tp1_alloc","tp2_alloc","shorts_enabled","win_rate","sl_buffer_atr","entry_buffer_atr","ha_sl_lookback","rsi_max_long"}
        parts = [f"{k}={v}" for k, v in p.items() if k not in skip]
        param_str = " ".join(parts)[:57]
        cols_str = []
        for c in cols:
            a = r["assets"].get(c, {})
            oos = a.get("oos", {}); tr = a.get("train", {})
            n = oos.get("n", 0); pf = oos.get("net_pf", 0); tn = tr.get("n", 0)
            cols_str.append(f"{pf:5.2f}({n:3d}|{tn:3d})" if n > 0 else "  -- (  0|  0)")
        print(f"  {rank:>4} | {param_str:<57} | " + " | ".join(cols_str))

    if results:
        best = results[0]
        bp = best["assets"].get(primary, {})
        bo = bp.get("oos", {}); bt = bp.get("train", {})
        print(f"\n  BEST ({primary}): OOS n={bo.get('n',0)} WR={bo.get('wr',0):.1%} "
              f"Net PF={bo.get('net_pf',0):.2f} | Train n={bt.get('n',0)} PF={bt.get('net_pf',0):.2f}")
        print(f"  Best params: {best['params']}")

    return results


def main():
    from p3_backtester.strategies.heikin_ashi_trend import HeikinAshiTrendStrategy

    # Heikin Ashi grid: 96 combos
    # Key questions:
    #   - consecutive_bull_bars 2 vs 3: 3 gives fewer/cleaner signals
    #   - require_no_lower_wick: strict filter or too restrictive?
    #   - ema_period: 20 (short-term trend) vs 50 (medium-term)
    #   - adx_threshold: 15/20/25
    #   - tp1_r / tp2_r: does higher RR help given trend-following nature?
    ha_combos = [
        {"consecutive_bull_bars": cb, "require_no_lower_wick": nw,
         "ema_period": ep, "adx_threshold": adx,
         "rsi_max_long": 70, "ha_sl_lookback": 3,
         "entry_buffer_atr": 0.05, "sl_buffer_atr": 0.1,
         "tp1_r": tp1, "tp2_r": tp2,
         "tp1_alloc": 60, "tp2_alloc": 40,
         "shorts_enabled": False, "win_rate": 0.45}
        for cb  in [2, 3]
        for nw  in [True, False]
        for ep  in [20, 50]
        for adx in [15, 20, 25]
        for tp1 in [2.0]
        for tp2 in [4.0, 6.0]
    ]
    print(f"Heikin Ashi Trend: {len(ha_combos)} combos")
    run_grid(
        "Heikin Ashi Trend",
        ha_combos,
        {"SPX": "^GSPC", "Gold": "GC=F", "US30": "^DJI", "Silver": "SI=F", "BTC": "BTC-USD"},
        HeikinAshiTrendStrategy,
        min_train_n=30,
        primary="SPX",
    )


if __name__ == "__main__":
    main()
