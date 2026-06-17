"""
run_new_strategies3_grid.py — Grid search for Donchian Breakout, Stochastic Bounce, Engulfing Candle.

Focus:
  - Donchian: optimize entry_period, tp ratios across Gold/SPX/US30/BTC/Silver (96 combos)
  - Stochastic: loosen oversold/overbought to 25-40 range to get more signals (72 combos)
  - Engulfing: loosen ema_zone_atr to 2.0-4.0 to get more signals (48 combos)

Scoring: best OOS Net PF on primary asset, require train_n >= 15

Usage:
    python run_new_strategies3_grid.py
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


def run_grid(name, combos, assets_def, klass, min_train_n=15, primary=None):
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
        if idx % 30 == 0:
            print(f"  {idx}/{len(combos)} combos done...")

    def score(r):
        a = r["assets"].get(primary, {})
        tr = a.get("train", {})
        oos = a.get("oos", {})
        if tr.get("n", 0) < min_train_n:
            return -999
        return oos.get("net_pf", 0.0)

    results.sort(key=score, reverse=True)

    cols = list(assets_def.keys())
    print(f"\n  TOP 15 by OOS Net PF on {primary}:")
    print(f"  {'Rank':>4} | {'Params':<55} | " + " | ".join(f"{c[:7]:>12}" for c in cols))
    print("  " + "-" * (65 + 15 * len(cols)))

    for rank, r in enumerate(results[:15], 1):
        p = r["params"]
        # Build compact param string
        param_parts = []
        for k, v in p.items():
            if k not in ("tp1_alloc","tp2_alloc","shorts_enabled","win_rate","sl_buffer_atr",
                         "htf_filter","volume_mult","require_close_above_prev_open",
                         "require_mother_direction","stoch_smooth","sl_lookback"):
                param_parts.append(f"{k}={v}")
        param_str = " ".join(param_parts)[:55]
        cols_str = []
        for c in cols:
            a = r["assets"].get(c, {})
            oos = a.get("oos", {})
            tr  = a.get("train", {})
            n   = oos.get("n", 0)
            pf  = oos.get("net_pf", 0)
            tn  = tr.get("n", 0)
            cols_str.append(f"{pf:5.2f}({n:3d}|{tn:3d})" if n > 0 else "  -- (  0|  0)")
        print(f"  {rank:>4} | {param_str:<55} | " + " | ".join(cols_str))

    # Print best combo summary
    if results:
        best = results[0]
        best_p = best["assets"].get(primary, {})
        best_oos = best_p.get("oos", {})
        best_tr  = best_p.get("train", {})
        print(f"\n  BEST ({primary}): OOS n={best_oos.get('n',0)} WR={best_oos.get('wr',0):.1%} "
              f"Net PF={best_oos.get('net_pf',0):.2f} | Train n={best_tr.get('n',0)} PF={best_tr.get('net_pf',0):.2f}")
        print(f"  Best params: {best['params']}")

    return results


def main():
    from p3_backtester.strategies.donchian_breakout import DonchianBreakoutStrategy
    from p3_backtester.strategies.stochastic_bounce import StochasticBounceStrategy
    from p3_backtester.strategies.engulfing_candle import EngulfingCandleStrategy

    # ── Donchian Breakout (96 combos) ─────────────────────────────────────────
    donchian_combos = [
        {"entry_period": ep, "exit_period": xp, "adx_threshold": adx,
         "max_breakout_atr": mb, "tp1_r": tp1, "tp2_r": tp2,
         "sl_buffer_atr": 0.1, "htf_filter": True,
         "tp1_alloc": 50, "tp2_alloc": 50, "shorts_enabled": False, "win_rate": 0.40}
        for ep  in [20, 30, 50]
        for xp  in [10]
        for adx in [15, 20]
        for mb  in [1.5, 2.5]
        for tp1 in [2.0, 3.0]
        for tp2 in [5.0, 8.0]
    ]
    print(f"Donchian: {len(donchian_combos)} combos")
    run_grid(
        "Donchian Channel Breakout",
        donchian_combos,
        {"SPX": "^GSPC", "Gold": "GC=F", "US30": "^DJI", "Silver": "SI=F", "BTC": "BTC-USD"},
        DonchianBreakoutStrategy,
        min_train_n=15,
        primary="SPX",
    )

    # ── Stochastic Bounce (72 combos) — looser thresholds ────────────────────
    stoch_combos = [
        {"stoch_k": sk, "stoch_d": 3, "stoch_smooth": 3,
         "oversold_level": ov, "overbought_level": ob,
         "ema_period": ep, "adx_threshold": adx,
         "sl_lookback": 10, "sl_buffer_atr": 0.15,
         "tp1_r": tp1, "tp2_r": tp2,
         "tp1_alloc": 60, "tp2_alloc": 40, "shorts_enabled": False, "win_rate": 0.45}
        for sk  in [14, 21]
        for ov  in [25.0, 30.0, 35.0]
        for ob  in [65.0, 70.0, 75.0]
        for ep  in [20, 50]
        for adx in [15, 20]
        for tp1 in [1.5]
        for tp2 in [3.0, 5.0]
    ]
    print(f"\nStochastic: {len(stoch_combos)} combos")
    run_grid(
        "Stochastic Bounce",
        stoch_combos,
        {"SPX": "^GSPC", "Gold": "GC=F", "BTC": "BTC-USD", "US30": "^DJI"},
        StochasticBounceStrategy,
        min_train_n=15,
        primary="SPX",
    )

    # ── Engulfing Candle (72 combos) — looser zone filter ────────────────────
    engulf_combos = [
        {"ema_period": ep, "ema_zone_atr": ez, "adx_threshold": adx,
         "rsi_max_long": 70, "rsi_min_short": 30,
         "volume_mult": 1.0, "require_close_above_prev_open": True,
         "htf_filter": htf, "sl_buffer_atr": 0.1,
         "tp1_r": tp1, "tp2_r": tp2,
         "tp1_alloc": 60, "tp2_alloc": 40, "shorts_enabled": False, "win_rate": 0.45}
        for ep  in [20, 50]
        for ez  in [2.0, 3.0, 5.0]
        for adx in [15, 20]
        for htf in [True, False]
        for tp1 in [2.0]
        for tp2 in [4.0, 6.0]
    ]
    print(f"\nEngulfing: {len(engulf_combos)} combos")
    run_grid(
        "Engulfing Candle",
        engulf_combos,
        {"SPX": "^GSPC", "Gold": "GC=F", "US30": "^DJI", "BTC": "BTC-USD"},
        EngulfingCandleStrategy,
        min_train_n=10,
        primary="SPX",
    )


if __name__ == "__main__":
    main()
