"""
run_new_strategies6_backtest.py — Baseline backtest for Ichimoku Cloud,
Triple EMA Cross, and Bollinger Band Bounce.

Train: 2019-01-01 to 2023-12-31
OOS:   2024-01-01 to today
Assets: SPX, Gold, US30, Silver, BTC (daily)

Usage:
    python run_new_strategies6_backtest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yfinance as yf

START     = "2015-01-01"   # Ichimoku needs 52-bar senkou_b — extend start
OOS_START = pd.Timestamp("2024-01-01")
ENTRY_TO  = 3
COST_R    = 0.03

ASSETS = {
    "SPX":    "^GSPC",
    "Gold":   "GC=F",
    "US30":   "^DJI",
    "Silver": "SI=F",
    "BTC":    "BTC-USD",
}


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


def _run_strat(strat, df, label):
    from p3_backtester.trade_simulator import simulate_trade
    tr_pnl, tr_net, oos_pnl, oos_net = [], [], [], []
    train_start = pd.Timestamp("2019-01-01")
    for i in range(150, len(df) - ENTRY_TO - 1):
        bar_ts = df.index[i]
        if bar_ts < train_start:
            continue
        out = strat.generate_setups(
            {"current_price": float(df["Close"].iloc[i]), "interval": "1d",
             "ticker": label, "atr_14": 0.0, "atr_pct": 0.0},
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


def _print_results(name, results):
    cols = list(ASSETS.keys())
    header = f"  {'Asset':>8} | {'Train n':>8} | {'Train WR':>9} | {'Train PF':>9} | {'OOS n':>7} | {'OOS WR':>8} | {'OOS PF':>8}"
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for asset in cols:
        tr, oos = results.get(asset, ({}, {}))
        if not tr:
            print(f"  {asset:>8} | {'N/A':>8}")
            continue
        print(f"  {asset:>8} | {tr['n']:>8} | {tr['wr']:>9.1%} | {tr['net_pf']:>9.2f} | "
              f"{oos['n']:>7} | {oos['wr']:>8.1%} | {oos['net_pf']:>8.2f}")


def main():
    from p3_backtester.strategies.ichimoku_cloud import IchimokuCloudStrategy
    from p3_backtester.strategies.triple_ema_cross import TripleEMACrossStrategy
    from p3_backtester.strategies.bb_bounce import BBBounceStrategy

    print("Loading data...")
    data = {label: _load(ticker) for label, ticker in ASSETS.items()}
    print("Data loaded.\n")

    strategies = [
        # Ichimoku variants
        ("Ichimoku TK Cross (above cloud, adx=20)", IchimokuCloudStrategy()),
        ("Ichimoku TK Cross (above cloud, adx=15)", IchimokuCloudStrategy(adx_threshold=15)),
        ("Ichimoku TK Cross (no cloud req, adx=15)", IchimokuCloudStrategy(
            require_above_cloud=False, adx_threshold=15)),

        # Triple EMA variants
        ("Triple EMA 9/21/50 (adx=20)", TripleEMACrossStrategy()),
        ("Triple EMA 9/21/50 (adx=15)", TripleEMACrossStrategy(adx_threshold=15)),
        ("Triple EMA 4/9/21 (adx=15)", TripleEMACrossStrategy(
            ema_fast=4, ema_medium=9, ema_slow=21, adx_threshold=15)),
        ("Triple EMA 5/13/50 (adx=15)", TripleEMACrossStrategy(
            ema_fast=5, ema_medium=13, ema_slow=50, adx_threshold=15)),

        # BB Bounce variants
        ("BB Bounce (default, adx=20, rsi<=65)", BBBounceStrategy()),
        ("BB Bounce (adx=15, rsi<=70)", BBBounceStrategy(adx_threshold=15, rsi_max_long=70)),
        ("BB Bounce (require_low_touch=True, adx=15)", BBBounceStrategy(
            require_low_touch=True, adx_threshold=15, rsi_max_long=70)),
    ]

    for name, strat in strategies:
        results = {}
        for label, df in data.items():
            if df is None:
                continue
            try:
                tr, oos = _run_strat(strat, df, label)
                results[label] = (tr, oos)
            except Exception as e:
                print(f"  Error on {label}: {e}")
        _print_results(name, results)


if __name__ == "__main__":
    main()
