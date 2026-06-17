"""
run_new_strategies4_backtest.py — Baseline backtest for Heikin Ashi Trend,
Parabolic SAR Flip, and Williams %R Reversal.

Train: 2019-01-01 to 2023-12-31
OOS:   2024-01-01 to today
Assets: SPX, Gold, US30, Silver, BTC (daily)

Usage:
    python run_new_strategies4_backtest.py
"""
import sys
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
    nw = [r for r in net_pnl if r > 0]; nl_neg = [r for r in net_pnl if r <= 0]
    np_ = sum(nw); nl_ = abs(sum(nl_neg))
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
    for i in range(100, len(df) - ENTRY_TO - 1):
        bar_ts = df.index[i]
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
    from p3_backtester.strategies.heikin_ashi_trend import HeikinAshiTrendStrategy
    from p3_backtester.strategies.parabolic_sar_flip import ParabolicSARFlipStrategy
    from p3_backtester.strategies.williams_r_reversal import WilliamsRReversalStrategy

    print("Loading data...")
    data = {label: _load(ticker) for label, ticker in ASSETS.items()}
    print("Data loaded.\n")

    strategies = [
        ("Heikin Ashi Trend (default)", HeikinAshiTrendStrategy()),
        ("Heikin Ashi Trend (loose: 2 bars, no-wick=False)", HeikinAshiTrendStrategy(
            consecutive_bull_bars=2, require_no_lower_wick=False, adx_threshold=15)),
        ("Parabolic SAR Flip (default)", ParabolicSARFlipStrategy()),
        ("Parabolic SAR Flip (adx=15)", ParabolicSARFlipStrategy(adx_threshold=15)),
        ("Williams %R Reversal (default)", WilliamsRReversalStrategy()),
        ("Williams %R Reversal (oversold=-70)", WilliamsRReversalStrategy(oversold_level=-70.0)),
        ("Williams %R Reversal (adx=15)", WilliamsRReversalStrategy(adx_threshold=15)),
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
