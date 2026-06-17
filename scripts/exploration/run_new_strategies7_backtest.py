"""
run_new_strategies7_backtest.py — Batch 7: ROC Momentum, Keltner Breakout, Consecutive Highs.

Strategies to baseline:
  - ROC Momentum: zero-line cross, signal-rich (30-60/year)
  - Keltner Breakout: KC upper-band close, momentum expansion
  - Consecutive Highs: N higher closes, raw candle momentum

Assets: SPX, Gold, US30, Silver, BTC
Train: 2019-2023, OOS: 2024-2026

Usage:
    python run_new_strategies7_backtest.py
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


def _run(strat, df, ticker):
    from p3_backtester.trade_simulator import simulate_trade
    ts = pd.Timestamp("2019-01-01")
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


def run_variant(name, strat, data):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  {'Asset':>8} | {'Train n':>7} | {'Train WR':>8} | {'Train PF':>8} | {'OOS n':>5} | {'OOS WR':>6} | {'OOS PF':>7}")
    print(f"  {'-'*70}")
    for label, df in data.items():
        if df is None:
            print(f"  {label:>8} | NO DATA")
            continue
        tr, oos = _run(strat, df, label)
        print(f"  {label:>8} | {tr['n']:>7} | {tr['wr']:>7.1%} | {tr['net_pf']:>8.2f} | "
              f"{oos['n']:>5} | {oos['wr']:>6.1%} | {oos['net_pf']:>7.2f}")


def main():
    print("Loading data...")
    data = {label: _load(ticker) for label, ticker in ASSETS.items()}
    print("Data loaded.\n")

    from p3_backtester.strategies.roc_momentum import ROCMomentumStrategy
    from p3_backtester.strategies.keltner_breakout import KeltnerBreakoutStrategy
    from p3_backtester.strategies.consecutive_highs import ConsecutiveHighsStrategy

    # ── ROC Momentum variants ─────────────────────────────────────────────────
    roc_variants = [
        ("ROC Momentum (roc=10, ema=50, adx=20)",
         ROCMomentumStrategy(roc_period=10, ema_period=50, adx_threshold=20, tp1_r=2.0, tp2_r=4.0)),
        ("ROC Momentum (roc=10, ema=20, adx=15)",
         ROCMomentumStrategy(roc_period=10, ema_period=20, adx_threshold=15, tp1_r=2.0, tp2_r=4.0)),
        ("ROC Momentum (roc=5, ema=20, adx=15)",
         ROCMomentumStrategy(roc_period=5, ema_period=20, adx_threshold=15, tp1_r=2.0, tp2_r=4.0)),
        ("ROC Momentum (roc=14, ema=50, adx=20, min=0.5%)",
         ROCMomentumStrategy(roc_period=14, ema_period=50, adx_threshold=20, min_roc_pct=0.5, tp1_r=2.0, tp2_r=6.0)),
    ]

    for name, strat in roc_variants:
        run_variant(name, strat, data)

    # ── Keltner Breakout variants ─────────────────────────────────────────────
    kc_variants = [
        ("Keltner Breakout (ema=20, mult=1.5, adx=20)",
         KeltnerBreakoutStrategy(ema_period=20, atr_mult=1.5, adx_threshold=20, tp1_r=2.0, tp2_r=4.0)),
        ("Keltner Breakout (ema=20, mult=2.0, adx=15)",
         KeltnerBreakoutStrategy(ema_period=20, atr_mult=2.0, adx_threshold=15, tp1_r=2.0, tp2_r=4.0)),
        ("Keltner Breakout (ema=50, mult=1.5, adx=15)",
         KeltnerBreakoutStrategy(ema_period=50, atr_mult=1.5, adx_threshold=15, tp1_r=2.0, tp2_r=6.0)),
        ("Keltner Breakout (ema=20, mult=1.0, adx=20, no fresh)",
         KeltnerBreakoutStrategy(ema_period=20, atr_mult=1.0, adx_threshold=20, require_fresh_cross=False, tp1_r=2.0, tp2_r=4.0)),
    ]

    for name, strat in kc_variants:
        run_variant(name, strat, data)

    # ── Consecutive Highs variants ────────────────────────────────────────────
    ch_variants = [
        ("Consecutive Highs (n=3, ema=50, adx=20)",
         ConsecutiveHighsStrategy(n_bars=3, ema_period=50, adx_threshold=20, tp1_r=2.0, tp2_r=4.0)),
        ("Consecutive Highs (n=3, ema=20, adx=15)",
         ConsecutiveHighsStrategy(n_bars=3, ema_period=20, adx_threshold=15, tp1_r=2.0, tp2_r=4.0)),
        ("Consecutive Highs (n=4, ema=50, adx=15)",
         ConsecutiveHighsStrategy(n_bars=4, ema_period=50, adx_threshold=15, tp1_r=2.0, tp2_r=4.0)),
        ("Consecutive Highs (n=2, ema=50, adx=20)",
         ConsecutiveHighsStrategy(n_bars=2, ema_period=50, adx_threshold=20, tp1_r=2.0, tp2_r=4.0)),
        ("Consecutive Highs (n=3, ema=20, adx=15, min=0.5%)",
         ConsecutiveHighsStrategy(n_bars=3, ema_period=20, adx_threshold=15, min_gain_pct=0.5, tp1_r=2.0, tp2_r=4.0)),
    ]

    for name, strat in ch_variants:
        run_variant(name, strat, data)


if __name__ == "__main__":
    main()
