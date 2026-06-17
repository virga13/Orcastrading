"""
run_new_strategies2_backtest.py — Baseline backtest for Fibonacci Pullback, RSI Divergence, Inside Bar.

Assets: Gold, SPX, Bitcoin, US30, Silver (daily)
Period: 2019-01-01 to now
Train/OOS split: 2024-01-01

Usage:
    python run_new_strategies2_backtest.py
"""
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yfinance as yf

ASSETS = {
    "Gold":   "GC=F",
    "SPX":    "^GSPC",
    "BTC":    "BTC-USD",
    "US30":   "^DJI",
    "Silver": "SI=F",
}

STRATEGIES = ["fibonacci_pullback", "rsi_divergence", "inside_bar"]
INTERVAL   = "1d"
START      = "2019-01-01"
OOS_START  = pd.Timestamp("2024-01-01")
ENTRY_TO   = 3
COST_R     = 0.03


def _quick_stats(pnl: list[float], net_pnl: list[float]) -> dict:
    if not pnl:
        return dict(n=0, wr=0.0, pf=0.0, net_pf=0.0, avg_r=0.0, max_dd=0.0)
    wins = [r for r in pnl if r > 0]
    loss = [r for r in pnl if r <= 0]
    gp   = sum(wins);  gl = abs(sum(loss))
    pf   = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    nw = [r for r in net_pnl if r > 0];  nl = [r for r in net_pnl if r <= 0]
    np_ = sum(nw);  nl_ = abs(sum(nl))
    npf = np_ / nl_ if nl_ > 0 else (999.0 if np_ > 0 else 0.0)

    eq = peak = max_dd = 0.0
    for r in pnl:
        eq += r
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    return dict(n=len(pnl), wr=len(wins)/len(pnl), pf=pf, net_pf=npf,
                avg_r=sum(pnl)/len(pnl), max_dd=max_dd)


def _load_data(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=START, interval=INTERVAL,
                         auto_adjust=True, progress=False)
        df.columns = df.columns.get_level_values(0)
        if len(df) < 100:
            return None
        return df
    except Exception as e:
        print(f"  Download failed for {ticker}: {e}")
        return None


def _run_asset_strategy(strat, df: pd.DataFrame, ticker: str):
    from p3_backtester.trade_simulator import simulate_trade

    train_pnl: list[float] = []
    train_net: list[float] = []
    oos_pnl:   list[float] = []
    oos_net:   list[float] = []

    for i in range(80, len(df) - ENTRY_TO - 1):
        bar_ts = df.index[i]
        df_sl  = df.iloc[:i + 1]
        price  = float(df["Close"].iloc[i])
        tech   = {"current_price": price, "interval": INTERVAL,
                  "ticker": ticker, "atr_14": 0.0, "adx_14": 25.0,
                  "atr_pct": 0.0}

        out = strat.generate_setups(tech, df_sl)
        if out is None:
            continue

        for setup in out.setups:
            trade = simulate_trade(
                setup              = setup,
                signal_bar_index   = i,
                signal_bar_time    = bar_ts.isoformat(),
                df                 = df,
                entry_timeout_bars = ENTRY_TO,
                claude_confidence  = 0.5,
                cost_r             = COST_R,
            )
            if trade.outcome == "EXPIRED":
                continue
            bucket_pnl = oos_pnl if bar_ts >= OOS_START else train_pnl
            bucket_net = oos_net if bar_ts >= OOS_START else train_net
            bucket_pnl.append(trade.pnl_r)
            bucket_net.append(trade.net_pnl_r)

    return _quick_stats(train_pnl, train_net), _quick_stats(oos_pnl, oos_net)


def main():
    from p3_backtester.strategies.registry import build_from_config

    all_rows: list[dict] = []

    hdr = (f"{'Strategy':<20} {'Asset':<8} {'TR n':>5} {'TR WR':>6} "
           f"{'TR PF':>7} {'TR Net':>8}  "
           f"{'OOS n':>5} {'OOS WR':>7} {'OOS PF':>8} {'OOS Net':>9}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for strat_id in STRATEGIES:
        strat = build_from_config(strat_id)

        for asset_name, ticker in ASSETS.items():
            df = _load_data(ticker)
            if df is None:
                print(f"  {strat_id:<20} {asset_name:<8}  -- no data --")
                continue

            try:
                tr, oo = _run_asset_strategy(strat, df, ticker)
            except Exception as exc:
                print(f"  {strat_id:<20} {asset_name:<8}  ERROR: {exc}")
                import traceback; traceback.print_exc()
                continue

            row = dict(strategy=strat_id, asset=asset_name, train=tr, oos=oo)
            all_rows.append(row)

            print(
                f"{strat_id:<20} {asset_name:<8} "
                f"{tr['n']:5}  {tr['wr']:5.1%}  {tr['pf']:6.2f}   {tr['net_pf']:7.2f}   "
                f"{oo['n']:5}  {oo['wr']:6.1%}  {oo['pf']:7.2f}   {oo['net_pf']:8.2f}"
            )

        print(sep)

    print("\nOOS WINNERS (Net PF > 1.0, sorted best to worst):")
    winners = [r for r in all_rows if r["oos"]["net_pf"] > 1.0]
    winners.sort(key=lambda r: r["oos"]["net_pf"], reverse=True)
    if not winners:
        print("  None profitable OOS with default params -- optimize via grid search.")
    for w in winners:
        oo = w["oos"]
        print(f"  {w['strategy']:<20} {w['asset']:<8} "
              f"OOS: {oo['n']} trades  {oo['wr']:.1%} WR  Net PF {oo['net_pf']:.2f}  "
              f"MaxDD -{oo['max_dd']:.1f}R")


if __name__ == "__main__":
    main()
