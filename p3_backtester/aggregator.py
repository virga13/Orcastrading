"""
aggregator.py — computes RunStats, equity curve, and drawdown from TradeRecords.
Also handles SQLite persistence.
"""
import sqlite3
import json
import math
from pathlib import Path

from p3_backtester.schema import BacktestConfig, TradeRecord, RunStats, CategoryStats, SignalRecord, WalkForwardWindow

RESULTS_DIR = Path(__file__).parent / "results"


def db_path(config: BacktestConfig) -> Path:
    safe = config.ticker.replace("=", "").replace("-", "").replace("^", "")
    return RESULTS_DIR / f"backtest_{safe}_{config.interval}_{config.start_date}_{config.end_date}.db"


def save_to_db(
    config: BacktestConfig,
    signals: list[SignalRecord],
    trades: list[TradeRecord],
    stats: RunStats,
) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = db_path(config)
    conn = sqlite3.connect(str(path))
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            bar_index INTEGER, bar_time TEXT, close REAL,
            directional_bias TEXT, bias_strength TEXT, confidence_score REAL,
            n_setups INTEGER, cache_file TEXT
        );
        CREATE TABLE IF NOT EXISTS trades (
            signal_bar_time TEXT, signal_bar_index INTEGER,
            setup_name TEXT, direction TEXT, trade_type TEXT, priority TEXT,
            entry_low REAL, entry_high REAL, fill_price REAL, fill_bar_index INTEGER,
            stop_loss REAL, trailing_sl_to_breakeven REAL,
            breakeven_activated INTEGER, effective_sl REAL,
            outcome TEXT, pnl_pts REAL, pnl_r REAL, bars_held INTEGER,
            claude_win_rate_est REAL, claude_ev_est REAL, claude_rr REAL, claude_confidence REAL
        );
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY, value TEXT
        );
        DELETE FROM signals;
        DELETE FROM trades;
        DELETE FROM run_metadata;
    """)

    cur.executemany(
        "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?)",
        [(s.bar_index, s.bar_time, s.close, s.directional_bias,
          s.bias_strength, s.confidence_score, s.n_setups, s.cache_file)
         for s in signals],
    )

    cur.executemany(
        "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(t.signal_bar_time, t.signal_bar_index, t.setup_name, t.direction,
          t.trade_type, t.priority, t.entry_low, t.entry_high,
          t.fill_price, t.fill_bar_index, t.stop_loss, t.trailing_sl_to_breakeven,
          int(t.breakeven_activated), t.effective_sl, t.outcome,
          t.pnl_pts, t.pnl_r, t.bars_held,
          t.claude_win_rate_est, t.claude_ev_est, t.claude_rr, t.claude_confidence)
         for t in trades],
    )

    cur.execute("INSERT OR REPLACE INTO run_metadata VALUES ('stats', ?)",
                (stats.model_dump_json(),))
    cur.execute("INSERT OR REPLACE INTO run_metadata VALUES ('config', ?)",
                (config.model_dump_json(),))

    conn.commit()
    conn.close()
    return path


def compute_stats(
    config: BacktestConfig,
    signals: list[SignalRecord],
    trades: list[TradeRecord],
    strategy_name: str = "Rule Engine",
) -> RunStats:
    filled = [t for t in trades if t.outcome != "EXPIRED"]
    expired = [t for t in trades if t.outcome == "EXPIRED"]

    def category_stats(subset: list[TradeRecord]) -> CategoryStats:
        if not subset:
            return CategoryStats(
                n_trades=0, n_wins=0, n_losses=0, n_breakeven=0, n_expired=0,
                win_rate=0.0, avg_pnl_r=0.0, profit_factor=0.0,
                avg_claude_win_rate=0.0, avg_claude_ev=0.0,
            )
        f = [t for t in subset if t.outcome != "EXPIRED"]
        wins   = [t for t in f if t.outcome in ("WIN", "PARTIAL_WIN")]
        losses = [t for t in f if t.outcome == "LOSS"]
        be     = [t for t in f if t.outcome == "BREAKEVEN"]

        gross_profit = sum(t.pnl_r for t in wins)
        gross_loss   = abs(sum(t.pnl_r for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        return CategoryStats(
            n_trades=len(f),
            n_wins=len(wins),
            n_losses=len(losses),
            n_breakeven=len(be),
            n_expired=len([t for t in subset if t.outcome == "EXPIRED"]),
            win_rate=round(len(wins) / len(f), 4) if f else 0.0,
            avg_pnl_r=round(sum(t.pnl_r for t in f) / len(f), 4) if f else 0.0,
            profit_factor=round(pf, 3),
            avg_claude_win_rate=round(sum(t.claude_win_rate_est for t in f) / len(f), 4) if f else 0.0,
            avg_claude_ev=round(sum(t.claude_ev_est for t in f) / len(f), 4) if f else 0.0,
        )

    wins   = [t for t in filled if t.outcome in ("WIN", "PARTIAL_WIN")]
    losses = [t for t in filled if t.outcome == "LOSS"]
    gp     = sum(t.pnl_r for t in wins)
    gl     = abs(sum(t.pnl_r for t in losses))
    pf     = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    # Net (after costs)
    net_gp = sum(t.net_pnl_r for t in wins)
    net_gl = abs(sum(t.net_pnl_r for t in losses))
    net_pf = net_gp / net_gl if net_gl > 0 else (999.0 if net_gp > 0 else 0.0)

    # Sharpe on R-multiples
    if len(filled) > 1:
        mean_r = sum(t.pnl_r for t in filled) / len(filled)
        var_r  = sum((t.pnl_r - mean_r) ** 2 for t in filled) / (len(filled) - 1)
        sharpe = (mean_r / math.sqrt(var_r)) * math.sqrt(len(filled)) if var_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown on equity curve
    equity = compute_equity_curve(filled)
    drawdown = compute_max_drawdown(equity)

    # Category breakdowns
    by_type = {
        tt: category_stats([t for t in trades if t.trade_type == tt])
        for tt in ("scalp", "intraday", "swing")
    }
    by_dir = {
        d: category_stats([t for t in trades if t.direction == d])
        for d in ("long", "short")
    }
    by_pri = {
        p: category_stats([t for t in trades if t.priority == p])
        for p in ("primary", "secondary", "conditional")
    }

    from p3_backtester.walk_forward import kelly_from_stats
    avg_net_pnl_r = round(sum(t.net_pnl_r for t in filled) / len(filled), 4) if filled else 0.0

    stats = RunStats(
        ticker=config.ticker,
        interval=config.interval,
        start_date=config.start_date,
        end_date=config.end_date,
        signal_mode=config.signal_mode,
        strategy_name=strategy_name,
        total_signals=len(signals),
        total_setups_generated=sum(s.n_setups for s in signals),
        total_trades_filled=len(filled),
        total_trades_expired=len(expired),
        actual_win_rate=round(len(wins) / len(filled), 4) if filled else 0.0,
        actual_avg_pnl_r=round(sum(t.pnl_r for t in filled) / len(filled), 4) if filled else 0.0,
        actual_profit_factor=round(pf, 3),
        actual_avg_net_pnl_r=avg_net_pnl_r,
        actual_net_profit_factor=round(net_pf, 3),
        max_drawdown_r=round(drawdown, 4),
        sharpe_r=round(sharpe, 4),
        avg_claude_win_rate=round(sum(t.claude_win_rate_est for t in filled) / len(filled), 4) if filled else 0.0,
        avg_claude_ev=round(sum(t.claude_ev_est for t in filled) / len(filled), 4) if filled else 0.0,
        by_trade_type=by_type,
        by_direction=by_dir,
        by_priority=by_pri,
    )
    stats.kelly_25pct = kelly_from_stats(stats, 0.25)
    return stats


def compute_equity_curve(filled_trades: list[TradeRecord]) -> list[float]:
    """Cumulative R-multiple curve, ordered by fill_bar_index."""
    ordered = sorted(filled_trades, key=lambda t: (t.fill_bar_index or 0))
    curve = []
    cumulative = 0.0
    for t in ordered:
        cumulative += t.pnl_r
        curve.append(round(cumulative, 4))
    return curve


def compute_max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd
