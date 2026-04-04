from pydantic import BaseModel, Field
from typing import Literal, Optional


class BacktestConfig(BaseModel):
    ticker: str
    interval: str                  # same values as P1: 1m, 5m, 15m, 30m, 1h, 1d, 1wk
    start_date: str                # ISO date string e.g. "2024-01-01"
    end_date: str                  # ISO date string e.g. "2024-12-31"
    signal_mode: Literal["every-n-bars", "session-open", "key-level-touch"] = "session-open"
    every_n_bars: int = 10         # used only for every-n-bars mode
    entry_timeout_bars: int = 20   # bars to wait for price to enter entry zone
    model: str = "claude-sonnet-4-6"
    force_regenerate: bool = False # re-run Pass 1 even if cache exists
    top_adx_pct: float = 1.0      # keep only top N% of signals by ADX (1.0 = keep all)


class SignalRecord(BaseModel):
    bar_index: int
    bar_time: str                  # ISO timestamp of the signal bar
    close: float
    directional_bias: str
    bias_strength: str
    confidence_score: float
    n_setups: int
    cache_file: str                # relative path to the JSON cache file


class TargetResult(BaseModel):
    price: float
    allocation_pct: int
    hit: bool
    pnl_pts: float


class TradeRecord(BaseModel):
    signal_bar_time: str
    signal_bar_index: int
    setup_name: str
    direction: Literal["long", "short"]
    trade_type: Literal["scalp", "intraday", "swing"]
    priority: Literal["primary", "secondary", "conditional"]

    # Entry
    entry_low: float
    entry_high: float
    fill_price: Optional[float] = None     # None = EXPIRED
    fill_bar_index: Optional[int] = None

    # Levels
    stop_loss: float
    trailing_sl_to_breakeven: float
    breakeven_activated: bool = False
    effective_sl: float                    # stop_loss or fill_price after BE activation

    # Exit
    outcome: Literal["WIN", "LOSS", "BREAKEVEN", "PARTIAL_WIN", "EXPIRED"]
    target_results: list[TargetResult] = []
    pnl_pts: float = 0.0                   # sum of realized points across all partial closes
    pnl_r: float = 0.0                     # pnl normalized by initial risk (fill - stop_loss)
    bars_held: Optional[int] = None

    # Claude's estimates (for comparison)
    claude_win_rate_est: float
    claude_ev_est: float
    claude_rr: float
    claude_confidence: float

    # Transaction costs
    cost_r: float = 0.0                     # round-trip spread+slippage in R-multiples
    net_pnl_r: float = 0.0                  # pnl_r - cost_r (what you actually keep)


class CategoryStats(BaseModel):
    n_trades: int
    n_wins: int
    n_losses: int
    n_breakeven: int
    n_expired: int
    win_rate: float
    avg_pnl_r: float
    profit_factor: float
    avg_claude_win_rate: float
    avg_claude_ev: float


class WalkForwardWindow(BaseModel):
    name: str                       # "train" | "validation" | "test"
    start_date: str
    end_date: str
    n_fills: int
    win_rate: float
    avg_pnl_r: float
    avg_net_pnl_r: float            # after costs
    profit_factor: float
    net_profit_factor: float        # after costs
    max_drawdown_r: float
    sharpe_r: float
    kelly_25pct: float              # 25% Kelly fraction


class RunStats(BaseModel):
    ticker: str
    interval: str
    start_date: str
    end_date: str
    signal_mode: str
    strategy_name: str = "Rule Engine"

    total_signals: int
    total_setups_generated: int
    total_trades_filled: int
    total_trades_expired: int

    actual_win_rate: float
    actual_avg_pnl_r: float
    actual_profit_factor: float
    actual_avg_net_pnl_r: float = 0.0    # after costs
    actual_net_profit_factor: float = 0.0
    max_drawdown_r: float
    sharpe_r: float

    avg_claude_win_rate: float     # mean of claude_win_rate_est across all filled trades
    avg_claude_ev: float           # mean of claude_ev_est

    by_trade_type: dict[str, CategoryStats]
    by_direction: dict[str, CategoryStats]
    by_priority: dict[str, CategoryStats]

    walk_forward: list[WalkForwardWindow] = []
    kelly_25pct: float = 0.0

    # Risk-adjusted performance metrics
    calmar_ratio: float = 0.0           # total_net_R / abs(max_drawdown_R) — higher is better
    max_drawdown_duration: int = 0      # consecutive filled trades spent in drawdown
    max_win_streak: int = 0             # longest consecutive winning streak
    max_loss_streak: int = 0            # longest consecutive losing streak
