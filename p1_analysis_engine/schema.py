from pydantic import BaseModel, Field
from typing import Literal, Optional


class KeyLevel(BaseModel):
    price: float
    label: str  # e.g. "support", "resistance", "fib_61.8", "52w_high"
    strength: Literal["weak", "moderate", "strong"]


class TechnicalSnapshot(BaseModel):
    current_price: float
    trend: Literal["uptrend", "downtrend", "sideways"]
    rsi_14: float
    macd_signal: Literal["bullish", "bearish", "neutral"]
    ema_alignment: Literal["bullish", "bearish", "mixed"]  # 20>50>200 = bullish
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    adx_14: float
    atr_14: float
    atr_pct: float  # ATR as % of price
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_position: Literal["above_upper", "upper", "middle", "lower", "below_lower"]
    stoch_k: float
    stoch_d: float
    stoch_signal: Literal["overbought", "oversold", "neutral"]
    cmf_20: float
    cmf_signal: Literal["accumulation", "distribution", "neutral"]
    volume_trend: Literal["increasing", "decreasing", "neutral"]


class MacroSnapshot(BaseModel):
    fed_funds_rate: Optional[float] = None
    yield_curve: Literal["normal", "inverted", "flat"]
    cpi_trend: Literal["rising", "falling", "stable"]
    vix_level: Optional[float] = None
    vix_regime: Literal["low", "elevated", "high"]  # <15, 15-25, >25
    usd_trend: Literal["strengthening", "weakening", "stable"]
    unemployment_rate: Optional[float] = None
    macro_bias: Literal["risk_on", "risk_off", "neutral"]


class NewsItem(BaseModel):
    title: str
    source: str
    published_at: str
    url: Optional[str] = None
    sentiment_hint: Optional[Literal["positive", "negative", "neutral"]] = None


class GeopoliticalRisk(BaseModel):
    risk_level: Literal["low", "moderate", "high", "extreme"]
    key_factors: list[str]
    relevant_headlines: list[str]


class ConfidenceBreakdown(BaseModel):
    technical_weight: float  # 0.0 - 1.0
    fundamental_weight: float
    geopolitical_weight: float
    note: str


class SetupTarget(BaseModel):
    price: float
    label: str              # e.g. "Target 1 (50%)", "Target 2 (30%)"
    allocation_pct: int     # % of position to close at this target


class TradingSetup(BaseModel):
    name: str               # "Setup A", "Setup B", "Setup C"
    label: str              # short description e.g. "Short scalp / bear flag"
    direction: Literal["long", "short"]
    trade_type: Literal["scalp", "intraday", "swing"]
    status: str             # "ACTIVE", "PRIMARY WATCH", "IF LEVEL BREAKS", etc.
    priority: Literal["primary", "secondary", "conditional"]
    rationale: str
    trigger: str            # exact condition to enter
    entry_low: float
    entry_high: float
    stop_loss: float
    trailing_sl_to_breakeven: float  # price level at which to move SL to entry
    targets: list[SetupTarget]
    rr_ratio: float = Field(ge=1.5)  # enforced minimum 1.5R
    win_rate_estimate: float = Field(ge=0.0, le=1.0)
    trade_duration: str     # e.g. "2-4 hours", "1-3 days", "1-2 weeks"
    ev: float               # expected value in price points
    profit_factor: float    # gross profit / gross loss
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_note: str


class InvalidationScenario(BaseModel):
    condition: str          # "Daily close below $X on high volume"
    description: str        # why this collapses the thesis
    price_trigger: float    # exact price level
    action: str             # "Stand aside entirely, no trades in either direction"


class DecisionTreeEntry(BaseModel):
    scenario: str           # "Price tags $X then bearish wick close below $Y"
    outcome: str            # "SHORT Setup A · $X entry"
    direction: Literal["long", "short", "no_trade"]
    setup_name: str         # which setup this triggers, or "NO TRADE"
    entry_price: Optional[float] = None


class SetupsOutput(BaseModel):
    """Trading setups and decision tree generated from BiasOutput."""
    asset: str
    current_price: float
    setups: list[TradingSetup]
    invalidation: InvalidationScenario
    decision_tree: list[DecisionTreeEntry]
    position_sizing_note: str   # ATR-based sizing warning


class FullAnalysisOutput(BaseModel):
    """Combined single-call schema: bias analysis + trade setups in one Claude call.
    Metadata fields (asset, asset_class, analysis_timestamp, current_price) are
    injected by the engine after the call — Claude does not fill them."""

    # ── Bias fields ──
    directional_bias: Literal["bullish", "bearish", "neutral"]
    bias_strength: Literal["weak", "moderate", "strong"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    key_levels: list[KeyLevel]
    nearest_support: float
    nearest_resistance: float
    primary_thesis: str
    bull_case: str
    bear_case: str
    key_risks: list[str]
    technical: TechnicalSnapshot
    macro: MacroSnapshot
    geopolitical: GeopoliticalRisk
    recent_news: list[NewsItem]
    confidence_breakdown: ConfidenceBreakdown
    data_quality_flags: list[str]
    suggested_timeframe: Literal["intraday", "swing_1-5d", "positional_1-4w", "macro_1-3m"]

    # ── Setup fields ──
    setups: list[TradingSetup]
    invalidation: InvalidationScenario
    decision_tree: list[DecisionTreeEntry]
    position_sizing_note: str


class BiasOutput(BaseModel):
    """Canonical P1 output — consumed by P2 Trading Setups Decision Tree."""

    # Metadata
    asset: str
    asset_class: Literal["equity", "forex", "crypto", "commodity"]
    analysis_timestamp: str  # ISO 8601

    # Core signal
    directional_bias: Literal["bullish", "bearish", "neutral"]
    bias_strength: Literal["weak", "moderate", "strong"]
    confidence_score: float = Field(ge=0.0, le=1.0)

    # Key levels
    key_levels: list[KeyLevel]
    nearest_support: float
    nearest_resistance: float

    # Claude's reasoning
    primary_thesis: str
    bull_case: str
    bear_case: str
    key_risks: list[str]  # 3-5 items

    # Sub-snapshots
    technical: TechnicalSnapshot
    macro: MacroSnapshot
    geopolitical: GeopoliticalRisk
    recent_news: list[NewsItem]

    # Meta
    confidence_breakdown: ConfidenceBreakdown
    data_quality_flags: list[str]
    suggested_timeframe: Literal["intraday", "swing_1-5d", "positional_1-4w", "macro_1-3m"]
