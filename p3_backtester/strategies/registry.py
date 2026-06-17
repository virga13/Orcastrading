"""
strategies/registry.py — strategy registry and natural language query parser.

Parses queries like:
  "Backtest 25 EMA continuation for Gold on 1h"
  "EMA pullback BTC 5m"
  "50 EMA trend NASDAQ daily"
  "rule engine Gold 1h"
  "default"
"""
import re
from p3_backtester.strategies.base import StrategyBase
from p3_backtester.strategies.ema_continuation import EMAContinuationStrategy
from p3_backtester.strategies.ema_pullback import EMAPullbackStrategy
from p3_backtester.strategies.mtf_trend import MTFTrendStrategy
from p3_backtester.strategies.orb import ORBStrategy
from p3_backtester.strategies.momentum_breakout import MomentumBreakoutStrategy
from p3_backtester.strategies.gold_scalper import GoldScalperStrategy
from p3_backtester.strategies.bb_squeeze import BBSqueezeStrategy
from p3_backtester.strategies.supertrend import SupertrendStrategy
from p3_backtester.strategies.macd_crossover import MACDCrossoverStrategy
from p3_backtester.strategies.fibonacci_pullback import FibonacciPullbackStrategy
from p3_backtester.strategies.rsi_divergence import RSIDivergenceStrategy
from p3_backtester.strategies.inside_bar import InsideBarStrategy
from p3_backtester.strategies.donchian_breakout import DonchianBreakoutStrategy
from p3_backtester.strategies.stochastic_bounce import StochasticBounceStrategy
from p3_backtester.strategies.engulfing_candle import EngulfingCandleStrategy
from p3_backtester.strategies.heikin_ashi_trend import HeikinAshiTrendStrategy
from p3_backtester.strategies.parabolic_sar_flip import ParabolicSARFlipStrategy
from p3_backtester.strategies.williams_r_reversal import WilliamsRReversalStrategy
from p3_backtester.strategies.adx_di_crossover import ADXDICrossoverStrategy
from p3_backtester.strategies.cci_reversal import CCIReversalStrategy
from p3_backtester.strategies.hammer_reversal import HammerReversalStrategy
from p3_backtester.strategies.ichimoku_cloud import IchimokuCloudStrategy
from p3_backtester.strategies.triple_ema_cross import TripleEMACrossStrategy
from p3_backtester.strategies.bb_bounce import BBBounceStrategy
from p3_backtester.strategies.roc_momentum import ROCMomentumStrategy
from p3_backtester.strategies.keltner_breakout import KeltnerBreakoutStrategy
from p3_backtester.strategies.consecutive_highs import ConsecutiveHighsStrategy
from p3_backtester.strategies.trix_signal import TRIXSignalStrategy
from p3_backtester.strategies.price_channel_retest import PriceChannelRetestStrategy

def _get_rule_engine_strategy():
    from p3_backtester.strategies.rule_engine_strategy import RuleEngineStrategy
    return RuleEngineStrategy


# ── Core config-driven factory ────────────────────────────────────────────────

def build_from_config(strategy_id: str, param_overrides: dict | None = None) -> StrategyBase:
    """
    Instantiate a strategy by its config ID (from config/strategies.yaml).
    Loads default params from config and applies any overrides.

    Usage:
        strategy = build_from_config("mtf_trend")
        strategy = build_from_config("orb", {"volume_multiplier": 2.0})
    """
    from core.config import get_strategy_params

    defaults = get_strategy_params(strategy_id)
    params   = {**defaults, **(param_overrides or {})}

    def _kw(cls, p):
        """Filter params to only those accepted by cls.__init__."""
        return {k: v for k, v in p.items() if k in cls.__init__.__code__.co_varnames}

    _MAP = {
        "mtf_trend":         lambda p: MTFTrendStrategy(**_kw(MTFTrendStrategy, p)),
        "orb":               lambda p: ORBStrategy(**_kw(ORBStrategy, p)),
        "momentum_breakout": lambda p: MomentumBreakoutStrategy(**_kw(MomentumBreakoutStrategy, p)),
        "ema_continuation":  lambda p: EMAContinuationStrategy(**_kw(EMAContinuationStrategy, p)),
        "ema_pullback":      lambda p: EMAPullbackStrategy(**_kw(EMAPullbackStrategy, p)),
        "gold_scalper":      lambda p: GoldScalperStrategy(**_kw(GoldScalperStrategy, p)),
        "bb_squeeze":           lambda p: BBSqueezeStrategy(**_kw(BBSqueezeStrategy, p)),
        "supertrend":           lambda p: SupertrendStrategy(**_kw(SupertrendStrategy, p)),
        "macd_crossover":       lambda p: MACDCrossoverStrategy(**_kw(MACDCrossoverStrategy, p)),
        "fibonacci_pullback":   lambda p: FibonacciPullbackStrategy(**_kw(FibonacciPullbackStrategy, p)),
        "rsi_divergence":       lambda p: RSIDivergenceStrategy(**_kw(RSIDivergenceStrategy, p)),
        "inside_bar":           lambda p: InsideBarStrategy(**_kw(InsideBarStrategy, p)),
        "donchian_breakout":    lambda p: DonchianBreakoutStrategy(**_kw(DonchianBreakoutStrategy, p)),
        "stochastic_bounce":    lambda p: StochasticBounceStrategy(**_kw(StochasticBounceStrategy, p)),
        "engulfing_candle":     lambda p: EngulfingCandleStrategy(**_kw(EngulfingCandleStrategy, p)),
        "heikin_ashi_trend":    lambda p: HeikinAshiTrendStrategy(**_kw(HeikinAshiTrendStrategy, p)),
        "parabolic_sar_flip":   lambda p: ParabolicSARFlipStrategy(**_kw(ParabolicSARFlipStrategy, p)),
        "williams_r_reversal":  lambda p: WilliamsRReversalStrategy(**_kw(WilliamsRReversalStrategy, p)),
        "adx_di_crossover":     lambda p: ADXDICrossoverStrategy(**_kw(ADXDICrossoverStrategy, p)),
        "cci_reversal":         lambda p: CCIReversalStrategy(**_kw(CCIReversalStrategy, p)),
        "hammer_reversal":      lambda p: HammerReversalStrategy(**_kw(HammerReversalStrategy, p)),
        "ichimoku_cloud":       lambda p: IchimokuCloudStrategy(**_kw(IchimokuCloudStrategy, p)),
        "triple_ema_cross":     lambda p: TripleEMACrossStrategy(**_kw(TripleEMACrossStrategy, p)),
        "bb_bounce":            lambda p: BBBounceStrategy(**_kw(BBBounceStrategy, p)),
        "roc_momentum":         lambda p: ROCMomentumStrategy(**_kw(ROCMomentumStrategy, p)),
        "keltner_breakout":          lambda p: KeltnerBreakoutStrategy(**_kw(KeltnerBreakoutStrategy, p)),
        "consecutive_highs":         lambda p: ConsecutiveHighsStrategy(**_kw(ConsecutiveHighsStrategy, p)),
        "trix_signal":               lambda p: TRIXSignalStrategy(**_kw(TRIXSignalStrategy, p)),
        "price_channel_retest":      lambda p: PriceChannelRetestStrategy(**_kw(PriceChannelRetestStrategy, p)),
        "rule_engine":               lambda p: _get_rule_engine_strategy()(),
    }

    if strategy_id not in _MAP:
        raise KeyError(
            f"Unknown strategy '{strategy_id}'. "
            f"Available: {list(_MAP.keys())}"
        )
    return _MAP[strategy_id](params)


# ── Strategy keyword map ─────────────────────────────────────────────────────
# Order matters — more specific phrases must come first
_STRATEGY_KEYWORDS: list[tuple[str, str]] = [
    ("ema continuation",   "EMAContinuation"),
    ("ema trend",          "EMAContinuation"),
    ("ema pullback",       "EMAPullback"),
    ("ema bounce",         "EMAPullback"),
    ("ema overshoot",      "EMAPullback"),
    ("mtf trend",          "MTFTrend"),
    ("multi timeframe",    "MTFTrend"),
    ("multi-timeframe",    "MTFTrend"),
    ("htf",                "MTFTrend"),
    ("ema retest",         "MTFTrend"),
    ("opening range",      "ORB"),
    ("orb",                "ORB"),
    ("momentum breakout",  "MomentumBreakout"),
    ("momentum",           "MomentumBreakout"),
    ("breakout",           "MomentumBreakout"),
    ("pullback",           "EMAContinuation"),
    ("trend",              "MTFTrend"),
    ("rule engine",        "RuleEngine"),
    ("default",            "RuleEngine"),
]

# ── Interval aliases ─────────────────────────────────────────────────────────
_INTERVAL_ALIASES: dict[str, str] = {
    "daily":   "1d",
    "weekly":  "1wk",
    "hourly":  "1h",
    "minute":  "1m",
}

_INTERVAL_PATTERN = re.compile(
    r'\b(1m|5m|15m|30m|1h|4h|1d|1wk|daily|weekly|hourly)\b',
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r'\b(\d+)\b')


def list_strategies() -> list[str]:
    """Return all available strategy names."""
    return [
        "mtf_trend         (1d)  — multi-timeframe EMA pullback with regime filter",
        "orb               (15m) — opening range breakout with volume + VWAP",
        "momentum_breakout (1d)  — N-day high breakout with volume expansion",
        "ema_continuation  (1d)  — EMA pullback in trending market with ADX filter",
        "ema_pullback      (1d)  — deep EMA overshoot bounce at structural support",
        "gold_scalper      (5m)  — BRK Gold Scalper v2: M30/M15/M5 composite score, ATR SL/TP",
        "bb_squeeze        (1d)  — Bollinger Band Squeeze breakout when BB expands outside Keltner",
        "supertrend        (1d)  — Supertrend ATR crossover with HTF trend filter",
        "macd_crossover    (1d)  — MACD signal-line crossover with RSI + volume + HTF filters",
        "fibonacci_pullback(1d)  — Fib retracement pullback at 38.2/50/61.8% after a trend move",
        "rsi_divergence    (1d)  — RSI divergence at swing pivots signals momentum reversal",
        "inside_bar        (1d)  — inside bar breakout after strong mother bar",
        "donchian_breakout (1d)  — Donchian channel N-period high/low breakout (Turtle Trading)",
        "stochastic_bounce (1d)  — Stochastic %K/%D cross from oversold/overbought with trend filter",
        "engulfing_candle  (1d)  — bullish/bearish engulfing candle near EMA — trend continuation",
        "heikin_ashi_trend (1d)  — consecutive bullish HA bars above EMA with no lower wicks",
        "parabolic_sar_flip(1d)  — Parabolic SAR flip from bearish to bullish in uptrend",
        "williams_r_reversal(1d) — Williams %R oversold cross-up in established uptrend",
        "adx_di_crossover  (1d)  — ADX DI+/DI- crossover with rising ADX and EMA trend filter",
        "cci_reversal       (1d)  — CCI oversold cross-up in uptrend — higher signal frequency than %R",
        "hammer_reversal    (1d)  — Hammer/pin bar rejection candle near EMA support",
        "rule_engine             — deterministic rule-based setups (default/legacy)",
    ]


def build_strategy(name: str, params: dict) -> StrategyBase:
    """Instantiate a strategy by class name with given params."""
    ema_period    = int(params.get("ema_period", 20))
    adx_threshold = int(params.get("adx_threshold", 20))

    if name == "MTFTrend":
        return MTFTrendStrategy(ema_fast=ema_period, adx_threshold=adx_threshold)
    if name == "ORB":
        return ORBStrategy(**{k: v for k, v in params.items()
                              if k in ORBStrategy.__init__.__code__.co_varnames})
    if name == "MomentumBreakout":
        return MomentumBreakoutStrategy()
    if name == "EMAContinuation":
        return EMAContinuationStrategy(ema_period=ema_period, adx_threshold=adx_threshold)
    if name == "EMAPullback":
        return EMAPullbackStrategy(ema_period=ema_period, adx_threshold=adx_threshold)
    # Default: rule engine
    return _get_rule_engine_strategy()()


def parse_query(query: str) -> dict:
    """
    Parse a natural language backtest query.

    Returns
    -------
    dict with keys:
      strategy_name : str   — "EMAContinuation" | "EMAPullback" | "RuleEngine" etc.
      params        : dict  — {"ema_period": int, "adx_threshold": int}
      ticker        : str | None  — resolved yfinance ticker, or None if not found
      interval      : str   — e.g. "1h"
      raw_query     : str

    Examples
    --------
    "Backtest 25 EMA continuation for Gold on 1h"
      → {strategy_name: "EMAContinuation", params: {ema_period:25}, ticker: "GC=F", interval: "1h"}

    "EMA pullback BTC 5m"
      → {strategy_name: "EMAPullback", params: {ema_period:20}, ticker: "BTC-USD", interval: "5m"}

    "50 EMA trend NASDAQ daily"
      → {strategy_name: "MTFTrend", params: {ema_period:50}, ticker: "^NDX", interval: "1d"}
    """
    from p1_analysis_engine.utils.asset_classifier import resolve_ticker

    q_lower = query.lower().strip()

    # ── Strategy ──────────────────────────────────────────────────────────────
    strategy_name = "RuleEngine"
    for keyword, name in _STRATEGY_KEYWORDS:
        if keyword in q_lower:
            strategy_name = name
            break

    # ── EMA period — first number in query ───────────────────────────────────
    numbers = _NUMBER_PATTERN.findall(query)
    ema_period = int(numbers[0]) if numbers else 20
    ema_period = max(5, min(ema_period, 500))

    # ── Interval ─────────────────────────────────────────────────────────────
    interval = "1h"  # default
    iv_match = _INTERVAL_PATTERN.search(query)
    if iv_match:
        raw = iv_match.group(1).lower()
        interval = _INTERVAL_ALIASES.get(raw, raw)

    # ── Ticker — try every word against asset_classifier ─────────────────────
    ticker = None
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-=^]*\b', query)
    stop_words = {
        "backtest", "test", "run", "for", "on", "at", "the", "a", "an",
        "with", "ema", "pullback", "continuation", "trend", "strategy",
        "timeframe", "interval", "daily", "weekly", "hourly", "minute",
        "and", "or", "in", "of", "to",
    }
    for word in words:
        if word.lower() in stop_words:
            continue
        candidate = resolve_ticker(word.upper())
        if candidate != word.upper():
            ticker = candidate
            break
        if word.upper() == word and 2 <= len(word) <= 7 and word.upper() not in stop_words:
            ticker = word.upper()
            break

    return {
        "strategy_name": strategy_name,
        "params":        {"ema_period": ema_period},
        "ticker":        ticker,
        "interval":      interval,
        "raw_query":     query,
    }
