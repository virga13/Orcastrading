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

# Lazy import to avoid circular imports
def _get_rule_engine_strategy():
    from p3_backtester.strategies.rule_engine_strategy import RuleEngineStrategy
    return RuleEngineStrategy


# ── Strategy keyword map ─────────────────────────────────────────────────────
# Order matters — more specific phrases must come first
_STRATEGY_KEYWORDS: list[tuple[str, str]] = [
    ("mtf trend",        "MTFTrend"),
    ("multi timeframe",  "MTFTrend"),
    ("multi-timeframe",  "MTFTrend"),
    ("htf",              "MTFTrend"),
    ("ema retest",       "MTFTrend"),
    ("ema continuation", "EMAContinuation"),
    ("ema trend",        "EMAContinuation"),
    ("ema pullback",     "EMAPullback"),
    ("ema bounce",       "EMAPullback"),
    ("ema overshoot",    "EMAPullback"),
    ("rule engine",      "RuleEngine"),
    ("default",          "RuleEngine"),
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
        "MTF Trend (recommended) — multi-timeframe: HTF EMA20/50 bias + ETF EMA20 retest",
        "EMA Continuation — single-timeframe EMA pullback with ADX filter",
        "EMA Pullback — deep overshoot bounce at S/R",
        "Rule Engine (default) — deterministic rule-based setups",
    ]


def build_strategy(name: str, params: dict) -> StrategyBase:
    """Instantiate a strategy by name with given params."""
    ema_period    = int(params.get("ema_period", 20))
    adx_threshold = int(params.get("adx_threshold", 20))

    if name == "MTFTrend":
        return MTFTrendStrategy(ema_fast=ema_period, adx_threshold=adx_threshold)
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
      strategy_name : str   — "EMAContinuation" | "EMAPullback" | "RuleEngine"
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
      → {strategy_name: "EMAContinuation", params: {ema_period:50}, ticker: "^NDX", interval: "1d"}
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
    # Clamp to sensible range
    ema_period = max(5, min(ema_period, 500))

    # ── Interval ─────────────────────────────────────────────────────────────
    interval = "1h"  # default
    iv_match = _INTERVAL_PATTERN.search(query)
    if iv_match:
        raw = iv_match.group(1).lower()
        interval = _INTERVAL_ALIASES.get(raw, raw)

    # ── Ticker — try every word against asset_classifier ─────────────────────
    ticker = None
    # Extract clean words (uppercase for ticker matching)
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-=^]*\b', query)
    # Skip common English words that will never be tickers
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
        # If resolve_ticker changed the word, it's a known alias (e.g. GOLD → GC=F)
        # If it stayed the same and looks like a real ticker (ALL CAPS, short), accept it
        if candidate != word.upper():
            ticker = candidate
            break
        if word.upper() == word and 2 <= len(word) <= 7 and word.upper() not in stop_words:
            # Looks like a ticker (AAPL, BTC, etc.) — accept as-is
            ticker = word.upper()
            break

    return {
        "strategy_name": strategy_name,
        "params":        {"ema_period": ema_period},
        "ticker":        ticker,
        "interval":      interval,
        "raw_query":     query,
    }
