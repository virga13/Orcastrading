import os
from datetime import datetime, timezone
import anthropic
from p1_analysis_engine.schema import BiasOutput, SetupsOutput, FullAnalysisOutput
from p1_analysis_engine.utils.formatting import (
    format_technical,
    format_macro,
    format_news,
    format_geopolitical,
)
from typing import Literal, Optional

BIAS_SYSTEM_PROMPT = """You are a professional quantitative analyst and trading intelligence engine.
Your role is to synthesize multi-source market data — technical indicators, macroeconomic
conditions, news sentiment, and geopolitical context — into a single structured directional
bias for a given asset.

Rules:
- Base your bias only on the data provided. Do not hallucinate prices or events.
- TIMEFRAME WEIGHTING: Scale source weights to the chart interval provided.
    1m–15m: technical=0.90, fundamental=0.05, geopolitical=0.05
    30m–1h: technical=0.80, fundamental=0.12, geopolitical=0.08
    1d:     technical=0.65, fundamental=0.25, geopolitical=0.10
    1wk:    technical=0.45, fundamental=0.40, geopolitical=0.15
  Reflect this in confidence_breakdown weights. Do not over-weight macro/geo for short intervals.
- DIRECTIONAL BIAS THRESHOLD: Only set directional_bias to "bullish" or "bearish" if
  confidence_score >= 0.60. Below 0.60, set directional_bias to "neutral" regardless of
  which direction has slightly more evidence.
- BB POSITION: Set bb_position strictly from the computed price vs band values in the data.
  Do not infer or override the label — use the value as given.
- FLAG CONTRADICTIONS between data sources explicitly in key_risks.
- SIZING: Reference ATR% only for position sizing warnings. Do not reference VIX for sizing —
  VIX is a macro sentiment indicator, not a per-trade sizing input.
- All prices must match the currency/unit of the input data.
- suggested_timeframe should reflect the chart interval and signal clarity:
    intraday intervals (1m–1h) -> suggested_timeframe = "intraday" or "swing_1-5d" at most.
    daily/weekly intervals -> full range applicable.
- primary_thesis must be 1-2 sentences maximum.
- key_risks must contain 3-5 items."""

SETUPS_RULES = """
TRADE SETUP RULES (mandatory — violating any makes the output invalid):
1. MINIMUM R:R: Never generate a setup with rr_ratio below 1.5.
2. STRUCTURAL STOP LOSS: SL must be placed just beyond a named key level. Never arbitrary distances.
   TIMEFRAME BUFFER: For 1d/1wk intervals, SL must be at least 0.5× ATR beyond the key level.
   For 30m/1h intervals, at least 0.25× ATR. Tight stops on higher timeframes are hit by normal
   daily noise before the setup can develop — this is not a soft guideline, it is a hard floor.
3. TRAILING SL TO BREAKEVEN: Set trailing_sl_to_breakeven at TP1 price or nearest midpoint between entry and TP1.
4. TRADE TYPE: Exactly one of: "scalp" (minutes-hours), "intraday" (within session), "swing" (days-weeks).
5. INVALIDATION SCENARIO: Exactly one node — price/condition that collapses the entire thesis. NOT a trade setup.
6. EV CALCULATION: EV = (win_rate x blended_reward_pts) - ((1-win_rate) x risk_pts). Must be positive.
7. PROFIT FACTOR: (win_rate x avg_reward) / ((1-win_rate) x risk). Must be >= 1.0.
8. PRIORITY TIERS: Exactly one setup is "primary" — highest conviction, best R:R, bias-aligned, trade it first.
   Others are "secondary" (valid, lower conviction) or "conditional" (requires level to break first).
   Never label the best setup as secondary.
9. DECISION TREE: 4-6 distinct price action scenarios (rejection, breakout, consolidation, gap, etc.)
   covering all plausible next moves including the invalidation trigger.
10. POSITION SIZING NOTE: Reference ATR% explicitly. Warn if ATR% > 2% (reduce size 50-70%).
    Do NOT reference VIX for sizing — VIX is macro context only.
11. WIN RATE: Round numbers only (45%, 50%, 55%, 60%). No false precision like 57.3%.
    Trend-following: 50-60%, counter-trend: 40-50%, breakout: 45-55%.
12. TARGETS: Split across 2-3 levels (e.g. 50/30/20). TP1 is always the nearest key level.
13. BREAKOUT ENTRIES ON DAILY/WEEKLY: Do not generate breakout entries (conditional setups that
    trigger above resistance or below support) on 1d or 1wk intervals unless the entry condition
    is explicitly a confirmed CLOSE beyond the level — not an intrabar touch. Intraday wicks
    routinely exceed key levels without closing there, making touch-based breakout entries
    systematically unprofitable on these timeframes. If a confirmed-close entry cannot be
    expressed cleanly, omit the breakout setup and note it in the decision tree instead.
14. COUNTER-TREND PROXIMITY GATE: Counter-trend setups (bounces at support, fades at resistance)
    are only valid when the current price is within 0.25× ATR of the key level being traded.
    If price is not already near the level, omit the counter-trend setup entirely — do not
    generate a setup that requires price to travel significantly before the entry is relevant."""

COMBINED_SYSTEM_PROMPT = BIAS_SYSTEM_PROMPT + "\n" + SETUPS_RULES


def _build_user_prompt(ticker, asset_class, technical, macro, news, geopolitical) -> str:
    interval = technical.get("interval", "1d")
    return f"""ASSET: {ticker} ({asset_class})
ANALYSIS DATE: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
CHART INTERVAL: {interval} candles — all technical indicators and key levels are computed on this timeframe.

## TECHNICAL DATA
{format_technical(technical)}

## MACROECONOMIC DATA
{format_macro(macro)}

## RECENT NEWS (last 72 hours)
{format_news(news)}

## GEOPOLITICAL CONTEXT
{format_geopolitical(geopolitical)}

Based on all of the above, produce a complete structured bias analysis for this asset.
Calibrate suggested_timeframe, key_risks, and primary_thesis to match the {interval} chart resolution."""


def _call_with_cache(client, model, system_prompt, tool_def, tool_name, user_prompt, extended_thinking):
    """Single API call with prompt caching on the system prompt."""
    cached_system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    cache_header = {"anthropic-beta": "prompt-caching-2024-07-31"}

    if extended_thinking:
        thinking_response = client.messages.create(
            model=model,
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        response = client.messages.create(
            model=model,
            max_tokens=6000,
            system=cached_system,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": tool_name},
            extra_headers=cache_header,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": thinking_response.content},
                {"role": "user", "content": f"Now record the complete analysis above using the {tool_name} tool."},
            ],
        )
    else:
        response = client.messages.create(
            model=model,
            max_tokens=6000,
            system=cached_system,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": tool_name},
            extra_headers=cache_header,
            messages=[{"role": "user", "content": user_prompt}],
        )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if not tool_block:
        raise RuntimeError(f"Claude did not return a {tool_name} tool_use block")
    return tool_block.input


def synthesize(
    ticker: str,
    asset_class: Literal["equity", "forex", "crypto", "commodity", "index"],
    technical: dict,
    macro: dict,
    news: list[dict],
    geopolitical: dict,
    model: str = "claude-sonnet-4-6",
    extended_thinking: bool = False,
) -> BiasOutput:
    """Bias-only synthesis (used when --report is not requested)."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    user_prompt = _build_user_prompt(ticker, asset_class, technical, macro, news, geopolitical)

    tool_def = {
        "name": "record_bias_output",
        "description": "Record the complete structured bias analysis for the asset.",
        "input_schema": BiasOutput.model_json_schema(),
    }

    raw = _call_with_cache(client, model, BIAS_SYSTEM_PROMPT, tool_def, "record_bias_output", user_prompt, extended_thinking)
    raw["asset"] = ticker
    raw["asset_class"] = asset_class
    raw["analysis_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return BiasOutput(**raw)


def synthesize_full(
    ticker: str,
    asset_class: Literal["equity", "forex", "crypto", "commodity", "index"],
    technical: dict,
    macro: dict,
    news: list[dict],
    geopolitical: dict,
    model: str = "claude-sonnet-4-6",
    extended_thinking: bool = False,
) -> tuple[BiasOutput, SetupsOutput]:
    """Combined single-call synthesis: bias + trade setups in one API call.
    Uses prompt caching on the system prompt to minimize input token costs."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    user_prompt = _build_user_prompt(ticker, asset_class, technical, macro, news, geopolitical)
    user_prompt += "\n\nAfter completing the bias analysis, also generate 2-4 trade setups and a 4-6 scenario decision tree."

    tool_def = {
        "name": "record_full_analysis",
        "description": "Record the complete bias analysis AND trade setups for the asset in one call.",
        "input_schema": FullAnalysisOutput.model_json_schema(),
    }

    raw = _call_with_cache(client, model, COMBINED_SYSTEM_PROMPT, tool_def, "record_full_analysis", user_prompt, extended_thinking)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    setup_keys = {"setups", "invalidation", "decision_tree", "position_sizing_note"}

    bias_raw = {k: v for k, v in raw.items() if k not in setup_keys}
    bias_raw["asset"] = ticker
    bias_raw["asset_class"] = asset_class
    bias_raw["analysis_timestamp"] = now
    bias = BiasOutput(**bias_raw)

    setups = SetupsOutput(
        asset=ticker,
        current_price=raw["technical"]["current_price"],
        setups=raw["setups"],
        invalidation=raw["invalidation"],
        decision_tree=raw["decision_tree"],
        position_sizing_note=raw["position_sizing_note"],
    )

    return bias, setups


# ── Strategy Scanner synthesis ────────────────────────────────────────────────

def synthesize_strategy_scan(
    asset_id: str,
    ticker: str,
    label: str,
    market_data: dict,
    strategy_defs: list[dict],
    model: str = "claude-sonnet-4-6",
    macro_context: dict | None = None,
    news_headlines: list[str] | None = None,
):
    """
    Single Claude call that evaluates all daily strategies for one asset.
    Returns a fully populated StrategyScannerOutput.
    """
    from p1_analysis_engine.schema import StrategySignal, StrategyScannerOutput
    from pydantic import BaseModel
    from typing import Literal

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    md = market_data

    # ── Market data block ────────────────────────────────────────────────────
    def _fmt(v, decimals=4):
        return f"{v:,.{decimals}f}" if v is not None else "N/A"

    data_block = f"""## MARKET DATA — {label} ({ticker}) — {md['date']}

Price:       {_fmt(md['price'])}
ATR(14):     {_fmt(md['atr'])}  ({md['atr_pct']}% of price)
RSI(14):     {md['rsi']}
EMA20:       {_fmt(md['ema20'])}  (price is {md['price_atr_from_ema20']:+.3f} ATR from EMA20)
EMA50:       {_fmt(md['ema50'])}
EMA200:      {_fmt(md['ema200'])}
ADX(14):     {md['adx_today']} today | {md['adx_prev']} yesterday

Volume:      {md['volume_today']:,} today vs {md['volume_20d_avg']:,} 20d avg → ratio {md['volume_ratio']}x

20-day high: {_fmt(md['high_20d'])}
20-day low:  {_fmt(md['low_20d'])}

Today's bar: open {_fmt(md['today_open'])} | low {_fmt(md['today_low'])} | high {_fmt(md['today_high'])} | close {_fmt(md['price'])}
Close position in today's range: {md['close_pct_of_range']:.1%}  (1.0 = top, 0.0 = bottom)

10-day swing high (prior bar): {_fmt(md['swing_high_10d'])}
Distance from swing high:      {md['distance_from_swing_high_atr']:.3f} ATR  (positive = price is below swing high)

## WEEKLY DATA
Weekly EMA20:  {_fmt(md['weekly_ema20'])}
Weekly EMA50:  {_fmt(md['weekly_ema50'])}
Weekly RSI:    {md['weekly_rsi'] if md['weekly_rsi'] else 'N/A'}
Weekly trend:  {md['weekly_trend']}  (bullish = weekly EMA20 > weekly EMA50)"""

    # ── Strategy definitions block ────────────────────────────────────────────
    strat_blocks = []
    for s in strategy_defs:
        strat_blocks.append(
            f"### {s['name']} (strategy_id: \"{s['id']}\")\n{s['signal_conditions'].strip()}"
        )
    strategies_block = "\n\n".join(strat_blocks)

    # ── Macro context block ───────────────────────────────────────────────────
    macro_block = ""
    if macro_context:
        vix     = macro_context.get("vix_level")
        vix_reg = macro_context.get("vix_regime", "")
        ffr     = macro_context.get("fed_funds_rate")
        yc      = macro_context.get("yield_curve", "")
        cpi     = macro_context.get("cpi_trend", "")
        usd     = macro_context.get("usd_trend", "")
        mbias   = macro_context.get("macro_bias", "")
        macro_block = f"""
## MACRO CONTEXT
VIX: {f'{vix:.1f}' if vix else 'N/A'}  ({vix_reg})   ← >25 = high fear, trend strategies underperform
Fed Funds Rate: {f'{ffr:.2f}%' if ffr else 'N/A'}
Yield curve: {yc}  |  CPI trend: {cpi}  |  USD trend: {usd}
Macro bias: {mbias.upper() if mbias else 'N/A'}

Note: VIX >30 or macro_bias=risk_off should reduce confidence in trend-following signals.
Macro_bias=risk_on supports momentum and trend signals; risk_off favours caution."""

    # ── News block ────────────────────────────────────────────────────────────
    news_block = ""
    if news_headlines:
        headlines_text = "\n".join(f"  - {h}" for h in news_headlines)
        news_block = f"""
## RECENT NEWS HEADLINES (last 72h)
{headlines_text}

Note: Strong contra-news (e.g. surprise rate hike when bullish setup) should reduce
confidence. Supportive news can add confidence but never replace technical conditions."""

    user_prompt = f"""{data_block}{macro_block}{news_block}

## STRATEGIES TO EVALUATE

For each strategy below, check every numbered condition precisely against the market data above. A signal fires only when ALL conditions are met simultaneously. If any single condition fails, fired must be false.

When assessing confidence, factor in macro context and news: a technically valid signal in a risk-off macro environment or contra-news should have confidence reduced by 0.1–0.2.

{strategies_block}

Evaluate all strategies and record results using the record_strategy_scan tool."""

    system_prompt = """You are a quantitative signal evaluator. Your only job is to determine whether each strategy's conditions have fired, based solely on the numbers in the market data block. Do not use external knowledge, current market prices, or assumptions.

Rules:
- Evaluate each condition as a precise numeric check against the data provided
- fired=true only when ALL listed conditions pass
- If any condition fails, fired=false for that strategy
- List every condition in conditions_met or conditions_failed (not both)
- When fired=true: compute entry/SL/TP exactly using the formulas in the strategy definition
- confidence: 0.9+ = all conditions clearly met with margin; 0.7-0.9 = met but some borderline; below 0.7 = should not fire
- market_regime: assess from EMA alignment and ADX (bull_trend = EMA20>EMA50>EMA200 and ADX>20; bear_trend = inverse; ranging = ADX<20; uncertain = mixed signals)
- rationale: 2-3 sentences explaining the key factors for your decision"""

    # Tool schema — Claude fills only market_regime and signals
    class _SignalSchema(BaseModel):
        strategy_id: str
        fired: bool
        direction: Optional[Literal["long", "short"]] = None
        entry_low: float | None = None
        entry_high: float | None = None
        stop_loss: float | None = None
        tp1: float | None = None
        tp2: float | None = None
        rr: float | None = None
        confidence: float | None = None
        rationale: str = ""
        conditions_met: list[str] = []
        conditions_failed: list[str] = []

    class _ScanSchema(BaseModel):
        market_regime: Literal["bull_trend", "bear_trend", "ranging", "uncertain"]
        signals: list[_SignalSchema]

    tool_def = {
        "name": "record_strategy_scan",
        "description": "Record the complete strategy signal evaluation for this asset.",
        "input_schema": _ScanSchema.model_json_schema(),
    }

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        tools=[tool_def],
        tool_choice={"type": "tool", "name": "record_strategy_scan"},
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if not tool_block:
        raise RuntimeError("Claude did not return a record_strategy_scan tool_use block")

    raw = tool_block.input
    return StrategyScannerOutput(
        asset_id=asset_id,
        ticker=ticker,
        label=label,
        current_price=md["price"],
        atr=md["atr"],
        rsi=md["rsi"],
        analysis_date=md["date"],
        market_regime=raw["market_regime"],
        signals=[StrategySignal(**s) for s in raw["signals"]],
    )
