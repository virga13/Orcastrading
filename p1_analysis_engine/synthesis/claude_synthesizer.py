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
from typing import Literal

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
12. TARGETS: Split across 2-3 levels (e.g. 50/30/20). TP1 is always the nearest key level."""

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
    asset_class: Literal["equity", "forex", "crypto", "commodity"],
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
    asset_class: Literal["equity", "forex", "crypto", "commodity"],
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
