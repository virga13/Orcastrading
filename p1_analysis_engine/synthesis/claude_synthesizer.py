import os
import json
from datetime import datetime, timezone
import anthropic
from p1_analysis_engine.schema import BiasOutput
from p1_analysis_engine.utils.formatting import (
    format_technical,
    format_macro,
    format_news,
    format_geopolitical,
)
from typing import Literal

SYSTEM_PROMPT = """You are a professional quantitative analyst and trading intelligence engine.
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
    intraday intervals (1m–1h) → suggested_timeframe = "intraday" or "swing_1-5d" at most.
    daily/weekly intervals → full range applicable.
- primary_thesis must be 1-2 sentences maximum.
- key_risks must contain 3-5 items."""


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
    """
    Assembles the full prompt from all fetcher outputs and calls Claude
    via tool use to produce a validated BiasOutput instance.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Build user prompt from formatted sections
    interval = technical.get("interval", "1d")
    user_prompt = f"""ASSET: {ticker} ({asset_class})
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

    # Derive JSON schema from BiasOutput Pydantic model
    bias_schema = BiasOutput.model_json_schema()

    # Use tool use to force structured output
    tool_def = {
        "name": "record_bias_output",
        "description": "Record the complete structured bias analysis for the asset.",
        "input_schema": bias_schema,
    }

    if extended_thinking:
        # Pass 1: extended thinking — no tools (incompatible with forced tool_choice)
        thinking_response = client.messages.create(
            model=model,
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Pass 2: structured extraction — feed thinking output back as context
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "record_bias_output"},
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": thinking_response.content},
                {"role": "user", "content": "Now record the complete analysis above using the record_bias_output tool."},
            ],
        )
    else:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "record_bias_output"},
            messages=[{"role": "user", "content": user_prompt}],
        )

    # Extract tool input from response
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if not tool_use_block:
        raise RuntimeError("Claude did not return a tool_use block")

    raw = tool_use_block.input

    # Inject metadata that Claude shouldn't guess
    raw["asset"] = ticker
    raw["asset_class"] = asset_class
    raw["analysis_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Validate and return typed BiasOutput
    return BiasOutput(**raw)
