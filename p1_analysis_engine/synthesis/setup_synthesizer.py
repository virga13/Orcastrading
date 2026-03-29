import os
import anthropic
from p1_analysis_engine.schema import BiasOutput, SetupsOutput

SYSTEM_PROMPT = """You are a professional quantitative trading strategist.
Given a structured market bias analysis, generate 2-4 concrete trade setups, one invalidation scenario, and a decision tree.

MANDATORY RULES — violating any of these makes the output invalid:
1. MINIMUM R:R: Never generate a setup with rr_ratio below 1.5. If a setup would be below 1.5R, do not include it.
2. STRUCTURAL STOP LOSS: SL must be placed just beyond a named key level from the data. Never arbitrary pip distances.
3. TRAILING SL TO BREAKEVEN: Set trailing_sl_to_breakeven at TP1 price or the nearest structural midpoint between entry and TP1. This is the price at which you move SL to breakeven.
4. TRADE TYPE: Classify each setup as exactly one of: "scalp" (minutes to hours, ATR-based tight SL), "intraday" (holds within session, respects session structure), "swing" (days to weeks, holds through noise).
5. INVALIDATION SCENARIO: Generate exactly one invalidation node — a price/condition that, if triggered, collapses the entire thesis and means standing aside completely. This is NOT a trade setup.
6. EV CALCULATION: EV = (win_rate × blended_reward_pts) - ((1-win_rate) × risk_pts). Must be positive.
7. PROFIT FACTOR: = (win_rate × avg_reward) / ((1-win_rate) × risk). Must be >= 1.0.
8. PRIORITY TIERS: Exactly one setup is "primary" — this must be the highest conviction, bias-aligned setup with the best R:R. The primary setup is the one you would trade first. Label others "secondary" (valid but lower conviction) or "conditional" (requires a specific level to break first). Never label the best setup as secondary.
9. DECISION TREE: 4-6 scenarios covering all plausible next price actions including the invalidation trigger. Each scenario must be a distinct price action event (rejection, breakout, consolidation, gap, etc.), not generic descriptions.
10. POSITION SIZING NOTE: Reference the ATR% value explicitly. Warn if ATR% > 2% (reduce size 50-70%). Do NOT reference VIX for sizing decisions — VIX is macro context only.
11. WIN RATE: Use round numbers only (e.g. 45%, 50%, 55%, 60%). Do not use false precision like 57.3%. Base on setup type: trend-following 50-60%, counter-trend 40-50%, breakout 45-55%.
12. Targets: split position across 2-3 levels with logical allocation (e.g. 50/30/20). TP1 is always the nearest key level."""


def generate_setups(bias: BiasOutput, model: str = "claude-sonnet-4-6", interval: str = "1d", extended_thinking: bool = False) -> SetupsOutput:
    """
    Takes a BiasOutput and generates concrete trade setups with full
    decision tree via Claude tool use.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    t = bias.technical
    key_levels_text = "\n".join(
        f"  {lv.label}: {lv.price} ({lv.strength})" for lv in bias.key_levels
    )

    user_prompt = f"""ASSET: {bias.asset} ({bias.asset_class})
CURRENT PRICE: {t.current_price}
CHART INTERVAL: {interval} candles — entries, SL distances, and trade_duration must be calibrated to this timeframe.
DIRECTIONAL BIAS: {bias.directional_bias} ({bias.bias_strength}) — confidence {bias.confidence_score:.0%}
SUGGESTED TIMEFRAME: {bias.suggested_timeframe}

PRIMARY THESIS:
{bias.primary_thesis}

TECHNICAL CONTEXT:
  Trend: {t.trend}
  RSI 14: {t.rsi_14} ({('oversold' if t.rsi_14 < 30 else 'overbought' if t.rsi_14 > 70 else 'neutral')})
  MACD: {t.macd_signal}
  EMA 20/50/200: {t.ema20} / {t.ema50} / {t.ema200}
  ATR 14: {t.atr_14} ({t.atr_pct}% of price)
  Stoch %K/%D: {t.stoch_k}/{t.stoch_d} ({t.stoch_signal})
  BB Upper/Lower: {t.bb_upper} / {t.bb_lower} (position: {t.bb_position})
  ADX: {t.adx_14}

NEAREST SUPPORT: {bias.nearest_support}
NEAREST RESISTANCE: {bias.nearest_resistance}

ALL KEY LEVELS:
{key_levels_text}

BULL CASE: {bias.bull_case}
BEAR CASE: {bias.bear_case}

KEY RISKS:
{chr(10).join(f'  - {r}' for r in bias.key_risks)}

Generate 2-4 trade setups and a 4-6 scenario decision tree for this asset right now."""

    tool_def = {
        "name": "record_setups",
        "description": "Record all trade setups and decision tree for the asset.",
        "input_schema": SetupsOutput.model_json_schema(),
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
            tool_choice={"type": "tool", "name": "record_setups"},
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": thinking_response.content},
                {"role": "user", "content": "Now record the complete trade setups above using the record_setups tool."},
            ],
        )
    else:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "record_setups"},
            messages=[{"role": "user", "content": user_prompt}],
        )

    tool_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if not tool_block:
        raise RuntimeError("Claude did not return setups")

    raw = tool_block.input
    raw["asset"] = bias.asset
    raw["current_price"] = t.current_price

    return SetupsOutput(**raw)
