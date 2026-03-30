"""
strategies/rule_engine_strategy.py — wraps the existing rule_engine.py
as a StrategyBase so it plugs into the new strategy system.
"""
import pandas as pd
from p3_backtester.strategies.base import StrategyBase
from p3_backtester.rule_engine import generate_setups_from_technicals
from p1_analysis_engine.schema import SetupsOutput


class RuleEngineStrategy(StrategyBase):
    """Original multi-indicator rule engine (trend + MACD + EMA + RSI + ADX)."""

    name = "Rule Engine"
    description = "Multi-indicator alignment (trend, MACD, EMA, RSI, ADX)"

    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        return generate_setups_from_technicals(tech)
