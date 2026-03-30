"""
strategies/base.py — abstract base class for all backtestable strategies.

Every strategy must implement generate_setups(tech, df) and expose
a name, description, and params dict.
"""
from abc import ABC, abstractmethod
import pandas as pd
from p1_analysis_engine.schema import SetupsOutput


class StrategyBase(ABC):
    name: str = "unnamed"
    description: str = ""

    @property
    def params(self) -> dict:
        return {}

    @abstractmethod
    def generate_setups(self, tech: dict, df: pd.DataFrame) -> SetupsOutput | None:
        """
        Generate trade setups for the current bar.

        Parameters
        ----------
        tech : dict
            Technical indicator dict from bar_slicer.recompute_technical().
            Keys include: current_price, ema20, ema50, ema200, atr_14, adx_14,
            rsi_14, macd_signal, trend, nearest_support, nearest_resistance, interval, etc.
        df : pd.DataFrame
            OHLCV slice up to and including the current bar (zero lookahead).
            Use this to compute indicators not present in `tech` (e.g. arbitrary EMA periods).

        Returns
        -------
        SetupsOutput | None
            None if quality gate fails (no setup generated).
        """
        ...

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str})"
