"""
tests/test_strategies.py — unit tests for strategy classes and registry.
"""
import pytest
import numpy as np
import pandas as pd
from p1_analysis_engine.schema import SetupsOutput, TradingSetup
from p3_backtester.strategies.registry import build_from_config, parse_query
from p3_backtester.strategies.base import StrategyBase


STRATEGY_IDS = ["mtf_trend", "orb", "ema_continuation", "ema_pullback", "momentum_breakout"]


# ── Instantiation ─────────────────────────────────────────────────────────────

class TestInstantiation:
    @pytest.mark.parametrize("sid", STRATEGY_IDS)
    def test_build_from_config(self, sid):
        s = build_from_config(sid)
        assert isinstance(s, StrategyBase)

    @pytest.mark.parametrize("sid", STRATEGY_IDS)
    def test_has_name_and_description(self, sid):
        s = build_from_config(sid)
        assert isinstance(s.name, str) and s.name
        assert isinstance(s.description, str)

    @pytest.mark.parametrize("sid", STRATEGY_IDS)
    def test_params_is_dict(self, sid):
        s = build_from_config(sid)
        assert isinstance(s.params, dict)

    def test_param_override_applied(self):
        s = build_from_config("mtf_trend", {"shorts_enabled": False})
        assert s.shorts_enabled is False

    def test_unknown_strategy_raises_key_error(self):
        with pytest.raises(KeyError):
            build_from_config("strategy_that_does_not_exist")

    def test_partial_param_override(self):
        default = build_from_config("ema_continuation")
        custom  = build_from_config("ema_continuation", {"adx_threshold": 30})
        assert custom.adx_threshold == 30
        assert custom.ema_period == default.ema_period  # unchanged


# ── generate_setups return type contract ─────────────────────────────────────

class TestGenerateSetups:
    @pytest.mark.parametrize("sid", ["ema_continuation", "mtf_trend"])
    def test_returns_setups_output_or_none(self, sid, tech_dict, synthetic_df):
        s = build_from_config(sid)
        result = s.generate_setups(tech_dict, synthetic_df)
        assert result is None or isinstance(result, SetupsOutput)

    @pytest.mark.parametrize("sid", ["ema_continuation", "mtf_trend"])
    def test_signal_structure_when_fired(self, sid, tech_dict, synthetic_df):
        s = build_from_config(sid)
        result = s.generate_setups(tech_dict, synthetic_df)
        if result is None:
            pytest.skip(f"{sid} produced no signal on synthetic data — conditions not met")
        assert len(result.setups) >= 1
        setup: TradingSetup = result.setups[0]
        assert setup.direction in ("long", "short")
        assert setup.entry_low < setup.entry_high, "entry_low must be < entry_high"
        assert setup.rr_ratio > 0

    @pytest.mark.parametrize("sid", ["ema_continuation", "mtf_trend"])
    def test_long_signal_levels_ordered(self, sid, tech_dict, synthetic_df):
        s = build_from_config(sid)
        result = s.generate_setups(tech_dict, synthetic_df)
        if result is None:
            pytest.skip("No signal produced")
        for setup in result.setups:
            if setup.direction == "long":
                assert setup.stop_loss < setup.entry_low, (
                    f"Long SL ({setup.stop_loss}) must be < entry_low ({setup.entry_low})"
                )
                assert setup.targets[0].price > setup.entry_high, (
                    f"Long TP1 ({setup.targets[0].price}) must be > entry_high ({setup.entry_high})"
                )

    @pytest.mark.parametrize("sid", STRATEGY_IDS)
    def test_no_crash_on_minimal_df(self, sid, tech_dict):
        """Strategy must not crash even on a short (50-bar) dataframe."""
        np.random.seed(0)
        n = 50
        p = np.linspace(2000, 2200, n)
        df_short = pd.DataFrame({
            "Open": p, "High": p * 1.001, "Low": p * 0.999,
            "Close": p, "Volume": np.ones(n) * 100_000,
        }, index=pd.date_range("2025-01-01", periods=n, freq="D"))
        s = build_from_config(sid)
        result = s.generate_setups(tech_dict, df_short)
        assert result is None or isinstance(result, SetupsOutput)

    @pytest.mark.parametrize("sid", STRATEGY_IDS)
    def test_no_crash_on_zero_atr(self, sid, tech_dict, synthetic_df):
        """Zero ATR must not raise ZeroDivisionError."""
        tech_zero_atr = dict(tech_dict, atr_14=0.0, atr_pct=0.0)
        s = build_from_config(sid)
        try:
            result = s.generate_setups(tech_zero_atr, synthetic_df)
            assert result is None or isinstance(result, SetupsOutput)
        except ZeroDivisionError:
            pytest.fail(f"{sid} raised ZeroDivisionError on atr_14=0")


# ── Registry: parse_query ─────────────────────────────────────────────────────

class TestParseQuery:
    @pytest.mark.parametrize("query,expected", [
        ("EMA continuation Gold 1d",    "EMAContinuation"),
        ("MTF trend SPX daily",         "MTFTrend"),
        ("momentum breakout BTC 4h",    "MomentumBreakout"),
        ("ema pullback 1h",             "EMAPullback"),
        ("opening range gold 15m",      "ORB"),
        ("orb Gold",                    "ORB"),
        ("default",                     "RuleEngine"),
    ])
    def test_strategy_name_extracted(self, query, expected):
        r = parse_query(query)
        assert r["strategy_name"] == expected, (
            f"Query {query!r} → got {r['strategy_name']}, expected {expected}"
        )

    def test_interval_extracted(self):
        r = parse_query("EMA continuation Gold 4h")
        assert r["interval"] == "4h"

    def test_daily_alias(self):
        r = parse_query("MTF trend daily")
        assert r["interval"] == "1d"

    def test_ema_period_extracted(self):
        r = parse_query("50 EMA continuation Gold")
        assert r["params"]["ema_period"] == 50

    def test_default_ema_period(self):
        r = parse_query("EMA continuation Gold 1d")
        assert r["params"]["ema_period"] == 20

    def test_returns_dict_with_required_keys(self):
        r = parse_query("EMA pullback Gold")
        for k in ("strategy_name", "params", "interval", "raw_query"):
            assert k in r, f"Missing key '{k}' in parse_query result"
