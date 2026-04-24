"""
tests/test_config.py — unit tests for core/config.py
"""
import pytest
from core.config import (
    get_all_assets, get_asset, get_ticker, get_all_strategies,
    get_strategy_params, get_strategy_timeframe, get_strategy_lookback,
    get_enabled_strategies, get_asset_strategy_params,
    get_watchlist, get_asset_baseline,
)


class TestAssets:
    def test_loads_assets(self):
        assets = get_all_assets()
        assert isinstance(assets, list)
        assert len(assets) > 0

    def test_get_asset_known(self):
        a = get_asset("gold")
        assert a["id"] == "gold"
        assert "tickers" in a

    def test_get_asset_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown asset"):
            get_asset("does_not_exist_xyz")

    def test_get_ticker_yfinance(self):
        t = get_ticker("gold", source="yfinance")
        assert isinstance(t, str)
        assert len(t) > 0

    def test_get_ticker_fallback(self):
        # requesting a source that isn't configured falls back to yfinance
        t = get_ticker("gold", source="nonexistent_source")
        assert isinstance(t, str)

    def test_enabled_assets_have_strategies(self):
        for a in get_all_assets():
            if a.get("enabled", True):
                strats = a.get("strategies", [])
                assert len(strats) > 0, f"{a['id']} is enabled but has no strategies"

    def test_all_asset_strategy_refs_exist(self):
        all_strats = get_all_strategies()
        for a in get_all_assets():
            for entry in a.get("strategies", []):
                sid = entry if isinstance(entry, str) else entry.get("id", "")
                assert sid in all_strats, (
                    f"Asset '{a['id']}' references unknown strategy '{sid}'"
                )

    def test_get_enabled_strategies(self):
        strats = get_enabled_strategies("gold")
        assert isinstance(strats, list)
        assert len(strats) > 0
        assert all(isinstance(s, str) for s in strats)

    def test_per_asset_param_overrides(self):
        # spx500 and us30 should have shorts_enabled=False for mtf_trend
        for asset_id in ("spx500", "us30"):
            params = get_asset_strategy_params(asset_id, "mtf_trend")
            assert params.get("shorts_enabled") is False, (
                f"{asset_id}/mtf_trend should have shorts_enabled=False"
            )

    def test_get_asset_baseline(self):
        b = get_asset_baseline("gold")
        assert b is not None
        assert "win_rate" in b
        assert "net_pf" in b
        assert "n_trades" in b

    def test_get_asset_baseline_missing(self):
        # should not crash for assets without baseline
        b = get_asset_baseline("does_not_exist_xyz")
        assert b is None


class TestStrategies:
    def test_loads_strategies(self):
        strats = get_all_strategies()
        assert isinstance(strats, dict)
        assert len(strats) > 0

    def test_known_strategies_present(self):
        strats = get_all_strategies()
        for sid in ("mtf_trend", "orb", "ema_continuation", "ema_pullback", "momentum_breakout"):
            assert sid in strats, f"'{sid}' missing from strategies.yaml"

    def test_get_strategy_params_returns_dict(self):
        p = get_strategy_params("mtf_trend")
        assert isinstance(p, dict)
        assert "ema_fast" in p
        assert "ema_slow" in p

    def test_get_strategy_timeframe(self):
        assert get_strategy_timeframe("mtf_trend") == "1d"
        assert get_strategy_timeframe("orb") == "15m"

    def test_get_strategy_lookback_positive(self):
        lb = get_strategy_lookback("mtf_trend")
        assert isinstance(lb, int)
        assert lb > 0

    def test_unknown_strategy_raises(self):
        from core.config import get_strategy_config
        with pytest.raises(KeyError, match="Unknown strategy"):
            get_strategy_config("strategy_xyz_does_not_exist")


class TestWatchlist:
    def test_watchlist_not_empty(self):
        wl = get_watchlist()
        assert len(wl) > 0

    def test_watchlist_structure(self):
        for asset_id, strategy_id, timeframe in get_watchlist():
            assert isinstance(asset_id, str) and asset_id
            assert isinstance(strategy_id, str) and strategy_id
            assert isinstance(timeframe, str) and timeframe

    def test_watchlist_only_enabled_assets(self):
        wl_asset_ids = {asset_id for asset_id, _, _ in get_watchlist()}
        for a in get_all_assets():
            if not a.get("enabled", True):
                assert a["id"] not in wl_asset_ids, (
                    f"Disabled asset '{a['id']}' appeared in watchlist"
                )

    def test_disabled_assets_not_in_watchlist(self):
        disabled = {a["id"] for a in get_all_assets() if not a.get("enabled", True)}
        wl_ids = {aid for aid, _, _ in get_watchlist()}
        overlap = disabled & wl_ids
        assert not overlap, f"Disabled assets in watchlist: {overlap}"
