"""
core/config.py — Central configuration loader.

Reads config/assets.yaml and config/strategies.yaml once at import time.
All other modules call get_asset(), get_strategy(), etc. rather than
touching the YAML files directly — so a switch from YAML to a DB or env
config is a one-file change.

Usage:
    from core.config import get_asset, get_all_assets, get_strategy_params

    asset  = get_asset("gold")            # dict from assets.yaml
    ticker = get_ticker("gold")           # resolves correct ticker for active provider
    params = get_strategy_params("orb")   # dict of default params
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"

# ── mtime-aware YAML cache ────────────────────────────────────────────────────
# Each entry: path_str → (data, mtime_float).
# Re-reads from disk only when the file's mtime changes — zero overhead on
# hot paths, instant pickup of edits without restarting the process.

_yaml_cache: dict[str, tuple[Any, float]] = {}


def _load_yaml(path: Path) -> Any:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    key   = str(path)
    entry = _yaml_cache.get(key)
    if entry is not None and entry[1] == mtime:
        return entry[0]
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _yaml_cache[key] = (data, mtime)
    return data


def _load_assets() -> dict:
    return _load_yaml(_CONFIG_DIR / "assets.yaml")


def _load_strategies() -> dict:
    return _load_yaml(_CONFIG_DIR / "strategies.yaml")


# ── Asset API ─────────────────────────────────────────────────────────────────

def get_all_assets() -> list[dict]:
    """Return all assets from assets.yaml as a list of dicts."""
    return _load_assets()["assets"]


def get_asset(asset_id: str) -> dict:
    """Return one asset dict by id. Raises KeyError if not found."""
    for a in get_all_assets():
        if a["id"] == asset_id:
            return a
    raise KeyError(f"Unknown asset id '{asset_id}'. Check config/assets.yaml.")


def get_asset_by_ticker(ticker: str) -> dict | None:
    """Find asset by any of its tickers (yfinance, eodhd, ccxt). Returns None if not found."""
    for a in get_all_assets():
        if ticker in a.get("tickers", {}).values():
            return a
    return None


def get_ticker(asset_id: str, source: str | None = None) -> str:
    """
    Return the ticker for an asset, using the given source.
    If source is None, uses the DATA_PROVIDER env var (default: yfinance).
    Falls back to yfinance ticker if the requested source isn't configured.
    """
    asset = get_asset(asset_id)
    tickers = asset.get("tickers", {})
    provider = (source or os.getenv("DATA_PROVIDER", "yfinance")).lower()

    if provider in tickers:
        return tickers[provider]
    # Fallback: try yfinance ticker
    if "yfinance" in tickers:
        return tickers["yfinance"]
    raise KeyError(
        f"No ticker configured for asset '{asset_id}' with provider '{provider}'. "
        f"Check config/assets.yaml."
    )


def get_enabled_strategies(asset_id: str) -> list[str]:
    """Return the list of strategy IDs enabled for an asset."""
    asset = get_asset(asset_id)
    strats = asset.get("strategies", [])
    return [s if isinstance(s, str) else s["id"] for s in strats]


def get_strategy_timeframes(asset_id: str, strategy_id: str) -> list[str]:
    """
    Return the timeframes configured for a strategy on a specific asset.
    Falls back to the strategy's default_timeframe if not explicitly configured.
    """
    asset = get_asset(asset_id)
    for entry in asset.get("strategies", []):
        if isinstance(entry, str):
            if entry == strategy_id:
                return [get_strategy_timeframe(strategy_id)]
        elif entry.get("id") == strategy_id:
            return entry.get("timeframes", [get_strategy_timeframe(strategy_id)])
    return [get_strategy_timeframe(strategy_id)]


def get_asset_strategy_params(asset_id: str, strategy_id: str) -> dict:
    """
    Return per-asset param overrides for a strategy, or {} if none defined.
    Asset entries may specify a 'params' dict that overrides the strategy defaults.

    Example in assets.yaml:
        strategies:
          - id: mtf_trend
            params:
              shorts_enabled: false
    """
    asset = get_asset(asset_id)
    for entry in asset.get("strategies", []):
        if isinstance(entry, dict) and entry.get("id") == strategy_id:
            return entry.get("params", {})
    return {}


def get_session(asset_id: str) -> dict:
    """Return the session timing dict for an asset."""
    return get_asset(asset_id).get("session", {})


def get_orb_config(asset_id: str) -> dict | None:
    """Return the ORB session config for an asset, or None if not configured."""
    return get_asset(asset_id).get("orb")


def get_asset_baseline(asset_id: str) -> dict | None:
    """
    Return the validated backtest baseline for an asset, or None if not set.
    Keys: win_rate, net_pf, avg_net_pnl_r, max_drawdown_r, kelly_25pct, n_trades, period.
    Set in config/assets.yaml under 'baseline:' — only edit after re-running walk-forward validation.
    """
    try:
        return get_asset(asset_id).get("baseline") or None
    except KeyError:
        return None


def get_asset_map() -> dict[str, dict]:
    """
    Return a dict keyed by yfinance ticker, value is the full asset dict.
    Used by scanner and report modules as a fast lookup.
    Only includes assets that have a yfinance ticker configured.
    """
    result = {}
    for a in get_all_assets():
        yf = a.get("tickers", {}).get("yfinance")
        if yf:
            result[yf] = a
    return result


# ── Strategy API ──────────────────────────────────────────────────────────────

def get_all_strategies() -> dict[str, dict]:
    """Return the full strategies dict from strategies.yaml."""
    return _load_strategies()["strategies"]


def get_strategy_config(strategy_id: str) -> dict:
    """Return the full strategy config dict. Raises KeyError if not found."""
    strats = get_all_strategies()
    if strategy_id not in strats:
        raise KeyError(
            f"Unknown strategy '{strategy_id}'. Check config/strategies.yaml."
        )
    return strats[strategy_id]


def get_strategy_params(strategy_id: str) -> dict:
    """Return just the default params dict for a strategy."""
    return get_strategy_config(strategy_id).get("params", {})


def get_strategy_timeframe(strategy_id: str) -> str:
    """Return the primary timeframe for a strategy (e.g. '1d', '15m')."""
    return get_strategy_config(strategy_id).get("default_timeframe", "1d")


def get_strategy_lookback(strategy_id: str) -> int:
    """Return how many calendar days of history the strategy needs for warmup."""
    return int(get_strategy_config(strategy_id).get("lookback_days", 400))


# ── Convenience: all (asset_id, strategy_id) pairs that are enabled ──────────

def get_watchlist() -> list[tuple[str, str, str]]:
    """
    Return all (asset_id, strategy_id, timeframe) triples that are currently enabled.
    Assets with enabled=false are skipped. Defaults to enabled if key absent.
    Each strategy entry may be a plain string (uses default_timeframe) or a dict
    with 'id' and 'timeframes' keys (expanded to one triple per timeframe).
    """
    triples = []
    for asset in get_all_assets():
        if not asset.get("enabled", True):
            continue
        for entry in asset.get("strategies", []):
            if isinstance(entry, str):
                strategy_id = entry
                triples.append((asset["id"], strategy_id, get_strategy_timeframe(strategy_id)))
            else:
                strategy_id = entry["id"]
                for tf in entry.get("timeframes", [get_strategy_timeframe(strategy_id)]):
                    triples.append((asset["id"], strategy_id, tf))
    return triples


def reload():
    """Force reload of all config files (clears mtime cache). Useful in tests."""
    _yaml_cache.clear()
