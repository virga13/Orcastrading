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
from functools import lru_cache
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── Raw loaders (cached) ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_assets() -> dict:
    path = _CONFIG_DIR / "assets.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_strategies() -> dict:
    path = _CONFIG_DIR / "strategies.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    return asset.get("strategies", [])


def get_session(asset_id: str) -> dict:
    """Return the session timing dict for an asset."""
    return get_asset(asset_id).get("session", {})


def get_orb_config(asset_id: str) -> dict | None:
    """Return the ORB session config for an asset, or None if not configured."""
    return get_asset(asset_id).get("orb")


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
    return get_strategy_config(strategy_id).get("timeframe", "1d")


def get_strategy_lookback(strategy_id: str) -> int:
    """Return how many calendar days of history the strategy needs for warmup."""
    return int(get_strategy_config(strategy_id).get("lookback_days", 400))


# ── Convenience: all (asset_id, strategy_id) pairs that are enabled ──────────

def get_watchlist() -> list[tuple[str, str]]:
    """
    Return all (asset_id, strategy_id) pairs that are currently enabled.
    Used by the scanner to know what to run each day.
    """
    pairs = []
    for asset in get_all_assets():
        for strategy_id in asset.get("strategies", []):
            pairs.append((asset["id"], strategy_id))
    return pairs


def reload():
    """Force reload of all config files (clears lru_cache). Useful in long-running processes."""
    _load_assets.cache_clear()
    _load_strategies.cache_clear()
