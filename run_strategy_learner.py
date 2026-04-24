"""
run_strategy_learner.py — Iterative regime-aware strategy optimizer.

Learns from per-trade outcome patterns:
  1. Runs a backtest and captures regime features for every filled trade
     (ADX, RSI, ATR%, EMA alignment, stochastic, volume trend, BB position).
  2. Analyzes win vs loss feature distributions on the TRAINING window only.
  3. Proposes filter improvements — both tightening existing params (adx_threshold,
     rsi_max) and adding new pre-filters (atr_pct_max, require_ema_aligned, etc.).
  4. Validates each improvement on the held-out TEST window.
  5. Keeps improvements that raise test PF by >= MIN_PF_GAIN; discards the rest.
  6. Iterates until no improvement is found or max iterations are reached.
  7. Optionally writes final params back to config/strategies.yaml (--apply).

Usage:
    python run_strategy_learner.py --strategy ema_continuation
    python run_strategy_learner.py --strategy ema_pullback
    python run_strategy_learner.py --strategy ema_continuation --tickers GC=F,^GSPC,BTC-USD
    python run_strategy_learner.py --strategy ema_continuation --iterations 6 --apply
"""
import sys
import io
import argparse
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from rich import box

from p3_backtester.market_data import fetch_ohlcv, split_backtest_window
from p3_backtester.signal_scheduler import get_signal_indices
from p3_backtester.strategies.ema_continuation import EMAContinuationStrategy
from p3_backtester.strategies.ema_pullback import EMAPullbackStrategy

# Re-use infrastructure from the optimizer
from run_ema_optimization import (
    precompute_technicals,
    run_backtest_inmem,
    split_trades,
    _compute_pf,
    MIN_TRADES_TEST,
)

console      = Console()
YAML_PATH    = Path("config/strategies.yaml")
MIN_PF_GAIN  = 0.08   # minimum absolute OOS PF gain to keep an improvement
MIN_CAT_N    = 4      # minimum trades in a category to make a suggestion


# ── Strategy helpers ──────────────────────────────────────────────────────────

def _strategy_param_names(strategy_name: str) -> set[str]:
    cls = EMAContinuationStrategy if strategy_name == "ema_continuation" else EMAPullbackStrategy
    import inspect
    sig = inspect.signature(cls.__init__)
    return set(sig.parameters) - {"self"}


def build_strategy(strategy_name: str, params: dict):
    valid_keys = _strategy_param_names(strategy_name)
    filtered   = {k: v for k, v in params.items() if k in valid_keys}
    cls = EMAContinuationStrategy if strategy_name == "ema_continuation" else EMAPullbackStrategy
    return cls(**filtered)


def load_params(strategy_name: str) -> dict:
    with open(YAML_PATH) as f:
        config = yaml.safe_load(f)
    return config["strategies"][strategy_name]["params"].copy()


def save_params(strategy_name: str, params: dict):
    """Write updated params back to strategies.yaml, preserving all other content."""
    try:
        from ruamel.yaml import YAML
        yml = YAML()
        yml.preserve_quotes = True
        with open(YAML_PATH) as f:
            config = yml.load(f)
        config["strategies"][strategy_name]["params"].update(params)
        with open(YAML_PATH, "w") as f:
            yml.dump(config, f)
        return True
    except ImportError:
        # Fallback: standard yaml (loses comments)
        with open(YAML_PATH) as f:
            config = yaml.safe_load(f)
        config["strategies"][strategy_name]["params"].update(params)
        with open(YAML_PATH, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        return True


# ── Pre-filter helpers ────────────────────────────────────────────────────────

def apply_pre_filters(precomputed: list, filters: list) -> list:
    """Filter precomputed entries using a list of (tech) -> bool callables."""
    if not filters:
        return precomputed
    return [
        (i, tech, df_slice)
        for i, tech, df_slice in precomputed
        if all(fn(tech) for fn in filters)
    ]


def _make_filter(filter_spec: dict):
    """Build a (tech) -> bool callable from a filter spec dict."""
    ftype = filter_spec["type"]
    value = filter_spec.get("value")

    if ftype == "atr_pct_max":
        return lambda t, v=value: (t.get("atr_pct") or 999) <= v
    if ftype == "require_ema_aligned":
        return lambda t: t.get("ema_alignment") != "mixed"
    if ftype == "stoch_not_overbought":
        return lambda t: t.get("stoch_signal") != "overbought"
    if ftype == "stoch_not_oversold":
        return lambda t: t.get("stoch_signal") != "oversold"
    if ftype == "volume_not_decreasing":
        return lambda t: t.get("volume_trend") != "decreasing"
    if ftype == "bb_lower_or_below":
        return lambda t: t.get("bb_position") in ("lower", "below_lower")
    if ftype == "macd_not_bearish":
        return lambda t: t.get("macd_signal") != "bearish"
    if ftype == "macd_not_bullish":
        return lambda t: t.get("macd_signal") != "bullish"
    if ftype == "cmf_not_distribution":
        return lambda t: t.get("cmf_signal") != "distribution"
    if ftype == "cmf_not_accumulation":
        return lambda t: t.get("cmf_signal") != "accumulation"
    if ftype == "require_williams_oversold":
        return lambda t: t.get("williams_r_sig") == "oversold"
    if ftype == "min_consec_bearish":
        return lambda t, v=value: (t.get("consec_bearish") or 0) >= v
    if ftype == "atr_ratio_max":
        return lambda t, v=value: (t.get("atr_ratio") or 1.0) <= v
    if ftype == "min_candle_close_pct":
        return lambda t, v=value: (t.get("candle_close_pct") or 0.5) >= v
    raise ValueError(f"Unknown filter type: {ftype}")


# ── Feature analysis ──────────────────────────────────────────────────────────

def _trades_to_df(trades: list[dict]) -> pd.DataFrame:
    filled = [t for t in trades if t["outcome"] != "EXPIRED"]
    if not filled:
        return pd.DataFrame()
    return pd.DataFrame([{
        "outcome":       t["outcome"],
        "pnl_r":         t["pnl_r"],
        "is_win":        int(t["outcome"] in ("WIN", "PARTIAL_WIN")),
        "adx":           t.get("adx"),
        "rsi":           t.get("rsi"),
        "atr_pct":       t.get("atr_pct"),
        "ema_alignment": t.get("ema_alignment", "mixed"),
        "stoch_signal":  t.get("stoch_signal",  "neutral"),
        "volume_trend":  t.get("volume_trend",  "neutral"),
        "bb_position":     t.get("bb_position",    "middle"),
        "trend":           t.get("trend",          "sideways"),
        "macd_signal":     t.get("macd_signal",    "neutral"),
        "cmf_signal":      t.get("cmf_signal",     "neutral"),
        "williams_r_sig":  t.get("williams_r_sig", "neutral"),
        "consec_bearish":  t.get("consec_bearish", 0),
        "atr_ratio":       t.get("atr_ratio",      1.0),
        "candle_close_pct":t.get("candle_close_pct", 0.5),
    } for t in filled])


def _pf_df(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    wins   = df[df["outcome"].isin(["WIN", "PARTIAL_WIN"])]["pnl_r"].sum()
    losses = abs(df[df["outcome"] == "LOSS"]["pnl_r"].sum())
    return round(wins / losses, 3) if losses > 0 else (999.0 if wins > 0 else 0.0)


def analyze_and_suggest(
    train_trades: list[dict],
    current_params: dict,
    active_pre_filters: list,
    strategy_name: str,
) -> list[dict]:
    """
    Analyze win/loss feature distributions on training trades.
    Returns filter suggestions sorted by expected PF improvement (descending).

    Each suggestion dict:
      type          : "tighten_param" | "add_filter"
      label         : human-readable name
      param         : param name or filter spec type
      new_value     : new param value (for tighten_param)
      filter_spec   : dict for _make_filter (for add_filter)
      train_pf_gain : expected PF gain on training data
      reason        : explanation string
    """
    df = _trades_to_df(train_trades)
    if df.empty or len(df) < 8:
        return []

    base_pf = _pf_df(df)
    suggestions = []

    # ── 1. ADX tightening ────────────────────────────────────────────────────
    adx_s = df["adx"].dropna()
    if len(adx_s) >= 10:
        current_adx = current_params.get("adx_threshold", 18)
        # Try the 40th, 50th, 60th percentile of observed ADX values
        for pct in [40, 50, 60]:
            thr = adx_s.quantile(pct / 100)
            if thr <= current_adx + 1:
                continue
            above = df[df["adx"] >= thr]
            below = df[df["adx"] <  thr]
            if len(above) < MIN_CAT_N or len(below) < MIN_CAT_N:
                continue
            pf_above = _pf_df(above)
            if pf_above > base_pf and _pf_df(above) > _pf_df(below):
                suggestions.append({
                    "type":          "tighten_param",
                    "label":         f"adx_threshold → {int(thr)}",
                    "param":         "adx_threshold",
                    "new_value":     int(thr),
                    "filter_spec":   None,
                    "train_pf_gain": round(pf_above - base_pf, 3),
                    "reason": (
                        f"ADX≥{int(thr)}: PF={pf_above:.2f} (n={len(above)})  "
                        f"ADX<{int(thr)}: PF={_pf_df(below):.2f} (n={len(below)})"
                    ),
                })
                break  # most conservative threshold first

    # ── 2. RSI tightening ────────────────────────────────────────────────────
    rsi_s = df["rsi"].dropna()
    current_rsi = current_params.get("rsi_max", 70)
    if len(rsi_s) >= 10:
        for thr in [45, 50, 55, 60]:
            if thr >= current_rsi:
                continue
            below = df[df["rsi"] <= thr]
            above = df[df["rsi"] >  thr]
            if len(below) < MIN_CAT_N or len(above) < MIN_CAT_N:
                continue
            pf_below = _pf_df(below)
            if pf_below > base_pf and pf_below > _pf_df(above):
                suggestions.append({
                    "type":          "tighten_param",
                    "label":         f"rsi_max → {thr}",
                    "param":         "rsi_max",
                    "new_value":     thr,
                    "filter_spec":   None,
                    "train_pf_gain": round(pf_below - base_pf, 3),
                    "reason": (
                        f"RSI≤{thr}: PF={pf_below:.2f} (n={len(below)})  "
                        f"RSI>{thr}: PF={_pf_df(above):.2f} (n={len(above)})"
                    ),
                })
                break

    # ── 3. ATR% cap (new pre-filter) ─────────────────────────────────────────
    atr_s = df["atr_pct"].dropna()
    if len(atr_s) >= 10:
        for thr in [1.2, 1.5, 2.0]:
            below = df[df["atr_pct"] <= thr]
            above = df[df["atr_pct"] >  thr]
            if len(below) < MIN_CAT_N or len(above) < MIN_CAT_N:
                continue
            pf_below = _pf_df(below)
            if pf_below > base_pf and pf_below > _pf_df(above):
                suggestions.append({
                    "type":          "add_filter",
                    "label":         f"atr_pct_max={thr}",
                    "param":         "atr_pct_max",
                    "new_value":     thr,
                    "filter_spec":   {"type": "atr_pct_max", "value": thr},
                    "train_pf_gain": round(pf_below - base_pf, 3),
                    "reason": (
                        f"ATR%≤{thr}: PF={pf_below:.2f} (n={len(below)})  "
                        f"ATR%>{thr}: PF={_pf_df(above):.2f} (n={len(above)})"
                    ),
                })
                break

    # ── 4. EMA alignment filter ───────────────────────────────────────────────
    if "ema_alignment" in df.columns:
        for exclude in ["mixed"]:
            sub = df[df["ema_alignment"] == exclude]
            rest = df[df["ema_alignment"] != exclude]
            if len(sub) < MIN_CAT_N or len(rest) < MIN_CAT_N:
                continue
            pf_rest = _pf_df(rest)
            if pf_rest > base_pf:
                suggestions.append({
                    "type":          "add_filter",
                    "label":         "require_ema_aligned",
                    "param":         "require_ema_aligned",
                    "new_value":     True,
                    "filter_spec":   {"type": "require_ema_aligned"},
                    "train_pf_gain": round(pf_rest - base_pf, 3),
                    "reason": (
                        f"Exclude '{exclude}' EMA alignment ({len(sub)} trades) "
                        f"→ PF {pf_rest:.2f} (was {base_pf:.2f})"
                    ),
                })

    # ── 5. Stochastic overbought/oversold filter ─────────────────────────────
    for stoch_cat in ["overbought", "oversold"]:
        sub  = df[df["stoch_signal"] == stoch_cat]
        rest = df[df["stoch_signal"] != stoch_cat]
        if len(sub) < MIN_CAT_N or len(rest) < MIN_CAT_N:
            continue
        pf_rest = _pf_df(rest)
        if pf_rest > base_pf:
            ftype = f"stoch_not_{stoch_cat}"
            suggestions.append({
                "type":          "add_filter",
                "label":         ftype,
                "param":         ftype,
                "new_value":     True,
                "filter_spec":   {"type": ftype},
                "train_pf_gain": round(pf_rest - base_pf, 3),
                "reason": (
                    f"Exclude stoch '{stoch_cat}' ({len(sub)} trades) "
                    f"→ PF {pf_rest:.2f} (was {base_pf:.2f})"
                ),
            })

    # ── 6. Volume trend filter ────────────────────────────────────────────────
    sub  = df[df["volume_trend"] == "decreasing"]
    rest = df[df["volume_trend"] != "decreasing"]
    if len(sub) >= MIN_CAT_N and len(rest) >= MIN_CAT_N:
        pf_rest = _pf_df(rest)
        if pf_rest > base_pf:
            suggestions.append({
                "type":          "add_filter",
                "label":         "volume_not_decreasing",
                "param":         "volume_not_decreasing",
                "new_value":     True,
                "filter_spec":   {"type": "volume_not_decreasing"},
                "train_pf_gain": round(pf_rest - base_pf, 3),
                "reason": (
                    f"Exclude 'decreasing' volume ({len(sub)} trades) "
                    f"→ PF {pf_rest:.2f} (was {base_pf:.2f})"
                ),
            })

    # ── 7. BB position filter ─────────────────────────────────────────────────
    for bb_keep in [["lower", "below_lower"], ["middle", "lower", "below_lower"]]:
        sub  = df[~df["bb_position"].isin(bb_keep)]
        rest = df[df["bb_position"].isin(bb_keep)]
        if len(sub) < MIN_CAT_N or len(rest) < MIN_CAT_N:
            continue
        pf_rest = _pf_df(rest)
        if pf_rest > base_pf:
            label = "bb_lower_or_below"
            suggestions.append({
                "type":          "add_filter",
                "label":         label,
                "param":         label,
                "new_value":     True,
                "filter_spec":   {"type": label},
                "train_pf_gain": round(pf_rest - base_pf, 3),
                "reason": (
                    f"BB position in {bb_keep} ({len(rest)} trades) "
                    f"→ PF {pf_rest:.2f} (was {base_pf:.2f})"
                ),
            })
            break

    # ── 8. MACD signal filter ─────────────────────────────────────────────────
    for macd_cat in ["bearish", "bullish"]:
        sub  = df[df["macd_signal"] == macd_cat]
        rest = df[df["macd_signal"] != macd_cat]
        if len(sub) < MIN_CAT_N or len(rest) < MIN_CAT_N:
            continue
        pf_rest = _pf_df(rest)
        if pf_rest > base_pf:
            ftype = f"macd_not_{macd_cat}"
            suggestions.append({
                "type":          "add_filter",
                "label":         ftype,
                "param":         ftype,
                "new_value":     True,
                "filter_spec":   {"type": ftype},
                "train_pf_gain": round(pf_rest - base_pf, 3),
                "reason": (
                    f"Exclude MACD '{macd_cat}' ({len(sub)} trades) "
                    f"→ PF {pf_rest:.2f} (was {base_pf:.2f})"
                ),
            })

    # ── 9. CMF signal filter ──────────────────────────────────────────────────
    for cmf_cat in ["distribution", "accumulation"]:
        sub  = df[df["cmf_signal"] == cmf_cat]
        rest = df[df["cmf_signal"] != cmf_cat]
        if len(sub) < MIN_CAT_N or len(rest) < MIN_CAT_N:
            continue
        pf_rest = _pf_df(rest)
        if pf_rest > base_pf:
            ftype = f"cmf_not_{cmf_cat}"
            suggestions.append({
                "type":          "add_filter",
                "label":         ftype,
                "param":         ftype,
                "new_value":     True,
                "filter_spec":   {"type": ftype},
                "train_pf_gain": round(pf_rest - base_pf, 3),
                "reason": (
                    f"Exclude CMF '{cmf_cat}' ({len(sub)} trades) "
                    f"→ PF {pf_rest:.2f} (was {base_pf:.2f})"
                ),
            })

    # ── 10. Williams %R filter ────────────────────────────────────────────────
    wr_oversold = df[df["williams_r_sig"] == "oversold"]
    wr_rest     = df[df["williams_r_sig"] != "oversold"]
    if len(wr_oversold) >= MIN_CAT_N and len(wr_rest) >= MIN_CAT_N:
        pf_oversold = _pf_df(wr_oversold)
        if pf_oversold > base_pf and pf_oversold > _pf_df(wr_rest):
            suggestions.append({
                "type":          "add_filter",
                "label":         "require_williams_oversold",
                "param":         "require_williams_oversold",
                "new_value":     True,
                "filter_spec":   {"type": "require_williams_oversold"},
                "train_pf_gain": round(pf_oversold - base_pf, 3),
                "reason": (
                    f"Williams %R oversold: PF={pf_oversold:.2f} (n={len(wr_oversold)})  "
                    f"not-oversold: PF={_pf_df(wr_rest):.2f} (n={len(wr_rest)})"
                ),
            })

    # ── 11. Consecutive bearish bars ─────────────────────────────────────────
    cb_s = df["consec_bearish"].dropna()
    if len(cb_s) >= 10:
        for thr in [2, 3, 4]:
            above = df[df["consec_bearish"] >= thr]
            below = df[df["consec_bearish"] <  thr]
            if len(above) < MIN_CAT_N or len(below) < MIN_CAT_N:
                continue
            pf_above = _pf_df(above)
            if pf_above > base_pf and pf_above > _pf_df(below):
                suggestions.append({
                    "type":          "add_filter",
                    "label":         f"min_consec_bearish={thr}",
                    "param":         "min_consec_bearish",
                    "new_value":     thr,
                    "filter_spec":   {"type": "min_consec_bearish", "value": thr},
                    "train_pf_gain": round(pf_above - base_pf, 3),
                    "reason": (
                        f"≥{thr} consec bearish bars: PF={pf_above:.2f} (n={len(above)})  "
                        f"<{thr}: PF={_pf_df(below):.2f} (n={len(below)})"
                    ),
                })
                break

    # ── 12. ATR ratio (volatility contracting = better bounce setup) ──────────
    atr_r = df["atr_ratio"].dropna()
    if len(atr_r) >= 10:
        for thr in [0.9, 1.0, 1.2]:
            below = df[df["atr_ratio"] <= thr]
            above = df[df["atr_ratio"] >  thr]
            if len(below) < MIN_CAT_N or len(above) < MIN_CAT_N:
                continue
            pf_below = _pf_df(below)
            if pf_below > base_pf and pf_below > _pf_df(above):
                suggestions.append({
                    "type":          "add_filter",
                    "label":         f"atr_ratio_max={thr}",
                    "param":         "atr_ratio_max",
                    "new_value":     thr,
                    "filter_spec":   {"type": "atr_ratio_max", "value": thr},
                    "train_pf_gain": round(pf_below - base_pf, 3),
                    "reason": (
                        f"ATR ratio ≤{thr} (contracting): PF={pf_below:.2f} (n={len(below)})  "
                        f">{thr}: PF={_pf_df(above):.2f} (n={len(above)})"
                    ),
                })
                break

    # ── 13. Candle close position (>0.5 = closed upper half = rejection) ──────
    cc_s = df["candle_close_pct"].dropna()
    if len(cc_s) >= 10:
        for thr in [0.4, 0.5, 0.6]:
            above = df[df["candle_close_pct"] >= thr]
            below = df[df["candle_close_pct"] <  thr]
            if len(above) < MIN_CAT_N or len(below) < MIN_CAT_N:
                continue
            pf_above = _pf_df(above)
            if pf_above > base_pf and pf_above > _pf_df(below):
                suggestions.append({
                    "type":          "add_filter",
                    "label":         f"min_candle_close_pct={thr}",
                    "param":         "min_candle_close_pct",
                    "new_value":     thr,
                    "filter_spec":   {"type": "min_candle_close_pct", "value": thr},
                    "train_pf_gain": round(pf_above - base_pf, 3),
                    "reason": (
                        f"Candle closed in top {100-int(thr*100)}%: PF={pf_above:.2f} (n={len(above)})  "
                        f"lower: PF={_pf_df(below):.2f} (n={len(below)})"
                    ),
                })
                break

    suggestions.sort(key=lambda s: s["train_pf_gain"], reverse=True)
    return suggestions


# ── Iteration display helpers ─────────────────────────────────────────────────

def _print_feature_summary(train_df: pd.DataFrame):
    """Print a quick win/loss feature comparison table."""
    if train_df.empty:
        return
    wins   = train_df[train_df["is_win"] == 1]
    losses = train_df[train_df["is_win"] == 0]

    t = Table(title="Feature summary (training window)", box=box.SIMPLE_HEAVY, padding=(0, 1))
    t.add_column("Feature",     style="bold")
    t.add_column("Win median",  justify="right")
    t.add_column("Loss median", justify="right")
    t.add_column("Δ",           justify="right")

    for col in ["adx", "rsi", "atr_pct"]:
        if col not in train_df.columns:
            continue
        wm = wins[col].dropna().median()
        lm = losses[col].dropna().median()
        if pd.notna(wm) and pd.notna(lm):
            delta = wm - lm
            color = "green" if abs(delta) > 2 else "dim"
            t.add_row(col, f"{wm:.2f}", f"{lm:.2f}",
                      f"[{color}]{delta:+.2f}[/{color}]")

    for col in ["consec_bearish", "atr_ratio", "candle_close_pct"]:
        if col not in train_df.columns:
            continue
        wm = wins[col].dropna().median()
        lm = losses[col].dropna().median()
        if pd.notna(wm) and pd.notna(lm):
            delta = wm - lm
            color = "green" if abs(delta) > 0.1 else "dim"
            t.add_row(col, f"{wm:.2f}", f"{lm:.2f}",
                      f"[{color}]{delta:+.2f}[/{color}]")

    for cat_col in ["ema_alignment", "stoch_signal", "volume_trend", "macd_signal", "cmf_signal", "williams_r_sig", "bb_position"]:
        if cat_col not in train_df.columns:
            continue
        wr_by = train_df.groupby(cat_col)["is_win"].agg(["mean", "count"])
        for cat, row in wr_by.iterrows():
            if row["count"] >= MIN_CAT_N:
                t.add_row(
                    f"{cat_col}={cat}",
                    f"WR {row['mean']:.0%} (n={int(row['count'])})",
                    "—", "—",
                )

    console.print(t)


def _print_iteration_table(history: list[dict]):
    t = Table(title="Learning progression", box=box.SIMPLE_HEAVY, padding=(0, 1))
    t.add_column("Iter",    style="dim", justify="right")
    t.add_column("Change",  width=40)
    t.add_column("Test PF", justify="right", style="bold")
    t.add_column("Test n",  justify="right")
    t.add_column("Test WR", justify="right")
    t.add_column("Result",  justify="center")

    def _c(v):
        return "green" if v > 1.0 else ("yellow" if v >= 0.9 else "red")

    for h in history:
        t.add_row(
            str(h["iteration"]),
            h["change"],
            f"[{_c(h['test_pf'])}]{h['test_pf']:.2f}[/{_c(h['test_pf'])}]",
            str(h["test_n"]),
            f"{h['test_wr']:.0%}",
            h["result"],
        )
    console.print(t)


# ── Core learning loop ────────────────────────────────────────────────────────

def run_learner(
    strategy_name: str,
    ticker:        str,
    interval:      str,
    start:         str,
    end:           str,
    initial_params: dict,
    n_iterations:  int,
    apply:         bool,
):
    console.rule(f"[bold]{strategy_name} — {ticker}[/bold]")

    # ── Fetch data & precompute ────────────────────────────────────────────────
    console.print("Fetching data and pre-computing technicals...")
    df = fetch_ohlcv(ticker, interval, start, end)
    first_valid, _ = split_backtest_window(df, start)
    signal_indices = get_signal_indices(df, "session-open", first_valid, every_n_bars=10)
    precomputed    = precompute_technicals(df, signal_indices, ticker, interval)
    console.print(f"  {len(df)} bars  |  {len(precomputed)} valid signal bars")

    n = len(df)
    train_end_dt = df.index[int(n * 0.60) - 1]
    test_start_dt = df.index[int(n * 0.80)]
    console.print(
        f"  Train ≤{train_end_dt.date()}  |  "
        f"Test >{df.index[int(n*0.80)-1].date()}  "
        f"({n - int(n*0.80)} bars)"
    )

    current_params      = initial_params.copy()
    active_pre_filters  = []   # list of (tech) -> bool callables
    active_filter_specs = []   # list of filter spec dicts (for reporting)
    history = []

    for iteration in range(n_iterations + 1):  # +1 for baseline
        # ── Run backtest with current state ───────────────────────────────────
        filtered_precomp = apply_pre_filters(precomputed, active_pre_filters)
        strategy = build_strategy(strategy_name, current_params)
        all_trades = run_backtest_inmem(strategy, filtered_precomp, df)

        train_t, val_t, test_t = split_trades(all_trades, df)
        test_pf, test_wr, test_n = _compute_pf(test_t)
        oos_pf, _, oos_n = _compute_pf(val_t + test_t)

        if iteration == 0:
            baseline_pf = test_pf
            history.append({
                "iteration": "Base",
                "change":    "Initial params",
                "test_pf":   test_pf,
                "test_wr":   test_wr,
                "test_n":    test_n,
                "result":    "—",
            })
            console.print(
                f"\n[bold]Baseline:[/bold] "
                f"Test PF={test_pf:.2f}  WR={test_wr:.0%}  n={test_n}  "
                f"OOS PF={oos_pf:.2f}  n={oos_n}"
            )
            if test_n < MIN_TRADES_TEST:
                console.print(
                    f"[yellow]Only {test_n} test trades — not enough data for reliable learning. "
                    f"Try a longer date range.[/yellow]"
                )
                break
        else:
            # We already applied and validated above — just record
            pass

        # ── Analyze training window ───────────────────────────────────────────
        train_df = _trades_to_df(train_t)
        if iteration == 0:
            _print_feature_summary(train_df)

        suggestions = analyze_and_suggest(
            train_t, current_params, active_pre_filters, strategy_name
        )

        if not suggestions:
            console.print("[dim]No further filter improvements found — converged.[/dim]")
            break

        # ── Try the best suggestion ───────────────────────────────────────────
        best = suggestions[0]
        console.print(
            f"\n[bold]Iter {iteration}:[/bold] "
            f"Trying [{best['label']}]  train gain +{best['train_pf_gain']:.2f}PF\n"
            f"  {best['reason']}"
        )

        # Build candidate state
        candidate_params  = current_params.copy()
        candidate_filters = list(active_pre_filters)

        if best["type"] == "tighten_param":
            candidate_params[best["param"]] = best["new_value"]
        else:
            candidate_filters.append(_make_filter(best["filter_spec"]))

        # Validate on test window
        cand_precomp = apply_pre_filters(precomputed, candidate_filters)
        cand_strategy = build_strategy(strategy_name, candidate_params)
        cand_trades   = run_backtest_inmem(cand_strategy, cand_precomp, df)
        _, _, cand_test_t = split_trades(cand_trades, df)
        cand_pf, cand_wr, cand_n = _compute_pf(cand_test_t)

        gain     = cand_pf - test_pf
        kept     = gain >= MIN_PF_GAIN and cand_n >= MIN_TRADES_TEST

        console.print(
            f"  Validation: Test PF {test_pf:.2f} → {cand_pf:.2f}  "
            f"(Δ{gain:+.2f})  n={cand_n}  "
            + ("[bold green]KEPT ✓[/bold green]" if kept else "[bold red]REJECTED ✗[/bold red]")
        )

        history.append({
            "iteration": str(iteration),
            "change":    best["label"],
            "test_pf":   cand_pf if kept else test_pf,
            "test_wr":   cand_wr if kept else test_wr,
            "test_n":    cand_n  if kept else test_n,
            "result":    "KEPT ✓" if kept else f"REJECTED (Δ{gain:+.2f}, n={cand_n})",
        })

        if kept:
            current_params = candidate_params
            active_pre_filters = candidate_filters
            if best["filter_spec"]:
                active_filter_specs.append(best["filter_spec"])
        else:
            break  # No improvement — stop iterating

    # ── Final summary ─────────────────────────────────────────────────────────
    console.print()
    _print_iteration_table(history)

    final_pf   = history[-1]["test_pf"]
    pf_vs_base = final_pf - baseline_pf
    console.print(
        f"\n[bold]Final result for {strategy_name} on {ticker}:[/bold]\n"
        f"  Test PF: {baseline_pf:.2f} → [bold]{final_pf:.2f}[/bold]  "
        f"(Δ{pf_vs_base:+.2f})\n"
        f"  Param changes: "
        + (str({k: v for k, v in current_params.items() if initial_params.get(k) != v}) or "none")
        + f"\n  Pre-filters: {[s['type'] for s in active_filter_specs] or 'none'}"
    )

    if apply and (current_params != initial_params):
        # Only write strategy-constructor params (not pre-filter specs)
        save_params(strategy_name, current_params)
        console.print(
            f"\n[bold green]✓ config/strategies.yaml updated for {strategy_name}[/bold green]"
        )
    elif apply:
        console.print("\n[dim]No param changes to write — strategies.yaml unchanged.[/dim]")

    if active_filter_specs:
        console.print(
            "\n[yellow]Note:[/yellow] Pre-filters (atr_pct_max, stoch, volume, etc.) are "
            "not yet in strategies.yaml.\nTo make them permanent, add them to the strategy "
            "class constructor and strategies.yaml manually."
        )

    return current_params, active_filter_specs


# ── Multi-ticker consensus ────────────────────────────────────────────────────

def run_multi_ticker(
    strategy_name: str,
    tickers:       list[str],
    interval:      str,
    start:         str,
    end:           str,
    initial_params: dict,
    n_iterations:  int,
    apply:         bool,
):
    """Run learner on each ticker and find consensus improvements."""
    per_ticker_final: dict[str, dict] = {}
    per_ticker_filters: dict[str, list] = {}

    for ticker in tickers:
        final_params, filter_specs = run_learner(
            strategy_name, ticker, interval, start, end,
            initial_params, n_iterations, apply=False,
        )
        per_ticker_final[ticker]   = final_params
        per_ticker_filters[ticker] = filter_specs

    if len(tickers) == 1:
        if apply and per_ticker_final:
            params = list(per_ticker_final.values())[0]
            if params != initial_params:
                save_params(strategy_name, params)
                console.print(f"\n[bold green]✓ strategies.yaml updated[/bold green]")
        return

    # ── Cross-ticker consensus ────────────────────────────────────────────────
    console.rule("[bold]Cross-ticker consensus improvements[/bold]")

    # Find param changes that appear in majority of tickers
    param_votes: dict[str, dict] = defaultdict(lambda: defaultdict(int))  # param -> value -> count
    for ticker, params in per_ticker_final.items():
        for k, v in params.items():
            if initial_params.get(k) != v:
                param_votes[k][v] += 1

    majority = len(tickers) // 2 + 1
    consensus_params = initial_params.copy()
    for param, votes in param_votes.items():
        best_value = max(votes, key=lambda v: votes[v])
        if votes[best_value] >= majority:
            consensus_params[param] = best_value
            console.print(
                f"  [green]Consensus:[/green] {param} → {best_value} "
                f"({votes[best_value]}/{len(tickers)} tickers)"
            )

    # Filter consensus
    filter_votes: dict[str, int] = defaultdict(int)
    for ticker, fspecs in per_ticker_filters.items():
        for spec in fspecs:
            filter_votes[spec["type"]] += 1
    consensus_filters = [
        ftype for ftype, count in filter_votes.items() if count >= majority
    ]
    if consensus_filters:
        console.print(f"  [green]Consensus filters:[/green] {consensus_filters}")

    if consensus_params == initial_params and not consensus_filters:
        console.print("  [dim]No consensus improvements found.[/dim]")
        return

    if apply and consensus_params != initial_params:
        save_params(strategy_name, consensus_params)
        console.print(f"\n[bold green]✓ strategies.yaml updated with consensus params[/bold green]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Iterative regime-aware strategy optimizer")
    parser.add_argument("--strategy",   choices=["ema_continuation", "ema_pullback"],
                        required=True)
    parser.add_argument("--tickers",    default="GC=F",
                        help="Comma-separated tickers (default: GC=F)")
    parser.add_argument("--interval",   default="1d")
    parser.add_argument("--start",      default="2018-01-01")
    parser.add_argument("--end",        default="2026-04-19")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Max improvement iterations per ticker (default: 5)")
    parser.add_argument("--apply",      action="store_true",
                        help="Write final params back to config/strategies.yaml")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]

    console.print(f"\n[bold]Strategy Learner[/bold]  — {args.strategy}")
    console.print(
        f"Assets: {', '.join(tickers)}  Interval: {args.interval}  "
        f"Window: {args.start} → {args.end}  "
        f"Max iterations: {args.iterations}"
        + ("  [bold green][--apply][/bold green]" if args.apply else "")
    )

    initial_params = load_params(args.strategy)
    console.print(f"Starting params: {initial_params}\n")

    run_multi_ticker(
        strategy_name=args.strategy,
        tickers=tickers,
        interval=args.interval,
        start=args.start,
        end=args.end,
        initial_params=initial_params,
        n_iterations=args.iterations,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()
