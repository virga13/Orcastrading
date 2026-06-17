"""
pass1_signal_gen.py — iterates over signal bar indices, computes technicals on
sliced OHLCV data, and generates trade setups.

Default mode: rule engine (free, deterministic, instant).
With use_claude=True: calls synthesize_full via Claude API (~$0.025-0.035/signal).

Cache files are named: {ticker}_{interval}_{unix_timestamp_ms}[_claude].json
"""
import json
import time
from pathlib import Path

import pandas as pd
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console

from p3_backtester.bar_slicer import recompute_technical
from p3_backtester.schema import BacktestConfig, SignalRecord
from p3_backtester.strategies.base import StrategyBase
from p3_backtester.strategies.rule_engine_strategy import RuleEngineStrategy

console = Console()

CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(ticker: str, interval: str, bar_time: pd.Timestamp, use_claude: bool, strategy_name: str = "rule_engine") -> Path:
    unix_ms  = int(bar_time.timestamp() * 1000)
    safe     = ticker.replace("=", "").replace("-", "").replace("^", "")
    strat    = strategy_name.lower().replace(" ", "_").replace("-", "_")
    suffix   = "_claude" if use_claude else ""
    return CACHE_DIR / f"{safe}_{interval}_{strat}_{unix_ms}{suffix}.json"


def run_pass1(
    config: BacktestConfig,
    df: pd.DataFrame,
    signal_indices: list[int],
    use_claude: bool = False,
    strategy: StrategyBase | None = None,
) -> list[SignalRecord]:
    """
    For each signal bar index:
    1. Slice df to df.iloc[:i+1] — strict no-lookahead
    2. Recompute technical indicators on the slice
    3. Generate setups via rule engine (default) or Claude API (use_claude=True)
    4. Write JSON cache

    Skips bars where cache already exists (unless config.force_regenerate).
    Returns list of SignalRecord (metadata only — used by Pass 2).
    """
    CACHE_DIR.mkdir(exist_ok=True)

    ticker   = config.ticker
    interval = config.interval

    # Default strategy: rule engine (backwards compatible)
    if strategy is None:
        strategy = RuleEngineStrategy()

    mode_label = "[bold cyan]Claude[/bold cyan]" if use_claude else f"[bold green]{strategy.name}[/bold green]"

    # Claude-mode: fetch macro/news/geo once (current state — not historical)
    macro = geo = news = None
    if use_claude:
        from p1_analysis_engine.synthesis.claude_synthesizer import synthesize_full
        from p1_analysis_engine.fetchers.macro import fetch_macro, MacroFetchError
        from p1_analysis_engine.fetchers.news import fetch_news
        from p1_analysis_engine.fetchers.geopolitical import fetch_geopolitical
        from p1_analysis_engine.utils.asset_classifier import classify_asset, get_asset_name

        asset_class = classify_asset(ticker)
        asset_name  = get_asset_name(ticker)
        try:
            macro = fetch_macro(asset_class)
            macro.pop("data_quality_flags", None)
        except MacroFetchError:
            macro = _empty_macro()
        try:
            news = fetch_news(ticker, asset_name)
        except Exception:
            news = []
        try:
            geo = fetch_geopolitical(asset_class, asset_name)
        except Exception:
            geo = _empty_geo()

    signals: list[SignalRecord] = []
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold]Pass 1[/bold] — generating signals ({mode_label})"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(signal_indices))

        for i in signal_indices:
            bar_time   = df.index[i]
            cache_file = _cache_path(ticker, interval, bar_time, use_claude, strategy.name)

            if cache_file.exists() and not config.force_regenerate:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                signals.append(SignalRecord(**raw["_signal_meta"]))
                skipped += 1
                progress.advance(task)
                continue

            # Slice — zero lookahead
            # Strategies that set uses_bar_slicer=False compute their own indicators
            # from df directly and don't use the tech dict — skip recompute_technical()
            # to avoid O(n²) indicator recomputation on every bar.
            if getattr(strategy, "uses_bar_slicer", True) or use_claude:
                df_slice  = df.iloc[: i + 1].copy()
                technical = recompute_technical(df_slice, interval)
                if technical is None:
                    progress.advance(task)
                    continue
            else:
                df_slice  = df.iloc[: i + 1]
                technical = {"current_price": float(df_slice["Close"].iloc[-1]), "interval": interval}

            # ── Generate setups ───────────────────────────────────────────────
            if use_claude:
                try:
                    bias_obj, setups_obj = synthesize_full(
                        ticker=ticker,
                        asset_class=asset_class,
                        technical=technical,
                        macro=macro,
                        news=news,
                        geopolitical=geo,
                        model=config.model,
                    )
                    directional_bias = bias_obj.directional_bias
                    bias_strength    = bias_obj.bias_strength
                    confidence       = bias_obj.confidence_score
                    setups_dump      = setups_obj.model_dump()
                    flags            = ["macro_not_historical", "news_not_historical"]
                except Exception as e:
                    console.print(f"[yellow]  Signal at {bar_time} failed: {e}[/yellow]")
                    progress.advance(task)
                    time.sleep(0.5)
                    continue
                time.sleep(0.3)   # Anthropic rate limiting
            else:
                technical["ticker"] = ticker
                setups_obj = strategy.generate_setups(technical, df_slice)
                if setups_obj is None:
                    progress.advance(task)
                    continue
                primary = next((s for s in setups_obj.setups if s.priority == "primary"), None)
                directional_bias = primary.direction if primary else "neutral"
                bias_strength    = "moderate"
                confidence       = primary.confidence if primary else 0.5
                setups_dump      = setups_obj.model_dump()
                flags            = [strategy.name.lower().replace(" ", "_")]

            signal_meta = SignalRecord(
                bar_index=i,
                bar_time=bar_time.isoformat(),
                close=technical["current_price"],
                directional_bias=directional_bias,
                bias_strength=bias_strength,
                confidence_score=confidence,
                n_setups=len(setups_obj.setups),
                cache_file=str(cache_file.relative_to(CACHE_DIR.parent)),
            )

            payload = {
                "_signal_meta":    signal_meta.model_dump(),
                "_backtest_flags": flags,
                "setups":          setups_dump,
            }
            cache_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            signals.append(signal_meta)
            progress.advance(task)

    # ── ADX percentile filter ─────────────────────────────────────────────────
    # Keep only the top N% of signals by confidence_score (proxy for ADX).
    # This reduces trade count but improves average quality.
    if config.top_adx_pct < 1.0 and signals:
        signals.sort(key=lambda s: s.confidence_score, reverse=True)
        keep_n = max(1, int(len(signals) * config.top_adx_pct))
        before = len(signals)
        signals = sorted(signals[:keep_n], key=lambda s: s.bar_index)
        console.print(
            f"  [dim]ADX filter (top {config.top_adx_pct:.0%}): "
            f"{before} -> {len(signals)} signals[/dim]"
        )

    console.print(f"  [dim]Pass 1 complete: {len(signals)} signals ({skipped} from cache)[/dim]")
    return signals


def _empty_macro() -> dict:
    return {
        "fed_funds_rate": None, "yield_curve": "flat", "cpi_trend": "stable",
        "vix_level": None, "vix_regime": "elevated", "usd_trend": "stable",
        "unemployment_rate": None, "macro_bias": "neutral",
    }


def _empty_geo() -> dict:
    return {"risk_level": "moderate", "key_factors": [], "relevant_headlines": []}
