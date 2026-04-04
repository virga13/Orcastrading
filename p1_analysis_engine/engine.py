import os
from dotenv import load_dotenv
from p1_analysis_engine.schema import BiasOutput, SetupsOutput
from p1_analysis_engine.utils.asset_classifier import classify_asset, get_asset_name, resolve_ticker
from p1_analysis_engine.fetchers.technical import fetch_technical, DataFetchError
from p1_analysis_engine.fetchers.macro import fetch_macro, MacroFetchError
from p1_analysis_engine.fetchers.news import fetch_news
from p1_analysis_engine.fetchers.geopolitical import fetch_geopolitical
from p1_analysis_engine.synthesis.claude_synthesizer import synthesize, synthesize_full

load_dotenv(override=True)


def analyze(ticker: str, model: str = "claude-sonnet-4-6", interval: str = "1d", extended_thinking: bool = False) -> BiasOutput:
    """
    Full P1 pipeline: fetch all data sources, synthesize via Claude, return BiasOutput.
    Partial failures are logged as data_quality_flags — the pipeline never crashes
    unless technical data (the core signal) is unavailable.
    """
    ticker = resolve_ticker(ticker)
    asset_class = classify_asset(ticker)
    fetcher_class = "equity" if asset_class == "index" else asset_class
    asset_name = get_asset_name(ticker)
    flags: list[str] = []

    # --- Technical (required) ---
    try:
        technical = fetch_technical(ticker, interval=interval)
    except DataFetchError as e:
        raise RuntimeError(f"Cannot analyze {ticker}: {e}")
    except Exception as e:
        raise RuntimeError(f"Technical fetch failed for {ticker}: {e}")

    # --- Macro (optional) ---
    try:
        macro = fetch_macro(fetcher_class)
        flags.extend(macro.pop("data_quality_flags", []))
    except MacroFetchError as e:
        macro = _empty_macro()
        flags.append(f"macro_unavailable: {e}")
    except Exception as e:
        macro = _empty_macro()
        flags.append(f"macro_error: {e}")

    # --- News (optional) ---
    try:
        news = fetch_news(ticker, asset_name)
        if not news:
            flags.append("news_unavailable: no articles returned")
    except Exception as e:
        news = []
        flags.append(f"news_error: {e}")

    # --- Geopolitical (optional) ---
    try:
        geopolitical = fetch_geopolitical(fetcher_class, asset_name)
    except Exception as e:
        geopolitical = _empty_geo()
        flags.append(f"geopolitical_error: {e}")

    # --- Synthesize ---
    bias = synthesize(
        ticker=ticker,
        asset_class=asset_class,
        technical=technical,
        macro=macro,
        news=news,
        geopolitical=geopolitical,
        model=model,
        extended_thinking=extended_thinking,
    )

    # Merge any flags collected during fetching
    bias.data_quality_flags = list(dict.fromkeys(bias.data_quality_flags + flags))

    return bias


def analyze_full(ticker: str, model: str = "claude-sonnet-4-6", interval: str = "1d", extended_thinking: bool = False) -> tuple[BiasOutput, SetupsOutput]:
    """
    Full P1 pipeline returning both BiasOutput and SetupsOutput from a single
    Claude API call. Use this when generating an HTML report.
    """
    ticker = resolve_ticker(ticker)
    asset_class = classify_asset(ticker)
    fetcher_class = "equity" if asset_class == "index" else asset_class
    asset_name = get_asset_name(ticker)
    flags: list[str] = []

    try:
        technical = fetch_technical(ticker, interval=interval)
    except DataFetchError as e:
        raise RuntimeError(f"Cannot analyze {ticker}: {e}")
    except Exception as e:
        raise RuntimeError(f"Technical fetch failed for {ticker}: {e}")

    try:
        macro = fetch_macro(fetcher_class)
        flags.extend(macro.pop("data_quality_flags", []))
    except MacroFetchError as e:
        macro = _empty_macro()
        flags.append(f"macro_unavailable: {e}")
    except Exception as e:
        macro = _empty_macro()
        flags.append(f"macro_error: {e}")

    try:
        news = fetch_news(ticker, asset_name)
        if not news:
            flags.append("news_unavailable: no articles returned")
    except Exception as e:
        news = []
        flags.append(f"news_error: {e}")

    try:
        geopolitical = fetch_geopolitical(fetcher_class, asset_name)
    except Exception as e:
        geopolitical = _empty_geo()
        flags.append(f"geopolitical_error: {e}")

    bias, setups = synthesize_full(
        ticker=ticker,
        asset_class=asset_class,
        technical=technical,
        macro=macro,
        news=news,
        geopolitical=geopolitical,
        model=model,
        extended_thinking=extended_thinking,
    )

    bias.data_quality_flags = list(dict.fromkeys(bias.data_quality_flags + flags))
    return bias, setups


def _empty_macro() -> dict:
    return {
        "fed_funds_rate": None,
        "yield_curve_spread": None,
        "yield_curve": "flat",
        "cpi_trend": "stable",
        "vix_level": None,
        "vix_regime": "elevated",
        "usd_trend": "stable",
        "unemployment_rate": None,
        "macro_bias": "neutral",
    }


def _empty_geo() -> dict:
    return {
        "risk_level": "moderate",
        "key_factors": [],
        "relevant_headlines": [],
        "asset_class_query_used": "",
    }
