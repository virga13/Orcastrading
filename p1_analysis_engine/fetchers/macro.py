import os
from fredapi import Fred
from typing import Literal


class MacroFetchError(Exception):
    pass


def _get_fred() -> Fred:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise MacroFetchError("FRED_API_KEY not set")
    return Fred(api_key=key)


def _last(fred: Fred, series_id: str, n: int = 2) -> list[float]:
    """Fetch the last n values of a FRED series."""
    s = fred.get_series(series_id)
    s = s.dropna()
    if s.empty:
        raise MacroFetchError(f"No data for FRED series {series_id}")
    return [float(v) for v in s.iloc[-n:]]


def fetch_macro(asset_class: Literal["equity", "forex", "crypto", "commodity"]) -> dict:
    """
    Fetch macroeconomic indicators from FRED.
    Returns a flat dict ready for formatting and Claude synthesis.
    """
    fred = _get_fred()
    flags = []

    # --- Fed Funds Rate ---
    try:
        fed_vals = _last(fred, "FEDFUNDS", 2)
        fed_funds_rate = fed_vals[-1]
    except Exception as e:
        fed_funds_rate = None
        flags.append(f"FRED_FEDFUNDS_unavailable: {e}")

    # --- Yield Curve (10Y - 2Y spread) ---
    try:
        spread_vals = _last(fred, "T10Y2Y", 3)
        spread = spread_vals[-1]
        if spread > 0.1:
            yield_curve = "normal"
        elif spread < -0.1:
            yield_curve = "inverted"
        else:
            yield_curve = "flat"
    except Exception as e:
        spread = None
        yield_curve = "flat"
        flags.append(f"FRED_T10Y2Y_unavailable: {e}")

    # --- CPI Trend ---
    try:
        cpi_vals = _last(fred, "CPIAUCSL", 3)
        cpi_trend_val = cpi_vals[-1] - cpi_vals[0]
        if cpi_trend_val > 0.1:
            cpi_trend = "rising"
        elif cpi_trend_val < -0.1:
            cpi_trend = "falling"
        else:
            cpi_trend = "stable"
    except Exception as e:
        cpi_trend = "stable"
        flags.append(f"FRED_CPIAUCSL_unavailable: {e}")

    # --- VIX ---
    try:
        vix_vals = _last(fred, "VIXCLS", 2)
        vix_level = vix_vals[-1]
        if vix_level < 15:
            vix_regime = "low"
        elif vix_level < 25:
            vix_regime = "elevated"
        else:
            vix_regime = "high"
    except Exception as e:
        vix_level = None
        vix_regime = "elevated"
        flags.append(f"FRED_VIXCLS_unavailable: {e}")

    # --- USD Trend (DXY proxy via trade-weighted index) ---
    try:
        usd_vals = _last(fred, "DTWEXBGS", 3)
        usd_change = usd_vals[-1] - usd_vals[0]
        if usd_change > 0.5:
            usd_trend = "strengthening"
        elif usd_change < -0.5:
            usd_trend = "weakening"
        else:
            usd_trend = "stable"
    except Exception as e:
        usd_trend = "stable"
        flags.append(f"FRED_DTWEXBGS_unavailable: {e}")

    # --- Unemployment ---
    try:
        unemp_vals = _last(fred, "UNRATE", 1)
        unemployment_rate = unemp_vals[-1]
    except Exception as e:
        unemployment_rate = None
        flags.append(f"FRED_UNRATE_unavailable: {e}")

    # --- Macro bias ---
    risk_on_signals = 0
    risk_off_signals = 0

    if vix_level and vix_level < 15:
        risk_on_signals += 1
    elif vix_level and vix_level > 25:
        risk_off_signals += 1

    if yield_curve == "normal":
        risk_on_signals += 1
    elif yield_curve == "inverted":
        risk_off_signals += 1

    if usd_trend == "weakening":
        risk_on_signals += 1
    elif usd_trend == "strengthening":
        risk_off_signals += 1

    if risk_on_signals > risk_off_signals:
        macro_bias = "risk_on"
    elif risk_off_signals > risk_on_signals:
        macro_bias = "risk_off"
    else:
        macro_bias = "neutral"

    # --- Commodity-specific extras ---
    wti_price = None
    gold_price = None
    if asset_class == "commodity":
        try:
            wti_price = _last(fred, "DCOILWTICO", 1)[-1]
        except Exception:
            flags.append("FRED_DCOILWTICO_unavailable")
        try:
            gold_price = _last(fred, "GOLDAMGBD228NLBM", 1)[-1]
        except Exception:
            flags.append("FRED_GOLD_unavailable")

    result = {
        "fed_funds_rate": round(fed_funds_rate, 2) if fed_funds_rate else None,
        "yield_curve_spread": round(spread, 3) if spread else None,
        "yield_curve": yield_curve,
        "cpi_trend": cpi_trend,
        "vix_level": round(vix_level, 2) if vix_level else None,
        "vix_regime": vix_regime,
        "usd_trend": usd_trend,
        "unemployment_rate": round(unemployment_rate, 1) if unemployment_rate else None,
        "macro_bias": macro_bias,
        "data_quality_flags": flags,
    }

    if asset_class == "commodity":
        result["wti_crude"] = round(wti_price, 2) if wti_price else None
        result["gold_fix"] = round(gold_price, 2) if gold_price else None

    return result
