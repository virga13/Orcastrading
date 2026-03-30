"""
costs.py — realistic round-trip transaction cost model.

Models spread + slippage as a fraction of ATR, per asset class and interval.
Short timeframes have proportionally larger costs because the ATR per bar
is smaller while the absolute spread (tick size, market impact) stays fixed.

Cost is expressed as R-multiples and subtracted from gross pnl_r to give
net_pnl_r — the number that matters for real trading.

Calibration (approximate, based on liquid futures/CFD markets):
  Gold 1h   : ATR ~10 pts, spread ~0.20 pts → ~0.02 ATR   → ~0.04R on 0.5R risk
  Gold 5m   : ATR ~1.5 pts, spread ~0.20 pts → ~0.13 ATR  → ~0.26R on 0.5R risk
  BTC 1h    : ATR ~500 USD, spread ~5 USD    → ~0.01 ATR
  BTC 5m    : ATR ~80 USD,  spread ~5 USD    → ~0.06 ATR
  SPX 1h    : ATR ~15 pts,  spread ~0.25 pts → ~0.017 ATR
  SPX 5m    : ATR ~2.5 pts, spread ~0.25 pts → ~0.10 ATR
  EUR/USD 1h: ATR ~0.0015, spread ~0.0001   → ~0.067 ATR

At 5m these costs completely destroy edges below +0.3R avg.
At 1h edges above +0.2R avg survive.
"""

# Round-trip cost as fraction of ATR, by (asset_class, interval)
# Includes: half-spread on entry + half-spread on exit + slippage
_COST_ATR_FRACTION: dict[tuple[str, str], float] = {
    ("commodity", "1m"):  0.28,
    ("commodity", "5m"):  0.14,
    ("commodity", "15m"): 0.08,
    ("commodity", "30m"): 0.05,
    ("commodity", "1h"):  0.03,
    ("commodity", "1d"):  0.01,
    ("commodity", "1wk"): 0.005,

    ("crypto",    "1m"):  0.20,
    ("crypto",    "5m"):  0.10,
    ("crypto",    "15m"): 0.06,
    ("crypto",    "30m"): 0.04,
    ("crypto",    "1h"):  0.025,
    ("crypto",    "1d"):  0.008,
    ("crypto",    "1wk"): 0.004,

    ("forex",     "1m"):  0.18,
    ("forex",     "5m"):  0.09,
    ("forex",     "15m"): 0.05,
    ("forex",     "30m"): 0.035,
    ("forex",     "1h"):  0.02,
    ("forex",     "1d"):  0.007,
    ("forex",     "1wk"): 0.003,

    ("index",     "1m"):  0.22,
    ("index",     "5m"):  0.11,
    ("index",     "15m"): 0.07,
    ("index",     "30m"): 0.045,
    ("index",     "1h"):  0.028,
    ("index",     "1d"):  0.010,
    ("index",     "1wk"): 0.005,

    ("equity",    "1m"):  0.15,
    ("equity",    "5m"):  0.08,
    ("equity",    "15m"): 0.05,
    ("equity",    "30m"): 0.03,
    ("equity",    "1h"):  0.02,
    ("equity",    "1d"):  0.008,
    ("equity",    "1wk"): 0.003,
}

# Fallback fraction when asset_class/interval combo not in table
_DEFAULT_FRACTION = 0.05


def round_trip_cost_r(
    asset_class: str,
    interval: str,
    atr: float,
    initial_risk: float,
) -> float:
    """
    Compute the round-trip transaction cost as an R-multiple.

    Parameters
    ----------
    asset_class  : "commodity" | "crypto" | "forex" | "index" | "equity"
    interval     : "1m" | "5m" | ... | "1d" | "1wk"
    atr          : ATR value in price units at the time of the trade
    initial_risk : abs(fill_price - stop_loss) in price units

    Returns
    -------
    float — cost in R-multiples (always >= 0)
    """
    if initial_risk < 1e-10 or atr < 1e-10:
        return 0.0
    fraction = _COST_ATR_FRACTION.get((asset_class, interval), _DEFAULT_FRACTION)
    cost_pts = atr * fraction
    return round(cost_pts / initial_risk, 4)


def cost_label(asset_class: str, interval: str) -> str:
    """Human-readable cost description for reports."""
    fraction = _COST_ATR_FRACTION.get((asset_class, interval), _DEFAULT_FRACTION)
    return f"~{fraction*100:.1f}% ATR round-trip ({asset_class}, {interval})"
