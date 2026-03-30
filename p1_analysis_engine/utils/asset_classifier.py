from typing import Literal

AssetClass = Literal["equity", "forex", "crypto", "commodity", "index"]

# Common-name aliases -> canonical yfinance ticker
# Allows users to type "SILVER", "OIL", "GOLD", "BTC", etc.
_ALIASES: dict[str, str] = {
    # Commodities
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "OIL": "CL=F",
    "CRUDE": "CL=F",
    "CRUDEOIL": "CL=F",
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "NATURALGAS": "NG=F",
    "GAS": "NG=F",
    "COPPER": "HG=F",
    "WHEAT": "ZW=F",
    "CORN": "ZC=F",
    "SOYBEANS": "ZS=F",
    "SOYBEAN": "ZS=F",
    "PLATINUM": "PL=F",
    "PALLADIUM": "PA=F",
    # Crypto shorthands (without -USD suffix)
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "BNB": "BNB-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
    "AVAX": "AVAX-USD",
    "DOT": "DOT-USD",
    "MATIC": "MATIC-USD",
    "LINK": "LINK-USD",
    "UNI": "UNI-USD",
    # Forex shorthands (without =X suffix)
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    # Indices
    "US30":      "^DJI",
    "DOW":       "^DJI",
    "DJI":       "^DJI",
    "NAS100":    "^NDX",
    "NASDAQ100": "^NDX",
    "NDX":       "^NDX",
    "SPX":       "^GSPC",
    "SP500":     "^GSPC",
    "SPX500":    "^GSPC",
    "GER40":     "^GDAXI",
    "DAX":       "^GDAXI",
    "DAX40":     "^GDAXI",
}

# Known commodity futures tickers (yfinance format)
_COMMODITIES: dict[str, str] = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "CL=F": "WTI Crude Oil",
    "BZ=F": "Brent Crude Oil",
    "NG=F": "Natural Gas",
    "HG=F": "Copper",
    "ZW=F": "Wheat",
    "ZC=F": "Corn",
    "ZS=F": "Soybeans",
    "PL=F": "Platinum",
    "PA=F": "Palladium",
}

# Known forex pairs (6-char pairs without =X suffix)
_FOREX_PAIRS: set[str] = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
}

# Known index tickers
_INDICES: dict[str, str] = {
    "^DJI":   "Dow Jones Industrial Average",
    "^NDX":   "NASDAQ 100",
    "^GSPC":  "S&P 500",
    "^GDAXI": "DAX 40",
}

# Well-known equity tickers mapped to readable names
_EQUITY_NAMES: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
    "AMZN": "Amazon", "NVDA": "Nvidia", "META": "Meta",
    "TSLA": "Tesla", "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan",
    "V": "Visa", "MA": "Mastercard", "UNH": "UnitedHealth",
    "XOM": "ExxonMobil", "JNJ": "Johnson & Johnson", "WMT": "Walmart",
}


def resolve_ticker(ticker: str) -> str:
    """Resolve common-name aliases to canonical yfinance tickers (e.g. SILVER -> SI=F)."""
    return _ALIASES.get(ticker.upper().strip(), ticker.upper().strip())


def classify_asset(ticker: str) -> AssetClass:
    t = resolve_ticker(ticker)

    if t in _INDICES:
        return "index"

    if t in _COMMODITIES:
        return "commodity"

    if t.endswith("=X") or t.rstrip("=X") in _FOREX_PAIRS or t in _FOREX_PAIRS:
        return "forex"

    if t.endswith("-USD") or t.endswith("-USDT") or t.endswith("-BTC"):
        return "crypto"

    return "equity"


def get_asset_name(ticker: str) -> str:
    """Return a human-readable name for use in news searches."""
    t = resolve_ticker(ticker)

    if t in _INDICES:
        return _INDICES[t]

    if t in _COMMODITIES:
        return _COMMODITIES[t]

    if t in _EQUITY_NAMES:
        return _EQUITY_NAMES[t]

    # Forex: strip =X suffix
    if t.endswith("=X"):
        return t[:-2]

    # Crypto: strip -USD / -USDT suffix
    for suffix in ("-USD", "-USDT", "-BTC"):
        if t.endswith(suffix):
            return t[: -len(suffix)]

    return t
