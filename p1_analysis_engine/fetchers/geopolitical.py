import os
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from typing import Literal


# Asset-class specific keyword sets for geopolitical signals
_KEYWORDS: dict[str, str] = {
    "equity": "Fed OR tariffs OR recession OR earnings OR inflation OR \"interest rates\" OR \"trade war\"",
    "forex": "FOMC OR \"dollar\" OR \"currency\" OR \"central bank\" OR \"interest rate\" OR \"trade deficit\"",
    "crypto": "\"crypto regulation\" OR \"SEC\" OR \"Bitcoin ETF\" OR \"stablecoin\" OR CBDC OR \"crypto ban\"",
    "commodity": "OPEC OR \"supply chain\" OR sanctions OR \"energy crisis\" OR drought OR \"commodity prices\"",
}

# Global macro risk keywords always fetched regardless of asset class
_GLOBAL_RISK_QUERY = (
    "war OR sanctions OR tariffs OR \"geopolitical risk\" OR \"trade war\" "
    "OR \"military conflict\" OR \"central bank\" OR recession"
)


def _query_newsapi(query: str, days_back: int = 5, max_articles: int = 5) -> list[str]:
    """Returns a list of headline strings."""
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        return []

    api = NewsApiClient(api_key=key)
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        response = api.get_everything(
            q=query,
            from_param=from_date,
            language="en",
            sort_by="relevancy",
            page_size=max_articles,
        )
        return [
            a["title"]
            for a in response.get("articles", [])
            if a.get("title") and "[Removed]" not in a["title"]
        ]
    except Exception:
        return []


def _assess_risk_level(headlines: list[str]) -> Literal["low", "moderate", "high", "extreme"]:
    """
    Simple keyword-based risk level assessment from headlines.
    Claude will do the deep reasoning — this is just a first-pass signal.
    """
    text = " ".join(headlines).lower()

    extreme_terms = ["world war", "nuclear", "invasion", "collapse", "default", "catastrophe"]
    high_terms = ["war", "conflict", "sanctions", "crisis", "crash", "recession confirmed"]
    moderate_terms = ["tariffs", "tension", "slowdown", "uncertainty", "rate hike", "inflation"]

    if any(t in text for t in extreme_terms):
        return "extreme"
    if any(t in text for t in high_terms):
        return "high"
    if any(t in text for t in moderate_terms):
        return "moderate"
    return "low"


def fetch_geopolitical(
    asset_class: Literal["equity", "forex", "crypto", "commodity"],
    asset_name: str,
) -> dict:
    """
    Fetch geopolitical and macro risk signals via NewsAPI keyword searches.
    Returns a dict with risk_level, key_factors, and relevant_headlines.
    """
    # Asset-class specific headlines
    class_query = _KEYWORDS.get(asset_class, _KEYWORDS["equity"])
    class_headlines = _query_newsapi(class_query, days_back=5, max_articles=5)

    # Global macro risk headlines
    global_headlines = _query_newsapi(_GLOBAL_RISK_QUERY, days_back=5, max_articles=5)

    # Combine, deduplicate
    all_headlines = list(dict.fromkeys(class_headlines + global_headlines))[:8]

    risk_level = _assess_risk_level(all_headlines)

    # Key factors: short labels derived from query keywords for Claude context
    key_factors_map = {
        "equity": ["Fed policy", "US tariff regime", "earnings season", "inflation trajectory"],
        "forex": ["FOMC stance", "USD strength", "central bank divergence", "trade balance"],
        "crypto": ["regulatory environment", "ETF flows", "stablecoin policy", "SEC actions"],
        "commodity": ["OPEC production decisions", "supply chain disruptions", "sanctions impact", "weather/drought"],
    }
    key_factors = key_factors_map.get(asset_class, [])

    return {
        "risk_level": risk_level,
        "key_factors": key_factors,
        "relevant_headlines": all_headlines[:6],
        "asset_class_query_used": class_query,
    }
