import os
from datetime import datetime, timedelta
import yfinance as yf
from newsapi import NewsApiClient


class NewsFetchError(Exception):
    pass


def _fetch_yfinance_news(ticker: str, max_articles: int = 10) -> list[dict]:
    """Primary: yfinance built-in news — ticker-specific, always financial context."""
    news = yf.Ticker(ticker).news
    if not news:
        return []

    results = []
    for item in news[:max_articles]:
        content = item.get("content", {})
        title = content.get("title", "")
        if not title:
            continue

        source = content.get("provider", {}).get("displayName", "Yahoo Finance")
        pub_date = content.get("pubDate", "")
        url = content.get("canonicalUrl", {}).get("url", "")

        results.append({
            "title": title,
            "source": source,
            "published_at": pub_date,
            "url": url,
        })
    return results


def _fetch_newsapi(query: str, days_back: int = 3, max_articles: int = 10) -> list[dict]:
    """Fallback: NewsAPI with a financial-context query."""
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        raise NewsFetchError("NEWSAPI_KEY not set")

    api = NewsApiClient(api_key=key)
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    response = api.get_everything(
        q=query,
        from_param=from_date,
        language="en",
        sort_by="publishedAt",
        page_size=max_articles,
    )

    articles = response.get("articles", [])
    return [
        {
            "title": a["title"],
            "source": a["source"]["name"],
            "published_at": a["publishedAt"],
            "url": a["url"],
        }
        for a in articles
        if a.get("title") and "[Removed]" not in a["title"]
    ]


def fetch_news(ticker: str, asset_name: str, max_articles: int = 8) -> list[dict]:
    """
    Fetch recent news headlines for an asset.
    Primary: yfinance .news (ticker-specific, financial context).
    Fallback: NewsAPI with financial-context query.
    Returns list of {title, source, published_at, url}.
    """
    # Primary: yfinance news
    try:
        articles = _fetch_yfinance_news(ticker, max_articles=max_articles)
        if articles:
            return articles[:max_articles]
    except Exception:
        pass

    # Fallback: NewsAPI
    query = f'"{asset_name}" AND (price OR market OR trading OR forecast OR rally OR selloff OR earnings OR analysis)'
    try:
        articles = _fetch_newsapi(query, days_back=3, max_articles=max_articles)
        if articles:
            return articles[:max_articles]
    except Exception:
        pass

    # Both failed — return empty, engine will flag it
    return []
