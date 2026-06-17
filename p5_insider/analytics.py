"""
p5_insider/analytics.py — Cluster detection, politician performance, price impact.
"""
from __future__ import annotations
import logging
from collections import defaultdict
from datetime import date, timedelta, datetime

log = logging.getLogger("orcastrading.insider.analytics")


# ── Cluster detection ─────────────────────────────────────────────────────────

def get_cluster_trades(since_days: int = 30, min_cluster: int = 3) -> list[dict]:
    """
    Find tickers where min_cluster+ distinct politicians traded within the window.
    A cluster of buys signals consensus bullishness; sells signal the opposite.
    Returns list sorted by cluster size descending.
    """
    from p5_insider.db import get_congress_trades
    since = (date.today() - timedelta(days=since_days)).isoformat()
    trades = get_congress_trades(since_date=since, limit=5000)

    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        ticker = t.get("ticker")
        if ticker:
            by_ticker[ticker].append(t)

    clusters = []
    for ticker, ticker_trades in by_ticker.items():
        buys  = [t for t in ticker_trades if "purchase" in (t.get("transaction_type") or "").lower()]
        sells = [t for t in ticker_trades if "sale"     in (t.get("transaction_type") or "").lower()]

        # Require min_cluster of the dominant side AND at least that many *distinct* politicians
        buy_pols  = {t.get("politician","") for t in buys}
        sell_pols = {t.get("politician","") for t in sells}

        if len(buy_pols) >= min_cluster or len(sell_pols) >= min_cluster:
            if len(buy_pols) >= len(sell_pols):
                action, dominant, pol_set = "BUY", buys, buy_pols
            else:
                action, dominant, pol_set = "SELL", sells, sell_pols

            total_low = sum(t.get("amount_low") or 0 for t in dominant)
            latest    = max((t.get("disclosure_date") or "" for t in dominant), default="")
            parties   = [t.get("party","") for t in dominant]
            party_bias = max(set(parties), key=parties.count) if parties else ""

            clusters.append({
                "ticker":           ticker,
                "action":           action,
                "cluster_size":     len(pol_set),
                "total_trades":     len(ticker_trades),
                "buys":             len(buys),
                "sells":            len(sells),
                "politicians":      sorted(pol_set)[:6],
                "party_bias":       party_bias,
                "total_amount_low": total_low,
                "latest_date":      latest,
            })

    clusters.sort(key=lambda x: (x["cluster_size"], x["total_amount_low"]), reverse=True)
    return clusters


# ── Politician leaderboard ────────────────────────────────────────────────────

def get_politician_stats(since_days: int = 365, min_trades: int = 2) -> list[dict]:
    """
    Per-politician summary for the look-back window.
    Returns list sorted by trade count descending.
    """
    from p5_insider.db import get_congress_trades
    since = (date.today() - timedelta(days=since_days)).isoformat()
    trades = get_congress_trades(since_date=since, limit=10000)

    by_pol: dict[str, dict] = {}
    for t in trades:
        pol = t.get("politician") or "Unknown"
        if pol not in by_pol:
            by_pol[pol] = {
                "politician":      pol,
                "party":           t.get("party",""),
                "state":           t.get("state",""),
                "source":          t.get("source",""),
                "buys":            0,
                "sells":           0,
                "tickers":         set(),
                "total_amount_low": 0.0,
                "latest":          "",
            }
        s  = by_pol[pol]
        tt = (t.get("transaction_type") or "").lower()
        if "purchase" in tt:
            s["buys"] += 1
        elif "sale" in tt:
            s["sells"] += 1
        if t.get("ticker"):
            s["tickers"].add(t["ticker"])
        s["total_amount_low"] += t.get("amount_low") or 0
        disc = t.get("disclosure_date") or ""
        if disc > s["latest"]:
            s["latest"] = disc

    result = []
    for s in by_pol.values():
        total = s["buys"] + s["sells"]
        if total < min_trades:
            continue
        result.append({
            **s,
            "tickers":      sorted(s["tickers"])[:8],
            "ticker_count": len(s["tickers"]),
            "total":        total,
            "buy_pct":      round(s["buys"] / total * 100) if total else 0,
        })

    result.sort(key=lambda x: x["total"], reverse=True)
    return result


# ── Hot tickers (congress-wide) ───────────────────────────────────────────────

def get_top_congress_tickers(since_days: int = 30, limit: int = 25) -> list[dict]:
    """
    Most-traded tickers by congress, regardless of user portfolio.
    Useful for discovering opportunities outside current positions.
    """
    from p5_insider.db import get_congress_trades
    since = (date.today() - timedelta(days=since_days)).isoformat()
    trades = get_congress_trades(since_date=since, limit=10000)

    by_ticker: dict[str, dict] = {}
    for t in trades:
        ticker = t.get("ticker")
        if not ticker:
            continue
        if ticker not in by_ticker:
            by_ticker[ticker] = {
                "ticker":           ticker,
                "buys":             0,
                "sells":            0,
                "politicians":      set(),
                "total_amount_low": 0.0,
                "latest":           "",
            }
        s  = by_ticker[ticker]
        tt = (t.get("transaction_type") or "").lower()
        if "purchase" in tt:
            s["buys"] += 1
        elif "sale" in tt:
            s["sells"] += 1
        s["politicians"].add(t.get("politician",""))
        s["total_amount_low"] += t.get("amount_low") or 0
        disc = t.get("disclosure_date") or ""
        if disc > s["latest"]:
            s["latest"] = disc

    result = []
    for s in by_ticker.values():
        total = s["buys"] + s["sells"]
        if total == 0:
            continue
        net  = s["buys"] - s["sells"]
        bias = "bullish" if net > 1 else "bearish" if net < -1 else "neutral"
        result.append({
            **s,
            "politicians":      sorted(s["politicians"])[:5],
            "politician_count": len(s["politicians"]),
            "total_trades":     total,
            "net_bias":         net,
            "bias":             bias,
        })

    result.sort(key=lambda x: x["total_trades"], reverse=True)
    return result[:limit]


# ── Price impact ──────────────────────────────────────────────────────────────

def get_price_impact(
    ticker: str,
    disclosure_date: str,
    forward_days: int = 30,
) -> dict | None:
    """
    Compute stock return at 5/10/20/30 days after disclosure_date using yfinance.
    Returns None on failure (no data, delisted, etc.).
    """
    try:
        import yfinance as yf
        import pandas as pd

        dt    = datetime.strptime(disclosure_date, "%Y-%m-%d").date()
        start = (dt - timedelta(days=5)).isoformat()
        end   = (dt + timedelta(days=forward_days + 10)).isoformat()

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None

        disc_ts   = pd.Timestamp(dt)
        at_or_after = closes[closes.index >= disc_ts]
        if at_or_after.empty:
            return None

        price_base = float(at_or_after.iloc[0])
        result: dict = {
            "ticker":              ticker,
            "disclosure_date":     disclosure_date,
            "price_at_disclosure": round(price_base, 2),
        }

        for days in [5, 10, 20, 30]:
            fwd_ts   = pd.Timestamp(dt + timedelta(days=days))
            fwd_sl   = closes[closes.index >= fwd_ts]
            if not fwd_sl.empty:
                price_fwd = float(fwd_sl.iloc[0])
                ret = (price_fwd - price_base) / price_base * 100
                result[f"return_{days}d"] = round(ret, 2)
            else:
                result[f"return_{days}d"] = None

        return result

    except Exception as e:
        log.debug(f"Price impact failed for {ticker} @ {disclosure_date}: {e}")
        return None


# ── Insider activity timeline (for charts) ────────────────────────────────────

def get_activity_timeline(
    ticker: str,
    since_days: int = 180,
) -> list[dict]:
    """
    Return a list of {date, action, count, source} for a ticker,
    suitable for plotting a buy/sell bar chart over time.
    """
    from p5_insider.db import get_congress_trades, get_form4_trades
    since = (date.today() - timedelta(days=since_days)).isoformat()
    rows: list[dict] = []

    for t in get_congress_trades(since_date=since, ticker=ticker, limit=500):
        tt = (t.get("transaction_type") or "").lower()
        action = "buy" if "purchase" in tt else "sell" if "sale" in tt else None
        if action:
            rows.append({
                "date":   t.get("disclosure_date",""),
                "action": action,
                "source": t.get("source","congress"),
                "who":    t.get("politician",""),
            })

    for t in get_form4_trades(since_date=since, ticker=ticker, limit=500):
        code = t.get("transaction_code","")
        action = {"P":"buy","S":"sell"}.get(code)
        if action:
            rows.append({
                "date":   t.get("filing_date",""),
                "action": action,
                "source": "form4",
                "who":    t.get("insider_name",""),
            })

    rows.sort(key=lambda x: x["date"])
    return rows
