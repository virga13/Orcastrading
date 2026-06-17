"""
p5_insider/correlator.py — Score insider trades against the user's portfolio.

Scoring (0–10):
  Congressional trades:
    Base 2 + amount tier (0–4) + portfolio conflict bonus (3) / confirmation (1)
  Form 4 trades:
    Base 1 + seniority (0–3) + amount tier (0–3) + portfolio conflict bonus (3)

"Portfolio conflict" = insider is selling a stock the user holds long,
  or buying a stock the user holds short. These are the highest-value signals.
"Portfolio confirmation" = insider buying what the user is long (or vice versa).

Alert threshold: score >= 5.
"""
from __future__ import annotations
from datetime import date, timedelta

from p5_insider.db import (
    get_congress_trades,
    get_form4_trades,
    get_portfolio,
)


# ── Amount scoring (congressional) ───────────────────────────────────────────

def _amount_score_congress(amount_low: float | None) -> int:
    if not amount_low:
        return 0
    if amount_low >= 1_000_000:
        return 4
    if amount_low >= 250_000:
        return 3
    if amount_low >= 50_000:
        return 2
    if amount_low >= 15_000:
        return 1
    return 0


# ── Amount scoring (Form 4) ───────────────────────────────────────────────────

def _amount_score_form4(total_value: float | None) -> int:
    if not total_value:
        return 0
    if total_value >= 1_000_000:
        return 3
    if total_value >= 250_000:
        return 2
    if total_value >= 50_000:
        return 1
    return 0


# ── Insider seniority scoring (Form 4) ────────────────────────────────────────

_SENIOR_KEYWORDS = ("ceo", "cfo", "coo", "president", "chairman", "chief")
_MID_KEYWORDS    = ("director", "vp ", "vice president", "evp", "svp")


def _seniority_score(title: str | None) -> int:
    if not title:
        return 0
    t = title.lower()
    if any(k in t for k in _SENIOR_KEYWORDS):
        return 3
    if any(k in t for k in _MID_KEYWORDS):
        return 2
    return 1   # any officer is still interesting


# ── Portfolio alignment ───────────────────────────────────────────────────────

def _portfolio_bonus(trade_type: str, portfolio_direction: str) -> int:
    """
    trade_type: "purchase"/"sale"/"sale_partial" (congress) or "P"/"S" (form4).
    Returns 3 for conflict (sell your long / buy your short),
            1 for confirmation (buy your long / sell your short),
            0 otherwise.
    """
    is_buy  = trade_type.lower() in ("purchase", "p")
    is_sell = trade_type.lower() in ("sale", "sale_partial", "s")

    if portfolio_direction == "long":
        if is_sell:
            return 3   # insider selling → bad for your long
        if is_buy:
            return 1   # insider buying → confirms your long
    elif portfolio_direction == "short":
        if is_buy:
            return 3   # insider buying → bad for your short
        if is_sell:
            return 1   # insider selling → confirms your short
    return 0


# ── Public scoring functions ──────────────────────────────────────────────────

def score_congress_trade(trade: dict, portfolio_item: dict) -> float:
    score = 2.0
    score += _amount_score_congress(trade.get("amount_low"))
    score += _portfolio_bonus(trade.get("transaction_type", ""), portfolio_item.get("direction", "long"))
    return min(score, 10.0)


def score_form4_trade(trade: dict, portfolio_item: dict) -> float:
    score = 1.0
    score += _seniority_score(trade.get("insider_title"))
    score += _amount_score_form4(trade.get("total_value"))
    score += _portfolio_bonus(trade.get("transaction_code", ""), portfolio_item.get("direction", "long"))
    return min(score, 10.0)


# ── Matched trades ────────────────────────────────────────────────────────────

def get_scored_trades(since_days: int = 90) -> list[dict]:
    """
    Return all insider trades that match portfolio tickers, scored and sorted
    by score descending.  Includes both congressional and Form 4 trades.
    """
    since = (date.today() - timedelta(days=since_days)).isoformat()
    portfolio = {p["ticker"]: p for p in get_portfolio()}
    if not portfolio:
        return []

    results: list[dict] = []

    # Congressional
    for trade in get_congress_trades(since_date=since):
        ticker = trade.get("ticker")
        if not ticker or ticker not in portfolio:
            continue
        p_item = portfolio[ticker]
        score = score_congress_trade(trade, p_item)
        results.append({
            **trade,
            "type":       "congress",
            "score":      score,
            "p_direction": p_item["direction"],
        })

    # Form 4
    for trade in get_form4_trades(since_date=since):
        ticker = trade.get("ticker")
        if not ticker or ticker not in portfolio:
            continue
        p_item = portfolio[ticker]
        score = score_form4_trade(trade, p_item)
        results.append({
            **trade,
            "type":       "form4",
            "score":      score,
            "p_direction": p_item["direction"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_portfolio_summary(since_days: int = 30) -> list[dict]:
    """
    Per-ticker summary of recent insider activity for the user's portfolio.
    Returns list of dicts sorted by activity score.
    """
    since = (date.today() - timedelta(days=since_days)).isoformat()
    portfolio = {p["ticker"]: p for p in get_portfolio()}
    summary: dict[str, dict] = {}

    for ticker in portfolio:
        summary[ticker] = {
            "ticker":    ticker,
            "label":     portfolio[ticker].get("label", ticker),
            "direction": portfolio[ticker].get("direction", "long"),
            "buys":      0,
            "sells":     0,
            "max_score": 0.0,
            "latest":    None,
            "signal":    "neutral",
        }

    for trade in get_congress_trades(since_date=since):
        t = trade.get("ticker")
        if t not in summary:
            continue
        tt = trade.get("transaction_type", "")
        is_buy = "purchase" in tt.lower()
        if is_buy:
            summary[t]["buys"] += 1
        elif "sale" in tt.lower():
            summary[t]["sells"] += 1
        score = score_congress_trade(trade, portfolio[t])
        if score > summary[t]["max_score"]:
            summary[t]["max_score"] = score
        dd = trade.get("disclosure_date", "")
        if not summary[t]["latest"] or dd > summary[t]["latest"]:
            summary[t]["latest"] = dd

    for trade in get_form4_trades(since_date=since):
        t = trade.get("ticker")
        if t not in summary:
            continue
        code = trade.get("transaction_code", "")
        if code == "P":
            summary[t]["buys"] += 1
        elif code == "S":
            summary[t]["sells"] += 1
        score = score_form4_trade(trade, portfolio[t])
        if score > summary[t]["max_score"]:
            summary[t]["max_score"] = score
        fd = trade.get("filing_date", "")
        if not summary[t]["latest"] or fd > summary[t]["latest"]:
            summary[t]["latest"] = fd

    # Determine signal
    for s in summary.values():
        direction = s["direction"]
        if s["sells"] > s["buys"] * 2 and s["max_score"] >= 5:
            s["signal"] = "bearish"   # heavy selling
        elif s["buys"] > s["sells"] * 2 and s["max_score"] >= 5:
            s["signal"] = "bullish"   # heavy buying
        elif s["sells"] > 0 or s["buys"] > 0:
            s["signal"] = "mixed"

        # Conflict detection
        if direction == "long" and s["sells"] > s["buys"] and s["max_score"] >= 5:
            s["signal"] = "conflict"  # insiders selling, you're long → watch out
        elif direction == "short" and s["buys"] > s["sells"] and s["max_score"] >= 5:
            s["signal"] = "conflict"  # insiders buying, you're short → watch out

    return sorted(summary.values(), key=lambda x: x["max_score"], reverse=True)
