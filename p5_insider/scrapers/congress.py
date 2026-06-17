"""
p5_insider/scrapers/congress.py — Congressional trade disclosures.

Sources (free, no API key):
  House : HouseStockWatcher S3 JSON feed
  Senate: SenateSockWatcher  S3 JSON feed

Both feeds contain all historical disclosures. New records are detected
by the UNIQUE constraint in the DB (source + date + politician + ticker + type).

Disclosure window: Congress members must report within 45 days of the trade.
We therefore pick up trades well before they appear on mainstream news/social media.
"""
from __future__ import annotations
import re
import time
import logging
import requests
from datetime import datetime

from p5_insider.db import insert_congress_trade

log = logging.getLogger("orcastrading.insider.congress")

_HOUSE_URL  = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"

_HEADERS = {"User-Agent": "Orcastrading/1.0 raul.s.gocan@gmail.com"}
_TIMEOUT = 30


# ── Amount parsing ────────────────────────────────────────────────────────────

def _parse_amount(s: str) -> tuple[float | None, float | None]:
    """Parse a range string like '$50,001 - $100,000' → (50001, 100000)."""
    if not s:
        return None, None
    s_clean = s.replace(",", "").replace("$", "").strip()
    if s_clean.lower().startswith("over"):
        nums = re.findall(r"\d+", s_clean)
        return (float(nums[0]), None) if nums else (None, None)
    nums = re.findall(r"\d+", s_clean)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), None
    return None, None


def _normalize_date(s: str) -> str | None:
    """Normalize various date formats to YYYY-MM-DD."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s  # return as-is if unparseable


def _normalize_ticker(t: str | None) -> str | None:
    if not t:
        return None
    t = t.strip().upper()
    # Skip non-stock entries
    if t in ("N/A", "NA", "--", "", "NONE"):
        return None
    # Some entries have suffixes like "SPY--" or "AAPL--"
    t = re.sub(r"-+$", "", t)
    return t or None


# ── House ─────────────────────────────────────────────────────────────────────

def _fetch_house() -> list[dict]:
    try:
        resp = requests.get(_HOUSE_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"House fetch failed: {e}")
        return []

    rows = []
    for item in data:
        ticker = _normalize_ticker(item.get("ticker"))
        amount_str = item.get("amount", "")
        low, high = _parse_amount(amount_str)
        rows.append({
            "source":           "house",
            "disclosure_date":  _normalize_date(item.get("disclosure_date", "")),
            "transaction_date": _normalize_date(item.get("transaction_date", "")),
            "politician":       item.get("representative", ""),
            "party":            item.get("party", ""),
            "state":            item.get("state", ""),
            "ticker":           ticker,
            "asset_description": item.get("asset_description", ""),
            "transaction_type": item.get("type", "").lower(),
            "amount_str":       amount_str,
            "amount_low":       low,
            "amount_high":      high,
            "comment":          item.get("comment", ""),
        })
    return rows


# ── Senate ────────────────────────────────────────────────────────────────────

def _fetch_senate() -> list[dict]:
    try:
        resp = requests.get(_SENATE_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"Senate fetch failed: {e}")
        return []

    rows = []
    for item in data:
        ticker = _normalize_ticker(item.get("ticker"))
        amount_str = item.get("amount", "")
        low, high = _parse_amount(amount_str)
        rows.append({
            "source":           "senate",
            "disclosure_date":  _normalize_date(item.get("disclosure_date", "")),
            "transaction_date": _normalize_date(item.get("transaction_date", "")),
            "politician":       item.get("senator", ""),
            "party":            item.get("party", ""),
            "state":            item.get("state", ""),
            "ticker":           ticker,
            "asset_description": item.get("asset_description", ""),
            "transaction_type": item.get("type", "").lower(),
            "amount_str":       amount_str,
            "amount_low":       low,
            "amount_high":      high,
            "comment":          item.get("comment", ""),
        })
    return rows


# ── Public refresh ────────────────────────────────────────────────────────────

def refresh() -> dict[str, int]:
    """
    Fetch both House and Senate feeds and store any new records.
    Returns {"house": n_new, "senate": n_new}.
    """
    results = {"house": 0, "senate": 0}

    house_rows = _fetch_house()
    for row in house_rows:
        if insert_congress_trade(row):
            results["house"] += 1
    log.info(f"Congress House: {results['house']} new trades from {len(house_rows)} total")

    time.sleep(1)  # polite pause between the two S3 requests

    senate_rows = _fetch_senate()
    for row in senate_rows:
        if insert_congress_trade(row):
            results["senate"] += 1
    log.info(f"Congress Senate: {results['senate']} new trades from {len(senate_rows)} total")

    return results
