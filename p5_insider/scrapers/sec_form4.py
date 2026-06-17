"""
p5_insider/scrapers/sec_form4.py — SEC Form 4 insider transaction scraper.

Strategy: portfolio-targeted.
  For each ticker in the user's portfolio, we look up the company's CIK via the
  SEC's public ticker→CIK map, then fetch recent Form 4 filings from EDGAR,
  parse the XML for transaction details, and store new records.

Form 4 must be filed within 2 business days of the transaction, making this
much more timely than the 45-day congressional disclosure window.

SEC rate limit: ≤10 req/s. We use 0.2s delays (5 req/s) to be conservative.
Requires User-Agent header per SEC Fair Access Policy.
"""
from __future__ import annotations
import re
import time
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta

from p5_insider.db import (
    insert_form4_trade,
    get_known_form4_accessions,
    get_portfolio_tickers,
)

log = logging.getLogger("orcastrading.insider.form4")

_HEADERS = {
    "User-Agent": "Orcastrading/1.0 raul.s.gocan@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT   = 20
_DELAY     = 0.2   # seconds between SEC requests

_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_RSS      = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=40&output=atom"
)
_FILING_INDEX   = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_stripped}/"


def _get(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                log.debug(f"HTTP error {e} for {url}")
                return None
        except Exception as e:
            log.debug(f"Request failed ({attempt+1}/{retries}): {e}")
            time.sleep(0.5)
    return None


# ── Ticker → CIK map (cached in memory per process) ──────────────────────────

_ticker_cik: dict[str, str] = {}


def _load_ticker_cik_map() -> None:
    global _ticker_cik
    if _ticker_cik:
        return
    resp = _get(_TICKER_CIK_URL)
    if not resp:
        log.warning("Failed to load SEC ticker→CIK map")
        return
    try:
        data = resp.json()
        for entry in data.values():
            t = entry.get("ticker", "").upper()
            c = str(entry.get("cik_str", ""))
            if t and c:
                _ticker_cik[t] = c
        log.info(f"Loaded {len(_ticker_cik)} ticker→CIK mappings from SEC")
    except Exception as e:
        log.warning(f"CIK map parse error: {e}")


def _cik_for_ticker(ticker: str) -> str | None:
    _load_ticker_cik_map()
    return _ticker_cik.get(ticker.upper())


# ── EDGAR Atom feed → accession numbers ──────────────────────────────────────

def _get_recent_accessions(cik: str) -> list[tuple[str, str, str]]:
    """
    Fetch recent Form 4 filings for a company CIK.
    Returns list of (accession_number, issuer_cik, filing_date).
    """
    url = _EDGAR_RSS.format(cik=cik)
    time.sleep(_DELAY)
    resp = _get(url)
    if not resp:
        return []

    try:
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
    except ET.ParseError as e:
        log.debug(f"Atom parse error for CIK {cik}: {e}")
        return []

    results = []
    for entry in entries:
        id_el = entry.find("atom:id", ns)
        updated_el = entry.find("atom:updated", ns)
        if id_el is None:
            continue
        # id text: "urn:tag:sec.gov,2008:accession-number=0001234567-24-000001"
        acc_match = re.search(r"accession-number=(.+)", id_el.text or "")
        if not acc_match:
            continue
        acc_no = acc_match.group(1).strip()

        filing_date = ""
        if updated_el is not None and updated_el.text:
            try:
                filing_date = updated_el.text[:10]  # YYYY-MM-DD
            except Exception:
                pass

        results.append((acc_no, cik, filing_date))

    return results


# ── Filing XML location ───────────────────────────────────────────────────────

def _find_form4_xml_url(issuer_cik: str, acc_no: str) -> str | None:
    """Scan the filing directory HTML to find the primary Form 4 XML."""
    acc_stripped = acc_no.replace("-", "")
    index_url = _FILING_INDEX.format(cik=issuer_cik, acc_stripped=acc_stripped)
    time.sleep(_DELAY)
    resp = _get(index_url)
    if not resp:
        return None

    # Find all .xml hrefs in the listing
    links = re.findall(r'href="([^"]+\.xml)"', resp.text, re.IGNORECASE)
    if not links:
        return None

    base = f"https://www.sec.gov/Archives/edgar/data/{issuer_cik}/{acc_stripped}/"
    for link in links:
        name = link.split("/")[-1].lower()
        # Prefer files that look like Form 4 primary documents
        if any(kw in name for kw in ("form4", "wf-form4", "primary")):
            return base + link.split("/")[-1]
    # Fallback: return first XML
    return base + links[0].split("/")[-1]


# ── Form 4 XML parser ─────────────────────────────────────────────────────────

def _xml_text(el, path: str) -> str:
    """Safe text extraction from XML element."""
    try:
        node = el.find(path)
        return (node.text or "").strip() if node is not None else ""
    except Exception:
        return ""


def _parse_form4_xml(xml_text: str, acc_no: str, filing_date: str) -> list[dict]:
    """
    Parse a Form 4 XML document and return a list of transaction dicts.
    One Form 4 can contain multiple transactions (e.g. sell of 3 tranches).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.debug(f"XML parse error for {acc_no}: {e}")
        return []

    company  = _xml_text(root, ".//issuerName")
    ticker   = _xml_text(root, ".//issuerTradingSymbol").upper() or None
    cik      = _xml_text(root, ".//issuerCik")
    owner    = _xml_text(root, ".//rptOwnerName")
    title    = _xml_text(root, ".//officerTitle")
    period   = _xml_text(root, ".//periodOfReport")   # YYYY-MM-DD

    trades = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code  = _xml_text(txn, ".//transactionCode")
        date_ = _xml_text(txn, ".//transactionDate/value") or period

        try:
            shares = float(_xml_text(txn, ".//transactionShares/value") or 0)
        except ValueError:
            shares = None

        try:
            price = float(_xml_text(txn, ".//transactionPricePerShare/value") or 0)
        except ValueError:
            price = None

        try:
            owned_after = float(
                _xml_text(txn, ".//sharesOwnedFollowingTransaction/value") or 0
            )
        except ValueError:
            owned_after = None

        total_value = (shares * price) if (shares and price) else None

        if not code:
            continue   # skip rows with no transaction code

        trades.append({
            "filing_date":       filing_date,
            "transaction_date":  date_,
            "company":           company,
            "ticker":            ticker,
            "cik":               cik,
            "insider_name":      owner,
            "insider_title":     title,
            "transaction_code":  code,   # P=purchase, S=sale, A=award, etc.
            "shares":            shares,
            "price_per_share":   price,
            "total_value":       total_value,
            "shares_owned_after": owned_after,
            "accession_number":  acc_no,
        })

    return trades


# ── Public refresh ────────────────────────────────────────────────────────────

def refresh() -> int:
    """
    For each ticker in the user's portfolio:
      1. Look up company CIK
      2. Get recent Form 4 filings from EDGAR
      3. Parse any new XML documents
      4. Store new transaction records

    Returns total number of new transaction records inserted.
    """
    tickers = get_portfolio_tickers()
    if not tickers:
        log.info("Form4: no portfolio tickers — skipping")
        return 0

    known_accessions = get_known_form4_accessions()
    total_new = 0

    for ticker in sorted(tickers):
        cik = _cik_for_ticker(ticker)
        if not cik:
            log.debug(f"Form4: no CIK found for {ticker} (may not be a US-listed stock)")
            continue

        accessions = _get_recent_accessions(cik)
        new_for_ticker = 0

        for acc_no, issuer_cik, filing_date in accessions:
            if acc_no in known_accessions:
                continue

            xml_url = _find_form4_xml_url(issuer_cik, acc_no)
            if not xml_url:
                log.debug(f"Form4: no XML found for {acc_no}")
                continue

            time.sleep(_DELAY)
            resp = _get(xml_url)
            if not resp:
                continue

            txns = _parse_form4_xml(resp.text, acc_no, filing_date)
            for txn in txns:
                if insert_form4_trade(txn):
                    new_for_ticker += 1
                    known_accessions.add(acc_no)

        if new_for_ticker:
            log.info(f"Form4: {ticker} → {new_for_ticker} new transaction(s)")
        total_new += new_for_ticker

    log.info(f"Form4 refresh done — {total_new} new records across {len(tickers)} portfolio tickers")
    return total_new
