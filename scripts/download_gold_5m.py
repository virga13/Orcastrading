"""
scripts/download_gold_5m.py — Download and cache multi-year 5m XAU/USD data.

Fetches from Twelve Data in 60-day windows, deduplicates, and saves to
data/XAUUSD_5m.parquet. On subsequent runs, only fetches the missing tail
(incremental update).

Usage:
    python scripts/download_gold_5m.py                  # fetch from 2022-01-01
    python scripts/download_gold_5m.py --from 2021-01-01
    python scripts/download_gold_5m.py --update          # extend existing file only

Output: data/XAUUSD_5m.parquet  (Open/High/Low/Close/Volume, DatetimeIndex UTC)
"""
import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

CACHE_PATH   = Path(__file__).parent.parent / "data" / "XAUUSD_5m.parquet"
TICKER       = "XAU/USD"
INTERVAL     = "5min"
CHUNK_DAYS   = 14       # XAU/USD trades ~288 bars/day; 14d × 288 = 4032 bars < 5000 TD limit
RATE_DELAY   = 8.0      # seconds between requests (Twelve Data: 8 req/min on free tier)
DEFAULT_FROM = "2022-01-01"


# ── Twelve Data fetcher ───────────────────────────────────────────────────────

def _fetch_chunk(api_key: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one chunk of 5m XAU/USD from Twelve Data. Returns empty df on failure."""
    import urllib.request, json

    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={TICKER}&interval={INTERVAL}&outputsize=5000"
        f"&start_date={start}&end_date={end}"
        f"&apikey={api_key}&format=JSON"
    )
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  [warn] HTTP error for {start}–{end}: {e}")
        return pd.DataFrame()

    if data.get("status") == "error":
        msg = data.get("message", "")
        if "No data" in msg or "not found" in msg.lower():
            return pd.DataFrame()
        print(f"  [warn] API error for {start}–{end}: {msg}")
        return pd.DataFrame()

    values = data.get("values", [])
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    vol_col = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
    df["Volume"] = pd.to_numeric(vol_col, errors="coerce").fillna(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    return df[df["Close"] > 0]


# ── Main ──────────────────────────────────────────────────────────────────────

def download(start_from: str, update_only: bool = False) -> pd.DataFrame:
    api_key = os.getenv("TWELVEDATA_API_KEY", "").strip()
    if not api_key:
        print("ERROR: TWELVEDATA_API_KEY not set in .env")
        sys.exit(1)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    existing = pd.DataFrame()
    if CACHE_PATH.exists():
        existing = pd.read_parquet(CACHE_PATH)
        print(f"Existing cache: {len(existing):,} bars  "
              f"({existing.index[0].date()} to {existing.index[-1].date()})")

    # Determine fetch window
    if update_only and not existing.empty:
        fetch_from = (existing.index[-1].date() - timedelta(days=1)).isoformat()
    else:
        fetch_from = start_from

    fetch_to   = date.today().isoformat()

    if fetch_from >= fetch_to:
        print("Cache is already up to date.")
        return existing

    # Build chunk list
    chunks: list[tuple[str, str]] = []
    cur = datetime.strptime(fetch_from, "%Y-%m-%d").date()
    end = datetime.strptime(fetch_to,   "%Y-%m-%d").date()
    while cur < end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end)
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + timedelta(days=1)

    print(f"\nFetching {TICKER} 5m  {fetch_from} to {fetch_to}")
    print(f"  {len(chunks)} chunk(s) × up to {CHUNK_DAYS} days each")
    print(f"  Rate limit: 1 request every {RATE_DELAY}s ({60/RATE_DELAY:.0f}/min)")
    print()

    frames: list[pd.DataFrame] = []
    for i, (c_start, c_end) in enumerate(chunks, 1):
        print(f"  [{i:2}/{len(chunks)}]  {c_start} to {c_end} ... ", end="", flush=True)
        df = _fetch_chunk(api_key, c_start, c_end)
        if df.empty:
            print("no data")
        else:
            print(f"{len(df):5} bars  "
                  f"({df.index[0].date()} to {df.index[-1].date()})")
            frames.append(df)

        if i < len(chunks):
            time.sleep(RATE_DELAY)

    if not frames:
        print("\nNo new data fetched.")
        return existing

    # Merge with existing cache, deduplicate
    new_data = pd.concat(frames).sort_index()
    combined = pd.concat([existing, new_data]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    combined.to_parquet(CACHE_PATH)
    print(f"\nSaved: {len(combined):,} bars  "
          f"({combined.index[0].date()} to {combined.index[-1].date()})")
    print(f"Path : {CACHE_PATH}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Download 5m XAU/USD history from Twelve Data")
    parser.add_argument("--from", dest="start_from", default=DEFAULT_FROM,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_FROM})")
    parser.add_argument("--update", action="store_true",
                        help="Only fetch bars newer than the cached tail")
    args = parser.parse_args()

    download(start_from=args.start_from, update_only=args.update)


if __name__ == "__main__":
    main()
