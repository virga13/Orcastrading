"""
p5_insider/momentum_scanner.py — Multi-index short-term momentum scanner.

Universe (combined, deduped ~2600 tickers):
  Russell 2000  — small caps      (~2000, from iShares IWM)
  S&P 500       — large caps      (~500,  from Wikipedia)
  NASDAQ 100    — tech/growth     (~100,  from Wikipedia)
  US30 / DJIA   — blue chips      (30,    from Wikipedia)

Fires Telegram alerts when:
  - Price moves >= 3% in the last 10 minutes  (momentum burst)
  - Current-hour volume >= 5x average hourly  (volume surge)

Thresholds configurable via .env (MOM_* prefix).
Alert history: data/momentum.db  (SQLite).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger("orcastrading.momentum")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).parent.parent
_DATA   = _ROOT / "data"
_DATA.mkdir(exist_ok=True)

_CACHE = {
    "russell2000": _DATA / "russell2000_tickers.csv",
    "sp500":       _DATA / "sp500_tickers.csv",
    "nasdaq100":   _DATA / "nasdaq100_tickers.csv",
    "us30":        _DATA / "us30_tickers.csv",
}
_VOL_CACHE = _DATA / "momentum_vol_cache.json"
_DB_PATH   = _DATA / "momentum.db"

# ── Thresholds (override via .env) ────────────────────────────────────────────
PRICE_SPIKE_PCT    = float(os.getenv("MOM_PRICE_SPIKE_PCT",    "3.0"))
PRICE_WINDOW_MIN   = int(os.getenv("MOM_PRICE_WINDOW_MIN",     "10"))
VOLUME_SURGE_RATIO = float(os.getenv("MOM_VOLUME_SURGE_RATIO", "5.0"))
COOLDOWN_HOURS     = float(os.getenv("MOM_COOLDOWN_HOURS",     "4.0"))
BATCH_SIZE         = int(os.getenv("MOM_BATCH_SIZE",           "100"))

# Index display labels (priority order for labelling overlapping tickers)
_INDEX_PRIORITY = ["us30", "nasdaq100", "sp500", "russell2000"]
_INDEX_LABEL    = {
    "us30":        "DJIA",
    "nasdaq100":   "NDX",
    "sp500":       "SPX",
    "russell2000": "R2K",
}


# ── Database ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS momentum_alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    signal_type  TEXT NOT NULL,
    direction    TEXT NOT NULL,
    pct_change   REAL,
    volume_ratio REAL,
    price        REAL,
    window_min   INTEGER,
    index_name   TEXT,
    triggered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cooldowns (
    key        TEXT PRIMARY KEY,
    last_alert TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mom_ticker_ts
    ON momentum_alerts(ticker, triggered_at DESC);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH))
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    # Migrate: add index_name column if this DB was created before it existed
    try:
        c.execute("ALTER TABLE momentum_alerts ADD COLUMN index_name TEXT")
        c.commit()
    except Exception:
        pass
    return c


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_recent_alerts(hours: int = 24, limit: int = 500) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM momentum_alerts "
        "WHERE triggered_at >= ? ORDER BY triggered_at DESC LIMIT ?",
        (cutoff, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _log_alert(signal: dict) -> None:
    conn = _conn()
    conn.execute(
        """INSERT INTO momentum_alerts
           (ticker, signal_type, direction, pct_change, volume_ratio,
            price, window_min, index_name, triggered_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            signal["ticker"], signal["signal_type"], signal["direction"],
            signal.get("pct_change"), signal.get("volume_ratio"),
            signal.get("price"), signal.get("window_min"),
            signal.get("index_name"), signal["triggered_at"],
        ),
    )
    conn.commit()
    conn.close()


# ── Cooldown cache ────────────────────────────────────────────────────────────

class _CooldownCache:
    def __init__(self) -> None:
        conn = _conn()
        rows = conn.execute("SELECT key, last_alert FROM cooldowns").fetchall()
        conn.close()
        self._data:  dict[str, str] = {r["key"]: r["last_alert"] for r in rows}
        self._dirty: dict[str, str] = {}

    def is_hot(self, ticker: str, signal_type: str) -> bool:
        ts_s = self._data.get(f"{ticker}:{signal_type}")
        if not ts_s:
            return False
        try:
            last = datetime.fromisoformat(ts_s)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - last).total_seconds() < COOLDOWN_HOURS * 3600
        except Exception:
            return False

    def mark(self, ticker: str, signal_type: str) -> None:
        key = f"{ticker}:{signal_type}"
        now = _now_utc()
        self._data[key]  = now
        self._dirty[key] = now

    def flush(self) -> None:
        if not self._dirty:
            return
        conn = _conn()
        for key, ts in self._dirty.items():
            conn.execute(
                "INSERT OR REPLACE INTO cooldowns (key, last_alert) VALUES (?,?)",
                (key, ts),
            )
        conn.commit()
        conn.close()
        self._dirty.clear()


# ── Ticker list helpers ───────────────────────────────────────────────────────

_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$")


def _clean_tickers(raw: list[str]) -> list[str]:
    """Uppercase, strip, remove non-ticker strings."""
    seen:    set[str] = set()
    result:  list[str] = []
    skip = {"CASH", "USD", "-", "N/A", ""}
    for t in raw:
        t = t.strip().upper()
        if t in skip or not _TICKER_RE.match(t) or t in seen:
            continue
        seen.add(t)
        result.append(t)
    return result


def _save_ticker_csv(path: Path, tickers: list[str]) -> None:
    try:
        path.write_text("\n".join(tickers), encoding="utf-8")
    except Exception as e:
        log.warning(f"Ticker CSV write failed ({path.name}): {e}")


def _load_ticker_csv(path: Path) -> list[str]:
    try:
        return _clean_tickers(path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return []


def _fetch_html(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        log.warning(f"HTTP fetch failed ({url}): {e}")
        return None


def _wiki_tickers(html: str, min_count: int = 10) -> list[str]:
    """
    Parse all tables in a Wikipedia HTML page and return the first ticker list
    found (column named 'symbol', 'ticker', or similar with >= min_count rows).
    """
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        log.warning(f"pd.read_html failed: {e}")
        return []

    for df in tables:
        ticker_col = next(
            (c for c in df.columns
             if any(kw in str(c).lower() for kw in ("symbol", "ticker"))),
            None,
        )
        if ticker_col is None:
            continue
        tickers = _clean_tickers(df[ticker_col].dropna().astype(str).tolist())
        if len(tickers) >= min_count:
            return tickers
    return []


def _cached_or_fetch(
    index_key: str,
    fetch_fn,
    max_age_days: int = 7,
    fallback: list[str] | None = None,
) -> list[str]:
    """Generic: return cached tickers if fresh, else call fetch_fn() to refresh."""
    path = _CACHE[index_key]
    if path.exists():
        age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
        if age <= max_age_days:
            tickers = _load_ticker_csv(path)
            if tickers:
                return tickers

    log.info(f"Refreshing {index_key} ticker list...")
    try:
        tickers = fetch_fn()
    except Exception as e:
        log.warning(f"{index_key} fetch failed: {e}")
        tickers = []

    if tickers:
        _save_ticker_csv(path, tickers)
        log.info(f"{index_key}: {len(tickers)} tickers")
        return tickers

    # Fall back to cache even if stale, then hardcoded fallback
    if path.exists():
        cached = _load_ticker_csv(path)
        if cached:
            log.warning(f"{index_key}: using stale cache ({len(cached)} tickers)")
            return cached

    if fallback:
        log.warning(f"{index_key}: using hardcoded fallback ({len(fallback)} tickers)")
        return fallback

    return []


# ── Russell 2000 — NASDAQ screener API, filtered by market cap ────────────────
# iShares IWM CSV is JS-protected; the NASDAQ public screener API returns all
# ~7000 US-listed stocks with market cap.  Filtering to $100M–$4B approximates
# the Russell 2000 small-cap universe (constituents ranked ~1001–3000 by mcap).

_NASDAQ_API = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=10000&offset=0&download=true"
)
_R2K_MCAP_MIN = 100_000_000    # $100M
_R2K_MCAP_MAX = 4_000_000_000  # $4B


def _fetch_russell2000() -> list[str]:
    import json as _j
    req = urllib.request.Request(
        _NASDAQ_API,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = _j.loads(resp.read())

    rows = data.get("data", {}).get("rows", [])
    tickers: list[str] = []
    for row in rows:
        sym  = row.get("symbol", "").strip().upper()
        mcap = row.get("marketCap", "") or ""
        try:
            mcap_val = float(str(mcap).replace(",", ""))
        except ValueError:
            continue
        if _TICKER_RE.match(sym) and _R2K_MCAP_MIN <= mcap_val <= _R2K_MCAP_MAX:
            tickers.append(sym)

    return tickers


def get_russell2000_tickers(max_age_days: int = 7) -> list[str]:
    return _cached_or_fetch("russell2000", _fetch_russell2000, max_age_days)


# ── S&P 500 (Wikipedia) ───────────────────────────────────────────────────────

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers(max_age_days: int = 7) -> list[str]:
    def _fetch():
        html = _fetch_html(_SP500_URL)
        return _wiki_tickers(html, min_count=400) if html else []

    return _cached_or_fetch("sp500", _fetch, max_age_days)


# ── NASDAQ 100 (Wikipedia) ────────────────────────────────────────────────────

_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

_NDX_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AMD","ADBE","PEP","CSCO","QCOM","TMUS","AMAT","AMGN",
    "INTU","CMCSA","TXN","ISRG","MU","HON","LRCX","REGN","KLAC","VRTX",
    "SNPS","CDNS","ADI","PANW","MELI","MDLZ","GILD","SBUX","CEG","PYPL",
    "ORLY","MNST","MRVL","FTNT","AEP","NXPI","ROST","PAYX","ABNB","WDAY",
    "DXCM","CTSH","FAST","ODFL","VRSK","PCAR","EA","IDXX","BKR","XEL",
    "ON","GEHC","CCEP","KHC","CPRT","CDW","MRNA","WBD","TTD","DLTR",
    "ZS","ROP","CHTR","TEAM","ANSS","TTWO","FANG","EXC","CSGP","GFS",
]


def get_nasdaq100_tickers(max_age_days: int = 7) -> list[str]:
    def _fetch():
        html = _fetch_html(_NDX_URL)
        return _wiki_tickers(html, min_count=80) if html else []

    return _cached_or_fetch("nasdaq100", _fetch, max_age_days, fallback=_NDX_FALLBACK)


# ── US30 / DJIA (Wikipedia) ───────────────────────────────────────────────────

_DJIA_URL = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"

_US30_FALLBACK = [
    "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
    "GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK",
    "MSFT","NKE","NVDA","PG","SHW","TRV","UNH","V","VZ","WMT",
]


def get_us30_tickers(max_age_days: int = 30) -> list[str]:
    def _fetch():
        html = _fetch_html(_DJIA_URL)
        return _wiki_tickers(html, min_count=28) if html else []

    return _cached_or_fetch("us30", _fetch, max_age_days, fallback=_US30_FALLBACK)


# ── Combined universe ─────────────────────────────────────────────────────────

def get_universe_tickers() -> tuple[list[str], dict[str, str]]:
    """
    Return (tickers, index_map) where index_map[ticker] = highest-priority index label.
    Priority: DJIA > NDX > SPX > R2K  (shown in alerts).
    """
    sets: dict[str, set[str]] = {}
    sets["russell2000"] = set(get_russell2000_tickers())
    sets["sp500"]       = set(get_sp500_tickers())
    sets["nasdaq100"]   = set(get_nasdaq100_tickers())
    sets["us30"]        = set(get_us30_tickers())

    # Build index map: each ticker gets the label of its most prestigious index
    index_map: dict[str, str] = {}
    for idx in reversed(_INDEX_PRIORITY):   # lower-priority first, higher overwrites
        for t in sets[idx]:
            index_map[t] = _INDEX_LABEL[idx]

    all_tickers = list(index_map.keys())
    total = sum(len(v) for v in sets.values())
    unique = len(all_tickers)
    log.info(
        f"Universe: {unique} unique tickers ({total} raw) — "
        + "  ".join(f"{_INDEX_LABEL[k]}: {len(v)}" for k, v in sets.items())
    )
    return all_tickers, index_map


# ── Volume baseline (cached once per day) ─────────────────────────────────────

def _load_vol_cache() -> dict:
    try:
        if _VOL_CACHE.exists():
            return json.loads(_VOL_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"date": "", "volumes": {}}


def _save_vol_cache(volumes: dict[str, float]) -> None:
    try:
        _VOL_CACHE.write_text(
            json.dumps({"date": date.today().isoformat(), "volumes": volumes}),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning(f"Vol cache write failed: {e}")


def get_avg_daily_volumes(tickers: list[str]) -> dict[str, float]:
    """20-day average daily volume per ticker. Rebuilt once per day."""
    cache = _load_vol_cache()
    if cache.get("date") == date.today().isoformat() and cache.get("volumes"):
        return cache["volumes"]

    log.info(f"Building 20-day volume baseline for {len(tickers)} tickers...")
    volumes: dict[str, float] = {}
    chunks = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for i, chunk in enumerate(chunks):
        try:
            df = yf.download(
                chunk, period="25d", interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
            if df.empty:
                continue
            if len(chunk) == 1:
                ticker = chunk[0]
                if "Volume" in df.columns:
                    v = float(df["Volume"].dropna().mean())
                    if v > 0:
                        volumes[ticker] = v
            else:
                for ticker in chunk:
                    try:
                        series = df[ticker]["Volume"]
                        v = float(series.dropna().mean())
                        if v > 0:
                            volumes[ticker] = v
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"Vol baseline batch {i}: {e}")
        time.sleep(0.5)

    log.info(f"Volume baseline ready: {len(volumes)} tickers")
    _save_vol_cache(volumes)
    return volumes


# ── Signal detection ──────────────────────────────────────────────────────────

def _detect_signals(
    ticker: str,
    bars: pd.DataFrame,
    avg_daily_vol: float,
    cooldowns: _CooldownCache,
    index_name: str = "",
) -> list[dict]:
    if bars is None or bars.empty:
        return []

    bars = bars.dropna(subset=["Close", "Volume"])
    if len(bars) < PRICE_WINDOW_MIN + 2:
        return []

    now_price  = float(bars["Close"].iloc[-1])
    past_price = float(bars["Close"].iloc[-(PRICE_WINDOW_MIN + 1)])
    if past_price <= 0:
        return []

    pct_chg   = (now_price - past_price) / past_price * 100
    direction = "up" if pct_chg >= 0 else "down"
    ts        = _now_utc()
    signals:  list[dict] = []

    base = dict(
        ticker=ticker, direction=direction,
        pct_change=round(pct_chg, 2), price=round(now_price, 4),
        window_min=PRICE_WINDOW_MIN, index_name=index_name, triggered_at=ts,
    )

    # ── Price spike ───────────────────────────────────────────────────────────
    if abs(pct_chg) >= PRICE_SPIKE_PCT and not cooldowns.is_hot(ticker, "price"):
        signals.append({**base, "signal_type": "price_spike"})
        cooldowns.mark(ticker, "price")

    # ── Volume surge ──────────────────────────────────────────────────────────
    if avg_daily_vol > 0:
        avg_hourly  = avg_daily_vol / 6.5
        one_hr_ago  = bars.index[-1] - pd.Timedelta(minutes=60)
        current_vol = float(bars.loc[bars.index >= one_hr_ago, "Volume"].sum())
        vol_ratio   = current_vol / avg_hourly if avg_hourly > 0 else 0.0

        if vol_ratio >= VOLUME_SURGE_RATIO and not cooldowns.is_hot(ticker, "volume"):
            signals.append({**base, "signal_type": "volume_surge", "volume_ratio": round(vol_ratio, 1)})
            cooldowns.mark(ticker, "volume")

    return signals


# ── Telegram alert ────────────────────────────────────────────────────────────

def _send_telegram(text: str) -> bool:
    token = (
        os.getenv("TELEGRAM_MOMENTUM_BOT_TOKEN", "").strip()
        or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    )
    chat_id = (
        os.getenv("TELEGRAM_MOMENTUM_CHAT_ID", "").strip()
        or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    )
    if not token or not chat_id:
        return False
    try:
        import json as _j
        payload = _j.dumps({
            "chat_id": chat_id, "text": text, "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return bool(_j.loads(resp.read()).get("ok"))
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        return False


def send_momentum_alerts(signals: list[dict]) -> int:
    """Send one Telegram message per signal, log each to the DB."""
    try:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
    except Exception:
        _ET = None

    sent = 0
    for sig in signals:
        ticker    = sig["ticker"]
        stype     = sig["signal_type"]
        pct       = sig.get("pct_change") or 0.0
        price     = sig.get("price") or 0.0
        vol_ratio = sig.get("volume_ratio")
        direction = sig.get("direction", "up")
        window    = sig.get("window_min", PRICE_WINDOW_MIN)
        idx_label = sig.get("index_name", "")
        pct_sign  = f"+{pct:.1f}" if pct >= 0 else f"{pct:.1f}"
        arrow     = "🟢" if direction == "up" else "🔴"
        idx_tag   = f"  <i>{idx_label}</i>" if idx_label else ""

        try:
            ts_utc = datetime.fromisoformat(sig["triggered_at"])
            ts_et  = ts_utc.astimezone(_ET).strftime("%H:%M ET") if _ET else ""
        except Exception:
            ts_et = ""

        if stype == "price_spike":
            icon  = "🚀" if direction == "up" else "🔻"
            title = f"{icon} <b>{ticker}</b>  {pct_sign}% in {window}min{idx_tag}"
            body  = (
                f"{arrow} Price: <b>${price:,.4g}</b>\n"
                f"Signal: MOMENTUM BURST\n"
                f"Time: {ts_et}"
            )
        elif stype == "volume_surge":
            icon  = "🔥"
            title = f"{icon} <b>{ticker}</b>  vol {vol_ratio:.1f}× avg{idx_tag}"
            body  = (
                f"{arrow} Price: <b>${price:,.4g}</b>  ({pct_sign}% vs {window}min ago)\n"
                f"Signal: VOLUME SURGE\n"
                f"Time: {ts_et}"
            )
        else:
            title = f"⚡ <b>{ticker}</b>  [{stype.upper()}]{idx_tag}"
            body  = f"Price: ${price:,.4g}  ({pct_sign}%)\n{ts_et}"

        ok = _send_telegram(f"{title}\n{body}")
        _log_alert(sig)
        if ok:
            sent += 1
            log.info(f"  Alert sent: {ticker} [{stype}] [{idx_label}]")
        else:
            log.warning(f"  Alert failed (no Telegram): {ticker} [{stype}]")

    return sent


# ── Main scan ─────────────────────────────────────────────────────────────────

def scan_momentum() -> list[dict]:
    """
    Full universe momentum scan (Russell 2000 + S&P 500 + NASDAQ 100 + US30).
    Downloads today's 1-min bars in batches, detects signals, returns fired list.
    Called every 5 minutes during US market hours.
    """
    tickers, index_map = get_universe_tickers()
    if not tickers:
        log.warning("Momentum scan aborted — no tickers available")
        return []

    avg_vols  = get_avg_daily_volumes(tickers)
    cooldowns = _CooldownCache()

    log.info(f"Momentum scan: {len(tickers)} tickers across 4 indices")
    all_signals: list[dict] = []
    chunks = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for i, chunk in enumerate(chunks):
        try:
            df = yf.download(
                chunk, period="1d", interval="1m",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True, prepost=False,
            )
            if df.empty:
                continue

            if len(chunk) == 1:
                t    = chunk[0]
                sigs = _detect_signals(t, df, avg_vols.get(t, 0.0), cooldowns, index_map.get(t, ""))
                all_signals.extend(sigs)
            else:
                # yfinance 1.x: MultiIndex is (ticker, field) — level 0 = tickers
                if "Close" not in df.columns.get_level_values(1):
                    continue
                for ticker in chunk:
                    try:
                        tdf  = df[ticker]
                        sigs = _detect_signals(ticker, tdf, avg_vols.get(ticker, 0.0), cooldowns, index_map.get(ticker, ""))
                        all_signals.extend(sigs)
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"Batch {i+1}/{len(chunks)}: {e}")

        time.sleep(0.3)

    cooldowns.flush()

    if all_signals:
        log.info(f"Momentum scan done — {len(all_signals)} signal(s)")
        for s in sorted(all_signals, key=lambda x: abs(x.get("pct_change") or 0), reverse=True):
            vol_s = f"  vol {s['volume_ratio']:.1f}x" if s.get("volume_ratio") else ""
            log.info(
                f"  {s['ticker']:6}  [{s.get('index_name',''):4}]  "
                f"{s['signal_type']:14}  {s.get('pct_change', 0):+.1f}%{vol_s}"
            )
    else:
        log.debug("Momentum scan done — no signals this cycle")

    return all_signals
