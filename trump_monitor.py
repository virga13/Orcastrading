"""
trump_monitor.py — Polls @realDonaldTrump on Truth Social every 2 s.

Telegram alerts fire only when a post is market-relevant:
  - Any company name detected (cashtag, corporate suffix, or curated list)
  - Any known CEO / executive mentioned
  - Any industry / sector keyword
  - Sentiment score >= TRUMP_TG_MIN_SCORE (default 6, set in .env)

Console prints ALL posts so nothing is missed in the logs.
RT wrappers and video/image-only posts are silently skipped.

Usage:
    python trump_monitor.py
    python trump_monitor.py --interval 3   # custom poll interval in seconds
    python trump_monitor.py --no-telegram  # console-only

Env vars:
    TRUMP_TG_MIN_SCORE   minimum score for Telegram (default 6)
    TRUMP_POLL_INTERVAL  poll interval in seconds (default 2)
    TRUMP_NO_TELEGRAM    set to 1 to disable Telegram
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import time
import urllib.request
from datetime import datetime

log = logging.getLogger("orcastrading.trump")

# ── Config ────────────────────────────────────────────────────────────────────

_ACCOUNT_ID = "107780257626128497"   # @realDonaldTrump on Truth Social
_BASE        = "https://truthsocial.com/api/v1"
_HEADERS     = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json",
}

# Minimum score to send a Telegram alert (console always prints everything).
_TG_MIN_SCORE = int(os.getenv("TRUMP_TG_MIN_SCORE", "6"))
_URGENT_FLOOR = 15


# ── Open company detection (no predefined list needed) ────────────────────────

# Cashtags: $AAPL, $NVDA, $MRVL — unambiguous stock references, score 10
_CASHTAG_PAT = re.compile(r"\$[A-Z]{1,5}\b")

# Corporate suffix: catches any "X Corp", "X Inc", "X Technologies" etc.
# Works for companies we've never heard of — if Trump names them, we catch them.
_CORP_SUFFIX_PAT = re.compile(
    r"\b[A-Z]\w+(?:\s+[A-Z]\w+)?\s+"
    r"(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Co\.?|Holdings?|Technologies?|"
    r"Pharmaceuticals?|Semiconductor|Biotech|Motors?|Airlines?|Airways?|"
    r"Bancorp|Brands?|Ventures?|Capital|Partners?|International|"
    r"Industries|Systems?|Solutions?|Communications?|Sciences?)\b"
)

# CEO title context: "CEO of X", "president of X", "founder of X"
_CEO_TITLE_PAT = re.compile(
    r"\b(?:ceo|chief executive|founder|chairman|president)\s+of\s+[A-Z]\w+",
    re.IGNORECASE,
)


# ── Curated lists (only companies/people whose names are hard to pattern-match) ─

# (score, regex_pattern, display_label)
_COMPANIES: list[tuple[int, str, str]] = [
    # Big Tech
    (9, r"\bapple\b",              "Apple"),
    (9, r"\btesla\b",              "Tesla"),
    (9, r"\bnvidia\b|\bnvda\b",   "Nvidia"),
    (9, r"\bmicrosoft\b|\bmsft\b", "Microsoft"),
    (9, r"\bamazon\b|\bamzn\b",   "Amazon"),
    (9, r"\bmeta\b|\bfacebook\b|\binstagram\b", "Meta"),
    (9, r"\bgoogle\b|\balphabet\b", "Google/Alphabet"),
    (9, r"\btiktok\b|\bbytedance\b", "TikTok"),
    (8, r"\bopenai\b|\bchatgpt\b", "OpenAI"),
    (8, r"\banthropics?\b",        "Anthropic"),
    # Industrial / Defense
    (9, r"\bboeing\b",             "Boeing"),
    (8, r"\blockheed\b",           "Lockheed"),
    (8, r"\braytheon\b",           "Raytheon"),
    (8, r"\bnorthrop\b",           "Northrop"),
    # Autos
    (8, r"\bford\b",               "Ford"),
    (8, r"\bgeneral motors\b|\bgm\b", "GM"),
    (8, r"\bstellantis\b",         "Stellantis"),
    # Energy
    (8, r"\bexxon\b|\bexxonmobil\b", "ExxonMobil"),
    (8, r"\bchevron\b",            "Chevron"),
    # Finance
    (8, r"\bjpmorgan\b|\bj\.p\. morgan\b", "JPMorgan"),
    (8, r"\bgoldman\b",            "Goldman"),
    (8, r"\bmorgan stanley\b",     "Morgan Stanley"),
    (8, r"\bblackrock\b",          "BlackRock"),
    # Retail / Consumer
    (7, r"\bwalmart\b",            "Walmart"),
    (7, r"\bdisney\b",             "Disney"),
    # Semis
    (8, r"\bintel\b",              "Intel"),
    (8, r"\bamd\b",                "AMD"),
    (9, r"\bmarvell\b|\bmrvl\b",  "Marvell"),
    (8, r"\bqualcomm\b",           "Qualcomm"),
    (8, r"\btsmc\b",               "TSMC"),
    # Crypto platforms
    (8, r"\bcoinbase\b",           "Coinbase"),
    (8, r"\bbinance\b",            "Binance"),
]

_CEOS: list[tuple[int, str, str]] = [
    (9, r"\belon\b|\bmusk\b",           "Elon Musk"),
    (9, r"\bjensen\b|\bjensen huang\b", "Jensen Huang"),
    (8, r"\bbezos\b",                    "Bezos"),
    (8, r"\bzuckerberg\b|\bzuck\b",     "Zuckerberg"),
    (8, r"\bdimon\b",                    "Dimon"),
    (8, r"\bbuffett\b",                  "Buffett"),
    (8, r"\btim cook\b",                "Tim Cook"),
    (7, r"\blarry fink\b|\bfink\b",     "Larry Fink"),
    (7, r"\bsundar\b",                   "Sundar Pichai"),
    (7, r"\bsam altman\b",              "Sam Altman"),
    (8, r"\bpowell\b|\bjerome powell\b", "Jerome Powell"),
]

_INDUSTRIES: list[tuple[int, str, str]] = [
    (8, r"\bsemiconductor\b|\bchipmaker\b|\bchip\b",        "semiconductors"),
    (8, r"\bartificial intelligence\b|\b\bai\b",             "AI"),
    (8, r"\bcrypto\b|\bbitcoin\b|\bethereum\b|\bblockchain\b", "crypto"),
    (7, r"\bpharmac\w+\b|\bbiotech\b|\bdrug\b|\bmedicine\b", "pharma/biotech"),
    (7, r"\bdefense\b|\bmilitary\b|\bweapon\b",              "defense"),
    (7, r"\boil\b|\bcrude\b|\bnatural gas\b|\bopec\b",       "oil/gas"),
    (7, r"\bsteel\b|\baluminum\b|\bmetal\b",                  "steel/metals"),
    (7, r"\belectric vehicle\b|\b\bev\b|\bself.driving\b|\bautonomous\b", "EV"),
    (7, r"\bairline\b|\baviation\b",                          "airlines"),
    (7, r"\bwall street\b|\bbanking\b|\bfinancial\b",         "finance/banking"),
    (6, r"\bhousing\b|\breal estate\b|\bmortgage\b",          "real estate"),
    (6, r"\bcoal\b|\benergy sector\b",                        "energy"),
    (6, r"\binfrastructure\b",                                "infrastructure"),
]

_POLICIES: list[tuple[int, str, str]] = [
    (8, r"\bfederal reserve\b|\bthe fed\b|\binterest rate\b|\brate cut\b|\brate hike\b", "Fed/rates"),
    (7, r"\btax cut\b|\btax reform\b|\btax break\b",         "tax reform"),
    (7, r"\bderegulat\w+\b",                                  "deregulation"),
    (7, r"\bexecutive order\b",                               "executive order"),
    (6, r"\bhealthcare\b|\bmedicare\b|\bmedicaid\b",          "healthcare"),
    (6, r"\bspending bill\b|\bstimulus\b|\brescue plan\b",   "stimulus"),
]


# ── Legacy sentiment scoring (kept, now part of the combined score) ────────────

_BULLISH: list[tuple[int, str]] = [
    (10, r"\btime to buy\b"),
    (10, r"\bgreat deal\b"),
    (10, r"\btrade deal\b"),
    (10, r"\bdeal is done\b"),
    (10, r"\bwe made a deal\b"),
    (8,  r"\bagreement\b"),
    (8,  r"\btariffs? (?:will be |are )?removed\b"),
    (8,  r"\bno more tariffs?\b"),
    (8,  r"\bopen (?:for )?business\b"),
    (7,  r"\bmarket (?:is )?great\b"),
    (7,  r"\bstock market\b"),
    (5,  r"\beconomy (?:is )?(?:booming|great|strong|winning)\b"),
]

_BEARISH: list[tuple[int, str]] = [
    (10, r"\btariff"),
    (10, r"\bsanctions?\b"),
    (10, r"\bimpose (?:new )?taxes?\b"),
    (9,  r"\bchina\b"),
    (9,  r"\btrade war\b"),
    (8,  r"\bban\b"),
    (8,  r"\bembargo\b"),
    (7,  r"\bpunish\b"),
    (6,  r"\bfight back\b"),
]


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(text: str) -> tuple[int, str, list[str], bool]:
    """
    Score a post for market relevance.

    Returns:
        score      — total relevance score
        direction  — 'bullish' | 'bearish' | 'neutral'
        hits       — list of matched label strings for display
        has_entity — True if a company / CEO / cashtag / industry was detected
                     (these always trigger Telegram regardless of total score)
    """
    low        = text.lower()
    bull       = 0
    bear       = 0
    hits:  list[str] = []
    has_entity = False

    # ── Open detection: cashtags ($AAPL, $MRVL) ──────────────────────────────
    cashtags = _CASHTAG_PAT.findall(text)
    for tag in cashtags:
        bull += 10
        hits.append(tag)
        has_entity = True

    # ── Open detection: corporate suffix ("Marvell Technologies", "X Corp") ──
    corp_matches = _CORP_SUFFIX_PAT.findall(text)
    for match in corp_matches:
        name = match.strip()
        bull += 8
        hits.append(f"[co: {name}]")
        has_entity = True

    # ── Open detection: CEO title context ("CEO of X") ───────────────────────
    title_matches = _CEO_TITLE_PAT.findall(text)
    for match in title_matches:
        bull += 8
        hits.append(f"[exec: {match.strip()}]")
        has_entity = True

    # ── Curated: known companies ──────────────────────────────────────────────
    for pts, pat, label in _COMPANIES:
        if re.search(pat, low):
            bull += pts
            hits.append(f"[{label}]")
            has_entity = True

    # ── Curated: known CEOs ───────────────────────────────────────────────────
    for pts, pat, label in _CEOS:
        if re.search(pat, low):
            bull += pts
            hits.append(f"[{label}]")
            has_entity = True

    # ── Industries / sectors ──────────────────────────────────────────────────
    for pts, pat, label in _INDUSTRIES:
        if re.search(pat, low):
            bull += pts
            hits.append(f"[{label}]")
            has_entity = True

    # ── Policy levers ─────────────────────────────────────────────────────────
    for pts, pat, label in _POLICIES:
        if re.search(pat, low):
            bull += pts
            hits.append(f"[{label}]")
            # policy alone doesn't set has_entity — still needs score gate

    # ── Legacy sentiment ──────────────────────────────────────────────────────
    for pts, pat in _BULLISH:
        if re.search(pat, low):
            bull += pts
            hits.append(f"[B+{pts}]")

    for pts, pat in _BEARISH:
        if re.search(pat, low):
            bear += pts
            hits.append(f"[Bear+{pts}]")

    score = bull + bear
    if bull > bear:
        direction = "bullish"
    elif bear > bull:
        direction = "bearish"
    else:
        direction = "neutral"

    return score, direction, hits, has_entity


# ── Truth Social API ──────────────────────────────────────────────────────────

def _get_json(url: str) -> list | dict | None:
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _fetch_new(since_id: str | None) -> list[dict]:
    url = f"{_BASE}/accounts/{_ACCOUNT_ID}/statuses?limit=20"
    if since_id:
        url += f"&since_id={since_id}"
    result = _get_json(url)
    return result if isinstance(result, list) else []


# ── Post filtering ────────────────────────────────────────────────────────────

_RT_PATTERN = re.compile(r"^rt:\s*https?://\S+$", re.I)


def _strip_html(raw: str) -> str:
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.I)
    raw = re.sub(r"<p[^>]*>",  "\n", raw, flags=re.I)
    raw = re.sub(r"</p>",      "",   raw, flags=re.I)
    raw = re.sub(r"<[^>]+>",   "",   raw)
    return html.unescape(raw).strip()


def _post_text(post: dict) -> str | None:
    raw  = post.get("content", "")
    text = _strip_html(raw)
    if not text:
        return None
    if _RT_PATTERN.match(text):
        return None
    return text


# ── Telegram ──────────────────────────────────────────────────────────────────

def _send_telegram(body: str) -> bool:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = (
        os.getenv("TELEGRAM_CHAT_ID", "").strip()
        or os.getenv("TELEGRAM_INSIDER_CHAT_ID", "").strip()
    )
    if not token or not chat_id:
        return False

    if len(body) > 4000:
        body = body[:4000] + "\n…"

    payload = json.dumps({
        "chat_id": chat_id, "text": body, "parse_mode": "HTML",
    }).encode()
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return bool(json.loads(r.read()).get("ok"))
    except Exception:
        return False


# ── Output ────────────────────────────────────────────────────────────────────

_RST = "\033[0m"
_RED = "\033[1;31m"
_GRN = "\033[1;32m"
_YLW = "\033[1;33m"
_CYN = "\033[1;36m"
_BLD = "\033[1m"
_DIM = "\033[2m"


def _console(
    post: dict, text: str,
    score: int, direction: str, hits: list[str],
    urgent: bool, will_alert: bool,
) -> None:
    ts = post.get("created_at", "")
    try:
        local = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S")
    except Exception:
        local = ts

    color      = _GRN if direction == "bullish" else (_RED if direction == "bearish" else _YLW)
    alert_flag = f"{color}[TELEGRAM]{_RST} " if will_alert else f"{_DIM}[no alert]{_RST} "
    url        = post.get("url", "")

    print()
    print(f"{color}{'='*68}{_RST}")
    if urgent:
        print(f"{color}{_BLD}  *** URGENT — {direction.upper()} ***  score={score}{_RST}")
    print(f"{_BLD}  @realDonaldTrump  [{local}]  {alert_flag}score={score}{_RST}")
    print(f"{color}{'='*68}{_RST}")
    print()
    for line in text.split("\n"):
        print(f"  {line}")
    print()
    if hits:
        print(f"  Matched: {' '.join(hits)}")
    if url:
        print(f"  {_CYN}{url}{_RST}")
    print(f"{color}{'-'*68}{_RST}")


def _tg_body(
    post: dict, text: str,
    score: int, direction: str, hits: list[str],
    urgent: bool,
) -> str:
    ts = post.get("created_at", "")
    try:
        local = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S %Z")
    except Exception:
        local = ts

    emoji  = {"bullish": "🟢", "bearish": "🔴"}.get(direction, "🟡")
    header = "🚨 URGENT" if urgent else "📢 TRUMP POST"
    lines  = [
        f"<b>{header} — @realDonaldTrump</b>",
        f"{emoji} {direction.upper()}  |  score {score}  |  {local}",
        "",
        text,
    ]
    if hits:
        lines += ["", f"<i>Detected: {', '.join(hits)}</i>"]
    url = post.get("url", "")
    if url:
        lines += ["", f'<a href="{url}">Open post ↗</a>']
    return "\n".join(lines)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(interval: float = 2.0, use_telegram: bool = True) -> None:
    log.info(
        f"Trump monitor starting — @realDonaldTrump  "
        f"interval={interval}s  tg_min_score={_TG_MIN_SCORE}"
    )

    seed    = _fetch_new(since_id=None)
    last_id: str | None = seed[0]["id"] if seed else None
    if last_id:
        preview = (_post_text(seed[0]) or "(video/RT)")[:80]
        log.info(f"Trump monitor seeded at id={last_id}: {preview!r}")
    else:
        log.warning("Trump monitor: could not seed last post ID — will alert on first poll")

    errors = 0

    while True:
        time.sleep(interval)

        posts = _fetch_new(since_id=last_id)
        if posts is None:
            errors += 1
            if errors % 10 == 1:
                log.warning(f"Trump monitor: API unreachable (attempt {errors})")
            continue

        errors = 0

        for post in reversed(posts):
            pid = post["id"]
            if last_id and pid <= last_id:
                continue
            last_id = pid

            text = _post_text(post)
            if text is None:
                continue

            score, direction, hits, has_entity = _score(text)
            urgent      = score >= _URGENT_FLOOR
            # Telegram fires when: a company/CEO/sector was found OR score is high enough
            will_alert  = has_entity or score >= _TG_MIN_SCORE

            log.info(
                f"Trump post — {direction.upper()}  score={score}  "
                f"entity={'YES' if has_entity else 'no'}  "
                f"alert={'YES' if will_alert else 'no'}  "
                f"{text[:80]!r}"
            )
            _console(post, text, score, direction, hits, urgent, will_alert)

            if use_telegram and will_alert:
                body = _tg_body(post, text, score, direction, hits, urgent)
                ok   = _send_telegram(body)
                log.info(f"Trump monitor: {'Telegram sent' if ok else 'Telegram FAILED'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor @realDonaldTrump on Truth Social")
    p.add_argument("--interval",    type=float, default=2.0, help="Poll interval in seconds (default: 2)")
    p.add_argument("--no-telegram", action="store_true",     help="Disable Telegram alerts")
    args = p.parse_args()

    print(f"\n{'='*68}")
    print(f"  ORCA — Trump Truth Social Monitor")
    print(f"  Telegram gate : score >= {_TG_MIN_SCORE}  OR  company/CEO/sector detected")
    print(f"  Telegram      : {'disabled' if args.no_telegram else 'enabled'}")
    print(f"{'='*68}\n")

    if not args.no_telegram:
        if not os.getenv("TELEGRAM_BOT_TOKEN"):
            print("  WARNING: TELEGRAM_BOT_TOKEN not set — Telegram skipped.")
        if not (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_INSIDER_CHAT_ID")):
            print("  WARNING: TELEGRAM_CHAT_ID not set — Telegram skipped.")

    try:
        run(args.interval, not args.no_telegram)
    except KeyboardInterrupt:
        print("\n\n  Stopped.\n")


if __name__ == "__main__":
    main()
