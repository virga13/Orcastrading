import sys
sys.path.insert(0, "C:/Users/admin/Desktop/Orcastrading")
from p5_insider.correlator import get_scored_trades, get_portfolio_summary
from p5_insider.db import was_alerted

print("=== SCORED TRADES (last 90 days, score >= 5) ===")
trades = get_scored_trades(since_days=90)
qualifying = [t for t in trades if t["score"] >= 5]
print(f"Total scored: {len(trades)}, qualifying (>=5): {len(qualifying)}")
print()
for t in qualifying:
    ticker = t.get("ticker", "?")
    score = t["score"]
    name = t.get("insider_name") or t.get("politician", "?")
    title = t.get("insider_title") or t.get("party", "")
    code = t.get("transaction_code") or t.get("transaction_type", "?")
    val = t.get("total_value") or t.get("amount_str", "--")
    date = t.get("filing_date") or t.get("disclosure_date", "?")
    ttype = t.get("type", "")
    val_str = str(val) if isinstance(val, str) else f"{int(val):,}"
    alerted = was_alerted(ttype, t["id"])
    flag = "" if alerted else " [NEW]"
    print(f"[{ttype}] {ticker} | score={score:.0f} | {name} ({title}) | {code} | ${val_str} | {date}{flag}")

print()
print("=== PORTFOLIO SIGNAL SUMMARY (last 30 days) ===")
for s in get_portfolio_summary(since_days=30):
    if s["max_score"] > 0:
        print(f'{s["ticker"]:8} | signal={s["signal"]:10} | buys={s["buys"]} sells={s["sells"]} | max_score={s["max_score"]:.0f}')
