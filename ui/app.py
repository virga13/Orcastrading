"""
ui/app.py — Orcastrading Trading Intelligence Platform v3
7 pages: Dashboard | Market Pulse | Market Analysis | Strategies | Journal | How to Use | Settings
"""
from __future__ import annotations
import sys, os, json, base64, time
from pathlib import Path
from datetime import date, timedelta, datetime, timezone

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv, set_key
import yaml

load_dotenv(ROOT / ".env")

# ── Romania timezone helper ───────────────────────────────────────────────────
try:
    from zoneinfo import ZoneInfo as _ZI
    _RO_TZ = _ZI("Europe/Bucharest")
except Exception:
    _RO_TZ = None

def _ro_time(ts_str: str | None, fmt: str = "%d %b %H:%M") -> str:
    """Convert an ISO UTC timestamp to Romania time (EEST/EET). Returns '' on failure."""
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if _RO_TZ:
            dt = dt.astimezone(_RO_TZ)
        return dt.strftime(fmt)
    except Exception:
        return ts_str[:16]


_STRATEGY_LABELS = {
    "mtf_trend":          "MTF Trend",
    "ema_continuation":   "EMA Continuation",
    "ema_pullback":       "EMA Pullback",
    "momentum_breakout":  "Momentum Breakout",
    "orb":                "ORB",
}

def _fmt_strategy(sid: str) -> str:
    """Return a properly capitalised display name for a strategy ID."""
    return _STRATEGY_LABELS.get(sid, sid.replace("_", " ").title())

st.set_page_config(page_title="Orcastrading", page_icon="🐋", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important}

/* ── Layout ──────────────────────────────────────────────────────────── */
.block-container{padding-top:1.4rem!important;padding-bottom:3rem!important;max-width:1400px}
.main .block-container{background:#060B14}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"]{background:#040810!important;border-right:1px solid #0C1524!important}
section[data-testid="stSidebar"] .block-container{padding:1.4rem .9rem}

/* Nav items (radio styled as links) */
section[data-testid="stSidebar"] [data-testid="stRadio"] label{
  display:flex;align-items:center;padding:.46rem .75rem;border-radius:8px;
  font-size:.83rem;font-weight:500;color:#64748B;cursor:pointer;
  transition:background .13s,color .13s;margin:1px 0;
  border-left:2px solid transparent!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{
  background:#0B1525!important;color:#CBD5E1!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){
  background:#0B1525!important;color:#E2E8F0!important;font-weight:600!important;
  border-left:2px solid #3B82F6!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] label input{display:none!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]{gap:0!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] > label{display:none!important}

/* Sidebar misc buttons */
section[data-testid="stSidebar"] .stButton button{
  background:transparent!important;border:1px solid #1A2840!important;
  color:#64748B!important;font-size:.78rem!important;border-radius:8px!important;
  transition:all .13s!important}
section[data-testid="stSidebar"] .stButton button:hover{
  background:#0B1525!important;color:#94A3B8!important;border-color:#1E3A5F!important}

/* ── Metric cards ─────────────────────────────────────────────────────── */
[data-testid="metric-container"]{
  background:#0A1220;border:1px solid #0C1524;border-radius:12px;
  padding:1rem 1.2rem;transition:border-color .2s,background .2s}
[data-testid="metric-container"]:hover{border-color:#1E3A5F;background:#0D1828}
[data-testid="stMetricValue"]{font-size:1.55rem!important;font-weight:700!important;
  letter-spacing:-.025em;color:#F1F5F9!important}
[data-testid="stMetricLabel"]{font-size:.66rem!important;text-transform:uppercase;
  letter-spacing:.09em;color:#64748B!important;font-weight:600}
[data-testid="stMetricDelta"]{font-size:.71rem!important}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton button{border-radius:8px!important;font-weight:500!important;
  font-size:.83rem!important;transition:all .13s!important;letter-spacing:.01em!important}
.stButton button[kind="primary"]{
  background:#2563EB!important;color:#fff!important;border:none!important;font-weight:600!important}
.stButton button[kind="primary"]:hover{
  background:#1D4ED8!important;box-shadow:0 4px 16px #2563EB28!important;
  transform:translateY(-1px)!important}
.stButton button[kind="secondary"]{
  background:transparent!important;border:1px solid #1A2840!important;color:#64748B!important}
.stButton button[kind="secondary"]:hover{border-color:#1E3A5F!important;color:#CBD5E1!important}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{background:transparent!important;
  border-bottom:1px solid #0C1524;gap:0;padding:0 0 0 0}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#64748B!important;
  border:none!important;padding:.5rem .95rem!important;font-size:.81rem!important;
  font-weight:500!important;border-bottom:2px solid transparent!important;
  margin-bottom:-1px!important;transition:color .13s!important}
.stTabs [data-baseweb="tab"]:hover{color:#94A3B8!important;background:transparent!important}
.stTabs [aria-selected="true"]{color:#E2E8F0!important;font-weight:600!important;
  border-bottom:2px solid #3B82F6!important}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.2rem!important}

/* ── Expander ────────────────────────────────────────────────────────── */
details{border:1px solid #0C1524!important;border-radius:12px!important;
  background:#0A1220!important;transition:border-color .2s!important}
details[open]{border-color:#1A2840!important}
details summary{font-weight:600!important;padding:.65rem 1rem!important;
  color:#CBD5E1!important;font-size:.87rem!important}
details summary:hover{color:#F1F5F9!important}

/* ── Inputs ──────────────────────────────────────────────────────────── */
.stTextInput input,.stNumberInput input,.stTextArea textarea{
  background:#0A1220!important;border:1px solid #0C1524!important;
  color:#E2E8F0!important;border-radius:8px!important;font-size:.84rem!important}
.stTextInput input:focus,.stNumberInput input:focus,.stTextArea textarea:focus{
  border-color:#2563EB!important;box-shadow:0 0 0 3px #2563EB18!important}
.stSelectbox [data-baseweb="select"]{background:#0A1220!important;
  border:1px solid #0C1524!important;border-radius:8px!important}
.stForm{border:1px solid #0C1524!important;border-radius:12px!important;
  padding:1rem!important;background:#080E1C!important}

/* ── Typography ──────────────────────────────────────────────────────── */
h1{font-size:1.55rem!important;font-weight:700!important;color:#F1F5F9!important;
  letter-spacing:-.035em!important;margin-bottom:.1rem!important}
h2{font-size:1.15rem!important;font-weight:600!important;color:#E2E8F0!important;letter-spacing:-.02em!important}
h3{font-size:.97rem!important;font-weight:600!important;color:#CBD5E1!important}
p{color:#94A3B8;line-height:1.65}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr{border:none!important;border-top:1px solid #0C1524!important;
  margin:1.3rem 0!important;opacity:1!important}

/* ── Dataframe ───────────────────────────────────────────────────────── */
.stDataFrame{border-radius:12px!important;overflow:hidden!important;
  border:1px solid #0C1524!important}
.stDataFrame thead tr th{background:#0A1220!important;color:#64748B!important;
  font-size:.7rem!important;text-transform:uppercase;letter-spacing:.07em!important;
  font-weight:600!important}

/* ── Code blocks ─────────────────────────────────────────────────────── */
.stCode pre,.stCodeBlock pre{background:#0A1220!important;border:1px solid #0C1524!important;
  border-radius:10px!important;font-size:.8rem!important}

/* ── Alerts ──────────────────────────────────────────────────────────── */
.stAlert{border-radius:10px!important}

/* ── Status badges ───────────────────────────────────────────────────── */
.orca-badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.67rem;
  font-weight:700;text-transform:uppercase;letter-spacing:.06em}
.b-win{background:#041E12;color:#10B981;border:1px solid #054830}
.b-partial_win{background:#091535;color:#60A5FA;border:1px solid #102060}
.b-loss{background:#1E0808;color:#F87171;border:1px solid #4a1010}
.b-breakeven{background:#1C1200;color:#FBBF24;border:1px solid #3a2800}
.b-pending{background:#0A1220;color:#475569;border:1px solid #1A2840}
.b-filled{background:#140828;color:#C084FC;border:1px solid #2D0E5A}
.b-expired{background:#080E18;color:#27374D;border:1px solid #0C1524}
.b-bullish{background:#041E12;color:#10B981;border:1px solid #054830}
.b-bearish{background:#1E0808;color:#F87171;border:1px solid #4a1010}
.b-neutral{background:#0A1220;color:#475569;border:1px solid #1A2840}
.b-active{background:#041E12;color:#10B981;border:1px solid #054830}
.b-warning{background:#1C1200;color:#FBBF24;border:1px solid #3a2800}
.b-tripped{background:#1E0808;color:#F87171;border:1px solid #4a1010}

/* ── Misc components ─────────────────────────────────────────────────── */
.signal-row{display:flex;align-items:center;gap:.6rem;padding:.5rem .85rem;
  background:#0A1220;border:1px solid #0C1524;border-radius:10px;
  margin-bottom:.3rem;font-size:.84rem;transition:border-color .15s}
.signal-row:hover{border-color:#1E3A5F}
.kpi-sub{font-size:.72rem;color:#475569;margin-top:-.4rem;margin-bottom:.9rem;letter-spacing:.02em}
.orca-logo{font-size:1.25rem;font-weight:800;color:#F1F5F9;letter-spacing:-.04em}
.orca-logo span{background:linear-gradient(135deg,#3B82F6,#8B5CF6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#0C1524;border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:#1A2840}

/* ── Restore Material Symbols font (broken by the * override above) ── */
/* Covers: sidebar collapse, metric delta arrows, expander chevrons,     */
/*         file uploader drop-zone icon                                  */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stMetricDelta"] *,
[data-testid="stMetricDeltaIcon"],
details summary span,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid*="stFileUploader"] span,
.material-symbols-rounded,
.material-symbols-outlined,
.material-symbols-sharp,
.material-icons {
  font-family:"Material Symbols Rounded","Material Symbols Outlined",
               "Material Icons"!important;
}
</style>""", unsafe_allow_html=True)

_PL = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94A3B8", size=12), margin=dict(l=0,r=0,t=36,b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#1A2840",font_color="#F1F5F9"),
    xaxis=dict(gridcolor="#1A2840",zeroline=False,color="#5A6E85"),
    yaxis=dict(gridcolor="#1A2840",zeroline=False,color="#5A6E85"),
)

# ── Cached loaders ────────────────────────────────────────────────────────────
def _trades():
    from p4_live.journal_supabase import get_all_trades
    return get_all_trades(st.session_state["supabase_client"])

def _manual_entries():
    from p4_live.journal_supabase import get_manual_entries
    return get_manual_entries(st.session_state["supabase_client"])

def _psychology_all():
    from p4_live.journal_supabase import get_all_psychology
    return get_all_psychology(st.session_state["supabase_client"])

@st.cache_data(ttl=120)
def _assets():
    from core.config import get_all_assets; return get_all_assets()

@st.cache_data(ttl=120)
def _strategies():
    from core.config import get_all_strategies; return get_all_strategies()

@st.cache_data(ttl=120)
def _scanner_assets():
    from p4_live.scanner import ASSETS; return ASSETS

@st.cache_data(ttl=300)
def _load_bt_stats(ticker_safe: str, interval: str, strategy_id: str | None = None):
    """Load stats from the most recent backtest DB for ticker + interval.
    Returns (stats_dict, db_filename) or (None, None) if not found.
    If strategy_id is given, tries to match strategy_name in metadata.
    """
    import sqlite3, json as _json
    bt_dir = ROOT / "p3_backtester" / "results"
    dbs = sorted(bt_dir.glob(f"backtest_{ticker_safe}_{interval}_*.db"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    for db in dbs:
        try:
            c = sqlite3.connect(str(db))
            row = c.execute("SELECT value FROM run_metadata WHERE key='stats'").fetchone()
            c.close()
            if not row:
                continue
            stats = _json.loads(row[0])
            if strategy_id:
                db_strat = stats.get("strategy_name","").lower().replace(" ","_")
                req_strat = strategy_id.lower().replace(" ","_")
                if db_strat != req_strat:
                    continue
            return stats, db.name
        except Exception:
            pass
    return None, None


def _render_chart_upload(t: dict, upload_key: str) -> None:
    """
    Show chart screenshot for a live trade, with an upload widget to add/replace it.
    Screenshot is saved to outputs/screenshots/ and path persisted in the journal DB.
    """
    from p4_live.journal_supabase import save_trade_chart_screenshot
    ss_dir = ROOT / "outputs" / "screenshots"
    ss_dir.mkdir(parents=True, exist_ok=True)

    existing_path = t.get("chart_screenshot_path")
    has_chart = bool(existing_path and Path(existing_path).exists())

    if has_chart:
        st.image(existing_path, use_container_width=True)

    uploaded = st.file_uploader(
        "Replace chart" if has_chart else "Chart screenshot (TradingView, etc.)",
        type=["png", "jpg", "jpeg", "webp"],
        key=upload_key,
        label_visibility="visible",
    )
    if uploaded:
        import datetime as _dt
        ts_s = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = ss_dir / f"{ts_s}_trade_{t['signal_date']}_{t['ticker']}_{uploaded.name}"
        save_path.write_bytes(uploaded.getvalue())
        save_trade_chart_screenshot(
            st.session_state["supabase_client"],
            t["signal_date"], t["ticker"],
            t.get("strategy", "mtf_trend"), t.get("timeframe", "1d"),
            str(save_path),
        )
        st.rerun()


@st.cache_data(ttl=60)
def _fetch_live_price(ticker_yf: str) -> float | None:
    """Fetch latest price for a ticker (60-second cache)."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker_yf).history(period="2d", interval="5m")
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None

# Live ticker → yfinance ticker (for price fetch)
_LIVE_TO_YF = {
    "XAU/USD": "GC=F", "XAG/USD": "SI=F",
    "BTC/USD": "BTC-USD", "BTC/USDT": "BTC-USD",
}

def _clear():
    pass  # Supabase-backed loaders are not cached; data refreshes on next rerun

def _clear_config():
    _assets.clear(); _strategies.clear(); _scanner_assets.clear()
    try:
        from core.config import reload; reload()
    except Exception: pass

def _pending_psych():
    try:
        from p4_live.journal_supabase import get_all_trades as _gat, get_all_psychology as _gap
        client = st.session_state.get("supabase_client")
        if not client: return 0
        closed=[t for t in _gat(client) if t["status"] not in ("pending","filled","expired")]
        already={p["trade_ref_id"] for p in _gap(client)}
        return sum(1 for t in closed
                   if f"{t['signal_date']}|{t['ticker']}|{t.get('strategy','')}" not in already)
    except Exception: return 0

def _save_analysis_cache(ticker, interval, bias, setups, aname=""):
    try:
        d=ROOT/"outputs"/"analysis_cache"; d.mkdir(parents=True,exist_ok=True)
        safe=ticker.replace("^","").replace("=","").replace("-","").replace("/","")
        data={"ticker":ticker,"interval":interval,"aname":aname,
              "timestamp":datetime.now().isoformat(),
              "bias":bias.model_dump(),"setups":setups.model_dump()}
        # Per-ticker cache (latest only, for auto-reload)
        (d/f"{safe}_{interval}.json").write_text(
            json.dumps(data,default=str),encoding="utf-8")
        # Permanent backlog — append to analysis_log.json
        log_p=ROOT/"outputs"/"analysis_log.json"
        try:
            log=json.loads(log_p.read_text(encoding="utf-8")) if log_p.exists() else []
        except Exception: log=[]
        log.insert(0,data)
        log_p.write_text(json.dumps(log[:200],default=str),encoding="utf-8")
    except Exception: pass

def _load_analysis_log():
    """Load the permanent analysis backlog (all analyses ever run)."""
    log_p=ROOT/"outputs"/"analysis_log.json"
    try:
        if not log_p.exists(): return []
        return json.loads(log_p.read_text(encoding="utf-8"))
    except Exception: return []

def _load_analysis_cache(ticker, interval):
    try:
        safe=ticker.replace("^","").replace("=","").replace("-","").replace("/","")
        p=ROOT/"outputs"/"analysis_cache"/f"{safe}_{interval}.json"
        if not p.exists(): return None,None,None,None
        data=json.loads(p.read_text(encoding="utf-8"))
        from p1_analysis_engine.schema import BiasOutput,SetupsOutput
        return (BiasOutput.model_validate(data["bias"]),
                SetupsOutput.model_validate(data["setups"]),
                data.get("aname",ticker), data.get("timestamp",""))
    except Exception: return None,None,None,None

# ── Account helpers ───────────────────────────────────────────────────────────
def _acct():
    try:
        from p4_live.journal_supabase import get_user_settings as _gus
        client = st.session_state.get("supabase_client")
        us = _gus(client) if client else {}
    except Exception:
        us = {}
    return {
        "balance":        float(us.get("account_balance") or os.getenv("ACCOUNT_BALANCE","7500")),
        "leverage":       int(os.getenv("ACCOUNT_LEVERAGE","100")),
        "risk_pct":       float(us.get("risk_per_trade_pct") or os.getenv("RISK_PER_TRADE_PCT","0.5")),
        "lot_oz":         float(os.getenv("LOT_OZ","100")),
        "min_lots":       float(os.getenv("MIN_LOTS","0.01")),
        "min_lot_margin": float(os.getenv("MIN_LOT_MARGIN","190")),
    }

def _save_acct(bal, lev, risk, lot_oz=100, min_lots=0.01, min_lot_margin=190.0):
    e = str(ROOT/".env")
    set_key(e,"ACCOUNT_LEVERAGE",str(lev))
    set_key(e,"LOT_OZ",str(lot_oz))
    set_key(e,"MIN_LOTS",str(min_lots))
    set_key(e,"MIN_LOT_MARGIN",str(min_lot_margin))
    os.environ.update({
        "ACCOUNT_LEVERAGE": str(lev),
        "LOT_OZ": str(lot_oz), "MIN_LOTS": str(min_lots),
        "MIN_LOT_MARGIN": str(min_lot_margin),
    })
    try:
        from p4_live.journal_supabase import save_user_settings as _sus
        client = st.session_state.get("supabase_client")
        if client:
            _sus(client, {"account_balance": bal, "risk_per_trade_pct": risk})
    except Exception:
        set_key(e,"ACCOUNT_BALANCE",str(bal))
        set_key(e,"RISK_PER_TRADE_PCT",str(risk))

# ── YAML helpers ──────────────────────────────────────────────────────────────
def _read_yaml(p):
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f) or {}

def _write_yaml(p, d):
    with open(p,"w",encoding="utf-8") as f:
        yaml.dump(d,f,allow_unicode=True,default_flow_style=False,sort_keys=False)

# ── Format helpers ────────────────────────────────────────────────────────────
_SC = {"win":"#10B981","partial_win":"#60A5FA","loss":"#F87171","breakeven":"#FBBF24",
       "filled":"#C084FC","pending":"#94A3B8","expired":"#4B5563"}

def _label(t): return t.get("label") or t.get("ticker","")

def _trades_df(tl):
    rows = []
    for t in tl:
        sig_ts  = _ro_time(t.get("signal_bar_ts") or t.get("recorded_at"), "%d %b %H:%M")
        fill_ts = _ro_time(t.get("fill_bar_ts"), "%d %b %H:%M")  # only show time if bar-precise ts exists
        rows.append({
            "Signal (RO)":  sig_ts or t["signal_date"],
            "TF":           t.get("timeframe", ""),
            "Asset":        _label(t),
            "Strategy":     _fmt_strategy(t.get("strategy","")),
            "Dir":          t["direction"].upper(),
            "Entry zone":   f"{t['entry_low']:,.1f}–{t['entry_high']:,.1f}",
            "SL":           f"{t['stop_loss']:,.1f}",
            "Fill (RO)":    fill_ts or "—",
            "Fill $":       f"{t['fill_price']:,.1f}" if t.get("fill_price") else "—",
            "Exit $":       f"{t['exit_price']:,.1f}" if t.get("exit_price") else "—",
            "Status":       t["status"].upper().replace("_"," "),
            "P&L":          f"{t['pnl_r']:+.2f}R" if t.get("pnl_r") is not None else "—",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def _calc_close_pnl_r(trade: dict, outcome: str, exit_price: float) -> float:
    """Compute pnl_r for a close operation, mirroring journal.py record_close logic."""
    fill = trade.get("fill_price")
    if fill is None:
        return 0.0
    direction = trade.get("direction", "long")
    risk = abs(fill - (trade.get("stop_loss") or fill))
    if risk < 1e-8:
        return 0.0
    if outcome == "loss":
        return -1.0
    if outcome == "breakeven":
        return 0.0
    alloc1 = (trade.get("tp1_alloc") or 70) / 100
    alloc2 = (trade.get("tp2_alloc") or 30) / 100
    tp2 = trade.get("tp2")
    if outcome == "partial_win" and tp2 is not None:
        tp1 = trade.get("tp1", exit_price)
        if direction == "long":
            return round((tp1 - fill) * alloc1 / risk + (exit_price - fill) * alloc2 / risk, 3)
        return round((fill - tp1) * alloc1 / risk + (fill - exit_price) * alloc2 / risk, 3)
    sign = 1 if direction == "long" else -1
    return round(sign * (exit_price - fill) / risk, 3)


def _calc_manual_r(e):
    """Calculate R for a manual entry. Returns (r_value, is_estimated).

    Priority for SL used in calculation:
      1. original_sl (stop at entry, before any trailing) — most accurate
      2. stop_loss — used only if on the PROTECTIVE side for the stored direction
         (i.e. not a trailed stop that's moved into profit)

    Direction uses the stored field; SL-position auto-detection only kicks in
    when the stored direction is 'unknown' or missing.
    """
    pnl=e.get("pnl_r")
    if pnl is not None:
        return pnl, False
    entry=e.get("entry_price"); exit_p=e.get("exit_price")
    # Try AI analysis for exit_price if not stored
    if exit_p is None and e.get("ai_analysis"):
        try:
            ai_d=json.loads(e["ai_analysis"])
            exit_p=(ai_d.get("merged") or ai_d.get("broker") or ai_d).get("exit_price")
        except Exception: pass
    if not (entry and exit_p is not None): return None, False

    direction=(e.get("direction") or "unknown").lower()

    # Choose the best SL: prefer original_sl, else stop_loss if on protective side
    orig_sl=e.get("original_sl"); stored_sl=e.get("stop_loss")
    if orig_sl and abs(entry-orig_sl)>1e-10:
        sl=orig_sl   # original protective stop — always reliable
    elif stored_sl and abs(entry-stored_sl)>1e-10:
        # Only use stored_sl if it's on the protective (non-profit) side
        sl_protective=((stored_sl < entry and direction in ("long","unknown")) or
                       (stored_sl > entry and direction in ("short","unknown")))
        sl = stored_sl if sl_protective else None
    else:
        sl=None

    if sl and abs(entry-sl)>1e-10:
        risk=abs(entry-sl)
        # If direction unknown, infer from SL position
        if direction=="unknown":
            direction="short" if sl > entry else "long"
        r=(entry-exit_p)/risk if direction=="short" else (exit_p-entry)/risk
        return round(r,3), True
    return None, False

def _compute_streak(closed):
    if not closed: return 0, "none"
    s = sorted(closed, key=lambda x: x["signal_date"])
    # Ignore breakeven trades (pnl_r == 0) for streak — they don't break or extend
    decisive = [t for t in s if (t.get("pnl_r") or 0) != 0]
    if not decisive: return 0, "none"
    kind = "win" if (decisive[-1].get("pnl_r") or 0) > 0 else "loss"
    streak = 1
    for t in reversed(decisive[:-1]):
        cur_win = (t.get("pnl_r") or 0) > 0
        if (cur_win and kind == "win") or (not cur_win and kind == "loss"):
            streak += 1
        else:
            break
    return streak, kind

def _rolling_wr(closed, window=20):
    if len(closed) < 3: return [], []
    s = sorted(closed, key=lambda x: x["signal_date"])
    dates, wrs = [], []
    for i in range(len(s)):
        w = s[max(0, i - window + 1):i + 1]
        decisive = [t for t in w if (t.get("pnl_r") or 0) != 0]
        if decisive:
            wrs.append(sum(1 for t in decisive if (t.get("pnl_r") or 0) > 0) / len(decisive))
        else:
            wrs.append(0.0)
        dates.append(s[i]["signal_date"])
    return dates, wrs

# ── Screenshot AI ─────────────────────────────────────────────────────────────
def _analyze_screenshot(img_bytes, media_type, notes, screenshot_type="broker"):
    """
    screenshot_type: "broker" = platform/order details, "chart" = price chart.
    Returns dict with extracted trade data + AI insights.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    b64 = base64.standard_b64encode(img_bytes).decode()

    if screenshot_type == "broker":
        prompt = (
            "You are an expert trading journal assistant analyzing a BROKER/PLATFORM screenshot "
            "(order ticket, trade history, position details, or P&L statement).\n"
            f"User notes: {notes or 'None'}\n\n"
            "STEP 1 — Identify the target trade:\n"
            "  - If the screenshot shows multiple rows/trades, look for:\n"
            "    a) A highlighted or selected row (different background color)\n"
            "    b) Any row underlined, circled, or annotated by the user\n"
            "    c) If no annotation, use the most recently closed trade\n"
            "  - Once you identify the target row, read ALL values (symbol, entry, SL, TP, "
            "exit, P&L, lot size) FROM THAT ROW ONLY. Do not mix values from different rows.\n\n"
            "STEP 2 — Sanity-check your extraction:\n"
            "  - SL and TP must be in the same price range as the entry price.\n"
            "    e.g. if entry is ~71, SL must also be near 71 — not 4610. If a number looks "
            "wrong for the asset, set it to null rather than using it.\n"
            "  - For a LONG: SL < entry < TP. For a SHORT: TP < entry < SL.\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"asset":"symbol or Unknown","timeframe":"e.g. 1h","direction":"long/short/unknown",'
            '"entry_price":null,"stop_loss":null,"take_profit":null,"exit_price":null,'
            '"pnl_dollars":null,"lot_size":null,'
            '"entry_reason":"what the broker data tells us about the trade setup",'
            '"trade_quality":5,'
            '"what_worked":"strengths visible from the trade data",'
            '"what_to_improve":"weaknesses visible from the trade data",'
            '"lessons":"one key lesson","journal_entry":"2-3 paragraph first-person journal entry"}'
        )
    else:  # chart
        prompt = (
            "You are an expert trading journal assistant analyzing a PRICE CHART screenshot.\n"
            f"User notes: {notes or 'None'}\n\n"
            "IMPORTANT: If there are any annotations visible on the chart:\n"
            "  1. Look for underlines, circles, arrows, or freehand drawings\n"
            "  2. Look for a specific candle or zone highlighted or pointed to\n"
            "  3. These annotations mark the trade entry or key area of interest\n\n"
            "Assess the trade quality based on:\n"
            "  - Was the entry at a good location (key level, structure, EMA)?\n"
            "  - Was the trend direction correct on this timeframe?\n"
            "  - Were there any warning signs visible before entry?\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"asset":"symbol or Unknown","timeframe":"e.g. 1h","direction":"long/short/unknown",'
            '"entry_price":null,"stop_loss":null,"take_profit":null,"exit_price":null,'
            '"market_structure":"describe price structure visible on the chart",'
            '"entry_reason":"what setup triggered the entry based on the chart",'
            '"trade_quality":5,'
            '"what_worked":"what the chart shows worked well",'
            '"what_to_improve":"what the chart shows could be improved",'
            '"lessons":"one key chart-based lesson","journal_entry":"2-3 paragraph first-person journal entry"}'
        )
    try:
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=2000,
            messages=[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":media_type,"data":b64}},
                {"type":"text","text":prompt}]}])
        text = resp.content[0].text.strip()
        if text.startswith("```"): text="\n".join(text.split("\n")[1:]).rstrip("`")
        return json.loads(text)
    except json.JSONDecodeError:
        return {"journal_entry":resp.content[0].text,"trade_quality":None}
    except Exception as e:
        return {"error":str(e),"journal_entry":f"Analysis failed: {e}"}

# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
def _run_scan_now(
    asset_ids: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> tuple[int, int, list[str]]:
    """
    Scan every (asset, strategy, timeframe) triple in the watchlist.
    Optionally filtered by asset_ids (core config IDs) and/or timeframes.
    Returns (fired_count, recorded_count, errors).
    """
    from p4_live.scanner import scan_strategy
    from p4_live.journal_supabase import (
        record_signal as _record_signal_sb,
        get_user_watchlist as _guw_scan,
    )
    # Prefer per-user Supabase watchlist; fall back to shared YAML watchlist
    try:
        watchlist = _guw_scan(st.session_state["supabase_client"])
        if not watchlist:
            from core.config import get_watchlist
            watchlist = get_watchlist()
    except Exception:
        from core.config import get_watchlist
        watchlist = get_watchlist()
    if asset_ids:
        watchlist = [(a, s, tf) for a, s, tf in watchlist if a in asset_ids]
    if timeframes:
        watchlist = [(a, s, tf) for a, s, tf in watchlist if tf in timeframes]

    _sb_client = st.session_state["supabase_client"]
    fired = 0; recorded = 0; errors: list[str] = []
    for asset_id, strategy_id, tf in watchlist:
        try:
            result = scan_strategy(asset_id, strategy_id, timeframe=tf)
        except Exception as e:
            errors.append(f"{asset_id}/{strategy_id}/{tf}: {e}")
            continue
        if not result.get("fired"):
            continue
        fired += 1
        try:
            if _record_signal_sb(_sb_client, result):
                recorded += 1
        except Exception as e:
            errors.append(str(e))
    return fired, recorded, errors


def _run_simulate_pending() -> dict:
    """
    Run the two-phase simulate_pending and return a summary dict:
      fills    — pending signals that entered the zone (now 'filled')
      active   — filled trades with no SL/TP hit yet (still running)
      closed   — trades that hit SL or TP (final outcome written)
      wins     — subset of closed that are win / partial_win
      expired  — signals that timed out without filling
      errors   — list of error strings
    """
    from p4_live.simulate import simulate_pending  # noqa
    results = simulate_pending(dry_run=False)
    fills   = sum(1 for r in results if r.get("status") == "filled")
    active  = sum(1 for r in results if r.get("status") == "active")
    closed  = sum(1 for r in results if r.get("status") not in
                  ("filled", "active", "in_progress", "no_data", "error", "expired"))
    wins    = sum(1 for r in results if r.get("status") in ("win", "partial_win"))
    expired = sum(1 for r in results if r.get("status") == "expired")
    errors  = [f"{r.get('ticker','')} {r.get('signal_date','')}: {r.get('reason','')}"
               for r in results if r.get("status") == "error"]
    return dict(fills=fills, active=active, closed=closed,
                wins=wins, expired=expired, errors=errors)


_GOLD_TICKERS = {"GC=F", "XAU/USD"}


def page_dashboard():
    # ── Auto-refresh + auto-simulate state ───────────────────────────────────
    now_ts   = time.time()
    auto_on  = st.session_state.get("dash_auto_refresh", False)
    interval = st.session_state.get("dash_refresh_secs", 60)
    last_ref = st.session_state.get("dash_last_refresh_ts", 0)

    # Auto-simulate on page load if >5 min since last simulation
    _last_sim = st.session_state.get("dash_last_sim_ts", 0)
    if (now_ts - _last_sim) > 300:
        try:
            _run_simulate_pending()
            st.session_state["dash_last_sim_ts"] = now_ts
            st.session_state.pop("dash_sim_error", None)
            _clear()
        except Exception as _e:
            st.session_state["dash_sim_error"] = str(_e)

    if auto_on and (now_ts - last_ref) >= interval:
        # Simulate first, then clear caches so UI shows fresh data
        try:
            _run_simulate_pending()
            st.session_state["dash_last_sim_ts"] = now_ts
            st.session_state.pop("dash_sim_error", None)
        except Exception as _e:
            st.session_state["dash_sim_error"] = str(_e)
        _clear()
        st.session_state["dash_last_refresh_ts"] = now_ts

    # ── Load all trades ONCE — reused everywhere in this function ─────────────
    _all_trades = _trades()
    _today_iso  = date.today().isoformat()

    # Derive status-bar data from the same list (no extra _trades() calls)
    _open_now     = [t for t in _all_trades if t["status"] in ("pending", "filled")]
    _closed_today = [t for t in _all_trades
                     if t.get("exit_date", "") == _today_iso and t.get("pnl_r") is not None]
    _today_r = sum(t["pnl_r"] for t in _closed_today)

    # Asset label set for filter dropdown
    _all_labels = sorted({
        t.get("label", "") or t.get("ticker", "")
        for t in _all_trades
        if t.get("label") or t.get("ticker")
    })

    st.title("Dashboard")

    # ── System status bar ─────────────────────────────────────────────────────
    _hb_f  = ROOT / "logs" / "scheduler_heartbeat"
    _lg_f  = ROOT / "logs" / "scheduler.log"
    _chk   = _hb_f if _hb_f.exists() else (_lg_f if _lg_f.exists() else None)
    _sched_on  = _chk and (time.time() - _chk.stat().st_mtime) < 120
    _sched_age = int((time.time() - _chk.stat().st_mtime) / 60) if _chk else None

    from p4_live import mt5_broker as _mt5b
    _mt5_info = _mt5b.get_account_info() if _mt5b.is_enabled() else None

    _sb_parts = []

    # MT5 status
    if _mt5b.is_enabled():
        if _mt5_info:
            _sb_parts.append(
                f'<span style="color:#10B981;font-weight:700">● MT5</span>'
                f'<span style="color:#CBD5E1"> #{_mt5_info["login"]} &nbsp;'
                f'${_mt5_info["balance"]:,.0f}</span>')
        else:
            _sb_parts.append('<span style="color:#EF4444;font-weight:700">● MT5 disconnected</span>')
    else:
        _sb_parts.append('<span style="color:#475569">○ MT5 disabled</span>')

    # Separator
    _sb_parts.append('<span style="color:#1E3A5F;margin:0 .4rem">│</span>')

    # Scheduler status
    if _sched_on:
        _sb_parts.append('<span style="color:#10B981;font-weight:700">● Scheduler</span>'
                         '<span style="color:#94A3B8"> running</span>')
    elif _sched_age is not None:
        _sb_parts.append(f'<span style="color:#F59E0B;font-weight:600">● Scheduler idle {_sched_age}m</span>')
    else:
        _sb_parts.append('<span style="color:#EF4444">● Scheduler OFF</span>')

    _sb_parts.append('<span style="color:#1E3A5F;margin:0 .4rem">│</span>')

    # Open trades
    _o_color = "#60A5FA" if _open_now else "#64748B"
    _sb_parts.append(f'<span style="color:{_o_color}">{len(_open_now)} open / pending</span>')

    _sb_parts.append('<span style="color:#1E3A5F;margin:0 .4rem">│</span>')

    # Today's P&L
    _r_color = "#10B981" if _today_r > 0 else ("#EF4444" if _today_r < 0 else "#64748B")
    _today_label = f'{_today_r:+.1f}R today' if _closed_today else 'no closes today'
    _sb_parts.append(f'<span style="color:{_r_color}">{_today_label}</span>')

    st.markdown(
        '<div style="background:#080E1C;border:1px solid #1A2840;border-radius:8px;'
        'padding:.45rem 1rem;margin-bottom:.8rem;font-size:.8rem;display:flex;'
        'align-items:center;gap:.5rem;flex-wrap:wrap">'
        + " ".join(_sb_parts) +
        '</div>', unsafe_allow_html=True)

    # ── Scan controls — row 1: buttons + filters ─────────────────────────────
    try:
        from core.config import get_all_assets as _gaa
        try:
            from p4_live.journal_supabase import get_user_watchlist as _gwl_sb
            _wl = _gwl_sb(st.session_state["supabase_client"])
            if not _wl:
                from core.config import get_watchlist as _gwl_yaml
                _wl = _gwl_yaml()
        except Exception:
            from core.config import get_watchlist as _gwl_yaml
            _wl = _gwl_yaml()
        _wl_asset_ids   = sorted({a for a, s, tf in _wl})
        _TF_ORDER = ["1m","5m","15m","30m","1h","4h","8h","1d","1wk"]
        _wl_tfs = sorted({tf for a, s, tf in _wl}, key=lambda x: _TF_ORDER.index(x) if x in _TF_ORDER else 99)
        _asset_id_label = {a["id"]: a.get("label", a["id"]) for a in _gaa()}
        _asset_sel_opts = ["All"] + _wl_asset_ids
        _asset_sel_fmt  = lambda i: "All assets" if i=="All" else _asset_id_label.get(i, i)
    except Exception:
        _wl_asset_ids=[]; _wl_tfs=[]; _asset_sel_opts=["All"]; _asset_sel_fmt=lambda i:i

    sc1, sc1b, sc2, sc3, sc4, sc5 = st.columns([2, 2, 3, 2, 1, 1])
    with sc1:
        scan_clicked = st.button("Run Scan Now", type="primary", use_container_width=True)
    with sc1b:
        sim_clicked = st.button("Simulate Pending", use_container_width=True,
                                help="Resolve pending→filled→closed for all open trades using real price data")
    with sc2:
        sel_assets = st.multiselect(
            "Assets", _wl_asset_ids,
            default=st.session_state.get("scan_asset_filter", []),
            format_func=lambda i: _asset_id_label.get(i, i),
            placeholder="All assets",
            key="scan_asset_ms",
            label_visibility="collapsed",
        )
        st.session_state["scan_asset_filter"] = sel_assets
    with sc3:
        sel_tfs = st.multiselect(
            "Timeframes", _wl_tfs,
            default=st.session_state.get("scan_tf_filter", []),
            placeholder="All timeframes",
            key="scan_tf_ms",
            label_visibility="collapsed",
        )
        st.session_state["scan_tf_filter"] = sel_tfs
    with sc4:
        new_auto = st.toggle("Auto", value=auto_on, key="dash_auto_toggle",
                             help="Auto-refresh: simulate pending trades and reload every N seconds")
        if new_auto != auto_on:
            st.session_state["dash_auto_refresh"] = new_auto
            st.session_state["dash_last_refresh_ts"] = now_ts
            st.rerun()
    with sc5:
        interval_label = st.selectbox(
            "Interval", ["30s", "1 min", "5 min", "15 min"],
            index=["30s","1 min","5 min","15 min"].index(
                {30:"30s",60:"1 min",300:"5 min",900:"15 min"}.get(interval,"1 min")
            ),
            label_visibility="collapsed", key="dash_interval_sel",
        )
        new_secs = {"30s":30,"1 min":60,"5 min":300,"15 min":900}[interval_label]
        if new_secs != interval:
            st.session_state["dash_refresh_secs"] = new_secs

    # Auto-refresh status caption — single compact line
    if auto_on:
        elapsed   = now_ts - last_ref
        remaining = max(0, interval - elapsed)
        _last_s   = st.session_state.get("dash_last_sim_ts", 0)
        st.caption(
            f"Auto ON — next in {int(remaining)}s  ·  "
            f"Last refresh: {datetime.fromtimestamp(last_ref).strftime('%H:%M:%S') if last_ref else '—'}  ·  "
            f"Last simulate: {datetime.fromtimestamp(_last_s).strftime('%H:%M:%S') if _last_s else '—'}"
        )
    else:
        last_scan = st.session_state.get("dash_last_scan_ts")
        _last_s   = st.session_state.get("dash_last_sim_ts", 0)
        _parts = []
        if last_scan: _parts.append(f"Last scan: {datetime.fromtimestamp(last_scan).strftime('%H:%M:%S')}")
        if _last_s:   _parts.append(f"Last simulate: {datetime.fromtimestamp(_last_s).strftime('%H:%M:%S')}")
        if _parts: st.caption("  ·  ".join(_parts))

    # Asset filter for display driven by scan multiselect — no separate selector needed
    _sel_labels = [_asset_id_label.get(a, a) for a in (sel_assets or [])]
    _asset_filter = _sel_labels[0] if len(_sel_labels) == 1 else "All assets"

    _sim_err = st.session_state.get("dash_sim_error")
    if _sim_err:
        st.warning(f"Auto-simulate error: {_sim_err}", icon="⚠️")

    # ── Run scan when button pressed ──────────────────────────────────────────
    if scan_clicked:
        _af = st.session_state.get("scan_asset_filter") or None
        _tf = st.session_state.get("scan_tf_filter") or None
        _desc = (
            ("assets: " + ", ".join(_asset_id_label.get(a,a) for a in _af) if _af else "all assets") +
            (" · TFs: " + ", ".join(_tf) if _tf else "")
        )
        with st.spinner(f"Scanning {_desc}..."):
            fired_n, rec_n, errs = _run_scan_now(asset_ids=_af, timeframes=_tf)
        st.session_state["dash_last_scan_ts"] = time.time()
        _clear()
        if fired_n == 0:
            st.info(f"Scan complete ({_desc}) — no new signals.")
        else:
            msg = f"Scan: {fired_n} signal(s) fired, {rec_n} new. ({_desc})"
            if errs: msg += f"  {len(errs)} error(s)"
            st.success(msg)
        st.rerun()

    if sim_clicked:
        with st.spinner("Simulating pending signals using real price data..."):
            sim = _run_simulate_pending()
        _clear()
        parts = []
        if sim["fills"]:
            parts.append(f"{sim['fills']} filled")
        if sim["active"]:
            parts.append(f"{sim['active']} running")
        if sim["closed"]:
            parts.append(f"{sim['closed']} closed ({sim['wins']} win)")
        if sim["expired"]:
            parts.append(f"{sim['expired']} expired")
        if parts:
            st.success("Simulate: " + " · ".join(parts))
        else:
            st.info("All signals still in progress — check back later.")
        if sim["errors"]:
            st.warning(f"Errors: {'; '.join(sim['errors'][:3])}")
        st.rerun()

    st.markdown("---")

    # ── Period filter ─────────────────────────────────────────────────────────
    from datetime import date as _dt_date
    _LIVE_START = _dt_date(2026, 4, 14)   # first day the live scanner fired signals
    _pf1, _pf2 = st.columns([2, 8])
    with _pf1:
        _since_val = st.session_state.get("dash_since_date", _LIVE_START)
        _since = st.date_input(
            "Since", value=_since_val,
            help="Filter all metrics and charts to trades with signal_date ≥ this date. "
                 "Default = live scanner start (2026-04-14). Set to 2024-01-01 to include full simulation history.",
            label_visibility="visible",
        )
        st.session_state["dash_since_date"] = _since
    with _pf2:
        _qcols = st.columns(4)
        _today = _dt_date.today()
        _quick = [("14d", _today - timedelta(days=14)), ("30d", _today - timedelta(days=30)),
                  ("Live start", _LIVE_START), ("All time", _dt_date(2024,1,1))]
        for _qc, (_ql, _qd) in zip(_qcols, _quick):
            if _qc.button(_ql, key=f"dash_q_{_ql}", use_container_width=True):
                st.session_state["dash_since_date"] = _qd
                st.rerun()
    _since_iso = _since.isoformat()

    # ── Active-combos filter ──────────────────────────────────────────────────
    _active_only = st.toggle(
        "Active combos only",
        value=st.session_state.get("dash_active_only", True),
        key="dash_active_only",
        help="When ON, only shows trades for (asset, strategy, timeframe) combos "
             "currently enabled in config/assets.yaml. Hides historical simulation "
             "trades from strategies that have since been disabled or removed.",
    )

    # Build the set of (ticker_yf, strategy, timeframe) currently in the watchlist
    _active_combos: set[tuple[str, str, str]] | None = None
    if _active_only:
        try:
            from core.config import get_ticker as _gtick
            try:
                from p4_live.journal_supabase import get_user_watchlist as _gwl_d
                _wl_d = _gwl_d(st.session_state["supabase_client"])
                if not _wl_d:
                    from core.config import get_watchlist as _gwl_yaml_d
                    _wl_d = _gwl_yaml_d()
            except Exception:
                from core.config import get_watchlist as _gwl_yaml_d
                _wl_d = _gwl_yaml_d()
            _active_combos = set()
            for _aid, _sid, _tf in _wl_d:
                try:
                    _active_combos.add((_gtick(_aid, source="yfinance"), _sid, _tf))
                except Exception:
                    pass
        except Exception:
            _active_combos = None

    # Apply asset + date filter — single pass over _all_trades
    _CLOSED_STATUSES  = {"win", "partial_win", "loss", "breakeven"}
    _af_match = _asset_filter != "All assets"
    all_t, closed, open_t, pending = [], [], [], []
    wins, losses = [], []
    total_r = 0.0
    for _t in _all_trades:
        if ((_t.get("signal_date") or "") < _since_iso):
            continue
        if _af_match and (_t.get("label","") or "") != _asset_filter \
                     and (_t.get("ticker","") or "") != _asset_filter:
            continue
        if _active_combos is not None:
            _combo = (_t.get("ticker",""), _t.get("strategy",""), _t.get("timeframe",""))
            if _combo not in _active_combos:
                continue
        all_t.append(_t)
        _st = _t["status"]
        if _st == "filled":
            open_t.append(_t)
        elif _st == "pending":
            pending.append(_t)
        elif _st in _CLOSED_STATUSES:
            closed.append(_t)
            _r = _t.get("pnl_r")
            if _r is not None:
                total_r += _r
                if _st in ("win", "partial_win"):
                    wins.append(_t)
                elif _st == "loss":
                    losses.append(_t)
    wr      = len(wins)/len(closed) if closed else 0.0
    streak, skind = _compute_streak(closed)
    acct        = _acct()
    dollar_risk = acct["balance"] * acct["risk_pct"] / 100
    min_lots    = acct.get("min_lots", 0.01)
    lot_oz      = acct.get("lot_oz", 100)

    # ── Pre-compute aggregations once — reused by charts + KPI cards ──────────
    _sp: dict[str, dict] = {}   # strategy → {n, r, w}
    _oc: dict[str, int]  = {}   # outcome   → count
    _ap: dict[str, float]= {}   # asset     → total_r
    _closed_sorted = sorted(closed, key=lambda x: x["signal_date"], reverse=True)
    _pending_sorted = sorted(pending, key=lambda x: x["signal_date"], reverse=True)
    _open_sorted    = sorted(open_t, key=lambda x: x.get("fill_date", ""), reverse=True)
    for _t in closed:
        _s  = _t.get("strategy", "unknown")
        _sl = _fmt_strategy(_s)
        _r  = _t.get("pnl_r") or 0.0
        _w  = _t["status"] in ("win", "partial_win")
        _ok = _t["status"]
        _lb = _label(_t)
        if _sl not in _sp: _sp[_sl] = {"n": 0, "r": 0.0, "w": 0}
        _sp[_sl]["n"] += 1; _sp[_sl]["r"] += _r
        if _w: _sp[_sl]["w"] += 1
        _ok_label = _ok.replace("_", " ").title()
        _oc[_ok_label] = _oc.get(_ok_label, 0) + 1
        if _t.get("pnl_r") is not None:
            _ap[_lb] = _ap.get(_lb, 0.0) + _t["pnl_r"]

    # ── KPI row ───────────────────────────────────────────────────────────────
    _since_label = "All time" if _since_iso <= "2024-01-02" else f"Since {_since.strftime('%d %b %Y')}"
    _dollar_total = total_r * dollar_risk

    k1,k2,k3,k4 = st.columns(4)
    k1.metric(
        "Total P&L",
        f"{total_r:+.1f}R",
        f"≈ ${_dollar_total:+.0f}",
        delta_color="normal" if total_r >= 0 else "inverse",
        help="Profit/Loss in R (risk units). 1R = one full risk per trade. "
             f"1R = ${dollar_risk:.0f} ({acct['risk_pct']:.1f}% of ${acct['balance']:,.0f} account).",
    )
    k2.metric(
        "Win Rate",
        f"{wr:.1%}",
        f"{len(wins)}W · {len(losses)}L · {len(closed)} closed",
        help="Percentage of closed trades that were wins or partial wins. "
             "Needs 30+ trades to be statistically meaningful.",
    )
    k3.metric(
        "Open / Pending",
        f"{len(open_t)} / {len(pending)}",
        f"${dollar_risk*len(open_t+pending):.0f} at risk" if open_t or pending else "no positions",
        help="Open = filled and running. Pending = waiting for price to enter entry zone.",
    )
    _streak_note = (
        f"{'🔥' if skind=='win' else '❄️'} {streak}-trade {skind} streak"
        if closed else "No trades yet"
    )
    k4.metric(
        "Streak",
        f"{streak}",
        _streak_note,
        delta_color="normal" if skind == "win" else "inverse",
        help="Consecutive wins or losses on the most recent closed trades.",
    )

    st.markdown(
        f'<p class="kpi-sub">{_since_label} · {len(closed)} closed trades'
        f' · 1R = ${dollar_risk:.0f}</p>',
        unsafe_allow_html=True,
    )

    # ── Strategy breakdown — uses pre-computed _sp ────────────────────────────
    if _sp:
        _need_more = len(closed) < 30
        sc = st.columns(max(len(_sp), 1))
        for col, (sname, sv) in zip(sc, _sp.items()):
            swr   = sv["w"] / sv["n"] if sv["n"] else 0
            pnl_c = "#10B981" if sv["r"] >= 0 else "#EF4444"
            dim   = "0.55" if sv["n"] < 5 else ("0.8" if sv["n"] < 10 else "1.0")
            n_tag = (f'<span style="color:#F59E0B;font-size:.62rem"> ·  {sv["n"]}/30</span>'
                     if sv["n"] < 30 else "")
            col.markdown(
                f'<div style="background:#0A1220;border:1px solid #0C1524;border-radius:10px;'
                f'padding:.55rem .8rem;text-align:center;opacity:{dim}">'
                f'<div style="font-size:.68rem;color:#64748B;font-weight:600;text-transform:uppercase'
                f';letter-spacing:.06em;margin-bottom:.25rem">{sname}{n_tag}</div>'
                f'<div style="font-size:1.15rem;font-weight:700;color:{pnl_c};line-height:1.1">'
                f'{sv["r"]:+.1f}R</div>'
                f'<div style="font-size:.7rem;color:#64748B;margin-top:.15rem">'
                f'{swr:.0%} WR · {sv["n"]} trades</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Portfolio exposure note ───────────────────────────────────────────────
    active_pos=open_t+pending
    if active_pos:
        total_exp=dollar_risk*len(active_pos)
        exp_pct=total_exp/acct["balance"]*100
        st.caption(
            f"At risk: ${total_exp:.0f} ({exp_pct:.1f}%) — "
            f"{len(open_t)} open · {len(pending)} pending"
        )

    st.markdown("---")

    # ── Equity + rolling WR ───────────────────────────────────────────────────
    chart_left, chart_right = st.columns([3, 2])
    with chart_left:
        if closed:
            # Use exit_date for realized P&L timing; fall back to signal_date
            pts = sorted(
                [(t.get("exit_date") or t["signal_date"], t["pnl_r"])
                 for t in closed if t["pnl_r"] is not None],
                key=lambda x: x[0],
            )
            cum,ex,ey=0,[],[]
            for d,r in pts: cum+=r; ex.append(d); ey.append(round(cum,3))
            c="#10B981" if ey[-1]>=0 else "#EF4444"
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines+markers",
                line=dict(color=c,width=2.5),fill="tozeroy",
                fillcolor=f"rgba({'16,185,129' if c=='#10B981' else '239,68,68'},.1)",
                hovertemplate="%{x}<br><b>%{y:+.2f}R</b><extra></extra>"))
            fig.add_hline(y=0,line_dash="dash",line_color="#2D4060",line_width=1)
            fig.update_layout(**_PL,title="Equity Curve (R)",height=240)
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("No closed trades yet — equity curve will appear here.")

    with chart_right:
        rdates, rwrs = _rolling_wr(closed)
        if rdates:
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(x=rdates,y=[w*100 for w in rwrs],
                mode="lines",line=dict(color="#60A5FA",width=2),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
            fig2.add_hline(y=50,line_dash="dot",line_color="#5A6E85",line_width=1)
            fig2.update_layout(**_PL,title="Rolling Win Rate (20-trade)",height=240)
            fig2.update_yaxes(ticksuffix="%",range=[0,100])
            st.plotly_chart(fig2,use_container_width=True)
        else:
            st.caption("Rolling win rate — needs at least 3 closed trades.")

    st.markdown("---")

    # ── Charts row: outcomes + P&L by asset + P&L by strategy ────────────────
    cc1,cc2,cc3 = st.columns(3)
    with cc1:
        st.subheader("Outcomes")
        if closed:
            clrs={"Win":"#10B981","Partial Win":"#60A5FA","Loss":"#EF4444","Breakeven":"#FBBF24"}
            fig3=go.Figure(go.Pie(labels=list(_oc.keys()),values=list(_oc.values()),
                marker_colors=[clrs.get(k,"#94A3B8") for k in _oc],hole=0.45,
                textinfo="label+percent",hovertemplate="%{label}: %{value}<extra></extra>"))
            fig3.update_layout(**_PL,height=250,showlegend=False,title="Trade Outcomes")
            st.plotly_chart(fig3,use_container_width=True)
            sel=st.selectbox("Drilldown",["All"]+list(_oc.keys()),key="d_out")
            if sel!="All":
                filt=[t for t in closed if t["status"].replace("_"," ").title()==sel]
                st.dataframe(_trades_df(filt),use_container_width=True,hide_index=True)
        else:
            st.markdown('<div style="height:250px;display:flex;align-items:center;'
                        'justify-content:center;color:#5A6E85">No closed trades</div>',
                        unsafe_allow_html=True)

    with cc2:
        st.subheader("P&L by Asset")
        if closed:
            sl=sorted(_ap.items(),key=lambda x:x[1],reverse=True)
            lb,vl=[x[0] for x in sl],[x[1] for x in sl]
            fig4=go.Figure(go.Bar(
                x=vl, y=lb, orientation="h",
                text=[f"{v:+.1f}R" for v in vl],
                textposition="outside",
                marker_color=["#10B981" if v>=0 else "#EF4444" for v in vl],
                hovertemplate="%{y}: %{x:+.2f}R<extra></extra>",
            ))
            fig4.add_vline(x=0, line_color="#2D4060", line_width=1)
            fig4.update_layout(**_PL,height=250,title="Net R / Asset",showlegend=False)
            st.plotly_chart(fig4,use_container_width=True)
            sel2=st.selectbox("Drilldown",["All"]+lb,key="d_asset")
            if sel2!="All":
                filt=[t for t in closed if _label(t)==sel2]
                st.dataframe(_trades_df(filt),use_container_width=True,hide_index=True)
        else:
            st.markdown('<div style="height:250px;display:flex;align-items:center;'
                        'justify-content:center;color:#5A6E85">No data</div>',unsafe_allow_html=True)

    with cc3:
        st.subheader("P&L by Strategy")
        if closed:
            slb=sorted(_sp.items(),key=lambda x:x[1]["r"],reverse=True)
            fig5=go.Figure(go.Bar(
                x=[x[0] for x in slb],
                y=[x[1]["r"] for x in slb],
                text=[f"{x[1]['r']:+.1f}R" for x in slb],
                textposition="outside",
                marker_color=["#10B981" if x[1]["r"]>=0 else "#EF4444" for x in slb],
                hovertemplate="%{x}: %{y:+.2f}R<extra></extra>"))
            fig5.add_hline(y=0, line_color="#2D4060", line_width=1)
            fig5.update_layout(**_PL,height=250,title="Net R / Strategy",showlegend=False)
            st.plotly_chart(fig5,use_container_width=True)
        else:
            st.markdown('<div style="height:250px;display:flex;align-items:center;'
                        'justify-content:center;color:#5A6E85">No data</div>',unsafe_allow_html=True)

    st.markdown("---")

    # ── Closed trades showcase ────────────────────────────────────────────────
    ch1,ch2=st.columns([4,1])
    ch1.subheader(f"Closed Trades ({len(closed)})")
    if ch2.button("View full history →",key="dash_jrnl"):
        st.session_state["page"]="Journal"; st.rerun()

    if closed:
        st.dataframe(_trades_df(_closed_sorted[:10]),use_container_width=True,hide_index=True)
    else:
        st.info("No closed trades yet. Log trades via the scanner or Journal → Manual Entry.")

    st.markdown("---")

    # ── Active Signals ────────────────────────────────────────────────────────
    st.markdown("## Active Signals")
    if pending:
        for t in _pending_sorted:
            dir_color = "#10B981" if t["direction"] == "long" else "#EF4444"
            exp_days  = t.get("expiry_days", 5)
            try:
                sig_d     = date.fromisoformat(t["signal_date"])
                days_left = max(0, exp_days - (date.today() - sig_d).days)
            except Exception:
                days_left = exp_days
            urgency_color = "#EF4444" if days_left <= 1 else ("#FBBF24" if days_left <= 2 else "#94A3B8")
            sig_date_fmt  = t.get("signal_date","")
            tf_str        = t.get("timeframe","")
            strat_str     = _fmt_strategy(t.get("strategy",""))

            # ── Signal card header ────────────────────────────────────────────
            st.markdown(
                f'<div style="background:#0D1824;border:1px solid {dir_color}44;'
                f'border-left:4px solid {dir_color};border-radius:10px;'
                f'padding:.8rem 1.1rem;margin-bottom:0">'

                # Row 1: direction badge + asset + strategy + timeframe + expiry
                f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem">'
                f'<span style="background:{dir_color};color:#fff;font-weight:800;'
                f'padding:3px 10px;border-radius:5px;font-size:.78rem;letter-spacing:.05em">'
                f'{t["direction"].upper()}</span>'
                f'<span style="color:#F1F5F9;font-weight:700;font-size:1rem">{_label(t)}</span>'
                f'<span style="color:#5A6E85;font-size:.8rem">{strat_str}</span>'
                + (f'<span style="background:#1A2840;color:#60A5FA;font-size:.7rem;font-weight:600;'
                   f'padding:1px 6px;border-radius:4px">{tf_str}</span>' if tf_str else '')
                + f'<span style="margin-left:auto;color:{urgency_color};font-size:.75rem;font-weight:600">'
                f'expires in {days_left}d</span>'
                f'</div>'

                # Row 2: entry / SL / TP levels in a clean horizontal layout
                f'<div style="display:flex;gap:1.5rem;font-size:.82rem;flex-wrap:wrap">'
                f'<span><span style="color:#5A6E85;font-size:.7rem">ENTRY&nbsp;</span>'
                f'<span style="color:#F1F5F9;font-weight:600">{t.get("entry_low",0):,.2f} – {t.get("entry_high",0):,.2f}</span></span>'
                f'<span><span style="color:#5A6E85;font-size:.7rem">STOP&nbsp;</span>'
                f'<span style="color:#EF4444;font-weight:600">{t.get("stop_loss",0):,.2f}</span></span>'
                f'<span><span style="color:#5A6E85;font-size:.7rem">TP1&nbsp;</span>'
                f'<span style="color:#10B981;font-weight:600">{t.get("tp1",0):,.2f}</span></span>'
                + (f'<span><span style="color:#5A6E85;font-size:.7rem">TP2&nbsp;</span>'
                   f'<span style="color:#34D399;font-weight:600">{t.get("tp2",0):,.2f}</span></span>'
                   if t.get("tp2") else '')
                + f'<span style="margin-left:auto"><span style="color:#5A6E85;font-size:.7rem">R:R&nbsp;</span>'
                f'<span style="color:#A78BFA;font-weight:700">{t.get("rr",0):.1f}R</span></span>'
                f'</div>'

                f'</div>', unsafe_allow_html=True)

            # Live execution detail panel
            entry_low  = float(t.get("entry_low", 0))
            entry_high = float(t.get("entry_high", 0))
            sl         = float(t.get("stop_loss", 0))
            tp1        = float(t.get("tp1", 0))
            tp2        = float(t.get("tp2", 0)) if t.get("tp2") else None
            is_long    = t["direction"] == "long"
            risk_pts   = abs(entry_high - sl) if is_long else abs(entry_low - sl)

            # Min-lot sizing (broker constraint: 0.01 lots = 1 oz for Gold)
            # dollar_risk, min_lots, lot_oz pre-computed above the loop
            min_oz       = min_lots * lot_oz          # 1 oz for Gold
            min_lot_risk = round(risk_pts * min_oz, 2) if risk_pts > 0 else 0

            # Ideal sizing (ignoring min lot constraint)
            ideal_oz   = round(dollar_risk / risk_pts, 4) if risk_pts > 0 else 0
            ideal_lots = round(ideal_oz / lot_oz, 4)

            # Tradability: can we trade this with at most 2x target risk?
            min_lot_margin = acct.get("min_lot_margin", 190.0)
            if min_lot_risk <= dollar_risk * 1.0:
                trade_status = ("TRADEABLE", "#10B981", "Risk fits budget exactly")
            elif min_lot_risk <= dollar_risk * 2.7:
                trade_status = ("TIGHT", "#FBBF24", f"0.01 lots = ${min_lot_risk:.0f} risk (over ${dollar_risk:.0f} budget but manageable)")
            else:
                trade_status = ("TOO WIDE", "#EF4444", f"0.01 lots = ${min_lot_risk:.0f} risk — SL too large for budget. Look for 1h signals.")

            notional   = round(min_oz * entry_high, 2)
            margin_req = round(notional / acct.get("leverage", 100), 2)

            # Fetch live price
            ticker_yf   = _LIVE_TO_YF.get(t.get("ticker",""), t.get("ticker",""))
            live_price  = _fetch_live_price(ticker_yf)

            with st.expander("Trade execution detail", expanded=False):

                # ── Live price status banner ──────────────────────────────────
                if live_price is not None:
                    in_zone = entry_low <= live_price <= entry_high
                    if is_long:
                        above_zone = live_price > entry_high
                        dist_to_zone = live_price - entry_high if above_zone else 0.0
                        below_zone = live_price < entry_low
                        dist_below  = entry_low - live_price if below_zone else 0.0
                    else:
                        above_zone  = live_price > entry_high
                        dist_to_zone = live_price - entry_high if above_zone else 0.0
                        below_zone  = live_price < entry_low
                        dist_below   = entry_low - live_price if below_zone else 0.0

                    if in_zone:
                        banner_color = "#10B981"; banner_bg = "#052E1C"
                        banner_icon  = "IN ENTRY ZONE"
                        banner_msg   = f"Price {live_price:,.2f} is inside the entry zone — place your order now"
                    elif is_long and above_zone:
                        banner_color = "#F59E0B"; banner_bg = "#1C1200"
                        banner_icon  = "ABOVE ZONE"
                        banner_msg   = f"Price {live_price:,.2f} is {dist_to_zone:.2f} pts above zone — wait for pullback to {entry_high:,.2f}"
                    elif is_long and below_zone:
                        banner_color = "#EF4444"; banner_bg = "#1C0505"
                        banner_icon  = "BELOW ZONE (SL RISK)"
                        banner_msg   = f"Price {live_price:,.2f} is {dist_below:.2f} pts below zone — approaching stop loss at {sl:,.2f}"
                    elif not is_long and below_zone:
                        banner_color = "#F59E0B"; banner_bg = "#1C1200"
                        banner_icon  = "BELOW ZONE"
                        banner_msg   = f"Price {live_price:,.2f} is {dist_below:.2f} pts below entry zone — wait for rally to {entry_low:,.2f}"
                    elif not is_long and above_zone:
                        # SHORT pending: price above entry zone, waiting to drop in
                        _pts_to_entry = dist_to_zone  # live - entry_high
                        if live_price > sl:
                            # Price has overshot above the SL — setup has deteriorated
                            _above_sl = live_price - sl
                            banner_color = "#EF4444"; banner_bg = "#1C0505"
                            banner_icon  = "ABOVE SL"
                            banner_msg   = (f"Price {live_price:,.2f} is {_above_sl:.2f} pts above SL ({sl:,.2f}) — "
                                            f"setup has deteriorated, consider skipping")
                        else:
                            banner_color = "#F59E0B"; banner_bg = "#1C1200"
                            banner_icon  = "WAITING FOR DROP"
                            banner_msg   = (f"Price {live_price:,.2f} needs to fall {_pts_to_entry:.2f} pts "
                                            f"to reach zone top {entry_high:,.2f}")
                    else:
                        banner_color = "#94A3B8"; banner_bg = "#101826"
                        banner_icon  = "NO LIVE DATA"
                        banner_msg   = "Could not determine price position relative to zone"

                    st.markdown(
                        f'<div style="padding:.6rem 1rem;background:{banner_bg};border:1px solid {banner_color}44;'
                        f'border-radius:8px;margin-bottom:.8rem;display:flex;align-items:center;gap:1rem">'
                        f'<span style="color:{banner_color};font-weight:800;font-size:.78rem;'
                        f'white-space:nowrap">{banner_icon}</span>'
                        f'<span style="color:#CBD5E1;font-size:.82rem">{banner_msg}</span>'
                        f'<span style="margin-left:auto;color:#60A5FA;font-weight:700;font-size:.9rem">'
                        f'Live: {live_price:,.2f}</span>'
                        f'</div>', unsafe_allow_html=True)

                    # ── Visual price gauge ────────────────────────────────────
                    # Build gauge: SL → entry_low → entry_high → TP1 [→ TP2]
                    # Normalise all prices to 0–100 scale for display
                    _all_levels = [sl, entry_low, entry_high, tp1]
                    if tp2: _all_levels.append(tp2)
                    _lo = min(_all_levels) - abs(tp1 - entry_high) * 0.1
                    _hi = max(_all_levels) + abs(tp1 - entry_high) * 0.1

                    def _norm(v):
                        return (v - _lo) / (_hi - _lo) * 100 if _hi > _lo else 50

                    sl_x     = _norm(sl)
                    el_x     = _norm(entry_low)
                    eh_x     = _norm(entry_high)
                    tp1_x    = _norm(tp1)
                    tp2_x    = _norm(tp2) if tp2 else None
                    price_x  = _norm(live_price)

                    # Build SVG-style bar using HTML
                    gauge_parts = []
                    # Background track
                    gauge_parts.append(
                        '<div style="position:relative;height:32px;background:#1A2840;border-radius:6px;'
                        'margin:.5rem 0 1.2rem 0;overflow:visible">')
                    # SL zone (SL → entry_low for long)
                    if is_long:
                        sl_w = el_x - sl_x
                        gauge_parts.append(
                            f'<div style="position:absolute;left:{sl_x:.1f}%;width:{sl_w:.1f}%;height:100%;'
                            f'background:#EF444433;border-radius:4px 0 0 4px"></div>')
                        # Entry zone
                        ez_w = eh_x - el_x
                        gauge_parts.append(
                            f'<div style="position:absolute;left:{el_x:.1f}%;width:{ez_w:.1f}%;height:100%;'
                            f'background:#10B98133;border:1px solid #10B98166"></div>')
                        # TP zone
                        tz_w = tp1_x - eh_x
                        gauge_parts.append(
                            f'<div style="position:absolute;left:{eh_x:.1f}%;width:{tz_w:.1f}%;height:100%;'
                            f'background:#60A5FA22"></div>')
                    else:
                        # Short: SL above entry_high
                        gauge_parts.append(
                            f'<div style="position:absolute;left:{eh_x:.1f}%;width:{sl_x-eh_x:.1f}%;height:100%;'
                            f'background:#EF444433;border-radius:0 4px 4px 0"></div>')
                        ez_w = eh_x - el_x
                        gauge_parts.append(
                            f'<div style="position:absolute;left:{el_x:.1f}%;width:{ez_w:.1f}%;height:100%;'
                            f'background:#10B98133;border:1px solid #10B98166"></div>')
                        tz_w = el_x - tp1_x
                        gauge_parts.append(
                            f'<div style="position:absolute;left:{tp1_x:.1f}%;width:{tz_w:.1f}%;height:100%;'
                            f'background:#60A5FA22"></div>')

                    # Level markers
                    for lx, lc, lt in [
                        (sl_x, "#EF4444", "SL"), (el_x, "#10B981", ""),
                        (eh_x, "#10B981", "Zone"), (tp1_x, "#60A5FA", "TP1"),
                    ]:
                        gauge_parts.append(
                            f'<div style="position:absolute;left:{lx:.1f}%;top:0;width:2px;height:100%;'
                            f'background:{lc};opacity:.8"></div>'
                            f'<div style="position:absolute;left:{lx:.1f}%;top:34px;'
                            f'transform:translateX(-50%);color:{lc};font-size:.6rem;white-space:nowrap">{lt}</div>'
                        )
                    if tp2_x:
                        gauge_parts.append(
                            f'<div style="position:absolute;left:{tp2_x:.1f}%;top:0;width:2px;height:100%;'
                            f'background:#A78BFA;opacity:.8"></div>'
                            f'<div style="position:absolute;left:{tp2_x:.1f}%;top:34px;'
                            f'transform:translateX(-50%);color:#A78BFA;font-size:.6rem">TP2</div>'
                        )

                    # Live price dot
                    price_pct = max(0, min(100, price_x))
                    price_col = banner_color
                    gauge_parts.append(
                        f'<div style="position:absolute;left:{price_pct:.1f}%;top:50%;'
                        f'transform:translate(-50%,-50%);width:14px;height:14px;border-radius:50%;'
                        f'background:{price_col};border:2px solid #fff;z-index:10;'
                        f'box-shadow:0 0 8px {price_col}88"></div>'
                    )
                    gauge_parts.append('</div>')
                    st.markdown("".join(gauge_parts), unsafe_allow_html=True)

                # ── Distance metrics ──────────────────────────────────────────
                ex1, ex2, ex3, ex4, ex5 = st.columns(5)
                if live_price is not None:
                    if in_zone:
                        dist_label = "IN ZONE"
                        dist_val   = f"{live_price:,.2f}"
                    elif is_long:
                        dist_to_entry = live_price - entry_high if above_zone else entry_low - live_price
                        dist_label = "pts to zone"
                        dist_val   = f"{dist_to_entry:.2f} pts"
                    else:
                        dist_to_entry = live_price - entry_high if above_zone else entry_low - live_price
                        dist_label = "pts to zone"
                        dist_val   = f"{dist_to_entry:.2f} pts"

                    dist_sl  = abs(live_price - sl)
                    dist_tp1 = abs(live_price - tp1)
                    # R:R from live price to TP1 vs live price to SL
                    rr_live = round(dist_tp1 / dist_sl, 2) if dist_sl > 0 else None
                    ex1.metric("Live price", f"{live_price:,.2f}", dist_label)
                    ex2.metric("Dist to SL",  f"{dist_sl:.2f} pts",  help=f"SL at {sl:,.2f}")
                    ex3.metric("Dist to TP1", f"{dist_tp1:.2f} pts", help=f"TP1 at {tp1:,.2f}")
                    ex4.metric("R:R (live→TP1)",
                               f"{rr_live:.2f}" if rr_live is not None else "—",
                               help="Distance to TP1 ÷ distance to SL from current live price")
                else:
                    ex1.metric("Live price", "unavailable")
                    ex2.metric("Risk pts (SL)", f"{risk_pts:.2f}")
                    ex3.metric("TP1", f"{tp1:,.2f}")
                    rr_sig = t.get("rr")
                    ex4.metric("R:R (signal)", f"{rr_sig:.2f}" if rr_sig else "—",
                               help="R:R as computed at signal time")

                # Min-lot sizing metric
                trd_color = {"TRADEABLE": "normal", "TIGHT": "off", "TOO WIDE": "inverse"}
                ex5.metric(
                    "0.01 lots risk",
                    f"${min_lot_risk:.0f}",
                    trade_status[0],
                    delta_color=trd_color.get(trade_status[0], "off"),
                    help=trade_status[2],
                )

                st.markdown("---")

                # ── Tradability warning ───────────────────────────────────────
                if trade_status[0] != "TRADEABLE":
                    st.markdown(
                        f'<div style="padding:.5rem .9rem;background:#1C1000;border:1px solid #78350F;'
                        f'border-radius:6px;color:#FCD34D;font-size:.78rem;margin-bottom:.6rem">'
                        f'<b>Position sizing note:</b> {trade_status[2]}'
                        f'</div>', unsafe_allow_html=True)

                lv1, lv2 = st.columns([2, 1])
                with lv1:
                    st.markdown("**Price levels**")
                    live_col = f"← **LIVE {live_price:,.2f}**" if live_price is not None else ""
                    if is_long:
                        def _dist_live(price):
                            if live_price is None: return ""
                            d = price - live_price
                            return f" ({d:+.2f} from live)"
                        rows = (
                            f"| Level | Price | Details |\n"
                            f"|-------|-------|----------|\n"
                            f"| Stop loss | `{sl:,.2f}` | -{risk_pts:.2f} pts from entry high{_dist_live(sl)} |\n"
                            f"| Entry low | `{entry_low:,.2f}` | bottom of zone{_dist_live(entry_low)} |\n"
                            f"| Entry high | `{entry_high:,.2f}` | top of zone — place limit here{_dist_live(entry_high)} |\n"
                            f"| TP1 ({t.get('tp1_alloc',70)}%) | `{tp1:,.2f}` | +{tp1-entry_high:.2f} pts{_dist_live(tp1)} |\n"
                        )
                        if tp2:
                            rows += f"| TP2 ({t.get('tp2_alloc',30)}%) | `{tp2:,.2f}` | +{tp2-entry_high:.2f} pts{_dist_live(tp2)} |\n"
                        st.markdown(rows)
                    else:
                        def _dist_live(price):
                            if live_price is None: return ""
                            d = price - live_price
                            return f" ({d:+.2f} from live)"
                        rows = (
                            f"| Level | Price | Details |\n"
                            f"|-------|-------|----------|\n"
                            f"| Stop loss | `{sl:,.2f}` | +{risk_pts:.2f} pts above entry high{_dist_live(sl)} |\n"
                            f"| Entry high | `{entry_high:,.2f}` | top of zone — **place limit SELL here**{_dist_live(entry_high)} |\n"
                            f"| Entry low | `{entry_low:,.2f}` | bottom of zone (add to position){_dist_live(entry_low)} |\n"
                            f"| TP1 ({t.get('tp1_alloc',70)}%) | `{tp1:,.2f}` | +{entry_low-tp1:.2f} pts below zone{_dist_live(tp1)} |\n"
                        )
                        if tp2:
                            rows += f"| TP2 ({t.get('tp2_alloc',30)}%) | `{tp2:,.2f}` | +{entry_low-tp2:.2f} pts below zone{_dist_live(tp2)} |\n"
                        st.markdown(rows)

                with lv2:
                    st.markdown("**How to trade this**")
                    entry_mid = (entry_low + entry_high) / 2
                    rec_lots = min_lots
                    rec_risk = min_lot_risk

                    if is_long:
                        order_price = entry_mid
                        order_type  = "limit BUY"
                        order_note  = f"`{order_price:,.2f}` (mid of zone)"
                    else:
                        order_price = entry_high
                        order_type  = "limit SELL"
                        order_note  = f"`{order_price:,.2f}` (top of zone)"

                    rr_val = t.get("rr")
                    rr_line = f"  \n5. R:R: **{rr_val:.1f}** (TP1 vs SL)  \n" if rr_val else "  \n"
                    margin_step = 6 if rr_val else 5
                    window_step = margin_step + 1

                    st.markdown(
                        f"1. Set **{order_type}** at {order_note}  \n"
                        f"2. **Stop loss** at `{sl:,.2f}`  \n"
                        f"3. Size: **{rec_lots} lots** = {rec_lots * lot_oz:.0f} oz  \n"
                        f"4. Risk: **${rec_risk:.2f}** at stop{rr_line}"
                        f"{margin_step}. Margin: **${min_lot_margin:.0f}** (broker min)  \n"
                        f"{window_step}. Window: {days_left} day(s) left"
                    )
                if t.get("rationale"):
                    st.caption(f"Why: {t['rationale'][:300]}")
    else:
        st.markdown(
            '<div style="padding:.8rem 1rem;background:#101826;border:1px solid #1A2840;'
            'border-radius:8px;color:#5A6E85;font-size:.84rem">'
            'No active signals. Click <b>Run Scan Now</b> above.</div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Open positions + recent signals ──────────────────────────────────────
    lft,rgt=st.columns([3,2])
    with lft:
        lh1,lh2=st.columns([5,2])
        lh1.subheader(f"Open Positions ({len(open_t)})")
        if lh2.button("Manage →",key="dash_mgr"):
            st.session_state["page"]="Journal"; st.rerun()
        if open_t:
            for t in _open_sorted:
                dir_color  = "#10B981" if t["direction"] == "long" else "#EF4444"
                fill_ro    = _ro_time(t.get("fill_bar_ts"), "%d %b %H:%M RO") or t.get("fill_date", "—")
                sig_ro     = _ro_time(t.get("signal_bar_ts") or t.get("recorded_at"), "%d %b %H:%M RO")
                fill_p     = float(t.get("fill_price",0))
                sl_p       = float(t.get("stop_loss",0))
                tp1_p      = float(t.get("tp1",0)) if t.get("tp1") else None
                sl_dist    = abs(fill_p - sl_p)
                tp_dist    = abs((tp1_p or fill_p) - fill_p)
                ticker_yf  = _LIVE_TO_YF.get(t.get("ticker",""), t.get("ticker",""))
                live_p     = _fetch_live_price(ticker_yf)
                live_pnl   = None
                if live_p and fill_p and sl_dist > 0:
                    raw = (live_p - fill_p) if t["direction"]=="long" else (fill_p - live_p)
                    live_pnl = round(raw / sl_dist, 2)
                pnl_color  = "#10B981" if (live_pnl or 0) >= 0 else "#EF4444"

                partial_pnl_r = t.get("partial_close_pnl_r")
                _exp_key = f"open_{t['signal_date']}_{t['ticker']}_{t.get('strategy','')}_{t.get('timeframe','')}"
                _tp1_badge = (
                    f"  ·  TP1 hit ({partial_pnl_r:+.2f}R locked)"
                    if partial_pnl_r is not None else ""
                )
                with st.expander(
                    f"{'🟢' if t['direction']=='long' else '🔴'} "
                    f"{_label(t)}  ·  {_fmt_strategy(t.get('strategy',''))}  "
                    f"[{t.get('timeframe','')}]  ·  Fill {fill_p:,.2f}"
                    + _tp1_badge
                    + (f"  ·  Live P&L: **{live_pnl:+.2f}R**" if live_pnl is not None else ""),
                    expanded=False,
                ):
                    if partial_pnl_r is not None:
                        _pc = "#10B981" if partial_pnl_r >= 0 else "#EF4444"
                        st.markdown(
                            f'<div style="background:#052E1C;border:1px solid #054830;'
                            f'border-radius:7px;padding:.4rem .8rem;font-size:.8rem;'
                            f'color:#10B981;margin-bottom:.6rem">'
                            f'<b>TP1 hit</b> — {partial_pnl_r:+.2f}R locked in on the first leg. '
                            f'Runner (second leg) still open in MT5.</div>',
                            unsafe_allow_html=True)
                    # Summary row
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Fill price",  f"{fill_p:,.2f}")
                    m2.metric("Stop loss",   f"{sl_p:,.2f}",
                              delta=f"{abs(fill_p-sl_p):,.2f} pts away", delta_color="off")
                    m3.metric("TP1",         f"{tp1_p:,.2f}" if tp1_p else "—",
                              delta=f"RR {tp_dist/sl_dist:.1f}R" if sl_dist>0 else None, delta_color="off")
                    m4.metric("Live price",  f"{live_p:,.2f}" if live_p else "—",
                              delta=f"{live_pnl:+.2f}R" if live_pnl is not None else None,
                              delta_color="normal" if (live_pnl or 0)>=0 else "inverse")
                    m5.metric("Signal at",  sig_ro or "—")
                    # Chart
                    _render_chart_upload(t, f"chart_open_{_exp_key}")
        else:
            st.info("No filled/open positions.")
    with rgt:
        cutoff=(date.today()-timedelta(days=7)).isoformat()
        recent=sorted([t for t in all_t if t["signal_date"]>=cutoff and t["status"] not in ("pending",)],
                      key=lambda x:x["signal_date"],reverse=True)[:10]
        st.subheader("Recent Activity (7d)")
        if recent:
            for t in recent:
                c2=_SC.get(t["status"],"#94A3B8")
                pnl=f" <b>{t['pnl_r']:+.2f}R</b>" if t.get("pnl_r") is not None else ""
                st.markdown(
                    f'<div class="signal-row"><span style="color:{c2};font-weight:700;'
                    f'font-size:.7rem">{t["status"].replace("_"," ").upper()}</span>'
                    f'<span style="color:#F1F5F9;font-weight:600">{_label(t)}</span>'
                    f'<span style="color:#5A6E85">{_fmt_strategy(t.get("strategy",""))}</span>'
                    f'<span style="color:#60A5FA">{t["direction"].upper()}</span>'
                    f'<span style="margin-left:auto;color:#5A6E85">{t["signal_date"]}{pnl}</span>'
                    f'</div>',unsafe_allow_html=True)
        else:
            st.info("No activity in the past 7 days.")

    # ── Auto-refresh loop ─────────────────────────────────────────────────────
    # Rerun without sleeping — the interval check at the top of this function
    # (now_ts - last_ref >= interval) prevents rapid re-renders and data thrash.
    if st.session_state.get("dash_auto_refresh", False):
        time.sleep(1)   # 1s yield so the browser can paint; no 5s UI freeze
        st.rerun()


# =============================================================================
# PAGE 2: MARKET ANALYSIS
# =============================================================================
def page_analysis():
    st.title("Market Analysis")

    _tab_sa, _tab_pb = st.tabs(["Single Asset Analysis", "Portfolio Briefing"])

    # ── Portfolio Briefing ────────────────────────────────────────────────────
    with _tab_pb:
        st.markdown("Enter any tickers to get an AI narrative explaining why your portfolio moved.")
        pb_c1, pb_c2, pb_c3 = st.columns([4, 2, 2])
        with pb_c1:
            pb_tickers_raw = st.text_area(
                "Tickers (comma or newline separated)",
                placeholder="e.g. GC=F, BTC-USD, ^GSPC, AAPL",
                height=80, key="pb_tickers",
            )
        with pb_c2:
            pb_period = st.selectbox("Look-back period",
                ["1 day","1 week","1 month","3 months"], index=2, key="pb_period")
            pb_model = st.selectbox("Model",["claude-sonnet-4-6","claude-opus-4-6"],
                format_func=lambda m:"Sonnet (fast)" if "sonnet" in m else "Opus (deep)",
                key="pb_model")
        with pb_c3:
            pb_context = st.text_area(
                "Optional context",
                placeholder="e.g. My portfolio: 40% Gold, 30% BTC, 30% SPX. I'm wondering why it went up ~15%.",
                height=80, key="pb_context",
            )

        pb_btn = st.button("Generate Briefing", type="primary", key="pb_run",
                           disabled=not pb_tickers_raw)

        if pb_btn and pb_tickers_raw:
            from core.demo_limit import try_consume as _dl_try, DEMO_LIMIT_MESSAGE as _dl_msg
            if not _dl_try(1):
                st.error(_dl_msg)
                st.stop()
            _pb_tickers = [t.strip().upper() for t in pb_tickers_raw.replace("\n",",").split(",") if t.strip()]
            _pb_days = {"1 day":2,"1 week":8,"1 month":35,"3 months":95}[pb_period]

            with st.spinner("Fetching price data…"):
                import yfinance as _yf
                from datetime import timedelta
                _pb_end = datetime.now()
                _pb_start = _pb_end - timedelta(days=_pb_days + 5)
                _pb_rows = []
                for _tk in _pb_tickers:
                    try:
                        _df = _yf.download(_tk, start=_pb_start.strftime("%Y-%m-%d"),
                                           end=_pb_end.strftime("%Y-%m-%d"),
                                           auto_adjust=True, progress=False)
                        if _df is None or len(_df) < 2:
                            _pb_rows.append({"Ticker":_tk,"Price":"N/A","1D%":"N/A","5D%":"N/A","1MO%":"N/A"})
                            continue
                        _close = _df["Close"].dropna()
                        _cur   = float(_close.iloc[-1])
                        _1d    = float((_close.iloc[-1]/_close.iloc[-2]-1)*100) if len(_close)>=2 else None
                        _5d    = float((_close.iloc[-1]/_close.iloc[max(0,len(_close)-6)]-1)*100) if len(_close)>=6 else None
                        _1mo   = float((_close.iloc[-1]/_close.iloc[0]-1)*100) if len(_close)>=5 else None
                        def _fmt(v): return f"{v:+.2f}%" if v is not None else "N/A"
                        _pb_rows.append({"Ticker":_tk,
                                         "Price":f"{_cur:,.4f}",
                                         "1D%":_fmt(_1d),"5D%":_fmt(_5d),
                                         f"{pb_period.replace(' ','').upper()}%":_fmt(_1mo)})
                    except Exception as _e:
                        _pb_rows.append({"Ticker":_tk,"Price":"err","1D%":"N/A","5D%":"N/A",
                                         f"{pb_period.replace(' ','').upper()}%":"N/A"})

            if _pb_rows:
                st.dataframe(pd.DataFrame(_pb_rows), use_container_width=True, hide_index=True)

            with st.spinner("Generating AI narrative…"):
                try:
                    import anthropic as _ant
                    _ant_client = _ant.Anthropic()
                    _perf_lines = "\n".join(
                        f"  - {r['Ticker']}: price={r['Price']}, " +
                        ", ".join(f"{k}={v}" for k,v in r.items() if k not in ("Ticker","Price"))
                        for r in _pb_rows
                    )
                    _pb_prompt = (
                        f"You are a portfolio analyst. Today is {datetime.now().strftime('%Y-%m-%d')}.\n\n"
                        f"Portfolio performance data (look-back: {pb_period}):\n{_perf_lines}\n\n"
                        + (f"User context: {pb_context}\n\n" if pb_context else "")
                        + "In 3–4 paragraphs, explain:\n"
                        "1. What macroeconomic, geopolitical, or sector-level forces drove the performance of each asset.\n"
                        "2. How the assets interacted (correlations, diversification effects, or concentration risk).\n"
                        "3. What the user should watch going forward given current market dynamics.\n\n"
                        "Be specific and factual. If a price move is unusual, flag it. "
                        "Avoid vague statements like 'markets were volatile'."
                    )
                    _pb_resp = _ant_client.messages.create(
                        model=pb_model,
                        max_tokens=1200,
                        messages=[{"role":"user","content":_pb_prompt}],
                    )
                    _pb_narrative = _pb_resp.content[0].text
                    st.session_state["pb_narrative"] = _pb_narrative
                    st.session_state["pb_narrative_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                except Exception as _e:
                    st.error(f"AI call failed: {_e}")

        if "pb_narrative" in st.session_state:
            ts_str = st.session_state.get("pb_narrative_ts","")
            if ts_str: st.caption(f"Generated {ts_str}")
            st.markdown(st.session_state["pb_narrative"])
            if st.button("Clear", key="pb_clear"):
                del st.session_state["pb_narrative"]
                st.rerun()

    # ── Single Asset Analysis ─────────────────────────────────────────────────
    with _tab_sa:
        st.markdown("AI-powered analysis combining technical indicators, macro data, news, "
                    "and geopolitical risk into a directional bias and concrete trade setups.")

        acct   = _acct()
        assets = _assets()
        amap   = {a["label"]:a for a in assets}

        c1,c2,c3,c4 = st.columns([3,2,2,2])
        with c1:
            opts=[a["label"] for a in assets]+["Custom..."]
            sel=st.selectbox("Asset",opts)
            if sel=="Custom...":
                custom=st.text_input("Ticker (e.g. AAPL, BTC-USD)")
                ticker=custom.strip() if custom else None; aname=custom or ""
            else:
                a_cfg=amap[sel]; ticker=a_cfg.get("tickers",{}).get("yfinance",sel); aname=sel
        with c2: tf=st.selectbox("Timeframe",["1d","1wk","1h","15m"])
        with c3: model=st.selectbox("Model",["claude-sonnet-4-6","claude-opus-4-6"],
                                     format_func=lambda m:"Sonnet (fast)" if "sonnet" in m else "Opus (deep)")
        with c4: st.metric("Account",f"${acct['balance']:,.0f}",f"{acct['risk_pct']:.1f}% risk")

        # ── Auto-load from cache when no result in session ────────────────────────
        if ticker and "analysis_result" not in st.session_state:
            b_c,s_c,n_c,ts_c=_load_analysis_cache(ticker,tf)
            if b_c:
                st.session_state["analysis_result"]=(b_c,s_c,ticker,aname)
                st.session_state["_cache_ts"]=ts_c

        ab1,ab2=st.columns([2,8])
        with ab1:
            run_clicked=st.button("🔍  Analyze",type="primary",disabled=not ticker,
                                   use_container_width=True)
        with ab2:
            cache_ts=st.session_state.get("_cache_ts","")
            if cache_ts:
                try:
                    age=(datetime.now()-datetime.fromisoformat(cache_ts)).days
                    st.caption(f"Showing cached result · {age}d ago — click Analyze to refresh")
                except Exception: pass

        if run_clicked:
            from core.demo_limit import try_consume as _dl_try, DEMO_LIMIT_MESSAGE as _dl_msg
            if not _dl_try(1):
                st.error(_dl_msg)
                st.stop()
            with st.spinner(f"Analyzing {aname} ({ticker}) on {tf}..."):
                try:
                    from p1_analysis_engine.engine import analyze_full
                    bias,setups=analyze_full(ticker,model=model,interval=tf)
                    hist=st.session_state.get("analysis_history",[])
                    hist.insert(0,{"bias":bias,"setups":setups,"ticker":ticker,"name":aname,"tf":tf,
                                    "ts":datetime.now().strftime("%H:%M")})
                    st.session_state["analysis_history"]=hist[:5]
                    st.session_state["analysis_result"]=(bias,setups,ticker,aname)
                    st.session_state.pop("_cache_ts",None)
                    _save_analysis_cache(ticker,tf,bias,setups,aname)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.session_state.pop("analysis_result",None)

        if "analysis_result" not in st.session_state:
            # ── Analysis History backlog ──────────────────────────────────────────
            log=_load_analysis_log()
            if log:
                st.markdown("### Analysis History")
                st.caption(f"{len(log)} saved analyses — click any card to reload it")
                by_date={}
                for entry in log:
                    ts_str=entry.get("timestamp","")
                    try: d=datetime.fromisoformat(ts_str).strftime("%Y-%m-%d")
                    except Exception: d="Unknown"
                    by_date.setdefault(d,[]).append(entry)
                for day,entries in by_date.items():
                    st.markdown(f'<div style="font-size:.78rem;color:#5A6E85;font-weight:600;'
                                f'margin:.5rem 0 .2rem">📅 {day}</div>',unsafe_allow_html=True)
                    cols=st.columns(min(len(entries),4))
                    for ci,entry in enumerate(entries):
                        b=entry.get("bias",{}); s=entry.get("setups",{})
                        direction=b.get("directional_bias","?")
                        conf=b.get("confidence_score",0)
                        n_setups=len(s.get("setups",[]))
                        aname_e=entry.get("aname",entry.get("ticker","?"))
                        interval_e=entry.get("interval","?")
                        try: ts_short=datetime.fromisoformat(entry.get("timestamp","")).strftime("%H:%M")
                        except Exception: ts_short=""
                        dir_color={"bullish":"#10B981","bearish":"#EF4444","neutral":"#FBBF24"}.get(direction,"#94A3B8")
                        with cols[ci % 4]:
                            if st.button(
                                f"{aname_e} · {interval_e}\n{direction.upper()} {conf:.0%} · {n_setups} setups · {ts_short}",
                                key=f"log_{day}_{ci}",use_container_width=True):
                                try:
                                    from p1_analysis_engine.schema import BiasOutput,SetupsOutput
                                    bias_r=BiasOutput.model_validate(b)
                                    setups_r=SetupsOutput.model_validate(s)
                                    st.session_state["analysis_result"]=(bias_r,setups_r,
                                        entry.get("ticker",aname_e),aname_e)
                                    st.session_state["_cache_ts"]=entry.get("timestamp","")
                                    st.rerun()
                                except Exception as ex:
                                    st.error(f"Could not reload: {ex}")
            else:
                st.markdown('<div style="text-align:center;padding:3rem;color:#5A6E85">'
                            '<div style="font-size:3rem">📊</div>'
                            '<div style="font-size:1.1rem;font-weight:600;margin:.5rem 0">Select an asset above</div>'
                            '<div>Run your first analysis — it will be saved to history automatically</div></div>',
                            unsafe_allow_html=True)
            return

        bias,setups,res_ticker,res_name=st.session_state["analysis_result"]
        b_dir=bias.directional_bias; b_conf=bias.confidence_score
        ts=getattr(bias,"analysis_timestamp","")

        col_h,col_c,col_x=st.columns([7,2,1])
        with col_h:
            st.markdown(
                f'<h2>{res_name} ({res_ticker}) &nbsp;'
                f'<span class="orca-badge b-{b_dir}">{b_dir.upper()}</span>'
                f'&nbsp;<span style="color:#5A6E85;font-size:.9rem">{b_conf:.0%} confidence · {ts[:16].replace("T"," ")}</span></h2>',
                unsafe_allow_html=True)
            if getattr(bias,"bias_narrative",None): st.caption(bias.bias_narrative)
        with col_c:
            lines=[f"ORCASTRADING ANALYSIS — {res_name} ({res_ticker}) — {ts[:16]}",
                   "="*60,
                   f"Bias: {b_dir.upper()}  |  Confidence: {b_conf:.0%}",
                   ""]
            if getattr(bias,"bias_narrative",None): lines+=[bias.bias_narrative,""]
            if setups and setups.setups:
                lines.append("TRADE SETUPS")
                lines.append("-"*40)
                for i,sp in enumerate(setups.setups,1):
                    lines.append(f"{i}. {sp.name} [{sp.direction.upper()}] — {sp.priority.upper()}")
                    lines.append(f"   Entry: {sp.entry_low:,.2f} – {sp.entry_high:,.2f}")
                    lines.append(f"   Stop: {sp.stop_loss:,.2f}  |  R/R: {sp.rr_ratio:.2f}x")
                    tgts=", ".join(f"{t.label}: {t.price:,.2f}" for t in (sp.targets or []))
                    if tgts: lines.append(f"   Targets: {tgts}")
                    lines.append(f"   Rationale: {sp.rationale}")
                    lines.append("")
            share_text="\n".join(lines)
            st.download_button("⬇ Export",share_text,
                file_name=f"{res_name.replace(' ','_')}_{ts[:10]}.txt",
                mime="text/plain",use_container_width=True,
                help="Download analysis as plain text — paste into chat or email to share")
        with col_x:
            if st.button("✕",help="Clear result"):
                del st.session_state["analysis_result"]; st.rerun()

        tab_s,tab_t,tab_m,tab_kl,tab_dt,tab_n=st.tabs(
            ["Trade Setups","Technical","Macro & Geo","Key Levels","Decision Tree","News"])

        with tab_s:
            if not setups or not setups.setups:
                st.info("No trade setups generated.")
            else:
                _ts_label=""
                _cache_ts_val=st.session_state.get("_cache_ts","") or ts
                if _cache_ts_val:
                    try: _ts_label=datetime.fromisoformat(_cache_ts_val).strftime("%Y-%m-%d %H:%M UTC")
                    except Exception: _ts_label=_cache_ts_val[:16].replace("T"," ")+" UTC"
                if _ts_label: st.caption(f"Analysis generated: {_ts_label}")
                dollar_risk=acct["balance"]*acct["risk_pct"]/100
                for setup in setups.setups:
                    pri_c={"primary":"#10B981","secondary":"#60A5FA","conditional":"#FBBF24"}.get(setup.priority,"#94A3B8")
                    pt_risk=abs(setup.entry_high-setup.stop_loss) if setup.entry_high and setup.stop_loss else None
                    pos_sz=round(dollar_risk/pt_risk,4) if pt_risk and pt_risk>0 else None
                    with st.expander(
                        f"{'📈' if setup.direction=='long' else '📉'}  {setup.name} — "
                        f"{getattr(setup,'label','')}  [{setup.trade_type.upper()}]",
                        expanded=(setup.priority=="primary")):
                        mc1,mc2,mc3,mc4=st.columns(4)
                        mc1.metric("Direction",setup.direction.upper())
                        mc2.metric("R/R",f"{setup.rr_ratio:.2f}x")
                        mc3.metric("Confidence",f"{setup.confidence:.0%}")
                        mc4.metric("Priority",setup.priority.title())
                        t1,t2=st.columns(2)
                        with t1:
                            st.markdown(f"**Entry:** `{setup.entry_low:,.2f}` – `{setup.entry_high:,.2f}`")
                            st.markdown(f"**Stop loss:** `{setup.stop_loss:,.2f}`")
                            if pt_risk: st.markdown(f"**Point risk:** `{pt_risk:,.2f}`")
                            for tgt in (setup.targets or []):
                                st.markdown(f"**{tgt.label}:** `{tgt.price:,.2f}` ({tgt.allocation_pct}%)")
                        with t2:
                            if pos_sz is not None:
                                st.markdown(f"**Position size:** `{pos_sz:.4f}` units  \n"
                                            f"Risk $: `${dollar_risk:.2f}` | "
                                            f"Win rate est: `{setup.win_rate_estimate:.0%}`")
                            st.markdown(f"**Trigger:** {setup.trigger}")
                        st.markdown(f"**Rationale:** {setup.rationale}")
                        inv=getattr(setup,"invalidation_scenario",None)
                        if inv: st.markdown(f"**Invalidated if:** {getattr(inv,'description',str(inv))}")
                        if st.button("📋 Plan This Trade",key=f"plan_{setup.name}"):
                            st.session_state["planner_prefill"]={
                                "asset":res_name,"ticker":res_ticker,
                                "direction":setup.direction,
                                "entry_low":setup.entry_low,"entry_high":setup.entry_high,
                                "stop_loss":setup.stop_loss,
                                "tp1":setup.targets[0].price if setup.targets else None,
                                "tp1_alloc":setup.targets[0].allocation_pct if setup.targets else 70,
                                "tp2":setup.targets[1].price if len(getattr(setup,"targets",[]))>1 else None,
                                "tp2_alloc":setup.targets[1].allocation_pct if len(getattr(setup,"targets",[]))>1 else 30,
                                "confidence":setup.confidence,
                                "rationale":setup.rationale,
                                "trade_type":setup.trade_type,
                            }
                            st.session_state["page"]="Journal"
                            st.rerun()

        with tab_t:
            tech=bias.technical
            tc1,tc2,tc3=st.columns(3)
            tc1.metric("Price",f"{tech.current_price:,.2f}")
            tc2.metric("RSI(14)",f"{tech.rsi_14:.1f}")
            tc3.metric("ADX(14)",f"{tech.adx_14:.1f}")
            tc4,tc5,tc6=st.columns(3)
            tc4.metric("Trend",tech.trend.upper())
            tc5.metric("EMA Align",tech.ema_alignment.upper())
            tc6.metric("MACD",tech.macd_signal.upper())
            tc7,tc8,tc9=st.columns(3)
            tc7.metric("ATR(14)",f"{tech.atr_14:.2f} ({tech.atr_pct:.1f}%)")
            tc8.metric("Stochastic",f"{tech.stoch_signal.upper()} ({tech.stoch_k:.0f}/{tech.stoch_d:.0f})")
            tc9.metric("CMF",f"{tech.cmf_20:.3f} {tech.cmf_signal.upper()}")
            if tech.ema20 or tech.ema50 or tech.ema200:
                ec1,ec2,ec3=st.columns(3)
                if tech.ema20: ec1.metric("EMA 20",f"{tech.ema20:,.2f}")
                if tech.ema50: ec2.metric("EMA 50",f"{tech.ema50:,.2f}")
                if tech.ema200: ec3.metric("EMA 200",f"{tech.ema200:,.2f}")
            if tech.bb_upper and tech.bb_lower:
                bc1,bc2,bc3=st.columns(3)
                bc1.metric("BB Upper",f"{tech.bb_upper:,.2f}")
                bc2.metric("BB Lower",f"{tech.bb_lower:,.2f}")
                bc3.metric("BB Position",tech.bb_position.upper())

        with tab_m:
            macro=bias.macro
            mc1,mc2,mc3=st.columns(3)
            mc1.metric("VIX",f"{macro.vix_level:.1f}" if macro.vix_level else "N/A",macro.vix_regime.upper())
            mc2.metric("Macro Bias",macro.macro_bias.upper().replace("_"," "))
            mc3.metric("USD Trend",macro.usd_trend.upper())
            mc4,mc5,mc6=st.columns(3)
            mc4.metric("Fed Funds",f"{macro.fed_funds_rate:.2f}%" if macro.fed_funds_rate else "N/A")
            mc5.metric("Yield Curve",macro.yield_curve.upper())
            mc6.metric("CPI",macro.cpi_trend.upper())
            geo=bias.geopolitical
            st.markdown(f"**Geopolitical Risk:** `{geo.risk_level.upper()}`")
            for f_ in (geo.key_factors or [])[:4]: st.markdown(f"- {f_}")
            if bias.data_quality_flags:
                with st.expander("Data quality flags"):
                    for fl in bias.data_quality_flags: st.caption(f"⚠ {fl}")

        with tab_kl:
            sup=getattr(bias,"key_support_levels",[])
            res2=getattr(bias,"key_resistance_levels",[])
            kc1,kc2=st.columns(2)
            with kc1:
                st.markdown("**Support**")
                if sup:
                    st.dataframe(pd.DataFrame([{"Price":f"{l.price:,.2f}","Label":l.label,
                                 "Strength":l.strength.upper()} for l in sup]),
                                 use_container_width=True,hide_index=True)
            with kc2:
                st.markdown("**Resistance**")
                if res2:
                    st.dataframe(pd.DataFrame([{"Price":f"{l.price:,.2f}","Label":l.label,
                                 "Strength":l.strength.upper()} for l in res2]),
                                 use_container_width=True,hide_index=True)

        with tab_dt:
            dt=getattr(setups,"decision_tree",None) if setups else None
            if dt:
                entries=getattr(dt,"entries",None) or (dt if isinstance(dt,list) else [])
                if entries:
                    rows=[{"Scenario":getattr(e,"scenario",""),
                           "Action":getattr(e,"outcome",""),
                           "Setup":getattr(e,"setup_name","")} for e in entries]
                    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                else:
                    st.code(str(dt),language=None)
            else:
                st.info("Decision tree not available for this analysis.")

        with tab_n:
            news_items=getattr(bias,"recent_news",None) or getattr(bias,"news",[])
            if news_items:
                for item in news_items[:10]:
                    t_=getattr(item,"title",str(item))
                    src=getattr(item,"source","")
                    ts_=getattr(item,"published_at","")[:10]
                    sent=getattr(item,"sentiment_hint",None)
                    sc={"positive":"#10B981","negative":"#EF4444","neutral":"#94A3B8"}.get(sent or "neutral","#94A3B8")
                    st.markdown(
                        f'<div style="padding:.5rem .75rem;background:#101826;border:1px solid #1A2840;'
                        f'border-radius:8px;margin-bottom:.35rem">'
                        f'<span style="color:{sc};font-size:.7rem;font-weight:700">'
                        f'{(sent or "neutral").upper()}</span> &nbsp;'
                        f'<span style="color:#E2E8F0">{t_}</span>'
                        f'<span style="color:#5A6E85;font-size:.75rem;float:right">{src} · {ts_}</span>'
                        f'</div>',unsafe_allow_html=True)
            else:
                st.info("No news items in this analysis result.")



# =============================================================================
# PAGE 4: STRATEGIES
# =============================================================================
def page_strategies():
    st.title("Strategies")
    strategies=_strategies(); scanner_assets=_scanner_assets()
    yaml_path=ROOT/"config"/"strategies.yaml"

    t_ov,t_edit,t_bt,t_cbt,t_ft,t_cmp=st.tabs(
        ["Overview","Add / Edit","Backtest","Custom Backtest","Forward Test","Comparison"])

    # ── Overview ──────────────────────────────────────────────────────────────
    with t_ov:
        # Figure out which assets use each strategy
        try:
            from core.config import get_all_assets as _gaa_s
            _asset_strat_map: dict[str,list[str]] = {}
            for _a in _gaa_s():
                if not _a.get("enabled", True): continue
                for _e in _a.get("strategies", []):
                    _sid = _e if isinstance(_e, str) else _e.get("id","")
                    _asset_strat_map.setdefault(_sid, []).append(_a.get("label", _a["id"]))
        except Exception:
            _asset_strat_map = {}

        try:
            from p4_live.report import compute_forward_stats as _cfs
        except Exception:
            _cfs = None

        cols = st.columns(2)
        for i, (sid, cfg) in enumerate(strategies.items()):
            tf   = cfg.get("default_timeframe", "?")
            name = cfg.get("name", sid)
            desc = cfg.get("description", "")
            exp  = cfg.get("expiry_days", "—")
            lb   = cfg.get("lookback_days", "—")
            used_by = _asset_strat_map.get(sid, [])
            is_active = bool(used_by)

            # Live stats
            n = 0; live_wr = "—"; live_pf = "—"; live_r = None
            if _cfs:
                try:
                    stats   = _cfs(ticker=None, strategy=sid)
                    n       = stats["n_closed"]
                    live_wr = f"{stats['win_rate']:.0%}" if n > 0 else "—"
                    live_pf = f"{stats['profit_factor']:.2f}" if n > 0 else "—"
                    live_r  = stats.get("total_r")
                except Exception:
                    pass

            status_color = "#10B981" if is_active else "#5A6E85"
            status_text  = "ACTIVE" if is_active else "INACTIVE"
            pf_color     = "#10B981" if (live_pf not in ("—","0.00") and float(live_pf.replace("—","0") or 0) >= 1) else "#EF4444"

            with cols[i % 2]:
                st.markdown(
                    f'<div style="background:#0D1824;border:1px solid #1A2840;border-radius:12px;'
                    f'padding:1rem 1.1rem;margin-bottom:.7rem">'

                    # Header row
                    f'<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.4rem">'
                    f'<div>'
                    f'<span style="font-size:1.05rem;font-weight:700;color:#F1F5F9">{name}</span>'
                    f'<span style="margin-left:.6rem;background:#1A2840;color:#60A5FA;'
                    f'font-size:.7rem;font-weight:600;padding:2px 7px;border-radius:4px">{tf}</span>'
                    f'</div>'
                    f'<span style="font-size:.65rem;font-weight:700;color:{status_color};'
                    f'background:{status_color}22;padding:2px 8px;border-radius:4px">{status_text}</span>'
                    f'</div>'

                    # Description
                    f'<div style="color:#64748B;font-size:.78rem;margin-bottom:.6rem;'
                    f'line-height:1.4">{desc}</div>'

                    # Stats row
                    f'<div style="display:flex;gap:1.2rem;margin-bottom:.5rem">'
                    f'<div><div style="font-size:.62rem;color:#5A6E85;text-transform:uppercase">Win Rate</div>'
                    f'<div style="font-size:.9rem;font-weight:700;color:#F1F5F9">{live_wr}</div></div>'
                    f'<div><div style="font-size:.62rem;color:#5A6E85;text-transform:uppercase">Profit Factor</div>'
                    f'<div style="font-size:.9rem;font-weight:700;color:{pf_color}">{live_pf}</div></div>'
                    f'<div><div style="font-size:.62rem;color:#5A6E85;text-transform:uppercase">Trades</div>'
                    f'<div style="font-size:.9rem;font-weight:700;color:#F1F5F9">{n}</div></div>'
                    f'<div><div style="font-size:.62rem;color:#5A6E85;text-transform:uppercase">Expiry</div>'
                    f'<div style="font-size:.9rem;font-weight:700;color:#F1F5F9">{exp}d</div></div>'
                    f'</div>'

                    # Active on markets
                    + (
                        f'<div style="font-size:.7rem;color:#5A6E85">Active on: '
                        + ", ".join(f'<span style="color:#A78BFA">{m}</span>' for m in used_by)
                        + '</div>'
                        if used_by else
                        '<div style="font-size:.7rem;color:#5A6E85;font-style:italic">Not assigned to any active market</div>'
                    )
                    + f'</div>', unsafe_allow_html=True)

        # ── Strategy Health tiles ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Strategy Health** *(from live journal — closed trades only)*")
        try:
            _all_closed = [t for t in _trades() if t["status"] not in ("pending","filled","expired")]
        except Exception:
            _all_closed = []
        _TF_ORDER_SH = ["1m","5m","15m","30m","1h","4h","8h","1d","1wk"]
        _TF_COLORS_SH = {"1m":"#F87171","5m":"#FBBF24","15m":"#FB923C","30m":"#E879F9",
                         "1h":"#34D399","4h":"#A78BFA","8h":"#818CF8","1d":"#60A5FA","1wk":"#93C5FD"}
        sh_strats = list(strategies.keys())
        if sh_strats:
            sh_cols = st.columns(max(len(sh_strats),1))
            for col, strat_id in zip(sh_cols, sh_strats):
                strat_label = strategies[strat_id].get("name", _fmt_strategy(strat_id))
                strat_trades = [t for t in _all_closed if t.get("strategy")==strat_id]
                n = len(strat_trades)
                wr_s = sum(1 for t in strat_trades if t["status"] in ("win","partial_win"))/n if n else 0
                pnl_s = sum(t["pnl_r"] for t in strat_trades if t.get("pnl_r") is not None)
                pnl_color = "#10B981" if pnl_s>=0 else "#EF4444"
                tf_rows=""
                for tf in _TF_ORDER_SH:
                    tf_t=[t for t in strat_trades if t.get("timeframe")==tf]
                    if not tf_t: continue
                    tf_n=len(tf_t)
                    tf_wr=sum(1 for t in tf_t if t["status"] in ("win","partial_win"))/tf_n
                    tf_r=sum(t["pnl_r"] for t in tf_t if t.get("pnl_r") is not None)
                    tf_color="#10B981" if tf_r>=0 else "#EF4444"
                    tf_c=_TF_COLORS_SH.get(tf,"#94A3B8")
                    tf_rows+=(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:1px 0;border-top:1px solid #1A2840;margin-top:2px">'
                        f'<span style="color:{tf_c};font-size:.62rem;font-weight:600">{tf}</span>'
                        f'<span style="color:{tf_color};font-size:.62rem;font-weight:700">{tf_r:+.1f}R</span>'
                        f'<span style="color:#5A6E85;font-size:.58rem">{tf_wr:.0%} · {tf_n}</span>'
                        f'</div>'
                    )
                col.markdown(
                    f'<div style="padding:.45rem .65rem;background:#101826;border:1px solid #1A2840;border-radius:8px">'
                    f'<div style="font-size:.72rem;font-weight:600;color:#94A3B8;margin-bottom:2px">{strat_label}</div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
                    f'<span style="font-size:.95rem;font-weight:700;color:{pnl_color}">{pnl_s:+.1f}R</span>'
                    f'<span style="font-size:.65rem;color:#5A6E85">{wr_s:.0%} WR · {n}</span>'
                    f'</div>{tf_rows}</div>', unsafe_allow_html=True)
        else:
            st.info("No strategies configured.")

    # ── Add / Edit ─────────────────────────────────────────────────────────────
    with t_edit:
        mode=st.radio("Mode",["Edit existing","Add new"],horizontal=True)
        if mode=="Edit existing":
            sel_sid=st.selectbox("Strategy",list(strategies.keys()),
                                 format_func=lambda s:strategies[s].get("name",s))
            cfg=strategies[sel_sid]
            with st.form(f"edit_{sel_sid}"):
                new_name=st.text_input("Name",value=cfg.get("name",sel_sid))
                new_desc=st.text_area("Description",value=cfg.get("description",""),height=60)
                c1e,c2e=st.columns(2)
                new_exp=c1e.number_input("Expiry days",value=int(cfg.get("expiry_days",5)),min_value=1,max_value=30)
                new_lb=c2e.number_input("Lookback days",value=int(cfg.get("lookback_days",60)),min_value=5)
                new_cond=st.text_area("Signal conditions",value=cfg.get("signal_conditions",""),height=120)
                st.markdown("**Parameters:**")
                params=cfg.get("params",{}); new_params={}
                pcols=st.columns(3)
                for j,(k,v) in enumerate(params.items()):
                    col=pcols[j%3]
                    if isinstance(v,bool):   new_params[k]=col.checkbox(k,value=v,key=f"e_{sel_sid}_{k}")
                    elif isinstance(v,int):  new_params[k]=int(col.number_input(k,value=v,step=1,key=f"e_{sel_sid}_{k}"))
                    elif isinstance(v,float):new_params[k]=col.number_input(k,value=v,step=0.01,format="%.4f",key=f"e_{sel_sid}_{k}")
                    else:                    new_params[k]=col.text_input(k,value=str(v),key=f"e_{sel_sid}_{k}")
                bc1,bc2=st.columns(2)
                if bc1.form_submit_button("Save Changes",type="primary"):
                    d=_read_yaml(yaml_path); d["strategies"][sel_sid].update(
                        {"name":new_name,"description":new_desc,"expiry_days":new_exp,
                         "lookback_days":new_lb,"signal_conditions":new_cond,"params":new_params})
                    _write_yaml(yaml_path,d); _clear_config()
                    st.success("Saved!"); st.rerun()
                if bc2.form_submit_button("Delete Strategy",type="secondary"):
                    st.session_state[f"confirm_del_{sel_sid}"]=True
            if st.session_state.get(f"confirm_del_{sel_sid}"):
                st.warning(f"Delete `{sel_sid}`? Existing trades are preserved.")
                if st.button("Yes, delete",type="primary"):
                    d=_read_yaml(yaml_path); d["strategies"].pop(sel_sid,None)
                    _write_yaml(yaml_path,d); _clear_config()
                    st.session_state.pop(f"confirm_del_{sel_sid}",None)
                    st.success("Deleted."); st.rerun()
        else:
            with st.form("add_strat"):
                na1,na2=st.columns(2)
                ns_id=na1.text_input("Strategy ID (snake_case)"); ns_name=na2.text_input("Display name")
                ns_desc=st.text_area("Description",height=60)
                nb1,nb2,nb3=st.columns(3)
                ns_tf=nb1.selectbox("Timeframe",["1d","1h","15m","1wk"])
                ns_lb=nb2.number_input("Lookback days",value=60,min_value=5)
                ns_exp=nb3.number_input("Expiry days",value=5,min_value=1)
                ns_cond=st.text_area("Signal conditions",height=100)
                ns_params=st.text_area("Parameters (one per line: key=value)",height=80)
                if st.form_submit_button("Add Strategy",type="primary"):
                    if not ns_id or not ns_name: st.error("ID and name required.")
                    elif ns_id in strategies: st.error("ID already exists.")
                    else:
                        pd2={}
                        for line in (ns_params or "").splitlines():
                            if "=" in line:
                                k,v=line.split("=",1); k,v=k.strip(),v.strip()
                                try: pd2[k]=int(v)
                                except ValueError:
                                    try: pd2[k]=float(v)
                                    except ValueError:
                                        pd2[k]=v.lower()=="true" if v.lower() in ("true","false") else v
                        d=_read_yaml(yaml_path)
                        d["strategies"][ns_id]={"name":ns_name,"description":ns_desc,
                            "default_timeframe":ns_tf,"lookback_days":int(ns_lb),"expiry_days":int(ns_exp),
                            "signal_conditions":ns_cond,"params":pd2}
                        _write_yaml(yaml_path,d); _clear_config()
                        st.success(f"Added '{ns_id}'."); st.rerun()

    # ── Backtest ───────────────────────────────────────────────────────────────
    with t_bt:
        st.markdown("Configure a backtest. Pass 1 (signal generation via Claude) takes time for large date ranges.")
        bc1,bc2,bc3=st.columns(3)
        _bt_assets=[(a.get("label",""),a.get("tickers",{}).get("yfinance","")) for a in _assets() if a.get("tickers",{}).get("yfinance")]
        _bt_labels=[f"{lbl} ({tick})" for lbl,tick in _bt_assets]
        _bt_idx=bc1.selectbox("Asset",range(len(_bt_labels)),format_func=lambda i:_bt_labels[i])
        bt_ticker=_bt_assets[_bt_idx][1] if _bt_assets else ""
        bt_strat=bc2.selectbox("Strategy",list(strategies.keys()))
        # Auto-select default_timeframe for the chosen strategy; user can override freely
        _all_intervals=["1m","5m","15m","30m","1h","4h","8h","1d","1wk"]
        _strat_default=strategies.get(bt_strat,{}).get("default_timeframe","1d")
        _def_idx=_all_intervals.index(_strat_default) if _strat_default in _all_intervals else 6
        bt_int=bc3.selectbox("Interval",_all_intervals,index=_def_idx)
        bd1,bd2,bd3=st.columns(3)
        bt_start=bd1.date_input("Start",value=date.today()-timedelta(days=365))
        bt_end=bd2.date_input("End",value=date.today())
        bt_wf=bd3.checkbox("Walk-forward",value=True)
        with st.expander("CLI command"):
            cmd=(f"python -m p3_backtester {bt_ticker} --interval {bt_int} "
                 f"--strategy {bt_strat} "
                 f"--start {bt_start} --end {bt_end} {'--walk-forward' if bt_wf else ''}")
            st.code(cmd,language="bash")
        if st.button("Run Backtest",type="primary",use_container_width=True):
            # Clear previous results so stale data is never shown
            st.session_state.pop("bt_result",None)
            with st.spinner("Running backtest (may take minutes for large date ranges)..."):
                import subprocess
                r=subprocess.run(
                    ["python","-m","p3_backtester",bt_ticker,"--interval",bt_int,
                     "--strategy",bt_strat,
                     "--start",str(bt_start),"--end",str(bt_end),"--json"]+
                    (["--walk-forward"] if bt_wf else []),
                    cwd=str(ROOT),capture_output=True,text=True,timeout=600)
            if r.returncode!=0:
                st.error("Backtest failed.")
                if r.stderr: st.code(r.stderr[-2000:],language=None)
            else:
                # Parse JSON from stdout
                import re as _re
                js=None
                m=_re.search(r'\{[\s\S]+\}',r.stdout)
                if m:
                    try: js=json.loads(m.group())
                    except Exception: pass
                if js:
                    st.session_state["bt_result"]=js
                    st.session_state["bt_ticker"]=bt_ticker
                    st.session_state["bt_interval"]=bt_int
                    st.session_state["bt_strat"]=bt_strat
                    st.session_state["bt_start"]=str(bt_start)
                    st.session_state["bt_end"]=str(bt_end)
                else:
                    st.warning("Could not parse structured results.")
                    st.code(r.stdout[-3000:],language=None)

        # ── Render results ────────────────────────────────────────────────────
        js=st.session_state.get("bt_result")
        if js:
            st.markdown("---")
            ticker_h=st.session_state.get("bt_ticker","")
            intv_h=st.session_state.get("bt_interval","")
            strat_h=st.session_state.get("bt_strat","")
            st.markdown(f"### Results — {ticker_h}  `{intv_h}`  `{strat_h}`  "
                        f"{st.session_state.get('bt_start','')} → {st.session_state.get('bt_end','')}")

            # ── KPI row ───────────────────────────────────────────────────────
            k1,k2,k3,k4,k5,k6=st.columns(6)
            wr=js.get("actual_win_rate",0)
            avgr=js.get("actual_avg_pnl_r",0)
            pf=js.get("actual_profit_factor",0)
            mdd=js.get("max_drawdown_r",0)
            sharpe=js.get("sharpe_r",0)
            kelly=js.get("kelly_25pct",0)
            fills=js.get("total_trades_filled",0)
            k1.metric("Win Rate",f"{wr:.0%}",delta=f"{wr-0.5:.0%} vs 50%",
                      delta_color="normal" if wr>=0.5 else "inverse")
            k2.metric("Avg R",f"{avgr:+.3f}R",delta="positive" if avgr>0 else None,
                      delta_color="normal" if avgr>0 else "inverse")
            k3.metric("Profit Factor",f"{pf:.2f}",
                      delta="good" if pf>1.5 else None,
                      delta_color="normal" if pf>1.5 else "inverse")
            k4.metric("Max Drawdown",f"-{mdd:.1f}R")
            k5.metric("Sharpe (R)",f"{sharpe:.2f}")
            k6.metric("Kelly 25%",f"{kelly:.1%}",help="Suggested fraction of account per trade")

            if fills<30:
                st.warning(f"Only {fills} filled trades — results have high variance. Need ≥ 30 for reliability.")

            # ── Charts ────────────────────────────────────────────────────────
            # Try load trades from DB for equity curve
            safe=ticker_h.replace("=","").replace("-","").replace("^","")
            bt_s=st.session_state.get("bt_start",""); bt_e=st.session_state.get("bt_end","")
            db_path=ROOT/"p3_backtester"/"results"/f"backtest_{safe}_{intv_h}_{bt_s}_{bt_e}.db"
            ch1,ch2=st.columns([2,1])
            with ch1:
                if db_path.exists():
                    import sqlite3 as _sq3
                    dbc=_sq3.connect(str(db_path))
                    dbc.row_factory=_sq3.Row
                    db_trades=dbc.execute(
                        "SELECT signal_bar_time,outcome,pnl_r FROM trades "
                        "WHERE outcome NOT IN ('EXPIRED','PENDING') ORDER BY fill_bar_index"
                    ).fetchall()
                    dbc.close()
                    if db_trades:
                        dates=[t["signal_bar_time"][:10] for t in db_trades]
                        pnls=[t["pnl_r"] or 0 for t in db_trades]
                        cum=[sum(pnls[:i+1]) for i in range(len(pnls))]
                        colors=["#10B981" if p>=0 else "#EF4444" for p in pnls]
                        fig=go.Figure()
                        fig.add_trace(go.Scatter(x=dates,y=cum,mode="lines+markers",
                            line=dict(color="#3B82F6",width=2),
                            marker=dict(color=colors,size=8),
                            hovertemplate="<b>%{x}</b><br>Cumulative: %{y:+.2f}R<extra></extra>"))
                        fig.add_hline(y=0,line_color="#1A2840",line_dash="dot")
                        fig.update_layout(**_PL,title="Equity Curve (R)",height=280)
                        st.plotly_chart(fig,use_container_width=True)
            with ch2:
                wins=sum(1 for t in (db_trades if db_path.exists() and db_trades else []) if (t["pnl_r"] or 0)>0)
                losses=sum(1 for t in (db_trades if db_path.exists() and db_trades else []) if (t["pnl_r"] or 0)<0)
                be=fills-wins-losses
                if wins+losses>0:
                    fig2=go.Figure(go.Pie(
                        labels=["Win","Loss","Breakeven"],values=[wins,losses,max(0,be)],
                        marker_colors=["#10B981","#EF4444","#94A3B8"],
                        hole=0.55,textinfo="label+percent",
                        hovertemplate="%{label}: %{value}<extra></extra>"))
                    fig2.update_layout(**_PL,title="Outcome Distribution",height=280,
                        showlegend=False)
                    st.plotly_chart(fig2,use_container_width=True)

            # ── Net vs Gross ──────────────────────────────────────────────────
            st.markdown("**Performance**")
            pg1,pg2,pg3,pg4=st.columns(4)
            pg1.metric("Gross Avg R",f"{avgr:+.3f}R")
            pg2.metric("Net Avg R",f"{js.get('actual_avg_net_pnl_r',0):+.3f}R",
                       help="After estimated transaction costs")
            pg3.metric("Net PF",f"{js.get('actual_net_profit_factor',0):.2f}")
            pg4.metric("Calmar",f"{js.get('calmar_ratio',0):.2f}",
                       help="Avg R / Max Drawdown")

            st.markdown(f"**Streaks** — Max win streak: **{js.get('max_win_streak',0)}**  |  "
                        f"Max loss streak: **{js.get('max_loss_streak',0)}**  |  "
                        f"Max DD duration: **{js.get('max_drawdown_duration',0)} bars**")

            # ── Walk-forward ──────────────────────────────────────────────────
            wf=js.get("walk_forward",[])
            if wf:
                st.markdown("**Walk-Forward Validation**")
                wf_rows=[]
                for w in wf:
                    wf_rows.append({
                        "Window":w.get("window",""),
                        "Period":f"{w.get('train_start','')[:7]} → {w.get('test_end','')[:7]}",
                        "Train fills":w.get("train_fills",0),
                        "Train WR":f"{w.get('train_win_rate',0):.0%}",
                        "Test fills":w.get("test_fills",0),
                        "Test WR":f"{w.get('test_win_rate',0):.0%}",
                        "Test Avg R":f"{w.get('test_avg_r',0):+.3f}R",
                        "Test PF":f"{w.get('test_pf',0):.2f}",
                    })
                st.dataframe(pd.DataFrame(wf_rows),use_container_width=True,hide_index=True)

            # ── By direction / type ───────────────────────────────────────────
            with st.expander("Breakdown by Direction & Trade Type"):
                dir_rows=[]
                for d,v in js.get("by_direction",{}).items():
                    if v.get("n_trades",0)>0:
                        dir_rows.append({"Direction":d.upper(),"Trades":v["n_trades"],
                            "Win %":f"{v['win_rate']:.0%}","Avg R":f"{v['avg_pnl_r']:+.3f}R",
                            "PF":f"{v['profit_factor']:.2f}"})
                if dir_rows:
                    st.markdown("**By Direction**")
                    st.dataframe(pd.DataFrame(dir_rows),use_container_width=True,hide_index=True)
                type_rows=[]
                for tt,v in js.get("by_trade_type",{}).items():
                    if v.get("n_trades",0)>0:
                        type_rows.append({"Type":tt.upper(),"Trades":v["n_trades"],
                            "Win %":f"{v['win_rate']:.0%}","Avg R":f"{v['avg_pnl_r']:+.3f}R",
                            "PF":f"{v['profit_factor']:.2f}"})
                if type_rows:
                    st.markdown("**By Trade Type**")
                    st.dataframe(pd.DataFrame(type_rows),use_container_width=True,hide_index=True)

    # ── Custom Backtest ────────────────────────────────────────────────────────
    with t_cbt:
        st.markdown("Write your own entry conditions and backtest them instantly. No files to edit.")
        import tempfile as _tmpfile, subprocess as _sp, re as _re2

        cb1,cb2,cb3=st.columns(3)
        _all_tickers=[(a.get("label",""),a.get("tickers",{}).get("yfinance",""))
                      for a in _assets() if a.get("tickers",{}).get("yfinance")]
        _all_labels=[f"{lbl} ({tick})" for lbl,tick in _all_tickers]
        cbt_idx=cb1.selectbox("Asset",range(len(_all_labels)),format_func=lambda i:_all_labels[i],key="cbt_asset")
        cbt_ticker=_all_tickers[cbt_idx][1] if _all_tickers else ""
        cbt_label=_all_tickers[cbt_idx][0] if _all_tickers else ""
        _cbt_intervals=["1m","5m","15m","30m","1h","4h","8h","1d","1wk"]
        cbt_int=cb2.selectbox("Interval",_cbt_intervals,index=5,key="cbt_int")
        cbt_wf=cb3.checkbox("Walk-forward (60/20/20)",value=True,key="cbt_wf")

        cd1,cd2=st.columns(2)
        cbt_start=cd1.date_input("Start",value=date.today()-timedelta(days=365*5),key="cbt_start")
        cbt_end=cd2.date_input("End",value=date.today(),key="cbt_end")

        ct1,ct2=st.columns(2)
        cbt_tp1=ct1.slider("TP1 allocation %",10,100,70,10,key="cbt_tp1")
        cbt_tp2=100-cbt_tp1
        ct2.metric("TP2 allocation %",cbt_tp2)

        _default_logic='''\
import ta

def entry_logic(df, tech):
    """
    Return (direction, entry_low, entry_high, stop_loss, tp1, tp2)  or  None to skip.
    direction: "long" or "short"
    """
    price = tech["current_price"]
    atr   = tech["atr_14"]
    rsi   = tech["rsi_14"]
    adx   = tech["adx_14"]
    ema20 = tech["ema20"]
    ema50 = tech["ema50"]

    # ── Your conditions here ─────────────────────────────────────────────────
    if ema20 <= ema50: return None       # need uptrend
    if rsi > 65:       return None       # not overbought
    if adx < 20:       return None       # trend must have strength
    if abs(price - ema20) > 1.5 * atr:  return None   # price near EMA20

    entry_low  = ema20 - 0.3 * atr
    entry_high = ema20 + 0.2 * atr
    stop_loss  = ema20 - 1.5 * atr
    risk = entry_high - stop_loss
    if risk <= 0: return None

    tp1 = entry_high + 2.5 * risk
    tp2 = entry_high + 4.0 * risk
    return "long", entry_low, entry_high, stop_loss, tp1, tp2
'''
        # ── AI Strategy Generator ─────────────────────────────────────────────
        with st.expander("✨ Describe your strategy — AI will write the code", expanded=False):
            _ai_desc = st.text_area(
                "Strategy description",
                placeholder=(
                    "e.g. Go long when EMA20 is above EMA50, RSI is below 65, ADX is above 22, "
                    "and price has pulled back within 1.5 ATR of EMA20. "
                    "Stop loss 1.5 ATR below EMA20. Take 70% off at 2.5R and remainder at 4R."
                ),
                height=110,
                key="ai_strat_desc",
                label_visibility="collapsed",
            )
            _ai_col1, _ai_col2 = st.columns([3, 1])
            with _ai_col2:
                _ai_direction = st.selectbox("Direction", ["Long only", "Short only", "Both"], key="ai_direction")
            with _ai_col1:
                _ai_btn = st.button("Generate code", type="primary", use_container_width=True, key="ai_gen_btn")

            if _ai_btn:
                from core.demo_limit import try_consume as _dl_try, DEMO_LIMIT_MESSAGE as _dl_msg
                if not _ai_desc.strip():
                    st.error("Please describe your strategy first.")
                elif not os.getenv("ANTHROPIC_API_KEY"):
                    st.error("No ANTHROPIC_API_KEY. Go to Settings → API Keys.")
                elif not _dl_try(1):
                    st.error(_dl_msg)
                else:
                    _direction_hint = {
                        "Long only": "Generate LONG signals only — no short trades.",
                        "Short only": "Generate SHORT signals only — no long trades.",
                        "Both": "Generate both long and short signals as appropriate.",
                    }[_ai_direction]

                    _BUILDER_SYS = (
                        "Generate a Python entry_logic() function for a trading backtest.\n\n"
                        "SIGNATURE: def entry_logic(df, tech)\n\n"
                        "PARAMETERS:\n"
                        "  df   — pandas DataFrame, OHLCV columns (Open,High,Low,Close,Volume), DatetimeIndex\n"
                        "  tech — dict with pre-computed indicators:\n"
                        "         current_price, ema20, ema50, ema200, atr_14, adx_14, rsi_14,\n"
                        "         nearest_support, nearest_resistance, interval, ticker, atr_pct,\n"
                        "         stoch_k, stoch_d, volume_trend, trend, macd_signal\n\n"
                        "RETURN VALUE: (direction, entry_low, entry_high, stop_loss, tp1, tp2) or None\n"
                        "  direction : 'long' or 'short'\n"
                        "  For longs : stop_loss < entry_low < entry_high < tp1 < tp2\n"
                        "  For shorts: stop_loss > entry_high > entry_low > tp1 > tp2\n\n"
                        "RULES:\n"
                        "  1. Start with 'import ta' on line 1 (ta-lib is available)\n"
                        "  2. Return None to skip a bar — never force a signal\n"
                        "  3. Use atr_14 for stop/target sizing unless user specified otherwise\n"
                        "  4. Risk = entry_high - stop_loss (longs). Must be > 0\n"
                        "  5. Add a short comment above each condition explaining what it checks\n"
                        "  6. Output ONLY the raw Python code — no markdown, no explanations\n\n"
                        f"DIRECTION: {_direction_hint}\n\n"
                        "EXAMPLE:\n"
                        "import ta\n\n"
                        "def entry_logic(df, tech):\n"
                        "    price = tech['current_price']\n"
                        "    atr   = tech['atr_14']\n"
                        "    rsi   = tech['rsi_14']\n"
                        "    adx   = tech['adx_14']\n"
                        "    ema20 = tech['ema20']\n"
                        "    ema50 = tech['ema50']\n\n"
                        "    # Uptrend: fast EMA above slow EMA\n"
                        "    if ema20 <= ema50: return None\n"
                        "    # Momentum not overbought\n"
                        "    if rsi > 65: return None\n"
                        "    # Trend must have strength\n"
                        "    if adx < 22: return None\n"
                        "    # Price pulled back within 1.5 ATR of EMA20\n"
                        "    if abs(price - ema20) > 1.5 * atr: return None\n\n"
                        "    entry_low  = ema20 - 0.3 * atr\n"
                        "    entry_high = ema20 + 0.2 * atr\n"
                        "    stop_loss  = ema20 - 1.5 * atr\n"
                        "    risk = entry_high - stop_loss\n"
                        "    if risk <= 0: return None\n\n"
                        "    tp1 = entry_high + 2.5 * risk\n"
                        "    tp2 = entry_high + 4.0 * risk\n"
                        "    return 'long', entry_low, entry_high, stop_loss, tp1, tp2\n"
                    )

                    try:
                        import anthropic as _anth
                        _anth_client = _anth.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                        with st.spinner("Claude is writing your strategy…"):
                            _resp = _anth_client.messages.create(
                                model="claude-opus-4-6",
                                max_tokens=1500,
                                system=_BUILDER_SYS,
                                messages=[{"role": "user", "content": _ai_desc.strip()}],
                            )
                        _gen_code = _resp.content[0].text.strip()
                        # Strip accidental markdown fences
                        _gen_code = _re2.sub(r"^```python\s*", "", _gen_code)
                        _gen_code = _re2.sub(r"^```\s*", "", _gen_code)
                        _gen_code = _re2.sub(r"\s*```$", "", _gen_code).strip()
                        st.session_state["cbt_code"] = _gen_code
                        st.success("Code generated — review it in the editor below, then run the backtest.")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Generation failed: {_e}")

        cbt_code=st.text_area(
            "entry_logic() — edit your conditions below",
            value=st.session_state.get("cbt_code", _default_logic),
            height=340, key="cbt_code",
            help="Write Python. You have: df (OHLCV), tech (price/ATR/RSI/ADX/EMA20/EMA50/support/resistance)"
        )

        if st.button("▶  Run Custom Backtest",type="primary",use_container_width=True,key="cbt_run"):
            st.session_state.pop("cbt_result",None)
            with st.spinner("Running… (data fetch + signal generation)"):
                # Write logic to a temp file
                with _tmpfile.NamedTemporaryFile(mode="w",suffix=".py",delete=False,encoding="utf-8") as tf:
                    tf.write("import pandas as pd\nimport ta\n\n")
                    tf.write(cbt_code)
                    logic_path=tf.name
                try:
                    cmd=[
                        sys.executable, str(ROOT/"run_custom_backtest.py"),
                        "--json",
                        "--logic-file", logic_path,
                        "--asset", cbt_ticker,
                        "--label", cbt_label,
                        "--interval", cbt_int,
                        "--start", str(cbt_start),
                        "--end",   str(cbt_end),
                        "--tp1-alloc", str(cbt_tp1),
                        "--tp2-alloc", str(cbt_tp2),
                    ]
                    if not cbt_wf:
                        cmd.append("--no-wf")
                    r=_sp.run(cmd,cwd=str(ROOT),capture_output=True,text=True,timeout=600)
                finally:
                    try: Path(logic_path).unlink()
                    except: pass

            if r.returncode!=0:
                st.error("Backtest failed.")
                st.code(r.stderr[-3000:] if r.stderr else "(no stderr)",language=None)
            else:
                js=None
                m=_re2.search(r'\{[\s\S]+\}',r.stdout)
                if m:
                    try: js=json.loads(m.group())
                    except: pass
                if js and "error" not in js:
                    st.session_state["cbt_result"]=js
                elif js and "error" in js:
                    st.error(f"Backtest error: {js['error']}")
                else:
                    st.warning("Could not parse results.")
                    st.code(r.stdout[-2000:],language=None)

        js=st.session_state.get("cbt_result")
        if js:
            st.markdown("---")
            wr=js.get("actual_win_rate",0); avgr=js.get("actual_avg_net_pnl_r",0)
            pf=js.get("actual_net_profit_factor",0); mdd=js.get("max_drawdown_r",0)
            sharpe=js.get("sharpe_r",0); kelly=js.get("kelly_25pct",0)
            fills=js.get("total_trades_filled",0)
            st.markdown(f"### {js.get('label','')} ({js.get('ticker','')})  "
                        f"`{cbt_int}`  {cbt_start} → {cbt_end}")
            k1,k2,k3,k4,k5,k6=st.columns(6)
            k1.metric("Win Rate",f"{wr:.0%}")
            k2.metric("Avg R (net)",f"{avgr:+.3f}R")
            k3.metric("Profit Factor",f"{pf:.2f}")
            k4.metric("Max Drawdown",f"-{mdd:.1f}R")
            k5.metric("Sharpe",f"{sharpe:.2f}")
            k6.metric("Kelly 25%",f"{kelly:.1%}")
            if fills<30:
                st.warning(f"Only {fills} filled trades — need ≥30 for reliable conclusions.")
            wf=js.get("walk_forward",[])
            if wf:
                st.markdown("**Walk-Forward (60% train / 20% val / 20% test)**")
                wf_rows=[{"Window":w.get("name","").upper(),
                          "Period":f"{w.get('start_date','')[:10]} → {w.get('end_date','')[:10]}",
                          "Fills":w.get("n_fills",0),
                          "Win %":f"{w.get('win_rate',0):.1%}",
                          "Net Avg R":f"{w.get('avg_net_pnl_r',0):+.3f}R",
                          "Net PF":f"{w.get('net_profit_factor',0):.2f}",
                          "Max DD":f"-{w.get('max_drawdown_r',0):.1f}R",
                          "Kelly 25%":f"{w.get('kelly_25pct',0):.1%}"}
                         for w in wf]
                st.dataframe(pd.DataFrame(wf_rows),use_container_width=True,hide_index=True)

    # ── Forward Test (Live Test) ───────────────────────────────────────────────
    with t_ft:
        from p4_live.report import compute_forward_stats
        from p4_live.risk import check_circuit_breaker,check_degradation,monte_carlo

        strat_tabs=st.tabs(list(strategies.keys()))
        for stab,sid in zip(strat_tabs,strategies.keys()):
            with stab:
                asset_names=["All Assets"]+[a["label"] for a in scanner_assets]
                asset_tickers=[None]+[a["ticker"] for a in scanner_assets]
                asset_bases=[None]+[a["baseline"] for a in scanner_assets]
                atabs=st.tabs(asset_names)
                for atab,atick,abase in zip(atabs,asset_tickers,asset_bases):
                    with atab:
                        stats=compute_forward_stats(ticker=atick,strategy=sid)
                        # Circuit breaker
                        if atick:
                            try:
                                breached,live_dd,max_dd=check_circuit_breaker(atick,sid)
                                pct=live_dd/max_dd if max_dd else 0
                                if breached:
                                    st.error(f"🛑 CIRCUIT BREAKER TRIPPED — {live_dd:.1f}R live DD ≥ {max_dd:.1f}R limit. "
                                             f"Do not take new {sid} trades on this asset.")
                                elif pct>=0.75:
                                    st.warning(f"⚠️ Approaching limit: {live_dd:.1f}R / {max_dd:.1f}R ({pct:.0%})")
                                else:
                                    st.success(f"✅ Circuit breaker ACTIVE — {live_dd:.1f}R / {max_dd:.1f}R used")
                            except Exception: pass

                        p1,p2,p3,p4,p5=st.columns(5)
                        p1.metric("Win Rate",f"{stats['win_rate']:.1%}",
                                  f"base {abase['win_rate']:.1%}" if abase else None)
                        p2.metric("Avg R",f"{stats['avg_r']:+.3f}R",
                                  f"base {abase.get('avg_net_pnl_r',0):+.3f}R" if abase else None)
                        p3.metric("Profit Factor",f"{stats['profit_factor']:.2f}",
                                  f"base {abase.get('net_pf',0):.2f}" if abase else None)
                        p4.metric("Max DD",f"-{stats['max_drawdown_r']:.1f}R",
                                  f"limit -{abase['max_drawdown_r']:.1f}R" if abase else None)
                        p5.metric("Total P&L",f"{stats['total_r']:+.1f}R")

                        # Degradation test
                        if atick:
                            try:
                                deg=check_degradation(atick,sid)
                                if deg.get("enough_data"):
                                    dc=":red[DEGRADED]" if deg["degraded"] else ":green[CONSISTENT]"
                                    st.markdown(f"**Degradation test:** {dc} — "
                                                f"Live WR {deg['live_wr']:.1%} vs Baseline {deg['baseline_wr']:.1%} "
                                                f"(z={deg['z_score']:.2f}, p={deg['p_value']:.3f})")
                                else:
                                    st.caption(f"Degradation test: {deg.get('reason','insufficient data')}")
                            except Exception: pass

                        if stats["n_closed"]==0:
                            st.info("No closed trades yet."); continue

                        pc1,pc2=st.columns([3,2])
                        with pc1:
                            pts=sorted([(t["signal_date"],t["pnl_r"]) for t in stats["trades"]
                                        if t["status"] not in ("pending","filled","expired")
                                        and t["pnl_r"] is not None],key=lambda x:x[0])
                            if pts:
                                cum,ex,ey=0,[],[]
                                for d,r in pts: cum+=r; ex.append(d); ey.append(round(cum,3))
                                c3="#10B981" if ey[-1]>=0 else "#EF4444"
                                fig=go.Figure()
                                fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines+markers",
                                    line=dict(color=c3,width=2),fill="tozeroy",
                                    fillcolor=f"rgba({'16,185,129' if c3=='#10B981' else '239,68,68'},.1)",
                                    hovertemplate="%{x}<br>%{y:+.2f}R<extra></extra>"))
                                fig.add_hline(y=0,line_dash="dash",line_color="#2D4060",line_width=1)
                                fig.update_layout(**_PL,title="Cumulative P&L",height=240)
                                st.plotly_chart(fig,use_container_width=True,key=f"ft_chart_{sid}_{atick}")
                        with pc2:
                            if st.button(f"Run Monte Carlo",key=f"mc_{sid}_{atick}"):
                                pnl_r_list=[t["pnl_r"] for t in stats["trades"] if t["pnl_r"] is not None]
                                if len(pnl_r_list)<5:
                                    st.warning("Need at least 5 closed trades.")
                                else:
                                    with st.spinner("Simulating 10,000 paths..."):
                                        mc=monte_carlo(pnl_r_list)
                                    st.metric("Median Final R",f"{mc['median_final_r']:+.2f}R")
                                    st.metric("Worst 5% DD",f"-{mc['worst5_dd_r']:.2f}R")
                                    st.metric("P(Ruin)",f"{mc['p_ruin']:.1%}")
                                    if len(pnl_r_list)<30:
                                        st.caption("Caution: <30 trades. Monte Carlo reliability improves with more data.")

    # ── Comparison: Backtest vs Forward Test ──────────────────────────────────
    with t_cmp:
        st.markdown(
            "**Backtest vs Live** — per viable combo (asset × strategy × timeframe). "
            "Backtest data from the last grid search CSV. Live data from the journal.")
        st.caption(
            "Grid search CSV: outputs/grid_results.csv · "
            "Need ≥5 live closed trades per combo for meaningful comparison.")

        # ── Load grid-search CSV (authoritative backtest source) ─────────────
        _grid_csv = ROOT / "outputs" / "grid_results.csv"
        _bt_lookup: dict[tuple, dict] = {}   # (asset_id, strategy, tf) → row
        if _grid_csv.exists():
            try:
                import csv as _csv
                with open(_grid_csv, newline="", encoding="utf-8") as _f:
                    for _row in _csv.DictReader(_f):
                        _key = (_row["asset"], _row["strategy"], _row["tf"])
                        _bt_lookup[_key] = _row
            except Exception:
                pass

        # ── Iterate actual viable combos from assets.yaml ────────────────────
        from core.config import get_all_assets as _gaa_cmp  # noqa: PLC0415
        _all_live = _trades()
        _closed_live = [t for t in _all_live
                        if t["status"] not in ("pending","filled","expired")]

        rows = []
        for _asset in _gaa_cmp():
            if not _asset.get("enabled", True):
                continue
            _asset_id  = _asset["id"]
            _asset_lbl = _asset.get("label", _asset_id)
            _ticker_yf = _asset.get("tickers", {}).get("yfinance", "")

            for _entry in _asset.get("strategies", []):
                _sid = _entry if isinstance(_entry, str) else _entry.get("id", "")
                _tfs = ([] if isinstance(_entry, str)
                        else _entry.get("timeframes", ["1d"]))
                for _tf in _tfs:
                    # ── Backtest row from grid CSV ────────────────────────────
                    _bt = _bt_lookup.get((_asset_id, _sid, _tf), {})
                    _bt_n    = int(_bt.get("trades") or 0)
                    _bt_pf   = float(_bt.get("net_pf") or 0)
                    _bt_wr   = float(_bt.get("win_rate") or 0)
                    _bt_avgr = float(_bt.get("avg_net_r") or 0)
                    _bt_oos  = float(_bt.get("oos_net_pf") or 0) or None

                    # ── Live trades for this combo ────────────────────────────
                    _live = [t for t in _closed_live
                             if t.get("ticker") == _ticker_yf
                             and t.get("strategy") == _sid
                             and t.get("timeframe") == _tf]
                    _lt_n = len(_live)
                    _lt_pnl  = [t["pnl_r"] for t in _live if t["pnl_r"] is not None]
                    _lt_wins = [t for t in _live if t["status"] in ("win","partial_win")]
                    _lt_wr   = len(_lt_wins) / _lt_n if _lt_n else None
                    _lt_avgr = sum(_lt_pnl) / len(_lt_pnl) if _lt_pnl else None
                    _lt_gp   = sum(r for r in _lt_pnl if r > 0)
                    _lt_gl   = abs(sum(r for r in _lt_pnl if r < 0))
                    _lt_pf   = (_lt_gp / _lt_gl if _lt_gl > 0
                                else (999.0 if _lt_gp > 0 else None))
                    _lt_tot  = sum(_lt_pnl) if _lt_pnl else None

                    # ── Status health ─────────────────────────────────────────
                    if _lt_n < 5:
                        _status = "insufficient data"
                        _sc = "#5A6E85"
                    elif _bt_pf > 0 and _lt_pf is not None:
                        _ratio = _lt_pf / _bt_pf
                        if _ratio >= 0.8:
                            _status = "on track"
                            _sc = "#10B981"
                        elif _ratio >= 0.5:
                            _status = "watch"
                            _sc = "#FBBF24"
                        else:
                            _status = "degrading"
                            _sc = "#EF4444"
                    else:
                        _status = "no BT data"
                        _sc = "#5A6E85"

                    rows.append({
                        "_asset_id": _asset_id, "_sid": _sid, "_tf": _tf,
                        "_ticker": _ticker_yf,
                        "_bt_pf": _bt_pf, "_lt_pf": _lt_pf,
                        "_bt_wr": _bt_wr, "_lt_wr": _lt_wr,
                        "_bt_avgr": _bt_avgr, "_lt_avgr": _lt_avgr,
                        "_bt_oos": _bt_oos, "_lt_n": _lt_n,
                        "_bt_n": _bt_n, "_lt_tot": _lt_tot,
                        "_status": _status, "_sc": _sc,
                    })

        if not rows:
            st.info("No viable combos found in assets.yaml.")
        else:
            # ── Summary status banner ─────────────────────────────────────────
            _n_ok  = sum(1 for r in rows if r["_status"] == "on track")
            _n_wat = sum(1 for r in rows if r["_status"] == "watch")
            _n_deg = sum(1 for r in rows if r["_status"] == "degrading")
            _n_ins = sum(1 for r in rows if r["_status"] in ("insufficient data","no BT data"))
            st.markdown(
                f'<div style="display:flex;gap:.6rem;margin-bottom:.8rem;flex-wrap:wrap">'
                f'<span style="background:#052E1C;color:#10B981;padding:3px 10px;'
                f'border-radius:5px;font-size:.75rem;font-weight:700">'
                f'{_n_ok} on track</span>'
                f'<span style="background:#1C1200;color:#FBBF24;padding:3px 10px;'
                f'border-radius:5px;font-size:.75rem;font-weight:700">'
                f'{_n_wat} watch</span>'
                f'<span style="background:#1E0808;color:#EF4444;padding:3px 10px;'
                f'border-radius:5px;font-size:.75rem;font-weight:700">'
                f'{_n_deg} degrading</span>'
                f'<span style="background:#0A1220;color:#5A6E85;padding:3px 10px;'
                f'border-radius:5px;font-size:.75rem;font-weight:700">'
                f'{_n_ins} insufficient data</span>'
                f'</div>', unsafe_allow_html=True)

            # ── Per-combo cards ───────────────────────────────────────────────
            for r in rows:
                _sc   = r["_sc"]
                _bt_pf_s  = f"{r['_bt_pf']:.2f}" if r["_bt_pf"] else "—"
                _lt_pf_s  = f"{r['_lt_pf']:.2f}" if r["_lt_pf"] is not None else "—"
                _bt_wr_s  = f"{r['_bt_wr']:.1%}"  if r["_bt_wr"] else "—"
                _lt_wr_s  = f"{r['_lt_wr']:.1%}"  if r["_lt_wr"] is not None else "—"
                _bt_ar_s  = f"{r['_bt_avgr']:+.3f}R" if r["_bt_avgr"] else "—"
                _lt_ar_s  = f"{r['_lt_avgr']:+.3f}R" if r["_lt_avgr"] is not None else "—"
                _lt_tot_s = f"{r['_lt_tot']:+.2f}R"  if r["_lt_tot"] is not None else "—"
                _oos_s    = f"OOS PF: {r['_bt_oos']:.2f}" if r["_bt_oos"] else ""

                with st.expander(
                    f"{r['_asset_id'].upper()}  ·  {_fmt_strategy(r['_sid'])}  "
                    f"[{r['_tf']}]   —   "
                    f"BT: {_bt_pf_s} PF  /  Live: {_lt_pf_s} PF  "
                    f"({r['_lt_n']} trades)   [{r['_status'].upper()}]",
                    expanded=False,
                ):
                    _ca, _cb = st.columns(2)
                    with _ca:
                        st.markdown(f"**Backtest** *(n={r['_bt_n']})*")
                        _bm1, _bm2, _bm3 = st.columns(3)
                        _bm1.metric("PF",    _bt_pf_s)
                        _bm2.metric("WR",    _bt_wr_s)
                        _bm3.metric("Avg R", _bt_ar_s)
                        if _oos_s:
                            st.caption(_oos_s)
                    with _cb:
                        _warn = " *(need 5+ trades)*" if r["_lt_n"] < 5 else ""
                        st.markdown(f"**Live forward test** *(n={r['_lt_n']})*{_warn}")
                        _lm1, _lm2, _lm3 = st.columns(3)

                        def _delta_color(live, bt):
                            if live is None or not bt: return "off"
                            return "normal" if live >= bt * 0.8 else "inverse"

                        _lm1.metric("PF",    _lt_pf_s,
                            delta=f"{r['_lt_pf']-r['_bt_pf']:+.2f}" if r["_lt_pf"] is not None and r["_bt_pf"] else None,
                            delta_color=_delta_color(r["_lt_pf"], r["_bt_pf"]))
                        _lm2.metric("WR",    _lt_wr_s,
                            delta=f"{r['_lt_wr']-r['_bt_wr']:+.1%}" if r["_lt_wr"] is not None and r["_bt_wr"] else None,
                            delta_color=_delta_color(r["_lt_wr"], r["_bt_wr"]))
                        _lm3.metric("Total R", _lt_tot_s)

            # ── PF comparison chart (combos with both BT and live data) ───────
            _chart_rows = [r for r in rows
                           if r["_bt_pf"] and r["_lt_pf"] is not None and r["_lt_n"] >= 5]
            if _chart_rows:
                st.markdown("---")
                st.markdown("**Profit Factor: Backtest vs Live** *(≥5 live trades)*")
                _clabels = [f"{r['_asset_id'].upper()} {r['_tf']}\n{_fmt_strategy(r['_sid'])}"
                            for r in _chart_rows]
                _fig_cmp = go.Figure()
                _fig_cmp.add_trace(go.Bar(
                    name="Backtest PF", x=_clabels,
                    y=[r["_bt_pf"] for r in _chart_rows],
                    marker_color="#60A5FA",
                    hovertemplate="%{x}<br>BT PF: %{y:.2f}<extra></extra>"))
                _fig_cmp.add_trace(go.Bar(
                    name="Live PF", x=_clabels,
                    y=[r["_lt_pf"] for r in _chart_rows],
                    marker_color=["#10B981" if r["_status"]=="on track"
                                  else ("#FBBF24" if r["_status"]=="watch"
                                        else "#EF4444")
                                  for r in _chart_rows],
                    hovertemplate="%{x}<br>Live PF: %{y:.2f}<extra></extra>"))
                _fig_cmp.add_hline(y=1.0, line_dash="dot", line_color="#5A6E85",
                                   line_width=1, annotation_text="Break-even",
                                   annotation_position="right")
                _fig_cmp.update_layout(**_PL, barmode="group", height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1))
                st.plotly_chart(_fig_cmp, use_container_width=True)


# =============================================================================
# PAGE 5: JOURNAL
# =============================================================================
def page_journal():
    st.title("Trade Journal")
    from p4_live.journal_supabase import (
        update_manual_entry, delete_manual_entry, save_manual_entry,
        save_psychology, add_note,
        record_fill_direct as _rf, record_close_direct as _rc, record_expired_direct as _re,
    )
    _jclient = st.session_state["supabase_client"]
    all_t=_trades(); assets=_assets(); strategies=_strategies()

    t_sim,t_rt,t_mgr,t_ss,t_psy,t_met=st.tabs(
        ["📊 Simulated Trades","⭐ Real Trades","Manage","Add Trade","Psychology","Metrics"])

    # ── Simulated Trades (scanner / backfill — not real money) ───────────────
    with t_sim:
        st.markdown(
            '<div style="background:#0A1628;border-left:3px solid #FBBF24;border-radius:6px;'
            'padding:.4rem .9rem;font-size:.78rem;color:#94A3B8;margin-bottom:.8rem">'
            '⚠️ <b style="color:#FBBF24">Simulated trades</b> — generated by the scanner and backfill engine. '
            'These are <b>not real money</b>. Entries/exits are pessimistic bar-by-bar fills on OHLCV data. '
            'Use them to evaluate strategy performance, not as an exact trade log.'
            '</div>', unsafe_allow_html=True)
        f1,f2,f3=st.columns(3)
        status_opts=["All","pending","filled","win","partial_win","loss","breakeven","expired"]
        asset_opts=["All"]+sorted({a["label"] for a in assets})
        strat_opts=["All"]+sorted(strategies.keys())
        sel_s=f1.selectbox("Status",status_opts,key="jl_s")
        sel_a=f2.selectbox("Asset",asset_opts,key="jl_a")
        sel_st=f3.selectbox("Strategy",strat_opts,key="jl_st")
        filt=all_t[:]
        if sel_s!="All": filt=[t for t in filt if t["status"]==sel_s]
        if sel_a!="All": filt=[t for t in filt if _label(t)==sel_a]
        if sel_st!="All": filt=[t for t in filt if t.get("strategy")==sel_st]
        filt.sort(key=lambda x:x["signal_date"],reverse=True)
        st.caption(f"{len(filt)} simulated trades · {len(all_t)} total in DB")
        if filt: st.dataframe(_trades_df(filt),use_container_width=True,hide_index=True)
        else: st.info("No trades match the filters.")
        if filt:
            csv=_trades_df(filt).to_csv(index=False)
            st.download_button("Export filtered to CSV",csv,"simulated_trades.csv","text/csv")

    # ── Real Trades (manual entries — real money, real execution) ─────────────
    with t_rt:
        manual=_manual_entries()
        if not manual:
            st.markdown(
                '<div style="text-align:center;padding:3rem;color:#5A6E85">'
                '<div style="font-size:3rem">📸</div>'
                '<div style="font-size:1.1rem;font-weight:600;margin:.5rem 0">No real trades logged yet</div>'
                '<div>Use the <b>Add Trade</b> tab to log a real trade via screenshot or manual entry.</div>'
                '</div>', unsafe_allow_html=True)
        else:
            # ── Summary bar ───────────────────────────────────────────────────
            def _eff_r(e):
                r,_=_calc_manual_r(e); return r
            eff_rs=[_eff_r(e) for e in manual]
            total_r_m=sum(r for r in eff_rs if r is not None)
            total_dollars=sum(e.get("pnl_dollars") or 0 for e in manual
                              if e.get("pnl_dollars") is not None)
            wins_m=sum(1 for r in eff_rs if r is not None and r>0)
            losses_m=sum(1 for r in eff_rs if r is not None and r<0)
            avg_q=sum(e.get("quality_score") or 0 for e in manual
                      if e.get("quality_score")) / max(sum(1 for e in manual if e.get("quality_score")),1)
            sr1,sr2,sr3,sr4,sr5=st.columns(5)
            sr1.metric("Total Real Trades",len(manual))
            sr2.metric("Total R",f"{total_r_m:+.2f}R",
                       delta_color="normal" if total_r_m>=0 else "inverse")
            dollar_color="normal" if total_dollars>=0 else "inverse"
            sr3.metric("Total $",f"{'+'if total_dollars>=0 else ''}${total_dollars:,.2f}",
                       delta_color=dollar_color)
            sr4.metric("Win / Loss",f"{wins_m} / {losses_m}")
            sr5.metric("Avg Quality",f"{avg_q:.1f}/10")
            st.markdown("---")

            # ── Filters ────────────────────────────────────────────────────────
            rf1,rf2,rf3=st.columns(3)
            sort_by=rf1.selectbox("Sort by",["Date (newest)","P&L (best)","Quality (highest)"],key="rt_sort")
            dir_filt=rf2.selectbox("Direction",["All","long","short","unknown"],key="rt_dir")
            asset_filt=rf3.text_input("Filter by asset",placeholder="Silver, Gold…",key="rt_af")

            view=list(manual)
            if dir_filt!="All": view=[e for e in view if (e.get("direction") or "")== dir_filt]
            if asset_filt.strip(): view=[e for e in view if asset_filt.lower() in (e.get("asset") or "").lower()]
            if sort_by=="P&L (best)": view=sorted(view,key=lambda e:e.get("pnl_r") or 0,reverse=True)
            elif sort_by=="Quality (highest)": view=sorted(view,key=lambda e:e.get("quality_score") or 0,reverse=True)
            else: view=sorted(view,key=lambda e:e.get("entry_date",""),reverse=True)

            st.caption(f"{len(view)} trade{'s' if len(view)!=1 else ''} shown")

            # ── Trade cards (2-up grid) ────────────────────────────────────────
            for i in range(0,len(view),2):
                cols=st.columns(2)
                for ci,e in enumerate(view[i:i+2]):
                    ai={}
                    if e.get("ai_analysis"):
                        try: ai=json.loads(e["ai_analysis"])
                        except: pass
                    with cols[ci]:
                        pnl_stored=e.get("pnl_r")
                        pnl,pnl_estimated=_calc_manual_r(e)
                        qs=e.get("quality_score")
                        pnl_color=("#10B981" if (pnl or 0)>0 else "#EF4444" if (pnl or 0)<0
                                   else "#FBBF24" if pnl==0 else "#5A6E85")
                        dir_v=(e.get("direction") or "unknown").lower()
                        dir_color="#10B981" if dir_v=="long" else "#EF4444" if dir_v=="short" else "#94A3B8"
                        asset_v=e.get("asset") or ai.get("asset","—")
                        # Quality bar (0-10)
                        q_bar=""
                        if qs is not None:
                            filled=round(qs); empty=10-filled
                            q_color=("#10B981" if qs>=7 else "#FBBF24" if qs>=5 else "#EF4444")
                            q_bar=(f'<div style="display:flex;align-items:center;gap:.4rem;margin:.3rem 0">'
                                   f'<div style="flex:1;height:4px;border-radius:2px;background:#1A2840;overflow:hidden">'
                                   f'<div style="width:{qs*10}%;height:100%;background:{q_color}"></div></div>'
                                   f'<span style="font-size:.72rem;color:{q_color};font-weight:600">{qs}/10</span></div>')
                        # Tags
                        tags=e.get("tags","")
                        tag_html=""
                        if tags:
                            tag_html='<div style="margin:.3rem 0">'+''.join(
                                f'<span style="background:#1A2840;color:#94A3B8;font-size:.65rem;'
                                f'padding:1px 7px;border-radius:10px;margin-right:3px">{t.strip()}</span>'
                                for t in tags.split(",") if t.strip())+'</div>'
                        # Metrics line
                        entry_v=e.get("entry_price"); exit_v=e.get("exit_price")
                        price_str=""
                        if entry_v: price_str+=f"Entry {entry_v:,.4f}"
                        if exit_v: price_str+=f" → {exit_v:,.4f}"

                        price_html=(f'<div style="font-size:.78rem;color:#5A6E85;margin:.15rem 0">'
                                    f'{price_str}</div>') if price_str else ""
                        pnl_dollars=e.get("pnl_dollars")
                        if pnl is not None:
                            est_label=(' <span style="font-size:.62rem;color:#5A6E85">(calc)</span>'
                                       if pnl_estimated else "")
                            dollar_str=(f' <span style="font-size:.9rem;color:{pnl_color};'
                                        f'font-weight:600">'
                                        f'{"+" if pnl_dollars>=0 else ""}${pnl_dollars:,.2f}'
                                        f'</span>' if pnl_dollars is not None else "")
                            pnl_display=(f'<span style="font-size:1.3rem;font-weight:700;color:{pnl_color}">'
                                         f'{pnl:+.2f}R</span>{dollar_str}{est_label}')
                        else:
                            pnl_display='<span style="font-size:.85rem;color:#5A6E85">P&L not set</span>'
                        st.markdown(
                            f'<div style="background:#101826;border:1px solid #1A2840;border-radius:12px;'
                            f'padding:.9rem 1rem;margin-bottom:.35rem">'
                            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.5rem">'
                            f'<div>'
                            f'<span style="font-weight:700;color:#F1F5F9;font-size:1rem">{asset_v}</span>&nbsp;&nbsp;'
                            f'<span style="background:{dir_color}22;color:{dir_color};font-size:.7rem;font-weight:700;'
                            f'padding:2px 8px;border-radius:4px">{dir_v.upper()}</span>'
                            f'</div>'
                            f'<span style="color:#5A6E85;font-size:.78rem">{e.get("entry_date","?")}</span>'
                            f'</div>'
                            f'<div style="display:flex;align-items:center;gap:1rem;margin:.2rem 0">'
                            f'{pnl_display}'
                            f'{q_bar}'
                            f'</div>'
                            f'{price_html}'
                            f'{tag_html}'
                            f'</div>',
                            unsafe_allow_html=True)

                        # Screenshot + AI insights + inline P&L update
                        has_ss=(e.get("screenshot_path") and Path(e["screenshot_path"]).exists())
                        has_chart=(e.get("chart_screenshot_path") and Path(e["chart_screenshot_path"]).exists())
                        # Handle both old (flat) and new (nested broker/chart) AI format
                        merged_ai=ai.get("merged",ai)
                        broker_ai=ai.get("broker",{})
                        chart_ai_d=ai.get("chart",{})
                        with st.expander("Details / Edit P&L",expanded=False):
                            if has_ss or has_chart:
                                img_cols=st.columns(2 if (has_ss and has_chart) else 1)
                                if has_ss:
                                    with img_cols[0]:
                                        st.caption("Broker / Order details")
                                        st.image(e["screenshot_path"],use_container_width=True)
                                if has_chart:
                                    with img_cols[-1]:
                                        st.caption("Chart")
                                        st.image(e["chart_screenshot_path"],use_container_width=True)
                            if e.get("notes"):
                                st.markdown(f'**Journal entry:**\n\n{e["notes"]}')
                            disp=merged_ai or ai
                            if disp.get("what_worked"): st.success(f"Worked: {disp['what_worked']}")
                            if disp.get("what_to_improve"): st.warning(f"Improve: {disp['what_to_improve']}")
                            if disp.get("lessons"): st.info(f"Lesson: {disp['lessons']}")
                            ms=chart_ai_d.get("market_structure") or disp.get("market_structure")
                            if ms: st.caption(f"Chart structure: {ms}")
                            # Inline edit form
                            st.markdown("---")
                            st.markdown("**Update trade result:**")
                            with st.form(f"pnl_edit_{e['id']}"):
                                uf0,uf0b=st.columns(2)
                                u_dir=uf0.selectbox("Direction",["long","short","unknown"],
                                    index=["long","short","unknown"].index(dir_v)
                                          if dir_v in ["long","short","unknown"] else 0,
                                    key=f"udir_{e['id']}")
                                u_exit=uf0b.number_input("Exit price",
                                    value=float(e.get("exit_price") or 0),
                                    step=0.01,format="%.4f",key=f"ue_{e['id']}")
                                uf1,uf2,uf3=st.columns(3)
                                u_orig_sl=uf1.number_input("Original SL (at entry)",
                                    value=float(e.get("original_sl") or 0),
                                    step=0.01,format="%.4f",key=f"uosl_{e['id']}",
                                    help="Protective stop placed at entry — before any trailing")
                                u_sl=uf2.number_input("Final SL (trailed)",
                                    value=float(e.get("stop_loss") or 0),
                                    step=0.01,format="%.4f",key=f"usl_{e['id']}",
                                    help="Where stop ended up when closed (may be in profit after trailing)")
                                u_pnl=uf3.number_input("P&L (R) — manual override",
                                    value=float(pnl or 0),
                                    step=0.001,format="%.3f",key=f"up_{e['id']}",
                                    help="Leave 0 to auto-calculate from Original SL")
                                # Live preview
                                _entry_v=float(e.get("entry_price") or 0)
                                _exit_v=float(e.get("exit_price") or 0)
                                if u_orig_sl and _entry_v and _exit_v and abs(_entry_v-u_orig_sl)>1e-10:
                                    _pr=abs(_entry_v-u_orig_sl)
                                    _pv=round((_entry_v-_exit_v)/_pr if u_dir=="short"
                                              else (_exit_v-_entry_v)/_pr, 3)
                                    out="WIN ✅" if _pv>0 else "LOSS ❌"
                                    st.caption(f"Preview: {_pv:+.3f}R  {out}  (risk={_pr:.4f} pts)")
                                u_dollars=st.number_input("P&L ($) — from broker",
                                    value=float(e.get("pnl_dollars") or 0),
                                    step=0.01,format="%.2f",key=f"udol_{e['id']}",
                                    help="Actual dollar gain/loss shown on your broker statement")
                                if st.form_submit_button("Save result", type="primary"):
                                    _e=_entry_v
                                    _ex_v=u_exit if u_exit else None
                                    _orig=u_orig_sl if u_orig_sl else None
                                    _sl_v=u_sl if u_sl else None
                                    auto_r=None
                                    if u_pnl==0 and _ex_v and _orig and _e and abs(_e-_orig)>1e-10:
                                        _risk=abs(_e-_orig)
                                        auto_r=round((_e-_ex_v)/_risk if u_dir=="short"
                                                     else (_ex_v-_e)/_risk, 3)
                                    final_r=u_pnl if u_pnl!=0 else auto_r
                                    final_d=u_dollars if u_dollars!=0 else None
                                    update_manual_entry(_jclient, e["id"],direction=u_dir,
                                                        exit_price=_ex_v,original_sl=_orig,
                                                        stop_loss=_sl_v,pnl_r=final_r,
                                                        pnl_dollars=final_d)
                                    _clear()
                                    st.success(f"Saved — {final_r:+.3f}R / {'+'if (final_d or 0)>=0 else ''}${final_d:,.2f}" if final_r and final_d else
                                               f"Saved — {final_r:+.3f}R" if final_r else "Saved")
                                    st.rerun()
                            # Delete
                            st.markdown("---")
                            if st.button("🗑 Delete this entry",key=f"del_{e['id']}",
                                         help="Permanently remove this trade from the journal"):
                                st.session_state[f"confirm_del_{e['id']}"]=True
                            if st.session_state.get(f"confirm_del_{e['id']}"):
                                st.warning("This cannot be undone. Are you sure?")
                                dc1,dc2=st.columns(2)
                                if dc1.button("Yes, delete",key=f"dely_{e['id']}",type="primary"):
                                    delete_manual_entry(_jclient, e["id"])
                                    st.session_state.pop(f"confirm_del_{e['id']}",None)
                                    _clear(); st.rerun()
                                if dc2.button("Cancel",key=f"deln_{e['id']}"):
                                    st.session_state.pop(f"confirm_del_{e['id']}",None); st.rerun()


    # ── Manage trades ──────────────────────────────────────────────────────────
    with t_mgr:
        if not all_t: st.info("No trades yet.")
        else:
            sorted_t=sorted(all_t,key=lambda x:x["signal_date"],reverse=True)
            trade_lbl=[f"{t['signal_date']} | {_label(t)} | {_fmt_strategy(t.get('strategy',''))} | {t['status'].upper()}"
                       for t in sorted_t]
            sel_idx=st.selectbox("Trade",range(len(trade_lbl)),format_func=lambda i:trade_lbl[i])
            sel=sorted_t[sel_idx]
            with st.expander("Details",expanded=True):
                d1,d2,d3=st.columns(3)
                d1.markdown(f"**Direction:** {sel['direction'].upper()}  \n"
                            f"**Entry:** `{sel['entry_low']:,.2f}`–`{sel['entry_high']:,.2f}`  \n"
                            f"**SL:** `{sel['stop_loss']:,.2f}`")
                d2.markdown(f"**TP1:** `{sel['tp1']:,.2f}`  \n"
                            f"**TP2:** `{sel['tp2']:,.2f}`" if sel.get("tp2") else "**TP1:**")
                d3.markdown(f"**Status:** {sel['status'].upper()}  \n"
                            f"**Fill:** `{sel['fill_price'] or '—'}`  \n"
                            f"**Confidence:** `{(sel.get('confidence') or 0):.0%}`")
                if sel.get("rationale"): st.caption(sel["rationale"][:400])
                if sel.get("notes"): st.info(f"Note: {sel['notes']}")
                src=sel.get("source","scanner")
                if src=="manual": st.caption("Source: Trade Planner (manual)")
            tt1,tt2,tt3,tt4=st.tabs(["Fill","Close","Expire","Note"])
            with tt1:
                if sel["status"]=="pending":
                    with st.form("ff"):
                        fa,fb=st.columns(2)
                        fp=fa.number_input("Fill price",min_value=0.01,step=0.01,format="%.2f")
                        fd=fb.date_input("Fill date",value=date.today())
                        if st.form_submit_button("Record Fill",type="primary"):
                            _rf(_jclient, sel["signal_date"], sel["ticker"],
                                sel.get("strategy","mtf_trend"), sel.get("timeframe","1d"),
                                float(fp), fd.isoformat())
                            _clear(); st.success("Fill recorded."); st.rerun()
                else: st.info(f"Only PENDING trades can be filled. Status: {sel['status'].upper()}")
            with tt2:
                if sel["status"]=="filled":
                    with st.form("fc"):
                        ca,cb,cc=st.columns(3)
                        out=ca.selectbox("Outcome",["win","partial_win","loss","breakeven"])
                        ep=cb.number_input("Exit price",min_value=0.01,step=0.01,format="%.2f")
                        ed=cc.date_input("Exit date",value=date.today())
                        if st.form_submit_button("Record Close",type="primary"):
                            _pnl = _calc_close_pnl_r(sel, out, float(ep))
                            _rc(_jclient, sel["signal_date"], sel["ticker"],
                                sel.get("strategy","mtf_trend"), sel.get("timeframe","1d"),
                                out, float(ep), ed.isoformat(), _pnl)
                            _clear(); st.success(f"Closed as {out}."); st.rerun()
                else: st.info(f"Trade must be FILLED. Status: {sel['status'].upper()}")
            with tt3:
                if sel["status"]=="pending":
                    st.warning(f"Mark {sel['signal_date']} {_label(sel)} as expired?")
                    if st.button("Expire",type="primary"):
                        _re(_jclient, sel["signal_date"], sel["ticker"],
                            sel.get("strategy","mtf_trend"), sel.get("timeframe","1d"))
                        _clear(); st.success("Expired."); st.rerun()
                else: st.info(f"Only PENDING signals can be expired. Status: {sel['status'].upper()}")
            with tt4:
                with st.form("fn"):
                    nt=st.text_area("Note",value=sel.get("notes",""),height=80)
                    if st.form_submit_button("Save Note"):
                        if nt.strip():
                            add_note(_jclient, sel["signal_date"], sel["ticker"],
                                    sel.get("strategy","mtf_trend"), sel.get("timeframe","1d"), nt.strip())
                            _clear(); st.success("Saved."); st.rerun()

    # ── Add Trade (screenshot or manual) ──────────────────────────────────────
    with t_ss:
        manual_ss=_manual_entries()
        st.markdown("Log a real trade from your broker — upload a screenshot or enter the details manually. "
                    "The AI will parse the chart and pre-fill the journal entry.")
        st1,st2,st3=st.tabs([f"New Entry","All Entries ({len(manual_ss)})","Import CSV"])
        manual=manual_ss   # alias so the rest of the block still works
        with st1:
            sc1,sc2=st.columns([1,1])
            with sc1:
                uploaded=st.file_uploader("Trade screenshot",type=["png","jpg","jpeg","webp"])
                if uploaded: st.image(uploaded,use_container_width=True)
            with sc2:
                uploaded_chart=st.file_uploader("Chart screenshot (optional)",
                                                type=["png","jpg","jpeg","webp"],key="ss_chart")
                st.caption("Annotate with Snipping Tool (underline/circle) the candle or trade row before uploading.")
                if uploaded_chart: st.image(uploaded_chart,use_container_width=True)

            # ── Trade details ─────────────────────────────────────────────────
            st.markdown("---")
            fd1,fd2=st.columns(2)
            with fd1:
                j_date=st.date_input("Trade date",value=date.today())
                j_asset=st.text_input("Asset",placeholder="EURUSD, BTC, SPX500...")
                j_dir=st.selectbox("Direction",["long","short","unknown"])
                j_pre=st.selectbox("Pre-trade state",["—","calm","optimal","anxious","overconfident","fomo","fearful","bored"])
                j_tags=st.text_input("Tags",placeholder="breakout, fomo, patient, revenge...")
            with fd2:
                c_e1,c_e2=st.columns(2)
                j_entry=c_e1.number_input("Entry",value=0.0,step=0.01,format="%.4f")
                j_sl=c_e2.number_input("Stop Loss",value=0.0,step=0.01,format="%.4f")
                c_e3,c_e4=st.columns(2)
                j_tp=c_e3.number_input("Take Profit",value=0.0,step=0.01,format="%.4f")
                j_exit=c_e4.number_input("Exit Price",value=0.0,step=0.01,format="%.4f")
                j_pnl=st.number_input("P&L (R) — leave 0 to auto-calculate",value=0.0,step=0.01,format="%.3f")
                j_notes=st.text_area("Your notes",height=68,placeholder="What did you see? Why did you enter?")
                # Warn if direction contradicts SL position
                if j_entry>0 and j_sl>0:
                    if j_dir=="long" and j_sl>j_entry:
                        st.warning("⚠️ SL is above entry — this looks like a SHORT, not a LONG.")
                    elif j_dir=="short" and j_sl<j_entry:
                        st.warning("⚠️ SL is below entry — this looks like a LONG, not a SHORT.")

            # ── AI analysis buttons ───────────────────────────────────────────
            ab1,ab2,ab3=st.columns(3)
            from core.demo_limit import try_consume as _dl_try, DEMO_LIMIT_MESSAGE as _dl_msg
            if ab1.button("Analyze broker screenshot",type="primary",disabled=not uploaded,
                          use_container_width=True):
                if not _dl_try(1):
                    st.error(_dl_msg)
                else:
                    with st.spinner("AI reading broker data…"):
                        mt=f"image/{uploaded.type.split('/')[-1]}".replace("image/jpg","image/jpeg")
                        analysis=_analyze_screenshot(uploaded.getvalue(),mt,j_notes,screenshot_type="broker")
                    st.session_state["ss_analysis"]=analysis
                    st.session_state.pop("ss_chart_analysis",None)
                    st.success("Broker data extracted — review below.")
            if ab2.button("Analyze chart screenshot",type="primary",disabled=not uploaded_chart,
                          use_container_width=True):
                if not _dl_try(1):
                    st.error(_dl_msg)
                else:
                    with st.spinner("AI reading chart…"):
                        mt2=f"image/{uploaded_chart.type.split('/')[-1]}".replace("image/jpg","image/jpeg")
                        chart_an=_analyze_screenshot(uploaded_chart.getvalue(),mt2,j_notes,screenshot_type="chart")
                    st.session_state["ss_chart_analysis"]=chart_an
                    st.success("Chart analyzed — review below.")
            if ab3.button("Analyze both",type="primary",
                          disabled=not (uploaded and uploaded_chart),use_container_width=True):
                if not _dl_try(2):
                    st.error(_dl_msg)
                else:
                    with st.spinner("AI analyzing both screenshots…"):
                        mt=f"image/{uploaded.type.split('/')[-1]}".replace("image/jpg","image/jpeg")
                        mt2=f"image/{uploaded_chart.type.split('/')[-1]}".replace("image/jpg","image/jpeg")
                        analysis=_analyze_screenshot(uploaded.getvalue(),mt,j_notes,screenshot_type="broker")
                        chart_an=_analyze_screenshot(uploaded_chart.getvalue(),mt2,j_notes,screenshot_type="chart")
                    st.session_state["ss_analysis"]=analysis
                    st.session_state["ss_chart_analysis"]=chart_an
                    st.success("Both analyzed — review below.")

            # ── AI results ────────────────────────────────────────────────────
            an=st.session_state.get("ss_analysis",{})
            chart_an=st.session_state.get("ss_chart_analysis",{})
            if an or chart_an:
                st.markdown("---")
                if "error" in an: st.error(f"Broker analysis error: {an['error']}")
                if "error" in chart_an: st.error(f"Chart analysis error: {chart_an['error']}")
                # Merge: broker data takes priority for prices, chart for quality/structure
                merged={**chart_an,**{k:v for k,v in an.items() if v is not None}}
                rc1,rc2,rc3,rc4=st.columns(4)
                rc1.metric("Asset",merged.get("asset","?"))
                rc2.metric("Direction",merged.get("direction","?").upper())
                rc3.metric("Quality",f"{merged.get('trade_quality','—')}/10")
                rc4.metric("Timeframe",merged.get("timeframe","?"))
                if an.get("entry_price"):
                    ri1,ri2,ri3,ri4=st.columns(4)
                    ri1.metric("Entry",f"{an['entry_price']:,.4f}" if an.get("entry_price") else "—")
                    ri2.metric("SL",f"{an['stop_loss']:,.4f}" if an.get("stop_loss") else "—")
                    ri3.metric("TP",f"{an['take_profit']:,.4f}" if an.get("take_profit") else "—")
                    ri4.metric("Exit",f"{an['exit_price']:,.4f}" if an.get("exit_price") else "—")
                if an.get("pnl_dollars"):
                    st.markdown(f'**Broker P&L:** `${an["pnl_dollars"]:,.2f}`  ·  '
                                f'Lot size: `{an.get("lot_size","?")}`')
                # Build combined journal entry
                broker_je=an.get("journal_entry","")
                chart_je=chart_an.get("journal_entry","")
                combined_je=(broker_je + ("\n\n**Chart analysis:**\n" + chart_je if chart_je else "")).strip()
                edited=st.text_area("Journal entry (edit before saving)",
                                    value=combined_je,height=200,key="ss_edited")
                if merged.get("what_worked"): st.success(f"Worked: {merged['what_worked']}")
                if merged.get("what_to_improve"): st.warning(f"Improve: {merged['what_to_improve']}")
                if merged.get("lessons"): st.info(f"Lesson: {merged['lessons']}")
                if chart_an.get("market_structure"):
                    st.caption(f"Chart structure: {chart_an['market_structure']}")

                if st.button("Save Entry",type="primary"):
                    ss_dir=ROOT/"outputs"/"screenshots"; ss_dir.mkdir(parents=True,exist_ok=True)
                    ts_s=datetime.now().strftime("%Y%m%d_%H%M%S")
                    ss_p=None; chart_p=None
                    if uploaded:
                        ss_p=ss_dir/f"{ts_s}_broker_{uploaded.name}"
                        ss_p.write_bytes(uploaded.getvalue())
                    if uploaded_chart:
                        chart_p=ss_dir/f"{ts_s}_chart_{uploaded_chart.name}"
                        chart_p.write_bytes(uploaded_chart.getvalue())
                    # Resolve prices: user input > broker AI > chart AI
                    _entry=j_entry or an.get("entry_price") or chart_an.get("entry_price")
                    _raw_sl=j_sl or an.get("stop_loss") or chart_an.get("stop_loss")
                    _raw_tp=j_tp or an.get("take_profit") or chart_an.get("take_profit")
                    _exit=j_exit or an.get("exit_price") or chart_an.get("exit_price") or None
                    # Sanity check: discard SL/TP that are clearly from a different asset
                    # (more than 50% away from entry price — cross-asset contamination)
                    def _sane(val, ref):
                        if not val or not ref: return val
                        return val if abs(float(val)-float(ref))/float(ref) < 0.50 else None
                    _sl=_sane(_raw_sl, _entry)
                    _tp=_sane(_raw_tp, _entry)
                    if _raw_sl and not _sl:
                        st.warning(f"Discarded SL {_raw_sl} — too far from entry {_entry} (likely from a different trade row).")
                    if _raw_tp and not _tp:
                        st.warning(f"Discarded TP {_raw_tp} — too far from entry {_entry}.")
                    _pnl=j_pnl if j_pnl else None
                    if _pnl is None and _entry and _sl and _exit and abs(float(_entry)-float(_sl))>1e-10:
                        _risk=abs(float(_entry)-float(_sl))
                        _pnl=round((float(_exit)-float(_entry))/_risk if j_dir=="long"
                                   else (float(_entry)-float(_exit))/_risk, 3)
                    combined_ai=json.dumps({"broker":an,"chart":chart_an,"merged":merged},default=str)
                    _pnl_dollars=an.get("pnl_dollars") or merged.get("pnl_dollars")
                    eid=save_manual_entry(_jclient, {
                        "entry_date":j_date.isoformat(),
                        "asset":j_asset or merged.get("asset",""),
                        "direction":j_dir if j_dir!="unknown" else merged.get("direction","unknown"),
                        "entry_price":_entry,"stop_loss":_sl,"take_profit":_tp,
                        "exit_price":_exit,"pnl_r":_pnl,
                        "pnl_dollars":float(_pnl_dollars) if _pnl_dollars is not None else None,
                        "quality_score":merged.get("trade_quality"),
                        "notes":edited or j_notes,
                        "ai_analysis":combined_ai,
                        "screenshot_path":str(ss_p) if ss_p else None,
                        "chart_screenshot_path":str(chart_p) if chart_p else None,
                        "tags":j_tags})
                    if j_pre and j_pre!="—":
                        save_psychology(_jclient, {"trade_ref_type":"manual","trade_ref_id":str(eid),
                                        "pre_state":j_pre,"lesson":merged.get("lessons","")})
                    st.session_state.pop("ss_analysis",None)
                    st.session_state.pop("ss_chart_analysis",None)
                    _clear(); st.success(f"Entry #{eid} saved."); st.rerun()
        with st2:
            st.markdown(
                '<div style="background:#0A1628;border-left:3px solid #7C3AED;border-radius:6px;'
                'padding:.4rem .8rem;font-size:.78rem;color:#C084FC;margin-bottom:.6rem">'
                '⭐ Go to <b>Real Trades</b> tab for the full card view with filters and stats.'
                '</div>', unsafe_allow_html=True)
            if not manual_ss: st.info("No entries yet.")
            else:
                for e in manual_ss:
                    ai={}
                    if e.get("ai_analysis"):
                        try: ai=json.loads(e["ai_analysis"])
                        except: pass
                    pnl_e=e.get("pnl_r"); pnl_str=f" · {pnl_e:+.2f}R" if pnl_e is not None else ""
                    with st.expander(
                        f"**{e['entry_date']}** · {e.get('asset','—')} · "
                        f"{(e.get('direction') or '').upper()}{pnl_str} · "
                        f"Q:{e.get('quality_score','—')}/10",expanded=False):
                        ec1,ec2=st.columns([1,1])
                        with ec1:
                            if e.get("screenshot_path") and Path(e["screenshot_path"]).exists():
                                st.image(e["screenshot_path"],use_container_width=True)
                        with ec2:
                            if e.get("entry_price"): st.markdown(f"**Entry:** `{e['entry_price']:.4f}`")
                            if e.get("pnl_r"):
                                c4="#10B981" if e["pnl_r"]>=0 else "#EF4444"
                                st.markdown(f'**P&L:** <span style="color:{c4};font-weight:700">{e["pnl_r"]:+.2f}R</span>',
                                            unsafe_allow_html=True)
                            if e.get("tags"): st.markdown(f"**Tags:** {e['tags']}")
                            if ai.get("what_worked"): st.success(f"Worked: {ai['what_worked']}")
                            if ai.get("lessons"): st.info(f"Lesson: {ai['lessons']}")
                        if e.get("notes"): st.markdown(f"**Entry:**\n\n{e['notes']}")

        # ── Import CSV ────────────────────────────────────────────────────────
        with st3:
            st.markdown(
                "Import historical trades from a broker CSV export (MT4/MT5 history, cTrader, "
                "or any spreadsheet). Map CSV columns to journal fields, then click **Import**."
            )
            csv_file=st.file_uploader("Upload CSV",type=["csv"],key="import_csv")
            if csv_file:
                try:
                    import io as _io
                    raw=csv_file.getvalue().decode("utf-8",errors="replace")
                    # Try to sniff delimiter
                    import csv as _csv
                    dialect=_csv.Sniffer().sniff(raw[:2048],delimiters=",;\t|")
                    df_imp=pd.read_csv(_io.StringIO(raw),sep=dialect.delimiter,
                                       dtype=str,keep_default_na=False)
                    df_imp.columns=[c.strip() for c in df_imp.columns]
                    st.caption(f"{len(df_imp)} rows · {len(df_imp.columns)} columns detected")
                    st.dataframe(df_imp.head(5),use_container_width=True,hide_index=True)
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")
                    df_imp=None
                else:
                    cols=["(skip)"]+list(df_imp.columns)
                    st.markdown("### Map columns")
                    cm1,cm2=st.columns(2)
                    with cm1:
                        m_date =cm1.selectbox("Date",        cols, key="m_date",
                                              index=next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["date","time","open time"])),0))
                        m_asset=cm1.selectbox("Asset/Symbol",cols, key="m_asset",
                                              index=next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["symbol","asset","pair","instrument","item"])),0))
                        m_dir  =cm1.selectbox("Direction",   cols, key="m_dir",
                                              index=next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["type","direction","side","action"])),0))
                        m_entry=cm1.selectbox("Entry price", cols, key="m_entry",
                                              index=next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["entry","open","price","open price"])),0))
                        m_sl   =cm1.selectbox("Stop Loss",   cols, key="m_sl",
                                              index=next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["sl","stop","s/l"])),0))
                    with cm2:
                        m_tp   =cm2.selectbox("Take Profit", cols, key="m_tp",
                                              index=next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["tp","target","t/p","take"])),0))
                        m_exit =cm2.selectbox("Exit price",  cols, key="m_exit",
                                              index=next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["exit","close","close price"])),0))
                        m_pnl  =cm2.selectbox("P&L (R)",     cols, key="m_pnl")
                        m_usd  =cm2.selectbox("P&L ($)",     cols, key="m_usd",
                                              index=next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["profit","pnl","p&l","usd","$"])),0))
                        m_notes=cm2.selectbox("Notes",       cols, key="m_notes",
                                              index=next((i for i,c in enumerate(cols) if any(k in c.lower() for k in ["note","comment","remark"])),0))

                    date_fmt=st.text_input("Date format (strptime)",value="%Y.%m.%d %H:%M:%S",
                                           help="e.g. %Y.%m.%d %H:%M:%S  or  %d/%m/%Y",
                                           key="m_datefmt")
                    dir_map_s=st.text_input("Direction map (buy→long, sell→short)",
                                            value="buy=long,sell=short,Buy=long,Sell=short,BUY=long,SELL=short",
                                            key="m_dirmap")

                    # Preview mapped data
                    def _get(row, col):
                        return row[col].strip() if col!="(skip)" and col in row and row[col].strip() else None

                    def _parse_date(s, fmt):
                        if not s: return None
                        for f in [fmt,"%Y-%m-%d","%d/%m/%Y","%Y.%m.%d","%d.%m.%Y","%m/%d/%Y"]:
                            try: return datetime.strptime(s,f).date().isoformat()
                            except: pass
                        return s[:10]  # fallback: first 10 chars

                    def _parse_dir(s, dmap):
                        if not s: return "unknown"
                        s=s.strip()
                        for pair in dmap.split(","):
                            if "=" in pair:
                                k,v=pair.split("=",1)
                                if s.lower()==k.lower(): return v.lower()
                        return s.lower()

                    def _parse_float(s):
                        if not s: return None
                        try: return float(s.replace(",",".").replace(" ",""))
                        except: return None

                    dir_map_parsed=dir_map_s

                    if st.button("Preview mapping (first 5 rows)",key="m_preview"):
                        prev=[]
                        for _,row in df_imp.head(5).iterrows():
                            prev.append({
                                "date":_parse_date(_get(row,m_date),date_fmt),
                                "asset":_get(row,m_asset),
                                "direction":_parse_dir(_get(row,m_dir),dir_map_parsed),
                                "entry":_parse_float(_get(row,m_entry)),
                                "sl":_parse_float(_get(row,m_sl)),
                                "tp":_parse_float(_get(row,m_tp)),
                                "exit":_parse_float(_get(row,m_exit)),
                                "pnl_r":_parse_float(_get(row,m_pnl)),
                                "pnl_$":_parse_float(_get(row,m_usd)),
                                "notes":_get(row,m_notes),
                            })
                        st.dataframe(pd.DataFrame(prev),use_container_width=True,hide_index=True)

                    st.markdown("---")
                    imp1,imp2=st.columns(2)
                    skip_dup=imp1.checkbox("Skip duplicates (same date + asset + entry)",value=True,key="m_skipdup")
                    if imp2.button("Import all rows",type="primary",key="m_import"):
                        imported=0; skipped=0; errors=[]
                        existing_keys={(e.get("entry_date",""),e.get("asset",""),
                                        str(round(e.get("entry_price") or 0,4)))
                                       for e in _manual_entries()} if skip_dup else set()
                        for _,row in df_imp.iterrows():
                            try:
                                d=_parse_date(_get(row,m_date),date_fmt)
                                asset_v=_get(row,m_asset) or "Unknown"
                                entry_v=_parse_float(_get(row,m_entry))
                                key=(d or "",asset_v,str(round(entry_v or 0,4)))
                                if skip_dup and key in existing_keys:
                                    skipped+=1; continue
                                save_manual_entry(_jclient, {
                                    "entry_date": d or date.today().isoformat(),
                                    "asset":      asset_v,
                                    "direction":  _parse_dir(_get(row,m_dir),dir_map_parsed),
                                    "entry_price":entry_v,
                                    "stop_loss":  _parse_float(_get(row,m_sl)),
                                    "take_profit":_parse_float(_get(row,m_tp)),
                                    "exit_price": _parse_float(_get(row,m_exit)),
                                    "pnl_r":      _parse_float(_get(row,m_pnl)),
                                    "pnl_dollars":_parse_float(_get(row,m_usd)),
                                    "notes":      _get(row,m_notes) or "",
                                    "tags":       "imported",
                                })
                                existing_keys.add(key)
                                imported+=1
                            except Exception as ex:
                                errors.append(str(ex))
                        _clear()
                        st.success(f"Imported {imported} trades · {skipped} duplicates skipped.")
                        if errors:
                            st.warning(f"{len(errors)} rows failed: {errors[:3]}")
                        st.rerun()

    # ── Psychology ─────────────────────────────────────────────────────────────
    with t_psy:
        st.markdown("Record your mental state before and after each trade. Identify patterns "
                    "between your psychology and your trading performance.")
        psych_all=_psychology_all()
        psy1,psy2=st.tabs(["New Entry","History & Patterns"])
        STATES=["calm","optimal","anxious","overconfident","fomo","fearful","bored","frustrated"]
        MISTAKES=["Moved SL","Exited early","Sized too large","Chased entry",
                  "Held past TP","Ignored setup rules","Revenge traded","None"]
        with psy1:
            ptab_t=sorted(all_t,key=lambda x:x["signal_date"],reverse=True)
            if not ptab_t: st.info("No trades to review.")
            else:
                already={p["trade_ref_id"] for p in psych_all}
                unreviewed=[t for t in ptab_t
                            if f"{t['signal_date']}|{t['ticker']}|{t.get('strategy','')}" not in already]
                if unreviewed:
                    st.markdown(f"**{len(unreviewed)} unreviewed trade(s):**")
                tlbl=[f"{t['signal_date']} | {_label(t)} | {t['status'].upper()}" for t in ptab_t]
                sel_pi=st.selectbox("Select trade to review",range(len(tlbl)),format_func=lambda i:tlbl[i])
                sel_pt=ptab_t[sel_pi]
                ref_id=f"{sel_pt['signal_date']}|{sel_pt['ticker']}|{sel_pt.get('strategy','')}"
                with st.form("psy_form"):
                    pp1,pp2=st.columns(2)
                    pp1.markdown("**Before the trade:**")
                    pre_s=pp1.selectbox("Pre-trade state",STATES,key="p_pre_s")
                    pre_c=pp1.slider("Pre-trade confidence",1,10,7,key="p_pre_c")
                    pre_n=pp1.text_area("Pre-trade notes",height=80,key="p_pre_n",
                                        placeholder="How were you feeling? Any hesitation?")
                    pp2.markdown("**After the trade:**")
                    post_s=pp2.selectbox("Post-trade state",STATES,key="p_post_s")
                    eq=pp2.slider("Execution quality (1=poor, 10=perfect)",1,10,7,key="p_eq")
                    post_n=pp2.text_area("Post-trade notes",height=80,key="p_post_n",
                                         placeholder="Did emotions affect your execution?")
                    mistakes=st.multiselect("Mistakes made",MISTAKES,key="p_mistakes")
                    lesson=st.text_input("One-sentence lesson",key="p_lesson")
                    if st.form_submit_button("Save Psychology Entry",type="primary"):
                        save_psychology(_jclient, {"trade_ref_type":"live","trade_ref_id":ref_id,
                            "pre_state":pre_s,"pre_confidence":pre_c,"pre_notes":pre_n,
                            "post_state":post_s,"post_notes":post_n,"execution_quality":eq,
                            "mistakes":mistakes,"lesson":lesson})
                        _clear(); st.success("Psychology entry saved."); st.rerun()
        with psy2:
            if not psych_all: st.info("No psychology entries yet.")
            else:
                import json as _json
                states=[p.get("pre_state","?") for p in psych_all if p.get("pre_state")]
                if states:
                    st.subheader("Pre-Trade State Distribution")
                    sc={}
                    for s in states: sc[s]=sc.get(s,0)+1
                    fig_p=go.Figure(go.Bar(x=list(sc.keys()),y=list(sc.values()),
                        marker_color=["#10B981" if s in ("calm","optimal") else "#FBBF24" if s in ("anxious","fearful") else "#EF4444" for s in sc.keys()],
                        hovertemplate="%{x}: %{y}<extra></extra>"))
                    fig_p.update_layout(**_PL,height=220,title="Pre-Trade States",showlegend=False)
                    st.plotly_chart(fig_p,use_container_width=True)
                # Mistakes frequency
                all_mistakes=[]
                for p in psych_all:
                    m=p.get("mistakes")
                    if m:
                        try: all_mistakes.extend(_json.loads(m) if isinstance(m,str) else m)
                        except: pass
                if all_mistakes:
                    mc={}
                    for m in all_mistakes: mc[m]=mc.get(m,0)+1
                    mc.pop("None",None)
                    if mc:
                        fig_m=go.Figure(go.Bar(x=list(mc.values()),y=list(mc.keys()),
                            orientation="h",marker_color="#EF4444",
                            hovertemplate="%{y}: %{x}×<extra></extra>"))
                        fig_m.update_layout(**_PL,height=200,title="Most Common Mistakes",showlegend=False)
                        st.plotly_chart(fig_m,use_container_width=True)
                for p in psych_all[:10]:
                    if p.get("lesson"):
                        st.markdown(f'- <span style="color:#5A6E85">{p["recorded_at"][:10]}</span>'
                                    f' &nbsp; {p["lesson"]}',unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────────────────────────────────
    with t_met:
        closed=[t for t in all_t if t["status"] not in ("pending","filled","expired")]
        if not closed: st.info("No closed trades yet."); return

        dr=st.selectbox("Date range",["All time","Last 90 days","Last 30 days","Last 10 trades"],key="met_dr")
        if dr=="Last 90 days": closed=[t for t in closed if t["signal_date"]>=(date.today()-timedelta(days=90)).isoformat()]
        elif dr=="Last 30 days": closed=[t for t in closed if t["signal_date"]>=(date.today()-timedelta(days=30)).isoformat()]
        elif dr=="Last 10 trades": closed=sorted(closed,key=lambda x:x["signal_date"],reverse=True)[:10]

        wins=[t for t in closed if t["status"] in ("win","partial_win")]
        losses=[t for t in closed if t["status"]=="loss"]
        total_r=sum(t["pnl_r"] for t in closed if t["pnl_r"] is not None)
        wr=len(wins)/len(closed) if closed else 0
        win_rs=[t["pnl_r"] for t in wins if t["pnl_r"]]
        loss_rs=[t["pnl_r"] for t in losses if t["pnl_r"]]
        avg_w=sum(win_rs)/len(win_rs) if win_rs else 0
        avg_l=sum(loss_rs)/len(loss_rs) if loss_rs else 0
        pf=abs(sum(win_rs)/sum(loss_rs)) if loss_rs and sum(loss_rs)!=0 else float("inf")
        streak,skind=_compute_streak(closed)

        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("Trades",len(closed))
        m2.metric("Win Rate",f"{wr:.1%}")
        m3.metric("Avg Win",f"{avg_w:+.2f}R")
        m4.metric("Avg Loss",f"{avg_l:+.2f}R")
        m5.metric("Profit Factor",f"{pf:.2f}" if pf!=float("inf") else "∞")
        m6.metric("Total R",f"{total_r:+.1f}R")

        st.markdown(f"Current streak: **{streak}-trade {skind}**")

        mc1,mc2=st.columns([3,2])
        with mc1:
            # Rolling WR chart
            rdates,rwrs=_rolling_wr(closed,window=min(20,len(closed)))
            if rdates:
                fig_rw=go.Figure()
                fig_rw.add_trace(go.Scatter(x=rdates,y=[w*100 for w in rwrs],mode="lines",
                    line=dict(color="#60A5FA",width=2),
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",name="Win Rate"))
                fig_rw.add_hline(y=50,line_dash="dot",line_color="#5A6E85",line_width=1)
                fig_rw.update_layout(**_PL,title=f"Rolling Win Rate ({min(20,len(closed))}-trade window)",
                                     height=240,showlegend=False)
                fig_rw.update_yaxes(ticksuffix="%",range=[0,100])
                st.plotly_chart(fig_rw,use_container_width=True)
        with mc2:
            # Day of week P&L
            if len(closed)>=5:
                dow_pnl={i:[] for i in range(7)}
                for t in closed:
                    if t.get("pnl_r"):
                        try: dow_pnl[datetime.strptime(t["signal_date"],"%Y-%m-%d").weekday()].append(t["pnl_r"])
                        except: pass
                dow_names=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                dow_avgs=[sum(v)/len(v) if v else 0 for v in dow_pnl.values()]
                fig_dow=go.Figure(go.Bar(x=dow_names,y=dow_avgs,
                    marker_color=["#10B981" if v>=0 else "#EF4444" for v in dow_avgs],
                    hovertemplate="%{x}: %{y:+.2f}R avg<extra></extra>"))
                fig_dow.update_layout(**_PL,title="Avg R by Day of Week",height=240,showlegend=False)
                st.plotly_chart(fig_dow,use_container_width=True)

        # Best / worst trades
        bc1,bc2=st.columns(2)
        with bc1:
            st.markdown("**Best 5 trades:**")
            best=sorted([t for t in closed if t.get("pnl_r")],key=lambda x:x["pnl_r"],reverse=True)[:5]
            if best: st.dataframe(_trades_df(best)[["Signal (RO)","Asset","Strategy","P&L"]],use_container_width=True,hide_index=True)
        with bc2:
            st.markdown("**Worst 5 trades:**")
            worst=sorted([t for t in closed if t.get("pnl_r")],key=lambda x:x["pnl_r"])[:5]
            if worst: st.dataframe(_trades_df(worst)[["Signal (RO)","Asset","Strategy","P&L"]],use_container_width=True,hide_index=True)

        # Execution analysis
        st.markdown("---")
        st.subheader("Execution Analysis")
        filled=[t for t in closed if t.get("fill_price") and t.get("entry_high")]
        if filled:
            slippage=[abs(t["fill_price"]-t["entry_high"]) for t in filled if t["direction"]=="long"]
            avg_slip=sum(slippage)/len(slippage) if slippage else 0
            ea1,ea2,ea3=st.columns(3)
            ea1.metric("Avg Entry Slippage",f"{avg_slip:.4f}",
                       help="(fill_price - entry_high) for longs — positive = chasing")
            ea2.metric("Avg Win R",f"{avg_w:+.2f}R",
                       help="Actual R realized on winning trades")
            ea3.metric("Avg Loss R",f"{avg_l:+.2f}R",
                       help="Actual R on losing trades (should be close to -1.0R)")
        else: st.caption("Fill prices needed for execution analysis.")


# =============================================================================
# PAGE 7: SETTINGS
# =============================================================================
def page_settings():
    st.title("Settings")
    t0,t1,t2,t3,t4,t5,t6,t7,t8=st.tabs(
        ["🔌 Broker","💰 Account","🛡️ Risk Rules","🔔 Alerts","📋 Assets","🧠 Analysis Engine","🔑 API Keys","⚙️ System","🔑 License"])

    # ── Broker (MT5) ───────────────────────────────────────────────────────────
    with t0:
        from p4_live import mt5_broker
        env = ROOT / ".env"

        # Live connection status
        mt5_enabled = os.getenv("MT5_ENABLED","").strip().lower() == "true"
        if mt5_enabled:
            acct_info = mt5_broker.get_account_info()
            if acct_info:
                st.markdown(
                    f'<div style="background:#052E1C;border:1px solid #10B98155;border-radius:10px;'
                    f'padding:.8rem 1.2rem;margin-bottom:1rem;display:flex;align-items:center;gap:1.5rem">'
                    f'<span style="color:#10B981;font-size:1.1rem;font-weight:800">● CONNECTED</span>'
                    f'<span style="color:#94A3B8;font-size:.85rem">'
                    f'Account <b style="color:#F1F5F9">{acct_info["login"]}</b> · '
                    f'{acct_info["server"]} · '
                    f'Balance <b style="color:#10B981">${acct_info["balance"]:,.2f}</b> · '
                    f'Equity <b style="color:#60A5FA">${acct_info["equity"]:,.2f}</b> · '
                    f'{acct_info["leverage"]}x leverage'
                    f'</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="background:#1C0505;border:1px solid #EF444455;border-radius:10px;'
                    'padding:.8rem 1.2rem;margin-bottom:1rem">'
                    '<span style="color:#EF4444;font-weight:700">● ENABLED BUT NOT CONNECTED</span>'
                    '<span style="color:#94A3B8;font-size:.82rem;margin-left:1rem">Check credentials and that MT5 terminal is running</span>'
                    '</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#101826;border:1px solid #1A2840;border-radius:10px;'
                'padding:.8rem 1.2rem;margin-bottom:1rem">'
                '<span style="color:#5A6E85;font-weight:600">● DISABLED</span>'
                '<span style="color:#5A6E85;font-size:.82rem;margin-left:1rem">Enable below to start live execution</span>'
                '</div>', unsafe_allow_html=True)

        with st.form("broker_form"):
            st.markdown("**MetaTrader 5 Connection**")
            en_col, _ = st.columns([1, 3])
            mt5_on = en_col.toggle("Enable live MT5 execution", value=mt5_enabled)

            bc1, bc2, bc3 = st.columns(3)
            mt5_acc  = bc1.text_input("Account number",
                                      value=os.getenv("MT5_ACCOUNT",""),
                                      placeholder="52840847")
            mt5_pwd  = bc2.text_input("Password",
                                      value=os.getenv("MT5_PASSWORD",""),
                                      type="password")
            mt5_srv  = bc3.text_input("Server",
                                      value=os.getenv("MT5_SERVER",""),
                                      placeholder="ICMarketsEU-Demo")

            bd1, bd2 = st.columns(2)
            mt5_path = bd1.text_input("Terminal path (leave blank to auto-detect)",
                                      value=os.getenv("MT5_TERMINAL_PATH",""),
                                      placeholder=r"C:\Program Files\MetaTrader 5\terminal64.exe")
            mt5_risk = bd2.number_input("Risk per trade (% of equity)",
                                        value=float(os.getenv("MT5_RISK_PCT","0.01"))*100,
                                        min_value=0.1, max_value=10.0, step=0.1, format="%.1f",
                                        help="1% means you risk 1% of account equity per signal")

            mt5_kelly = st.checkbox(
                "Use Kelly position sizing",
                value=os.getenv("MT5_KELLY_SIZING", "false").lower() == "true",
                help="When enabled, position size is scaled by each asset's 25% fractional Kelly "
                     "fraction (from assets.yaml). Overrides the flat risk % above for assets that "
                     "have kelly_25pct configured.",
            )

            st.caption("Risk example: $10,000 account × 1% = $100 risked per trade. "
                       "Position size is calculated automatically from the signal's stop-loss distance.")

            if st.form_submit_button("Save Broker Settings", type="primary"):
                set_key(str(env), "MT5_ENABLED",       "true" if mt5_on else "false")
                set_key(str(env), "MT5_ACCOUNT",        mt5_acc)
                set_key(str(env), "MT5_PASSWORD",       mt5_pwd)
                set_key(str(env), "MT5_SERVER",         mt5_srv)
                set_key(str(env), "MT5_TERMINAL_PATH",  mt5_path)
                set_key(str(env), "MT5_RISK_PCT",       f"{mt5_risk/100:.4f}")
                set_key(str(env), "MT5_KELLY_SIZING",   "true" if mt5_kelly else "false")
                os.environ["MT5_ENABLED"]      = "true" if mt5_on else "false"
                os.environ["MT5_ACCOUNT"]      = mt5_acc
                os.environ["MT5_PASSWORD"]     = mt5_pwd
                os.environ["MT5_SERVER"]       = mt5_srv
                os.environ["MT5_RISK_PCT"]     = f"{mt5_risk/100:.4f}"
                os.environ["MT5_KELLY_SIZING"] = "true" if mt5_kelly else "false"
                st.success("Broker settings saved. Restart the scheduler to apply.")

        st.markdown("---")
        st.markdown("**Test Connection**")
        if st.button("Connect to MT5 & show account info", use_container_width=False):
            with st.spinner("Connecting…"):
                info = mt5_broker.get_account_info()
            if info:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Balance",  f"${info['balance']:,.2f}")
                c2.metric("Equity",   f"${info['equity']:,.2f}")
                c3.metric("Margin",   f"${info['margin']:,.2f}")
                c4.metric("Leverage", f"{info['leverage']}x")
                st.success(f"Connected — {info['server']}  account #{info['login']}")
            else:
                st.error("Connection failed. Check credentials and ensure MT5 terminal is running.")

    # ── Account ────────────────────────────────────────────────────────────────
    with t1:
        acct=_acct()
        with st.form("acct_form"):
            sc1,sc2,sc3=st.columns(3)
            new_bal=sc1.number_input("Account Balance ($)",value=acct["balance"],step=100.0)
            new_lev=sc2.number_input("Leverage",value=acct["leverage"],min_value=1,max_value=1000)
            new_rsk=sc3.number_input("Risk / Trade (%)",value=acct["risk_pct"],min_value=0.1,max_value=10.0,step=0.1,format="%.1f")
            st.markdown(f"→ Dollar risk per trade: **${new_bal*new_rsk/100:.2f}**")
            if st.form_submit_button("Save Account Settings",type="primary"):
                _save_acct(new_bal,int(new_lev),new_rsk)
                st.success("Saved.")

    # ── Risk rules ─────────────────────────────────────────────────────────────
    with t2:
        st.subheader("Risk Rules & Limits")
        env=ROOT/".env"
        try:
            from p4_live.journal_supabase import get_user_settings as _gus
            _rr_us = _gus(st.session_state["supabase_client"])
        except Exception:
            _rr_us = {}
        with st.form("risk_rules"):
            rr1,rr2=st.columns(2)
            max_trades=rr1.number_input("Max simultaneous open trades",
                                         value=int(_rr_us.get("max_open_trades") or os.getenv("MAX_OPEN_TRADES","5")),min_value=1)
            daily_loss=rr2.number_input("Daily loss limit (R) — 0 = disabled",
                                          value=float(_rr_us.get("daily_loss_limit_r") or os.getenv("DAILY_LOSS_LIMIT_R","0")),min_value=0.0,step=0.5)
            rr3,rr4=st.columns(2)
            weekly_dd=rr3.number_input("Weekly drawdown limit (R) — 0 = disabled",
                                         value=float(_rr_us.get("weekly_dd_limit_r") or os.getenv("WEEKLY_DD_LIMIT_R","0")),min_value=0.0,step=0.5)
            max_pos_risk=rr4.number_input("Max single position risk (%)",
                                            value=float(os.getenv("MAX_POSITION_RISK_PCT","2.0")),min_value=0.1,step=0.1)
            if st.form_submit_button("Save Risk Rules",type="primary"):
                try:
                    from p4_live.journal_supabase import save_user_settings as _sus
                    _sus(st.session_state["supabase_client"], {
                        "max_open_trades": max_trades,
                        "daily_loss_limit_r": daily_loss,
                        "weekly_dd_limit_r": weekly_dd,
                    })
                except Exception:
                    pass
                set_key(str(env),"MAX_OPEN_TRADES",str(max_trades))
                set_key(str(env),"DAILY_LOSS_LIMIT_R",str(daily_loss))
                set_key(str(env),"WEEKLY_DD_LIMIT_R",str(weekly_dd))
                set_key(str(env),"MAX_POSITION_RISK_PCT",str(max_pos_risk))
                st.success("Risk rules saved.")
        # Status check
        all_t=_trades()
        closed_today=[t for t in all_t if t.get("exit_date")==date.today().isoformat() and t.get("pnl_r") is not None]
        daily_pnl=sum(t["pnl_r"] for t in closed_today)
        st.markdown("**Current status:**")
        sc1,sc2,sc3=st.columns(3)
        open_ct=len([t for t in all_t if t["status"] in ("pending","filled")])
        sc1.metric("Open trades",f"{open_ct} / {max_trades}",delta_color="off")
        sc2.metric("Today's P&L",f"{daily_pnl:+.2f}R",
                   f"Limit: -{daily_loss:.1f}R" if daily_loss>0 else "No limit")
        sc3.metric("Max pos risk",f"{max_pos_risk:.1f}%")

    # ── Alerts ─────────────────────────────────────────────────────────────────
    with t3:
        from p4_live.alerts import configured_channels
        channels=configured_channels()
        if channels:
            for ch in channels: st.success(f"{ch} — configured")
        else: st.warning("No alert channels configured.")
        with st.form("tg_form"):
            st.markdown("**Telegram**")
            try:
                from p4_live.journal_supabase import get_user_settings as _gus
                _tg_us = _gus(st.session_state["supabase_client"])
            except Exception:
                _tg_us = {}
            tg1,tg2=st.columns(2)
            tg_tok=tg1.text_input("Bot Token",value=_tg_us.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN",""),type="password")
            tg_cid=tg2.text_input("Chat ID",value=_tg_us.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID",""))
            if st.form_submit_button("Save Telegram"):
                try:
                    from p4_live.journal_supabase import save_user_settings as _sus
                    _sus(st.session_state["supabase_client"], {"telegram_bot_token": tg_tok, "telegram_chat_id": tg_cid})
                except Exception:
                    pass
                os.environ["TELEGRAM_BOT_TOKEN"]=tg_tok; os.environ["TELEGRAM_CHAT_ID"]=tg_cid
                st.success("Saved.")
        with st.form("em_form"):
            st.markdown("**Email (SMTP)**")
            em1,em2=st.columns(2)
            em_from=em1.text_input("From",value=os.getenv("ALERT_EMAIL_FROM",""))
            em_to=em2.text_input("To",value=os.getenv("ALERT_EMAIL_TO",""))
            em_pw=st.text_input("App password",value=os.getenv("ALERT_EMAIL_PASSWORD",""),type="password")
            if st.form_submit_button("Save Email"):
                e=str(ROOT/".env"); set_key(e,"ALERT_EMAIL_FROM",em_from); set_key(e,"ALERT_EMAIL_TO",em_to); set_key(e,"ALERT_EMAIL_PASSWORD",em_pw)
                st.success("Saved.")
        if st.button("Send Test Alert"):
            with st.spinner("Sending..."):
                from p4_live.alerts import alert_test
                res=alert_test()
            for ch,ok in res.items():
                if ok: st.success(f"{ch}: delivered")
                else: st.error(f"{ch}: failed or not configured")

    # ── Assets ─────────────────────────────────────────────────────────────────
    with t4:
        assets_path = ROOT / "config" / "assets.yaml"
        assets      = _assets()
        strat_ids   = list(_strategies().keys())

        def _strat_labels(strat_list):
            """Normalise mixed string/dict strategy entries to a list of IDs."""
            return [s if isinstance(s, str) else s.get("id","?") for s in strat_list]

        CAT_ICON = {"index":"📈","crypto":"₿","commodity":"🥇","forex":"💱","stock":"🏢"}

        # ── My Watchlist (per-user, Supabase-backed) ──────────────────────────
        st.markdown("#### My Watchlist")
        st.caption(
            "Choose which markets and strategies the scanner runs for your account. "
            "Each user has their own selection — changes here don't affect other users."
        )
        try:
            from p4_live.journal_supabase import (
                get_user_watchlist as _guwl,
                replace_user_watchlist as _ruwl,
            )
            _wl_client = st.session_state["supabase_client"]
            _cur_wl = set(_guwl(_wl_client))
        except Exception:
            _cur_wl = set()
            _wl_client = None

        _all_combos: list[tuple[str, str, str, str]] = []  # (asset_id, label, strat_id, tf)
        for _wa in [a for a in assets if a.get("enabled", True)]:
            _waid = _wa["id"]
            for _wentry in _wa.get("strategies", []):
                if isinstance(_wentry, str):
                    _wsid = _wentry
                    _wtfs = [_strategies().get(_wsid, {}).get("default_timeframe", "1d")]
                else:
                    _wsid = _wentry.get("id", "")
                    _wtfs = _wentry.get("timeframes", ["1d"])
                for _wtf in _wtfs:
                    _all_combos.append((_waid, _wa.get("label", _waid), _wsid, _wtf))

        if _all_combos:
            _wl_cols = st.columns(3)
            for _ci, (_waid, _walabel, _wsid, _wtf) in enumerate(_all_combos):
                _wkey = f"wl_{_waid}_{_wsid}_{_wtf}"
                _wdefault = (_waid, _wsid, _wtf) in _cur_wl
                _icon = CAT_ICON.get(
                    next((a.get("category","") for a in assets if a["id"]==_waid), ""), "📊"
                )
                _wl_cols[_ci % 3].checkbox(
                    f"{_icon} {_walabel} · {_wsid} / {_wtf}",
                    value=_wdefault,
                    key=_wkey,
                )

            if st.button("Save My Watchlist", type="primary"):
                if _wl_client:
                    _new_entries = [
                        {"asset_id": waid, "strategy_id": wsid, "timeframe": wtf,
                         "params": {}, "enabled": True}
                        for waid, _, wsid, wtf in _all_combos
                        if st.session_state.get(f"wl_{waid}_{wsid}_{wtf}", False)
                    ]
                    _ruwl(_wl_client, _new_entries)
                    st.success(f"Watchlist saved — {len(_new_entries)} active combination(s).")
                    st.rerun()
                else:
                    st.error("Supabase not connected.")
        else:
            st.info("No enabled assets in the global catalog yet. Add one below.")

        st.markdown("---")

        # ── Asset cards grid ──────────────────────────────────────────────────
        st.markdown("#### Global Catalog")
        n_cols = 3
        card_cols = st.columns(n_cols)
        for idx, a in enumerate(assets):
            enabled   = a.get("enabled", True)
            strats    = _strat_labels(a.get("strategies", []))
            mt5_sym   = a.get("tickers", {}).get("mt5", "")
            cat_icon  = CAT_ICON.get(a.get("category",""), "📊")
            border_c  = "#1E3A5F" if enabled else "#1A2840"
            status_c  = "#10B981" if enabled else "#5A6E85"
            status_t  = "ACTIVE" if enabled else "DISABLED"

            with card_cols[idx % n_cols]:
                st.markdown(
                    f'<div style="background:#0D1824;border:1px solid {border_c};border-radius:10px;'
                    f'padding:.75rem 1rem;margin-bottom:.5rem">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem">'
                    f'<span style="font-weight:700;color:#F1F5F9;font-size:.95rem">{cat_icon} {a["label"]}</span>'
                    f'<span style="font-size:.65rem;font-weight:700;color:{status_c};'
                    f'background:{status_c}22;padding:2px 7px;border-radius:4px">{status_t}</span>'
                    f'</div>'
                    f'<div style="font-size:.72rem;color:#5A6E85;margin-bottom:.3rem">'
                    f'<span style="color:#3B82F6">{a.get("tickers",{}).get("yfinance","—")}</span>'
                    + (f' &nbsp;·&nbsp; MT5: <span style="color:#A78BFA">{mt5_sym}</span>' if mt5_sym else '')
                    + f'</div>'
                    f'<div style="font-size:.7rem;color:#94A3B8">'
                    + " ".join(f'<span style="background:#1A2840;padding:1px 5px;border-radius:3px;margin-right:3px">{s}</span>' for s in strats)
                    + f'</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Edit an asset ─────────────────────────────────────────────────────
        st.markdown("#### Edit asset")
        sel_aid = st.selectbox("Select asset", [a["id"] for a in assets],
                               format_func=lambda i: next((a["label"] for a in assets if a["id"]==i), i),
                               key="assets_sel_edit")
        a_cfg = next((a for a in assets if a["id"] == sel_aid), None)
        if a_cfg:
            with st.form(f"edit_asset_{sel_aid}"):
                ea1, ea2 = st.columns(2)
                new_enabled = ea1.toggle("Enabled", value=a_cfg.get("enabled", True))
                new_mt5 = ea2.text_input("MT5 symbol",
                                         value=a_cfg.get("tickers",{}).get("mt5",""),
                                         placeholder="e.g. XAUUSD")
                cur_strats = _strat_labels(a_cfg.get("strategies", []))
                new_s = st.multiselect("Active strategies", strat_ids, default=cur_strats,
                                       help="Select which strategies scan this market")
                if st.form_submit_button("Save changes", type="primary"):
                    d = _read_yaml(assets_path)
                    for a in d["assets"]:
                        if a["id"] == sel_aid:
                            a["enabled"] = new_enabled
                            a["strategies"] = new_s
                            if new_mt5:
                                a.setdefault("tickers", {})["mt5"] = new_mt5
                            break
                    _write_yaml(assets_path, d)
                    _clear_config()
                    st.success("Saved.")
                    st.rerun()

            if st.button("🗑 Delete asset", key=f"del_btn_{sel_aid}"):
                st.session_state[f"del_asset_{sel_aid}"] = True
            if st.session_state.get(f"del_asset_{sel_aid}"):
                st.warning(f"Delete '{sel_aid}'? Existing trades are preserved.")
                dc1, dc2 = st.columns(2)
                if dc1.button("Confirm delete", type="primary", key=f"del_confirm_{sel_aid}"):
                    d = _read_yaml(assets_path)
                    d["assets"] = [a for a in d["assets"] if a["id"] != sel_aid]
                    _write_yaml(assets_path, d)
                    _clear_config()
                    st.session_state.pop(f"del_asset_{sel_aid}", None)
                    st.success(f"Deleted '{sel_aid}'.")
                    st.rerun()
                if dc2.button("Cancel", key=f"del_cancel_{sel_aid}"):
                    st.session_state.pop(f"del_asset_{sel_aid}", None)
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Add new market")
        with st.form("add_asset"):
            na1, na2, na3 = st.columns(3)
            na_id    = na1.text_input("ID (lowercase, no spaces)", placeholder="nasdaq100")
            na_label = na2.text_input("Display name", placeholder="NASDAQ 100")
            na_cat   = na3.selectbox("Category", ["index","commodity","crypto","forex","stock"])
            nb1, nb2, nb3 = st.columns(3)
            na_yf    = nb1.text_input("yfinance ticker", placeholder="^NDX")
            na_mt5   = nb2.text_input("MT5 symbol", placeholder="US100")
            na_strats = nb3.multiselect("Strategies", strat_ids, default=["mtf_trend"])
            if st.form_submit_button("Add market", type="primary"):
                if na_id and na_label and na_yf:
                    d = _read_yaml(assets_path)
                    new_entry: dict = {
                        "id": na_id, "label": na_label,
                        "tickers": {"yfinance": na_yf},
                        "category": na_cat,
                        "enabled": True,
                        "strategies": na_strats,
                    }
                    if na_mt5:
                        new_entry["tickers"]["mt5"] = na_mt5
                    d["assets"].append(new_entry)
                    _write_yaml(assets_path, d)
                    _clear_config()
                    st.success(f"Added '{na_label}'.")
                    st.rerun()
                else:
                    st.error("ID, display name and yfinance ticker are required.")

    # ── Analysis Engine ────────────────────────────────────────────────────────
    with t5:
        st.subheader("Analysis Engine Configuration")
        env=ROOT/".env"

        # API key status
        st.markdown("**Data source status:**")
        ss1,ss2,ss3,ss4,ss5=st.columns(5)
        ss1.metric("Anthropic API",  "✅ Set" if os.getenv("ANTHROPIC_API_KEY") else "❌ Missing")
        ss2.metric("FRED API",       "✅ Set" if os.getenv("FRED_API_KEY") else "⚠️ Optional")
        ss3.metric("NewsAPI",        "✅ Set" if os.getenv("NEWSAPI_KEY") else "⚠️ Optional")
        ss4.metric("Twelve Data",    "✅ Set" if os.getenv("TWELVEDATA_API_KEY") else "⚠️ Optional")
        ss5.metric("Data Provider",  os.getenv("DATA_PROVIDER","yfinance").upper())

        st.markdown("---")
        st.markdown("**Engine components:**")
        st.markdown("""
| Component | What it fetches | Impact |
|-----------|----------------|--------|
| **Technical** | OHLCV, EMA 20/50/200, RSI, MACD, ATR, ADX, Bollinger, Stochastic, CMF | Required — core signal |
| **Macro** | FRED: Fed Funds, yield curve, CPI, VIX, USD, unemployment | Filters bull/bear regime |
| **News** | yfinance headlines + NewsAPI articles (48h) | Sentiment context |
| **Geopolitical** | Asset-class risk scoring (equity/crypto/commodity) | Risk flag |
""")
        with st.form("ae_form"):
            ae1,ae2=st.columns(2)
            inc_macro=ae1.checkbox("Include Macro",value=os.getenv("ANALYSIS_INCLUDE_MACRO","1")=="1")
            inc_news=ae2.checkbox("Include News",value=os.getenv("ANALYSIS_INCLUDE_NEWS","1")=="1")
            ae3,ae4=st.columns(2)
            inc_geo=ae3.checkbox("Include Geopolitical",value=os.getenv("ANALYSIS_INCLUDE_GEO","1")=="1")
            ext_think=ae4.checkbox("Extended Thinking (Opus only)",value=os.getenv("ANALYSIS_EXTENDED_THINKING","0")=="1")
            def_model=st.selectbox("Default Model",["claude-sonnet-4-6","claude-opus-4-6"],
                                   index=0 if os.getenv("DEFAULT_ANALYSIS_MODEL","claude-sonnet-4-6")=="claude-sonnet-4-6" else 1)
            news_hours=st.slider("News lookback (hours)",12,72,int(os.getenv("NEWS_LOOKBACK_HOURS","48")))
            if st.form_submit_button("Save Engine Config",type="primary"):
                e=str(env)
                set_key(e,"ANALYSIS_INCLUDE_MACRO","1" if inc_macro else "0")
                set_key(e,"ANALYSIS_INCLUDE_NEWS","1" if inc_news else "0")
                set_key(e,"ANALYSIS_INCLUDE_GEO","1" if inc_geo else "0")
                set_key(e,"ANALYSIS_EXTENDED_THINKING","1" if ext_think else "0")
                set_key(e,"DEFAULT_ANALYSIS_MODEL",def_model)
                set_key(e,"NEWS_LOOKBACK_HOURS",str(news_hours))
                st.success("Engine config saved.")
        if st.button("Clear Analysis Cache"):
            st.cache_data.clear(); st.success("All cached data cleared.")

    # ── API Keys ───────────────────────────────────────────────────────────────
    with t6:
        def _mk(k): v=os.getenv(k,""); return v[:8]+"…" if len(v)>10 else ("(not set)" if not v else v)
        st.markdown(f"**Anthropic:** `{_mk('ANTHROPIC_API_KEY')}`")
        st.markdown(f"**FRED:** `{_mk('FRED_API_KEY')}`")
        st.markdown(f"**NewsAPI:** `{_mk('NEWSAPI_KEY')}`")
        st.markdown(f"**Twelve Data:** `{_mk('TWELVEDATA_API_KEY')}`")
        st.caption("Twelve Data — free tier at twelvedata.com · 800 credits/day · best for Gold, Silver, Forex")
        with st.form("key_form"):
            nk1=st.text_input("Anthropic key",type="password",placeholder="sk-ant-...")
            nk2=st.text_input("FRED key",type="password")
            nk3=st.text_input("NewsAPI key",type="password")
            nk4=st.text_input("Twelve Data key",type="password",placeholder="your_twelvedata_key")
            ka1,ka2=st.columns(2)
            new_dp=ka1.selectbox("Data Provider (for backtester/scanner)",
                                 ["yfinance","twelvedata","eodhd"],
                                 index=["yfinance","twelvedata","eodhd"].index(
                                     os.getenv("DATA_PROVIDER","yfinance")))
            if st.form_submit_button("Save Keys",type="primary"):
                e=str(ROOT/".env")
                if nk1: set_key(e,"ANTHROPIC_API_KEY",nk1)
                if nk2: set_key(e,"FRED_API_KEY",nk2)
                if nk3: set_key(e,"NEWSAPI_KEY",nk3)
                if nk4: set_key(e,"TWELVEDATA_API_KEY",nk4)
                set_key(e,"DATA_PROVIDER",new_dp)
                os.environ["DATA_PROVIDER"]=new_dp
                _clear_config(); st.success("Keys saved. Reload the page to apply.")

    # ── System ─────────────────────────────────────────────────────────────────
    with t7:
        # DB stats (Supabase)
        try:
            all_t2=_trades(); man=_manual_entries(); psych_s=_psychology_all()
            ds1,ds2,ds3=st.columns(3)
            ds1.metric("Trades",len(all_t2))
            ds2.metric("Manual Entries",len(man))
            ds3.metric("Psychology Entries",len(psych_s))
        except Exception:
            st.caption("DB stats unavailable")

        # Export
        st.markdown("---")
        st.subheader("Export & Backup")
        ec1,ec2=st.columns(2)
        with ec1:
            all_t2=_trades()
            if all_t2:
                csv=_trades_df(all_t2).to_csv(index=False)
                st.download_button("Export Trades CSV",csv,"all_trades.csv","text/csv",use_container_width=True)
        with ec2:
            man=_manual_entries()
            if man:
                mdf=pd.DataFrame([{k:v for k,v in e.items() if k!="ai_analysis"} for e in man])
                st.download_button("Export Journal CSV",mdf.to_csv(index=False),"journal_entries.csv","text/csv",use_container_width=True)

        st.markdown("---")
        st.subheader("Scheduler Log")
        log=ROOT/"logs"/"scheduler.log"
        if log.exists():
            lines=log.read_text(encoding="utf-8",errors="replace").splitlines()
            _lc1,_lc2,_lc3=st.columns([2,3,2])
            n=_lc1.slider("Lines",10,500,100,step=10)
            _lvl=_lc2.selectbox("Level filter",["ALL","ERROR","WARNING","INFO","DEBUG"],index=0)
            _auto=_lc3.checkbox("Auto-refresh (30s)",value=False)
            filtered=lines
            if _lvl!="ALL":
                _lvl_map={"ERROR":["ERROR"],"WARNING":["WARNING","ERROR"],"INFO":["INFO","WARNING","ERROR"],"DEBUG":["DEBUG","INFO","WARNING","ERROR"]}
                _keep=_lvl_map.get(_lvl,[])
                filtered=[l for l in lines if any(k in l for k in _keep)]
            st.code("\n".join(filtered[-n:]),language=None)
            st.caption(f"{log} — {len(lines)} total lines, {len(filtered)} matching")
            if _auto:
                import time as _time
                _time.sleep(30)
                st.rerun()
        else: st.info("No scheduler log. Start: `python run_scheduler.py`")

        st.markdown("---")
        st.code("# Run UI\nstreamlit run ui/app.py\n\n"
                "# Scanner (once)\npython run_scheduler.py --once\n\n"
                "# Continuous scheduler\npython run_scheduler.py\n\n"
                "# Backtest\npython -m p3_backtester\n\n"
                "# Market analysis CLI\npython -m p1_analysis_engine",language="bash")

    # ── License ────────────────────────────────────────────────────────────────
    with t8:
        from core import license as _lic_s
        info = _lic_s.get_info()
        _card_s = "background:#0A1220;border:1px solid #0C1524;border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:.8rem"
        _lbl_s  = "font-size:.68rem;color:#2D3D50;font-weight:600;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.15rem"

        if info:
            # Status
            ok_now, status_now = _lic_s.check()
            badge = '<span class="orca-badge b-active">Active</span>' if ok_now else '<span class="orca-badge b-tripped">Expired</span>'
            offline_note = ""
            if ok_now and status_now.startswith("offline:"):
                days_left = status_now.split(":")[1]
                offline_note = f' <span style="font-size:.72rem;color:#FBBF24">· Offline grace: {days_left}d left</span>'

            st.markdown(
                f'<div style="{_card_s}">'
                f'<div style="{_lbl_s}">Status</div>'
                f'{badge}{offline_note}'
                f'</div>',
                unsafe_allow_html=True)

            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown(
                    f'<div style="{_card_s}">'
                    f'<div style="{_lbl_s}">Licensed to</div>'
                    f'<div style="color:#E2E8F0;font-size:.88rem">{info["email"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            with lc2:
                st.markdown(
                    f'<div style="{_card_s}">'
                    f'<div style="{_lbl_s}">Machine ID</div>'
                    f'<div style="color:#94A3B8;font-size:.83rem;font-family:monospace">{info["machine_id"]}…</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            st.markdown(
                f'<div style="{_card_s}">'
                f'<div style="{_lbl_s}">License key</div>'
                f'<div style="color:#94A3B8;font-size:.83rem;font-family:monospace">{info["key_preview"]}</div>'
                f'</div>',
                unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Transfer to a new machine")
            st.markdown(
                '<div style="font-size:.81rem;color:#3B4D61;margin-bottom:.8rem">'
                'Deactivate this machine to free up your activation slot. '
                'Then enter your license key on the new machine.</div>',
                unsafe_allow_html=True)
            if st.button("Deactivate this machine", type="secondary"):
                ok_d, msg_d = _lic_s.deactivate()
                if ok_d:
                    st.success(msg_d)
                    st.rerun()
                else:
                    st.error(msg_d)
        else:
            st.info("No license found on this machine. Enter your key below.")
            key_in = st.text_input("License key", placeholder="XXXX-XXXX-XXXX-XXXX")
            if st.button("Activate", type="primary"):
                if key_in.strip():
                    ok_a, msg_a = _lic_s.activate(key_in.strip())
                    if ok_a:
                        st.success(f"✓ {msg_a}")
                        st.rerun()
                    else:
                        st.error(msg_a)


# =============================================================================
# PAGE: MARKET PULSE
# =============================================================================
def page_pulse():
    st.title("Market Pulse")
    st.markdown("Real-time macro snapshot. Catch sector stress and regime shifts before they become obvious.")

    import yfinance as _yf
    import requests as _req

    _WATCH = [
        # (label, ticker, group)
        ("VIX",    "^VIX",     "Fear & Rates"),
        ("US10Y",  "^TNX",     "Fear & Rates"),
        ("HYG",    "HYG",      "Fear & Rates"),
        ("TLT",    "TLT",      "Fear & Rates"),
        ("SPX500", "^GSPC",    "Equity Indices"),
        ("US30",   "^DJI",     "Equity Indices"),
        ("GER40",  "^GDAXI",   "Equity Indices"),
        ("Gold",   "GC=F",     "Commodities & Crypto"),
        ("Silver", "SI=F",     "Commodities & Crypto"),
        ("BTC",    "BTC-USD",  "Commodities & Crypto"),
        ("XLF",    "XLF",      "Sector ETFs"),
        ("XLE",    "XLE",      "Sector ETFs"),
        ("XHB",    "XHB",      "Sector ETFs"),
        ("XLK",    "XLK",      "Sector ETFs"),
    ]

    # Twelve Data only supports forex/crypto reliably on the free tier.
    # Everything else (VIX, rates, ETFs, indices) goes through yfinance.
    _TD_PULSE = {
        # yfinance symbol → Twelve Data symbol (free tier confirmed)
        "GC=F":    "XAU/USD",
        "SI=F":    "XAG/USD",
        "BTC-USD": "BTC/USD",
    }

    col_r, col_b = st.columns([1,5])
    refresh = col_r.button("⟳  Refresh", key="pulse_ref")
    td_key = os.getenv("TWELVEDATA_API_KEY","").strip()

    if refresh or "pulse_data" not in st.session_state:
        with st.spinner("Fetching market data…"):
            rows=[]; errors=[]

            # ── Twelve Data: Gold / Silver / BTC only ─────────────────────────
            td_results={}
            if td_key:
                from core.data import fetch_twelvedata_quotes
                td_syms=list(_TD_PULSE.values())
                td_results=fetch_twelvedata_quotes(td_syms, td_key)

            # ── yfinance: all tickers (TD overrides where available) ──────────
            for lbl,tick,grp in _WATCH:
                td_sym=_TD_PULSE.get(tick)
                if td_sym and td_sym in td_results and "error" not in td_results[td_sym]:
                    r=td_results[td_sym]
                    rows.append({"label":lbl,"ticker":tick,"group":grp,
                                 "price":r["price"],"1W%":r["1W%"],"1M%":r["1M%"],"3M%":r["3M%"],
                                 "source":"TD"})
                    continue
                # yfinance path
                try:
                    d=_yf.Ticker(tick).history(period="100d",auto_adjust=True)
                    if d.empty or len(d)<5: errors.append(f"{lbl}: no data from yfinance"); continue
                    close=d["Close"]
                    price=float(close.iloc[-1])
                    def _pct_yf(idx, _c=close, _p=price):
                        return float((_p-_c.iloc[-idx])/_c.iloc[-idx]*100) if len(_c)>=idx else None
                    rows.append({"label":lbl,"ticker":tick,"group":grp,"price":price,
                                 "1W%":_pct_yf(6),"1M%":_pct_yf(22),"3M%":_pct_yf(65),
                                 "source":"yf"})
                except Exception as ex:
                    errors.append(f"{lbl}: {ex}")

            st.session_state["pulse_data"]=rows
            st.session_state["pulse_errors"]=errors
            st.session_state["pulse_ts"]=datetime.now().strftime("%H:%M")
            td_count=sum(1 for r in rows if r.get("source")=="TD")
            provider_lbl=f"yfinance + Twelve Data ({td_count} tickers)" if (td_key and td_count) else "yfinance"
            st.session_state["pulse_provider"]=provider_lbl
    else:
        provider_lbl=st.session_state.get("pulse_provider","yfinance")
        col_b.caption(f"Last updated {st.session_state.get('pulse_ts','?')} · {provider_lbl} · click Refresh to reload")
    if not td_key:
        st.info("Add TWELVEDATA_API_KEY in Settings → API Keys for better Gold/Silver data.")
    errs=st.session_state.get("pulse_errors",[])
    if errs:
        with st.expander(f"⚠️ {len(errs)} ticker(s) failed to load",expanded=False):
            for e in errs: st.caption(e)

    rows=st.session_state.get("pulse_data",[])
    if not rows: st.warning("No data loaded."); return

    # ── Stress signals ────────────────────────────────────────────────────────
    alerts=[]
    for r in rows:
        m1=r.get("1M%"); m3=r.get("3M%")
        if r["label"]=="VIX":
            if r["price"]>35: alerts.append(("🚨",f"VIX {r['price']:.1f} — CRISIS regime. Trend strategies paused.","#EF4444"))
            elif r["price"]>25: alerts.append(("⚠️",f"VIX {r['price']:.1f} — Stress regime. Size reduced 50%.","#FBBF24"))
        if r["label"]=="HYG" and m1 and m1<-3:
            alerts.append(("⚠️",f"High Yield Bonds (HYG) {m1:+.1f}% in 1M — credit stress signal.","#FBBF24"))
        if r["label"]=="TLT" and m1 and m1<-5:
            alerts.append(("⚠️",f"Long Bonds (TLT) {m1:+.1f}% in 1M — rates rising fast.","#FBBF24"))
        if m1 and m1<-15:
            alerts.append(("🚨",f"{r['label']} {m1:+.1f}% in 1 month — trend collapse / momentum breakdown.","#EF4444"))
        elif m1 and m1<-10:
            alerts.append(("⚠️",f"{r['label']} {m1:+.1f}% in 1 month — momentum breakdown forming.","#FBBF24"))
        if m3 and m3<-25:
            alerts.append(("🚨",f"{r['label']} {m3:+.1f}% in 3 months — major structural move.","#EF4444"))

    if alerts:
        st.markdown("**Active Signals:**")
        for icon,msg,color in alerts:
            st.markdown(
                f'<div style="padding:.45rem .9rem;background:#111;border-left:3px solid {color};'
                f'border-radius:4px;font-size:.85rem;color:{color};margin:.25rem 0">{icon} {msg}</div>',
                unsafe_allow_html=True)
        st.markdown("---")
    else:
        st.success("No stress signals — markets appear calm.")

    # ── Grouped market tiles ──────────────────────────────────────────────────
    def _tile(col, r):
        p=r["price"]; w=r.get("1W%"); m=r.get("1M%"); q=r.get("3M%")
        def _fc(v): return "#10B981" if (v or 0)>0 else "#EF4444" if (v or 0)<0 else "#5A6E85"
        def _fs(v): return (f"+{v:.1f}%" if v>0 else f"{v:.1f}%") if v is not None else "—"
        col.markdown(
            f'<div style="background:#101826;border:1px solid #1A2840;border-radius:10px;'
            f'padding:.65rem .85rem;text-align:center;min-height:95px">'
            f'<div style="font-size:.68rem;color:#5A6E85;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.05em">{r["label"]}</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:#F1F5F9;margin:3px 0">'
            f'{p:,.2f}</div>'
            f'<div style="font-size:.7rem">'
            f'<span style="color:{_fc(w)}">1W {_fs(w)}</span>&nbsp;&nbsp;'
            f'<span style="color:{_fc(m)}">1M {_fs(m)}</span>&nbsp;&nbsp;'
            f'<span style="color:{_fc(q)}">3M {_fs(q)}</span>'
            f'</div></div>', unsafe_allow_html=True)

    groups={}
    for r in rows:
        groups.setdefault(r["group"],[]).append(r)
    for gname,grows in groups.items():
        st.markdown(f"**{gname}**")
        cols=st.columns(len(grows))
        for col,r in zip(cols,grows): _tile(col,r)
        st.markdown("")

    # ── Headlines ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Market Headlines")
    newsapi_key=os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        try:
            resp=_req.get("https://newsapi.org/v2/top-headlines",
                          params={"category":"business","language":"en",
                                  "pageSize":15,"apiKey":newsapi_key},timeout=10)
            for a in resp.json().get("articles",[])[:12]:
                title=a.get("title",""); src=a.get("source",{}).get("name","")
                pub=(a.get("publishedAt","") or "")[:10]
                if title and "[Removed]" not in title:
                    st.markdown(
                        f'<div style="padding:.45rem .75rem;background:#101826;border:1px solid #1A2840;'
                        f'border-radius:8px;margin-bottom:.3rem">'
                        f'<span style="color:#E2E8F0">{title}</span>'
                        f'<span style="color:#5A6E85;font-size:.73rem;float:right">{src} · {pub}</span>'
                        f'</div>',unsafe_allow_html=True)
        except Exception as e: st.caption(f"News fetch failed: {e}")
    else:
        try:
            tk=_yf.Ticker("^GSPC")
            for item in (tk.news or [])[:10]:
                title=item.get("title","")
                pub=datetime.fromtimestamp(item.get("providerPublishTime",0)).strftime("%Y-%m-%d")
                pub_n=item.get("publisher","")
                if title:
                    st.markdown(
                        f'<div style="padding:.45rem .75rem;background:#101826;border:1px solid #1A2840;'
                        f'border-radius:8px;margin-bottom:.3rem">'
                        f'<span style="color:#E2E8F0">{title}</span>'
                        f'<span style="color:#5A6E85;font-size:.73rem;float:right">{pub_n} · {pub}</span>'
                        f'</div>',unsafe_allow_html=True)
        except Exception:
            st.caption("Add NEWSAPI_KEY in Settings → API Keys for broader news coverage.")


# =============================================================================
# HOW TO USE
# =============================================================================
def page_howto():
    st.title("📖 How to Use Orcastrading")
    st.markdown(
        '<div style="background:#0D1929;border:1px solid #1A2840;border-radius:10px;'
        'padding:1rem 1.2rem;margin-bottom:1.2rem;font-size:.9rem;color:#94A3B8">'
        'This app finds high-probability trade setups, sizes them correctly, and keeps you honest. '
        'Follow this routine every day and you will have a real statistical edge over time.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── DAILY ROUTINE ─────────────────────────────────────────────────────────
    st.markdown("## 🗓️ Your Daily Routine (10 minutes)")
    st.markdown(
        '<div style="background:#0A1F38;border-left:3px solid #3B82F6;border-radius:6px;'
        'padding:.7rem 1rem;margin-bottom:.8rem;font-size:.85rem;color:#CBD5E1">'
        '<b>Do this every morning before the market opens.</b><br>'
        'Open the app, go through steps 1–3. Takes 10 minutes max.'
        '</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("1", "🌊 Market Pulse",
         "Check first — is the market healthy or stressed?",
         [
             "Green tiles = risk-on, normal conditions, fine to trade.",
             "Red tiles = stress (VIX high, bonds falling, credit spreads blowing out).",
             "If more than half the tiles are red → <b>trade smaller or skip the day.</b>",
             "Read the headlines. If there is a major event (Fed, earnings, geopolitical shock) → be extra careful.",
         ]),
        ("2", "🏠 Dashboard",
         "See your current P&L and open positions at a glance.",
         [
             "Check your <b>total R</b> (profit/loss in risk units). Positive = you are ahead.",
             "Check if any open trade hit its target or stop overnight.",
             "If you are on a losing streak (red streak counter) → reduce your risk per trade by 25–50%.",
             "The 30-trade progress bar shows how many trades you have in the statistics sample. Under 30 = too early to judge the strategy.",
         ]),
        ("3", "📊 Market Analysis → find a setup",
         "This is where you find what to trade today.",
         [
             "Pick an asset from the list (or type a ticker).",
             "Pick a timeframe: <b>1d</b> for swing trades (days–weeks), <b>1h/4h</b> for shorter trades.",
             "Click <b>Run Analysis</b>. The AI will read the chart and tell you if there is a valid setup.",
             "If the setup says <b>ACTIVE</b> → move to step 4. If it says no setup → skip this asset today.",
             "The result is saved automatically. Next time you open it, you see the cached result with a timestamp.",
         ]),
    ]

    for num, title, subtitle, bullets in steps:
        bullet_html = "".join(f'<li style="margin-bottom:.3rem">{b}</li>' for b in bullets)
        st.markdown(
            f'<div style="background:#101826;border:1px solid #1A2840;border-radius:10px;'
            f'padding:.9rem 1.1rem;margin-bottom:.7rem">'
            f'<div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.5rem">'
            f'<div style="background:#1E3A5F;color:#60A5FA;font-weight:700;font-size:1rem;'
            f'width:2rem;height:2rem;border-radius:50%;display:flex;align-items:center;justify-content:center">'
            f'{num}</div>'
            f'<div><b style="color:#F1F5F9">{title}</b><br>'
            f'<span style="font-size:.78rem;color:#5A6E85">{subtitle}</span></div>'
            f'</div>'
            f'<ul style="margin:.2rem 0 0 1rem;padding:0;font-size:.83rem;color:#94A3B8">'
            f'{bullet_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── PLANNING A TRADE ──────────────────────────────────────────────────────
    st.markdown("## 📋 Planning a Trade")
    st.markdown(
        '<div style="background:#0D1929;border:1px solid #1A2840;border-radius:10px;'
        'padding:.8rem 1rem;margin-bottom:.8rem;font-size:.84rem;color:#94A3B8">'
        '<b style="color:#F1F5F9">Before you enter any trade, do these 3 things:</b>'
        '</div>',
        unsafe_allow_html=True,
    )

    plan_steps = [
        ("Journal → Manual Entry → fill in the numbers",
         "Entry zone, stop loss, and targets come from the Market Analysis page. "
         "Copy them exactly — do not adjust them based on hope."),
        ("Check the position size",
         "Dollar risk = balance × risk%. Position size = dollar risk ÷ point risk. "
         "<b>Never risk more than 1–2% per trade.</b> This is the single most important rule."),
        ("Only take the trade if R:R ≥ 2:1",
         "That means your potential profit is at least 2× your potential loss. "
         "If the numbers do not give you a 2:1 ratio → skip the trade. There will be another one."),
    ]

    for i, (title, body) in enumerate(plan_steps, 1):
        st.markdown(
            f'<div style="display:flex;gap:.8rem;padding:.65rem .8rem;background:#101826;'
            f'border:1px solid #1A2840;border-radius:8px;margin-bottom:.4rem;font-size:.83rem">'
            f'<div style="color:#3B82F6;font-weight:700;font-size:1.1rem;min-width:1.4rem">{i}.</div>'
            f'<div><b style="color:#E2E8F0">{title}</b><br>'
            f'<span style="color:#94A3B8">{body}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── CLOSING A TRADE ───────────────────────────────────────────────────────
    st.markdown("## ✅ Closing a Trade")

    close_rows = [
        ("Price hits Target 1 (TP1)", "Close 60–70% of your position. Move your stop to breakeven on the rest."),
        ("Price hits Target 2 (TP2)", "Close the remaining position. Record the trade as a WIN."),
        ("Price hits your Stop Loss", "Close immediately. Do not move the stop down. Record as LOSS."),
        ("After closing", "Go to <b>Journal → Psychology</b> and log how you felt and what happened. This is not optional — it is how you get better."),
    ]

    for trigger, action in close_rows:
        st.markdown(
            f'<div style="display:flex;gap:1rem;padding:.5rem .8rem;background:#101826;'
            f'border:1px solid #1A2840;border-radius:8px;margin-bottom:.35rem;font-size:.83rem;align-items:flex-start">'
            f'<div style="color:#FBBF24;min-width:14rem;font-weight:600">{trigger}</div>'
            f'<div style="color:#94A3B8">{action}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── WHERE YOUR EDGE COMES FROM ────────────────────────────────────────────
    st.markdown("## ⚡ Where Your Edge Comes From")
    st.markdown(
        '<div style="background:#0A1F38;border-left:3px solid #7C3AED;border-radius:6px;'
        'padding:.7rem 1rem;margin-bottom:.8rem;font-size:.84rem;color:#CBD5E1">'
        'Most traders lose because they trade randomly and size incorrectly. '
        'This app removes both of those problems — but only if you follow the process.'
        '</div>',
        unsafe_allow_html=True,
    )

    edge_items = [
        ("📐", "Fixed risk per trade",
         "You always risk the same % of your account. One bad trade cannot blow you up. "
         "This alone puts you ahead of 80% of retail traders."),
        ("🎯", "Only high-probability setups",
         "The strategies only fire when specific conditions are met — trend direction, momentum, volume. "
         "You are not guessing. You are waiting for the conditions to line up."),
        ("📊", "You track everything",
         "Every trade is logged with entry, exit, result, and your emotional state. "
         "After 30 trades you will see patterns — which setups work for you, which do not."),
        ("🧠", "You make fewer emotional decisions",
         "The app tells you when to trade and when not to. "
         "If the Market Pulse is red, you sit on your hands. "
         "If the circuit breaker fires, you stop. "
         "Discipline is built into the system."),
    ]

    cols = st.columns(2)
    for i, (icon, title, body) in enumerate(edge_items):
        with cols[i % 2]:
            st.markdown(
                f'<div style="background:#101826;border:1px solid #1A2840;border-radius:10px;'
                f'padding:.9rem 1rem;margin-bottom:.6rem;height:100%">'
                f'<div style="font-size:1.4rem;margin-bottom:.3rem">{icon}</div>'
                f'<div style="font-weight:700;color:#E2E8F0;margin-bottom:.3rem">{title}</div>'
                f'<div style="font-size:.82rem;color:#94A3B8">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── WHEN TO USE BACKTEST ──────────────────────────────────────────────────
    st.markdown("## 🔬 When to Use the Backtest (Strategies page)")
    st.markdown(
        '<div style="background:#101826;border:1px solid #1A2840;border-radius:10px;'
        'padding:.8rem 1rem;font-size:.84rem;color:#94A3B8">'
        'The backtest is <b style="color:#E2E8F0">not a daily tool</b>. Use it when:<br><br>'
        '<ul style="margin:.2rem 0 0 1rem;padding:0">'
        '<li style="margin-bottom:.3rem">You want to understand how a strategy performed historically before trusting it with real money.</li>'
        '<li style="margin-bottom:.3rem">You are considering changing strategy parameters and want to see the impact.</li>'
        '<li style="margin-bottom:.3rem">You have 30+ live trades and want to compare your live results to the historical baseline.</li>'
        '<li>You are evaluating a new asset (e.g. crypto vs equities) for a strategy.</li>'
        '</ul><br>'
        '<b style="color:#FBBF24">Warning:</b> Good backtest results do not guarantee future profits. '
        'The live forward test (shown in the Strategies page) is more important than the backtest.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── WHEN TO STOP ─────────────────────────────────────────────────────────
    st.markdown("## 🚨 When to Stop Trading")
    st.markdown(
        '<div style="background:#3C1010;border:1px solid #7f1d1d;border-radius:10px;'
        'padding:.8rem 1rem;font-size:.84rem;color:#FCA5A5">'
        '<b>Stop trading for the day (or the week) if any of these are true:</b><br><br>'
        '<ul style="margin:.2rem 0 0 1rem;padding:0;color:#F87171">'
        '<li style="margin-bottom:.3rem">You are on a 3+ losing streak and the streak counter is red.</li>'
        '<li style="margin-bottom:.3rem">Market Pulse has 3+ red tiles — the market is in risk-off mode.</li>'
        '<li style="margin-bottom:.3rem">A circuit breaker has fired for this strategy (shown in the Strategies page).</li>'
        '<li style="margin-bottom:.3rem">You feel angry, revenge-trading, or desperate to "make it back".</li>'
        '<li>There is a major economic event today (Fed meeting, CPI, NFP) — wait for the dust to settle.</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── QUICK REFERENCE ───────────────────────────────────────────────────────
    st.markdown("## 📌 Quick Reference")
    ref_rows = [
        ("What is R?", "R = your risk per trade. 1R = the amount you are willing to lose on one trade. A 2R winner = you made 2× what you risked."),
        ("What is a good win rate?", "Anything above 40% is fine if your average winner is 2× your average loser. You do not need to win most trades."),
        ("How many trades per week?", "Quality over quantity. 1–3 good setups per week is plenty. The app will not generate signals unless conditions are right — trust that."),
        ("What timeframe should I use?", "1d (daily) for swing trades held days to weeks. 1h/4h for shorter trades. Start with daily — it is slower and easier to manage."),
        ("What if no signal fires today?", "Do nothing. Sit on your hands. The market will set up again. Forcing a trade is how accounts get blown."),
        ("What does confidence score mean?", "A number the strategy gives to each signal (0–1). Higher = more conditions aligned. 0.7+ is high confidence. Do not skip low-confidence setups — they are filtered for you already."),
    ]

    for q, a in ref_rows:
        with st.expander(q):
            st.markdown(f'<div style="font-size:.85rem;color:#94A3B8;padding:.3rem 0">{a}</div>',
                        unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        '<div style="text-align:center;color:#3B4D61;font-size:.78rem;padding:1rem 0">'
        'The edge is not in the indicators. It is in following the process consistently — every single day.'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# LICENSE ACTIVATION PAGE
# =============================================================================

def page_activate(reason: str = "") -> None:
    """Full-screen license activation — shown when no valid license is found."""
    from core import license as _lic

    # Centre everything with a max-width card
    st.markdown("""
    <style>
    .lic-wrap{display:flex;flex-direction:column;align-items:center;
              justify-content:center;min-height:80vh;padding:2rem}
    .lic-card{background:#0A1220;border:1px solid #0C1524;border-radius:16px;
              padding:2.5rem 2.8rem;width:100%;max-width:440px}
    .lic-logo{font-size:1.6rem;font-weight:800;color:#F1F5F9;letter-spacing:-.04em;
              margin-bottom:.25rem}
    .lic-logo span{background:linear-gradient(135deg,#3B82F6,#8B5CF6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .lic-sub{font-size:.78rem;color:#2D3D50;margin-bottom:2rem}
    .lic-reason{background:#1E0808;border:1px solid #4a1010;border-radius:8px;
                padding:.55rem .8rem;font-size:.78rem;color:#F87171;margin-bottom:1.2rem}
    .lic-buy{font-size:.8rem;color:#2D3D50;text-align:center;margin-top:1.2rem}
    .lic-buy a{color:#3B82F6;text-decoration:none}
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="lic-logo">Orca<span>trading</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="lic-sub">Trading Intelligence Platform</div>', unsafe_allow_html=True)

        # Reason banner
        _reason_text = {
            "no_license":    "No license found on this machine.",
            "wrong_machine": "This license key is already active on a different machine.<br>"
                             "Deactivate the other machine first, then re-enter your key.",
        }
        if reason and reason not in ("dev", "valid") and not reason.startswith("offline"):
            label = _reason_text.get(reason, f"License error: {reason}")
            st.markdown(f'<div class="lic-reason">🔒 {label}</div>', unsafe_allow_html=True)

        st.markdown("#### Activate your license")
        st.markdown(
            '<div style="font-size:.8rem;color:#3B4D61;margin-bottom:.8rem">'
            'Enter the license key from your purchase confirmation email.</div>',
            unsafe_allow_html=True)

        key_input = st.text_input(
            "License key",
            placeholder="XXXX-XXXX-XXXX-XXXX",
            label_visibility="collapsed",
            key="lic_key_input",
        )

        if st.button("Activate", type="primary", use_container_width=True):
            if not key_input.strip():
                st.error("Please enter your license key.")
            else:
                with st.spinner("Validating…"):
                    ok, msg = _lic.activate(key_input.strip())
                if ok:
                    st.success(f"✓ {msg}")
                    st.rerun()
                else:
                    st.error(f"Activation failed: {msg}")

        st.markdown(
            '<div class="lic-buy">'
            'Don\'t have a license? '
            '<a href="https://orcastrading.lemonsqueezy.com" target="_blank">Purchase here →</a>'
            '</div>',
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Having trouble?"):
            st.markdown(
                '<div style="font-size:.79rem;color:#3B4D61;line-height:1.7">'
                '<b style="color:#64748B">Wrong machine error?</b><br>'
                'Your license is active on another computer. Open Orcastrading on that machine, '
                'go to Settings → License → Deactivate, then come back here.<br><br>'
                '<b style="color:#64748B">Lost your key?</b><br>'
                'Check your purchase confirmation email or log into '
                '<a href="https://app.lemonsqueezy.com/my-orders" '
                'style="color:#3B82F6" target="_blank">LemonSqueezy My Orders</a>.<br><br>'
                '<b style="color:#64748B">Still stuck?</b><br>'
                'Email <a href="mailto:support@orcastrading.com" style="color:#3B82F6">'
                'support@orcastrading.com</a>'
                '</div>',
                unsafe_allow_html=True)


# =============================================================================
# INSIDER ACTIVITY PAGE
# =============================================================================

def page_insider():  # noqa: C901
    # ── Header + refresh strip ────────────────────────────────────────────────
    _hdr_l, _hdr_r = st.columns([4, 1])
    _hdr_l.markdown(
        '<h2 style="color:#E2E8F0;font-size:1.35rem;font-weight:700;margin-bottom:.1rem">'
        'Insider Activity</h2>'
        '<div style="color:#475569;font-size:.78rem;margin-bottom:.6rem">'
        'Congressional STOCK Act disclosures · SEC Form 4 corporate insiders · portfolio correlation'
        '</div>',
        unsafe_allow_html=True,
    )
    if _hdr_r.button("⟳ Refresh", use_container_width=True, key="ins_hdr_refresh"):
        with st.spinner("Fetching insider data…"):
            try:
                from p5_insider.scrapers.congress import refresh as _cr
                from p5_insider.scrapers.sec_form4 import refresh as _f4r
                from p5_insider.alerts import send_new_alerts as _san
                _rc = _cr()
                _rf = _f4r()
                _ra = _san(since_days=7)
                st.success(
                    f"House +{_rc['house']} · Senate +{_rc['senate']} · "
                    f"Form 4 +{_rf} · alerts sent: {_ra}"
                )
            except Exception as _re:
                st.error(f"Refresh failed: {_re}")

    try:
        import importlib, p5_insider.db as _p5db, p5_insider.correlator as _p5cor, p5_insider.analytics as _p5an
        importlib.reload(_p5db)
        importlib.reload(_p5cor)
        importlib.reload(_p5an)
        from p5_insider.db import (
            get_portfolio, add_portfolio_item, remove_portfolio_item,
            get_congress_trades, get_form4_trades,
        )
        get_alert_history = _p5db.get_alert_history
        from p5_insider.correlator import get_scored_trades, get_portfolio_summary
        from p5_insider.analytics import (
            get_cluster_trades, get_politician_stats,
            get_top_congress_tickers, get_price_impact, get_activity_timeline,
        )
    except ImportError as e:
        st.error(f"p5_insider module not found: {e}")
        return

    (
        _ins_tab_port,
        _ins_tab_summary,
        _ins_tab_trades,
        _ins_tab_scored,
        _ins_tab_analytics,
        _ins_tab_alerts,
    ) = st.tabs([
        "Portfolio",
        "Signal Summary",
        "Trade Explorer",
        "Scored Trades",
        "Analytics",
        "Alert History",
    ])

    # =========================================================================
    # TAB 1 — Portfolio management
    # =========================================================================
    with _ins_tab_port:
        st.markdown("#### Watched Tickers")
        st.caption(
            "Tickers here are monitored for congressional disclosures and SEC Form 4 filings. "
            "Direction (long/short) drives the conflict/confirmation scoring."
        )

        portfolio = get_portfolio()

        with st.form("ins_add_ticker", clear_on_submit=True):
            _c1, _c2, _c3, _c4 = st.columns([2, 3, 1.2, 1])
            _new_tick  = _c1.text_input("Ticker", placeholder="AAPL")
            _new_label = _c2.text_input("Label (optional)", placeholder="Apple Inc.")
            _new_dir   = _c3.selectbox("Direction", ["long", "short"])
            if _c4.form_submit_button("Add", use_container_width=True) and _new_tick.strip():
                add_portfolio_item(_new_tick.strip(), _new_label.strip(), _new_dir)
                st.rerun()

        if not portfolio:
            st.info("No tickers watched yet. Add one above to start tracking insider trades.")
        else:
            st.markdown(
                '<div style="display:grid;grid-template-columns:1fr 2.5fr 1fr .6fr;'
                'gap:.3rem;padding:.3rem .5rem;font-size:.72rem;color:#283848;'
                'font-weight:600;text-transform:uppercase;letter-spacing:.06em">'
                '<span>Ticker</span><span>Label</span><span>Direction</span><span></span></div>',
                unsafe_allow_html=True,
            )
            for _p in portfolio:
                _col_t, _col_l, _col_d, _col_rm = st.columns([1, 2.5, 1, .6])
                _col_t.markdown(
                    f'<span style="color:#E2E8F0;font-weight:700;font-size:.9rem">{_p["ticker"]}</span>',
                    unsafe_allow_html=True,
                )
                _col_l.markdown(
                    f'<span style="color:#64748B;font-size:.83rem">{_p.get("label","")}</span>',
                    unsafe_allow_html=True,
                )
                _dir_color = "#10B981" if _p["direction"] == "long" else "#EF4444"
                _col_d.markdown(
                    f'<span style="color:{_dir_color};font-size:.83rem;font-weight:600">'
                    f'{_p["direction"].upper()}</span>',
                    unsafe_allow_html=True,
                )
                if _col_rm.button("✕", key=f"ins_rm_{_p['ticker']}", help=f"Remove {_p['ticker']}"):
                    remove_portfolio_item(_p["ticker"])
                    st.rerun()

    # =========================================================================
    # TAB 2 — Signal Summary (per-ticker cards + activity timeline)
    # =========================================================================
    with _ins_tab_summary:
        _sum_c1, _sum_c2 = st.columns([2, 2])
        _sum_days = _sum_c1.selectbox(
            "Look-back window", [7, 14, 30, 60, 90], index=2, key="ins_sum_days",
            format_func=lambda x: f"Last {x} days",
        )
        _sum_ticker_filter = _sum_c2.text_input(
            "Activity chart — ticker", placeholder="e.g. AAPL", key="ins_sum_ticker"
        ).strip().upper()

        summary = get_portfolio_summary(since_days=_sum_days)

        if not summary:
            st.info("No portfolio tickers or no insider activity found for this window.")
        else:
            _SIGNAL_STYLES = {
                "conflict": ("#FCA5A5", "#1C0000", "⚠️ CONFLICT"),
                "bearish":  ("#FCA5A5", "#180000", "↓ Bearish"),
                "bullish":  ("#6EE7B7", "#001A0E", "↑ Bullish"),
                "mixed":    ("#FCD34D", "#1A1200", "~ Mixed"),
                "neutral":  ("#475569", "#0A1220", "— Neutral"),
            }
            _ncols = min(len(summary), 3)
            _cols  = st.columns(_ncols)
            for _i, _s in enumerate(summary):
                _col = _cols[_i % _ncols]
                _sig = _s.get("signal", "neutral")
                _sig_color, _sig_bg, _sig_label = _SIGNAL_STYLES.get(_sig, _SIGNAL_STYLES["neutral"])
                _dir_icon = "↑" if _s["direction"] == "long" else "↓"
                _col.markdown(
                    f'<div style="background:{_sig_bg};border:1px solid {_sig_color}33;'
                    f'border-radius:8px;padding:.75rem .9rem;margin:.3rem 0">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="color:#E2E8F0;font-weight:700;font-size:1rem">{_s["ticker"]}</span>'
                    f'<span style="color:{_sig_color};font-size:.72rem;font-weight:600;'
                    f'background:{_sig_color}22;padding:.1rem .4rem;border-radius:4px">{_sig_label}</span></div>'
                    f'<div style="color:#475569;font-size:.73rem;margin:.18rem 0">'
                    f'{_s.get("label",_s["ticker"])} &nbsp;·&nbsp; {_dir_icon} {_s["direction"].upper()}</div>'
                    f'<div style="display:flex;gap:.9rem;margin-top:.4rem;align-items:center">'
                    f'<span style="font-size:.73rem;color:#10B981;font-weight:600">▲ {_s["buys"]}</span>'
                    f'<span style="font-size:.73rem;color:#EF4444;font-weight:600">▼ {_s["sells"]}</span>'
                    f'<span style="font-size:.73rem;color:#94A3B8">Score&nbsp;'
                    f'<b style="color:#E2E8F0">{_s["max_score"]:.0f}</b>/10</span></div>'
                    f'<div style="font-size:.68rem;color:#1E3040;margin-top:.25rem">'
                    f'Latest disclosure: {_s.get("latest") or "—"}</div></div>',
                    unsafe_allow_html=True,
                )

        # Activity timeline chart for a selected ticker
        if _sum_ticker_filter:
            _tl = get_activity_timeline(_sum_ticker_filter, since_days=180)
            if _tl:
                import plotly.graph_objects as _go_ins
                _buy_dates  = [r["date"] for r in _tl if r["action"] == "buy"]
                _sell_dates = [r["date"] for r in _tl if r["action"] == "sell"]
                _buy_who    = [r.get("who","") for r in _tl if r["action"] == "buy"]
                _sell_who   = [r.get("who","") for r in _tl if r["action"] == "sell"]
                _fig_tl = _go_ins.Figure()
                _fig_tl.add_trace(_go_ins.Scatter(
                    x=_buy_dates, y=[1]*len(_buy_dates),
                    mode="markers", marker=dict(color="#10B981", size=10, symbol="triangle-up"),
                    name="Buy", text=_buy_who, hovertemplate="%{text}<br>%{x}<extra></extra>",
                ))
                _fig_tl.add_trace(_go_ins.Scatter(
                    x=_sell_dates, y=[-1]*len(_sell_dates),
                    mode="markers", marker=dict(color="#EF4444", size=10, symbol="triangle-down"),
                    name="Sell", text=_sell_who, hovertemplate="%{text}<br>%{x}<extra></extra>",
                ))
                _fig_tl.update_layout(
                    title=f"{_sum_ticker_filter} — insider activity (180d)",
                    paper_bgcolor="#060B14", plot_bgcolor="#060B14",
                    font_color="#94A3B8",
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(visible=False, range=[-2, 2]),
                    xaxis=dict(gridcolor="#0C1524"),
                    showlegend=True,
                    legend=dict(bgcolor="#0A1220", bordercolor="#0C1524"),
                )
                st.plotly_chart(_fig_tl, use_container_width=True)
            else:
                st.info(f"No activity data for {_sum_ticker_filter} in last 180 days.")

    # =========================================================================
    # TAB 3 — Trade Explorer
    # =========================================================================
    with _ins_tab_trades:
        _rt_r1c1, _rt_r1c2, _rt_r1c3, _rt_r1c4 = st.columns([1.5, 1.5, 1.5, 1])
        _rt_src       = _rt_r1c1.selectbox("Source", ["All", "House", "Senate", "Form 4"], key="ins_rt_src")
        _rt_days      = _rt_r1c2.selectbox(
            "Period", [7, 14, 30, 60, 90, 180], index=2, key="ins_rt_days",
            format_func=lambda x: f"Last {x} days",
        )
        _rt_ticker    = _rt_r1c3.text_input("Filter ticker", placeholder="SPY (blank = all)", key="ins_rt_ticker").strip().upper()
        _rt_port_only = _rt_r1c4.toggle("Portfolio only", value=True, key="ins_rt_port_only")

        _since_iso    = (date.today() - timedelta(days=_rt_days)).isoformat()
        _port_tickers = {p["ticker"] for p in get_portfolio()} if _rt_port_only else None

        _rt_rows: list[dict] = []
        if _rt_src in ("All", "House", "Senate"):
            for _t in get_congress_trades(since_date=_since_iso, limit=2000):
                if _port_tickers is not None and _t.get("ticker") not in _port_tickers:
                    continue
                if _rt_ticker and _t.get("ticker") != _rt_ticker:
                    continue
                _src_name = (_t.get("source") or "").capitalize()
                if _rt_src == "House"  and _src_name != "House":
                    continue
                if _rt_src == "Senate" and _src_name != "Senate":
                    continue
                _rt_rows.append({
                    "Source":    _src_name,
                    "Date":      _t.get("disclosure_date",""),
                    "Ticker":    _t.get("ticker",""),
                    "Person":    _t.get("politician",""),
                    "Party":     _t.get("party",""),
                    "Action":    (_t.get("transaction_type") or "").upper(),
                    "Amount":    _t.get("amount_str",""),
                    "Asset":     (_t.get("asset_description") or "")[:45],
                })
        if _rt_src in ("All", "Form 4"):
            for _t in get_form4_trades(since_date=_since_iso, limit=2000):
                if _port_tickers is not None and _t.get("ticker") not in _port_tickers:
                    continue
                if _rt_ticker and _t.get("ticker") != _rt_ticker:
                    continue
                _code    = _t.get("transaction_code","")
                _action  = {"P": "PURCHASE", "S": "SALE", "A": "AWARD"}.get(_code, _code)
                _total   = _t.get("total_value")
                _rt_rows.append({
                    "Source":    "Form 4",
                    "Date":      _t.get("filing_date",""),
                    "Ticker":    _t.get("ticker",""),
                    "Person":    _t.get("insider_name",""),
                    "Party":     _t.get("insider_title",""),
                    "Action":    _action,
                    "Amount":    f"${_total:,.0f}" if _total else "",
                    "Asset":     (_t.get("company") or "")[:45],
                })

        if not _rt_rows:
            st.info("No trades found. Try widening the date range or disabling 'Portfolio only'.")
        else:
            st.caption(f"{len(_rt_rows)} trade(s)")
            _df_rt = pd.DataFrame(_rt_rows).sort_values("Date", ascending=False)
            st.dataframe(_df_rt, use_container_width=True, height=480, hide_index=True)

            # Download
            _csv_rt = _df_rt.to_csv(index=False).encode()
            st.download_button(
                "Download CSV", _csv_rt, file_name="insider_trades.csv",
                mime="text/csv", key="ins_dl_csv",
            )

    # =========================================================================
    # TAB 4 — Scored Trades (portfolio-correlated, with price impact lookup)
    # =========================================================================
    with _ins_tab_scored:
        _sc_c1, _sc_c2 = st.columns([2, 2])
        _sc_days   = _sc_c1.selectbox(
            "Look-back", [7, 14, 30, 60, 90], index=2, key="ins_sc_days",
            format_func=lambda x: f"Last {x} days",
        )
        _sc_thresh = _sc_c2.slider("Min score", 0, 10, 5, key="ins_sc_thresh")

        _scored = [s for s in get_scored_trades(since_days=_sc_days) if s["score"] >= _sc_thresh]

        if not _scored:
            st.info("No matching insider trades for current portfolio + score filter.")
        else:
            st.caption(f"{len(_scored)} trade(s) · score ≥ {_sc_thresh} · sorted by score")
            for _idx, _s in enumerate(_scored):
                _stype = _s["type"]
                _sc    = _s["score"]
                _tick  = _s.get("ticker","?")
                _dir   = _s.get("p_direction","long").upper()
                _sc_color = "#EF4444" if _sc >= 8 else "#FCD34D" if _sc >= 6 else "#94A3B8"
                _dir_color = "#10B981" if _dir == "LONG" else "#EF4444"
                _bg = "#1C0000" if _sc >= 8 else "#1A1200" if _sc >= 6 else "#0A1220"

                if _stype == "congress":
                    _who     = _s.get("politician","?")
                    _act     = (_s.get("transaction_type") or "").upper()
                    _amt     = _s.get("amount_str","?")
                    _trade_date = _s.get("disclosure_date","?")
                    _detail  = f"{_who} · {_act} · {_amt} · disclosed {_trade_date}"
                    _src_badge = f'<span style="color:#60A5FA;font-size:.68rem;font-weight:600">{(_s.get("source") or "").upper()}</span>'
                else:
                    _who     = _s.get("insider_name","?")
                    _ttl     = _s.get("insider_title","")
                    _code    = _s.get("transaction_code","")
                    _act     = {"P":"PURCHASE","S":"SALE","A":"AWARD"}.get(_code,_code)
                    _tv      = _s.get("total_value")
                    _tv_s    = f"${_tv:,.0f}" if _tv else ""
                    _trade_date = _s.get("filing_date","?")
                    _detail  = f"{_who} ({_ttl}) · {_act} {_tv_s} · filed {_trade_date}"
                    _src_badge = '<span style="color:#A78BFA;font-size:.68rem;font-weight:600">FORM 4</span>'

                _card_l, _card_r = st.columns([5, 1])
                _card_l.markdown(
                    f'<div style="background:{_bg};border-left:3px solid {_sc_color};'
                    f'border-radius:6px;padding:.65rem .9rem">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="color:#E2E8F0;font-weight:700;font-size:.95rem">{_tick}</span>'
                    f'<span>{_src_badge} &nbsp; '
                    f'<span style="color:{_sc_color};font-weight:700">Score {_sc:.0f}/10</span></span></div>'
                    f'<div style="font-size:.78rem;color:#64748B;margin:.2rem 0">{_detail}</div>'
                    f'<div style="font-size:.72rem;color:{_dir_color}">Your position: {_dir}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if _card_r.button("Price impact", key=f"ins_pi_{_idx}", help="Fetch price return after disclosure"):
                    _pi = get_price_impact(_tick, _trade_date)
                    if _pi:
                        _base = _pi["price_at_disclosure"]
                        _pi_parts = [f"${_base:.2f} at disclosure"]
                        for _d in [5, 10, 20, 30]:
                            _r = _pi.get(f"return_{_d}d")
                            if _r is not None:
                                _rc_color = "green" if _r >= 0 else "red"
                                _pi_parts.append(f"{_d}d: **:{_rc_color}[{_r:+.1f}%]**")
                        st.markdown("  ·  ".join(_pi_parts))
                    else:
                        st.warning(f"No price data for {_tick} around {_trade_date}")

    # =========================================================================
    # TAB 5 — Analytics (clusters, hot tickers, politician leaderboard)
    # =========================================================================
    with _ins_tab_analytics:
        _an_sub1, _an_sub2, _an_sub3 = st.tabs(["Cluster Detector", "Hot Tickers", "Politician Leaderboard"])

        # ── Cluster Detector ─────────────────────────────────────────────────
        with _an_sub1:
            st.markdown(
                "**Cluster** = multiple distinct congress members trading the same ticker "
                "in the selected window. High cluster size = consensus signal."
            )
            _cl_c1, _cl_c2 = st.columns(2)
            _cl_days = _cl_c1.selectbox(
                "Window", [14, 30, 60, 90], index=1, key="ins_cl_days",
                format_func=lambda x: f"Last {x} days",
            )
            _cl_min = _cl_c2.slider("Min politicians", 2, 10, 3, key="ins_cl_min")

            _clusters = get_cluster_trades(since_days=_cl_days, min_cluster=_cl_min)

            if not _clusters:
                st.info(f"No clusters found with ≥ {_cl_min} politicians in last {_cl_days} days.")
            else:
                st.caption(f"{len(_clusters)} cluster(s) detected")
                for _cl in _clusters:
                    _cl_action = _cl["action"]
                    _cl_color  = "#10B981" if _cl_action == "BUY" else "#EF4444"
                    _cl_bg     = "#001A0E" if _cl_action == "BUY" else "#1C0000"
                    _amt_s     = f"${_cl['total_amount_low']:,.0f}+" if _cl["total_amount_low"] else ""
                    _pols_s    = ", ".join(_cl["politicians"][:4])
                    if len(_cl["politicians"]) > 4:
                        _pols_s += f" +{len(_cl['politicians'])-4} more"
                    st.markdown(
                        f'<div style="background:{_cl_bg};border:1px solid {_cl_color}44;'
                        f'border-radius:8px;padding:.7rem .9rem;margin:.4rem 0">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<span style="color:#E2E8F0;font-weight:700;font-size:.95rem">{_cl["ticker"]}</span>'
                        f'<span style="color:{_cl_color};font-weight:700">{_cl_action} cluster · '
                        f'{_cl["cluster_size"]} politicians</span></div>'
                        f'<div style="font-size:.77rem;color:#64748B;margin:.2rem 0">{_pols_s}</div>'
                        f'<div style="display:flex;gap:1.2rem;margin-top:.3rem;font-size:.73rem">'
                        f'<span style="color:#10B981">▲ {_cl["buys"]} buys</span>'
                        f'<span style="color:#EF4444">▼ {_cl["sells"]} sells</span>'
                        f'<span style="color:#94A3B8">{_amt_s}</span>'
                        f'<span style="color:#475569">Latest: {_cl["latest_date"]}</span>'
                        f'<span style="color:#475569">Party: {_cl["party_bias"]}</span></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Hot Tickers ───────────────────────────────────────────────────────
        with _an_sub2:
            st.markdown(
                "Most-traded tickers by congress members — "
                "regardless of your portfolio. Useful for discovering opportunities."
            )
            _ht_c1, _ht_c2 = st.columns(2)
            _ht_days  = _ht_c1.selectbox(
                "Window", [7, 14, 30, 60, 90], index=2, key="ins_ht_days",
                format_func=lambda x: f"Last {x} days",
            )
            _ht_limit = _ht_c2.slider("Show top N", 5, 30, 15, key="ins_ht_limit")

            _hot = get_top_congress_tickers(since_days=_ht_days, limit=_ht_limit)
            if not _hot:
                st.info("No congressional trading data. Run a refresh first.")
            else:
                _hot_df = pd.DataFrame([
                    {
                        "Ticker":       h["ticker"],
                        "Total trades": h["total_trades"],
                        "Buys":         h["buys"],
                        "Sells":        h["sells"],
                        "Net":          h["net_bias"],
                        "Bias":         h["bias"].capitalize(),
                        "Politicians":  h["politician_count"],
                        "Est. Amount":  f"${h['total_amount_low']:,.0f}+" if h["total_amount_low"] else "",
                        "Latest":       h["latest"],
                    }
                    for h in _hot
                ])

                def _bias_style(v: str) -> str:
                    if v == "Bullish":
                        return "color: #10B981"
                    if v == "Bearish":
                        return "color: #EF4444"
                    return "color: #94A3B8"

                st.dataframe(
                    _hot_df.style.map(_bias_style, subset=["Bias"]),
                    use_container_width=True,
                    height=420,
                    hide_index=True,
                )

        # ── Politician Leaderboard ────────────────────────────────────────────
        with _an_sub3:
            st.markdown("Most active congress members over the last 365 days.")
            _pl_min = st.slider("Min trades", 1, 20, 3, key="ins_pl_min")
            _pol_stats = get_politician_stats(since_days=365, min_trades=_pl_min)

            if not _pol_stats:
                st.info("No data. Run a refresh to populate congressional trades.")
            else:
                _pol_df = pd.DataFrame([
                    {
                        "Politician":   p["politician"],
                        "Party":        p["party"],
                        "State":        p["state"],
                        "Total":        p["total"],
                        "Buys":         p["buys"],
                        "Sells":        p["sells"],
                        "Buy %":        f'{p["buy_pct"]}%',
                        "Tickers":      p["ticker_count"],
                        "Est. Amount":  f"${p['total_amount_low']:,.0f}+" if p["total_amount_low"] else "",
                        "Latest":       p["latest"],
                    }
                    for p in _pol_stats
                ])

                # Party colouring
                def _party_style(v: str) -> str:
                    if v in ("R", "Republican"):
                        return "color: #F87171"
                    if v in ("D", "Democrat"):
                        return "color: #60A5FA"
                    return ""

                st.dataframe(
                    _pol_df.style.map(_party_style, subset=["Party"]),
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                )

    # =========================================================================
    # TAB 6 — Alert History
    # =========================================================================
    with _ins_tab_alerts:
        _ah_limit = st.slider("Show last N alerts", 10, 200, 50, key="ins_ah_limit")
        _hist = get_alert_history(limit=_ah_limit)

        if not _hist:
            st.info("No alerts have been sent yet.")
        else:
            st.caption(f"{len(_hist)} alert(s) on record")
            _ah_df = pd.DataFrame([
                {
                    "Sent":       h.get("sent_at","")[:16].replace("T"," "),
                    "Ticker":     h.get("ticker",""),
                    "Type":       h.get("trade_type","").capitalize(),
                    "Who":        h.get("who","") or "",
                    "Action":     (h.get("action") or "").upper(),
                    "Amount":     h.get("amount","") or "",
                    "Trade Date": h.get("trade_date","") or "",
                }
                for h in _hist
            ]).sort_values("Sent", ascending=False)
            st.dataframe(_ah_df, use_container_width=True, height=460, hide_index=True)


# =============================================================================
# PAGE: MOMENTUM ALERTS
# =============================================================================

def page_momentum():  # noqa: C901
    """Russell 2000 short-term momentum alerts — price spikes and volume surges."""
    st.markdown("## Momentum Alerts")
    st.markdown(
        '<p style="color:#64748B;margin-top:-.5rem">Russell 2000 — real-time price & volume burst detection</p>',
        unsafe_allow_html=True,
    )

    try:
        from p5_insider.momentum_scanner import (
            get_recent_alerts,
            PRICE_SPIKE_PCT, PRICE_WINDOW_MIN,
            VOLUME_SURGE_RATIO, COOLDOWN_HOURS,
        )
    except Exception as e:
        st.error(f"Momentum scanner unavailable: {e}")
        return

    # ── Config banner ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    _card = "background:#0A1220;border:1px solid #0C1524;border-radius:10px;padding:.7rem 1rem"
    _lbl  = "font-size:.6rem;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:.08em"
    _val  = "font-size:1.1rem;font-weight:700;color:#E2E8F0"
    c1.markdown(
        f'<div style="{_card}"><div style="{_lbl}">Price Spike</div>'
        f'<div style="{_val}">&ge;{PRICE_SPIKE_PCT:.0f}% in {PRICE_WINDOW_MIN}min</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div style="{_card}"><div style="{_lbl}">Volume Surge</div>'
        f'<div style="{_val}">&ge;{VOLUME_SURGE_RATIO:.0f}&times; avg hourly</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div style="{_card}"><div style="{_lbl}">Alert Cooldown</div>'
        f'<div style="{_val}">{COOLDOWN_HOURS:.0f}h per ticker</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div style="{_card}"><div style="{_lbl}">Universe</div>'
        f'<div style="{_val}">Russell 2000</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([2, 2, 3])
    hours_back = f1.selectbox("Show last", [4, 8, 24, 48, 168], index=2,
                              format_func=lambda h: f"{h}h" if h < 168 else "7 days")
    sig_filter = f2.selectbox("Signal type", ["All", "price_spike", "volume_surge"])
    f3.write("")

    # ── Load alerts ───────────────────────────────────────────────────────────
    try:
        alerts = get_recent_alerts(hours=hours_back, limit=500)
    except Exception as e:
        st.error(f"Could not load alert history: {e}")
        return

    if sig_filter != "All":
        alerts = [a for a in alerts if a.get("signal_type") == sig_filter]

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_total  = len(alerts)
    n_spikes = sum(1 for a in alerts if a.get("signal_type") == "price_spike")
    n_surges = sum(1 for a in alerts if a.get("signal_type") == "volume_surge")
    n_up     = sum(1 for a in alerts if a.get("direction") == "up")
    n_down   = n_total - n_up

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total alerts", n_total)
    m2.metric("Price spikes", n_spikes)
    m3.metric("Volume surges", n_surges)
    m4.metric("Bullish", n_up)
    m5.metric("Bearish", n_down)

    st.markdown("---")

    # ── Alert feed ────────────────────────────────────────────────────────────
    if not alerts:
        st.info("No alerts in this window. Scanner fires every 5 minutes during US market hours.")
        return

    try:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
    except Exception:
        _ET = None

    def _fmt_ts(iso: str) -> str:
        try:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if _ET:
                dt = dt.astimezone(_ET)
            return dt.strftime("%b %d  %H:%M ET")
        except Exception:
            return iso[:16]

    def _sig_badge(stype: str) -> str:
        if stype == "price_spike":
            return (
                '<span class="orca-badge" style="background:#0D1F3C;color:#60A5FA;'
                'border:1px solid #1E3A5F">PRICE SPIKE</span>'
            )
        if stype == "volume_surge":
            return (
                '<span class="orca-badge" style="background:#1C0D3C;color:#C084FC;'
                'border:1px solid #3D1A7A">VOL SURGE</span>'
            )
        return f'<span class="orca-badge b-neutral">{stype.upper()}</span>'

    def _dir_badge(direction: str) -> str:
        if direction == "up":
            return '<span class="orca-badge b-win">UP</span>'
        return '<span class="orca-badge b-loss">DOWN</span>'

    for a in alerts:
        pct       = a.get("pct_change") or 0.0
        price     = a.get("price") or 0.0
        vol_ratio = a.get("volume_ratio")
        vol_s     = f"  &nbsp;&nbsp;vol <b>{vol_ratio:.1f}x</b>" if vol_ratio else ""
        pct_color = "#10B981" if pct >= 0 else "#EF4444"
        pct_sign  = f"+{pct:.1f}" if pct >= 0 else f"{pct:.1f}"
        ts_s      = _fmt_ts(a.get("triggered_at", ""))

        st.markdown(
            f'<div class="signal-row" style="justify-content:space-between">'
            f'<div style="display:flex;align-items:center;gap:.75rem">'
            f'<span style="font-weight:700;color:#F1F5F9;font-size:.95rem;'
            f'font-family:monospace;min-width:4rem">{a["ticker"]}</span>'
            f'{_sig_badge(a.get("signal_type",""))}'
            f'{_dir_badge(a.get("direction","up"))}'
            f'<span style="color:{pct_color};font-weight:600">{pct_sign}%</span>'
            f'{vol_s}'
            f'<span style="color:#64748B;font-size:.8rem">price: ${price:,.4g}</span>'
            f'</div>'
            f'<span style="color:#334155;font-size:.78rem">{ts_s}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Raw table (expandable) ────────────────────────────────────────────────
    with st.expander("Raw data table"):
        df_raw = pd.DataFrame([
            {
                "Time (ET)":   _fmt_ts(a.get("triggered_at", "")),
                "Ticker":      a.get("ticker", ""),
                "Signal":      a.get("signal_type", "").replace("_", " ").title(),
                "Direction":   a.get("direction", "").capitalize(),
                "% Change":    round(a.get("pct_change") or 0, 2),
                "Vol Ratio":   a.get("volume_ratio"),
                "Price":       a.get("price"),
            }
            for a in alerts
        ])
        st.dataframe(df_raw, use_container_width=True, hide_index=True)


# =============================================================================
# AUTH GATE — checked once per session before any UI renders
# =============================================================================
from core.auth import is_authenticated  # noqa: E402
from ui.login import render_login_page  # noqa: E402

if not is_authenticated():
    render_login_page()
    st.stop()


# =============================================================================
# SIDEBAR + ROUTING
# =============================================================================
PAGES=["Dashboard","Market Pulse","Market Analysis","Strategies","Journal","Insider","Momentum","How to Use","Settings"]
ICONS={"Dashboard":"🏠","Market Pulse":"🌊","Market Analysis":"📊",
       "Strategies":"⚡","Journal":"📓","Insider":"🕵️","Momentum":"🚀","How to Use":"📖","Settings":"⚙️"}
_NAV_LABELS=[f"{ICONS[p]}  {p}" for p in PAGES]

def _on_nav_change():
    """Sync radio selection → page state."""
    lbl = st.session_state.get("nav_radio")
    if lbl and lbl in _NAV_LABELS:
        st.session_state["page"] = PAGES[_NAV_LABELS.index(lbl)]

with st.sidebar:
    # ── Logo ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="orca-logo">Orca<span>trading</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:.7rem;color:#283848;margin-bottom:.9rem;letter-spacing:.03em">'
        'Trading Intelligence Platform</div>',
        unsafe_allow_html=True)

    from core.demo_limit import is_demo_mode as _is_demo_mode, render_owner_unlock as _render_owner_unlock
    if _is_demo_mode():
        _render_owner_unlock()

    # ── Alert banners ─────────────────────────────────────────────────────────
    try:
        from p4_live.journal_supabase import get_user_settings as _gus_sb
        _sb_us = _gus_sb(st.session_state["supabase_client"])
        _tg_ok = bool(_sb_us.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN"))
    except Exception:
        _tg_ok = bool(os.getenv("TELEGRAM_BOT_TOKEN"))
    if not _tg_ok:
        st.markdown(
            '<div style="background:#1C1000;border:1px solid #3a2800;border-radius:6px;'
            'padding:.38rem .6rem;font-size:.72rem;color:#FBBF24;margin:.35rem 0">'
            '⚠️ Alerts not configured — '
            '<span style="color:#78580A">Settings → Alerts</span></div>',
            unsafe_allow_html=True)
    _pp = _pending_psych()
    if _pp > 0:
        st.markdown(
            f'<div style="background:#120828;border:1px solid #2D0E5A;border-radius:6px;'
            f'padding:.38rem .6rem;font-size:.72rem;color:#C084FC;margin:.35rem 0">'
            f'📝 {_pp} trade{"s" if _pp>1 else ""} need review — '
            f'<span style="color:#4c1d95">Journal → Psychology</span></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Navigation (radio styled as nav links via CSS `:has(input:checked)`) ──
    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]
    # Keep radio widget in sync when other pages navigate programmatically
    st.session_state["nav_radio"] = _NAV_LABELS[PAGES.index(st.session_state["page"])]
    st.radio(
        "Navigation",
        _NAV_LABELS,
        key="nav_radio",
        on_change=_on_nav_change,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Quick stats grid ──────────────────────────────────────────────────────
    try:
        _sb_t = _trades()
        _sb_closed = [x for x in _sb_t if x["status"] not in ("pending", "filled", "expired")]
        _sb_total_r = sum(x["pnl_r"] for x in _sb_closed if x["pnl_r"] is not None)
        _sb_open = len([x for x in _sb_t if x["status"] in ("pending", "filled")])
        _sb_streak, _sb_skind = _compute_streak(_sb_closed)
        _sb_cr = "#10B981" if _sb_total_r >= 0 else "#EF4444"
        _sb_cs = "#10B981" if _sb_skind == "win" else "#EF4444"
        _card = "background:#0A1220;border:1px solid #0C1524;border-radius:8px;padding:.5rem .65rem"
        _lbl  = "font-size:.6rem;color:#1E2D40;font-weight:600;text-transform:uppercase;letter-spacing:.08em"
        _val  = "font-size:.92rem;font-weight:700"
        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin:.1rem 0">'
            f'<div style="{_card}"><div style="{_lbl}">All-Time P&L</div>'
            f'<div style="{_val};color:{_sb_cr}">{_sb_total_r:+.2f}R</div>'
            f'<div style="font-size:.6rem;color:#1E2D40">all history</div></div>'
            f'<div style="{_card}"><div style="{_lbl}">Active / Pending</div>'
            f'<div style="{_val};color:#E2E8F0">{_sb_open}</div>'
            f'<div style="font-size:.6rem;color:#1E2D40">open positions</div></div>'
            f'</div>'
            f'<div style="{_card};margin-top:.4rem"><div style="{_lbl}">Streak</div>'
            f'<div style="{_val};color:{_sb_cs}">{_sb_streak}× {_sb_skind}</div>'
            f'<div style="font-size:.6rem;color:#1E2D40">recent trades</div></div>',
            unsafe_allow_html=True)
        st.markdown("")
    except Exception:
        pass

    if st.button("⟳  Refresh", use_container_width=True):
        _clear()
        st.rerun()
    st.caption(datetime.now().strftime("%H:%M:%S"))

    st.markdown("---")
    _user_info = st.session_state.get("user", {})
    st.markdown(
        f'<div style="font-size:.7rem;color:#475569;margin-bottom:.4rem">'
        f'Signed in as<br><span style="color:#94A3B8">{_user_info.get("email","")}</span></div>',
        unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True):
        from core.auth import logout as _logout
        _logout()
        st.rerun()

# Route
page=st.session_state.get("page","Dashboard")
if   page=="Dashboard":       page_dashboard()
elif page=="Market Pulse":     page_pulse()
elif page=="Market Analysis":  page_analysis()
elif page=="Strategies":       page_strategies()
elif page=="Journal":          page_journal()
elif page=="Insider":          page_insider()
elif page=="Momentum":         page_momentum()
elif page=="How to Use":       page_howto()
elif page=="Settings":         page_settings()
