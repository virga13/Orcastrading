"""
ui/app.py — Orcastrading Trading Intelligence Platform v3
7 pages: Dashboard | Market Analysis | Trade Planner | Strategies | Risk Manager | Journal | Settings
"""
from __future__ import annotations
import sys, os, json, base64
from pathlib import Path
from datetime import date, timedelta, datetime, timezone

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv, set_key
import yaml

load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Orcastrading", page_icon="🐋", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.block-container{padding-top:1.2rem!important;padding-bottom:2rem!important;max-width:1440px}
section[data-testid="stSidebar"]{background:#080E1C!important;border-right:1px solid #172035!important}
section[data-testid="stSidebar"] .block-container{padding:1rem .8rem}
[data-testid="metric-container"]{background:#101826;border:1px solid #1A2840;border-radius:10px;
  padding:.9rem 1.1rem;transition:border-color .2s}
[data-testid="metric-container"]:hover{border-color:#3B82F6}
[data-testid="stMetricValue"]{font-size:1.6rem!important;font-weight:700!important}
[data-testid="stMetricLabel"]{font-size:.7rem!important;text-transform:uppercase;
  letter-spacing:.07em;color:#5A6E85!important}
[data-testid="stMetricDelta"]{font-size:.73rem!important}
.stButton button[kind="primary"]{background:linear-gradient(135deg,#2563EB,#7C3AED)!important;
  color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important}
details{border:1px solid #1A2840!important;border-radius:10px!important;background:#101826!important}
details summary{font-weight:600;padding:.5rem .75rem}
h1{font-size:1.7rem!important;font-weight:700!important;margin-bottom:.2rem!important}
h2{font-size:1.3rem!important;font-weight:600!important}
h3{font-size:1.1rem!important;font-weight:600!important}
hr{border-color:#1A2840!important;margin:1rem 0!important}
.orca-badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.68rem;
  font-weight:700;text-transform:uppercase;letter-spacing:.06em}
.b-win{background:#054030;color:#10B981;border:1px solid #065f46}
.b-partial_win{background:#0F2456;color:#60A5FA;border:1px solid #1e3a8a}
.b-loss{background:#3C1010;color:#F87171;border:1px solid #7f1d1d}
.b-breakeven{background:#3A2500;color:#FBBF24;border:1px solid #5c3a00}
.b-pending{background:#1A2030;color:#94A3B8;border:1px solid #293548}
.b-filled{background:#28103C;color:#C084FC;border:1px solid #4c1d95}
.b-expired{background:#111822;color:#4B5563;border:1px solid #1f2937}
.b-bullish{background:#054030;color:#10B981;border:1px solid #065f46}
.b-bearish{background:#3C1010;color:#F87171;border:1px solid #7f1d1d}
.b-neutral{background:#1A2030;color:#94A3B8;border:1px solid #293548}
.b-active{background:#054030;color:#10B981;border:1px solid #065f46}
.b-warning{background:#3A2500;color:#FBBF24;border:1px solid #5c3a00}
.b-tripped{background:#3C1010;color:#F87171;border:1px solid #7f1d1d}
.signal-row{display:flex;align-items:center;gap:.6rem;padding:.5rem .75rem;
  background:#101826;border:1px solid #1A2840;border-radius:8px;margin-bottom:.35rem;font-size:.85rem}
.signal-row:hover{border-color:#2563EB}
.cb-tile{padding:.6rem .9rem;border-radius:8px;border:1px solid;margin:.2rem 0;font-size:.82rem}
.kpi-sub{font-size:.76rem;color:#5A6E85;margin-top:-.3rem;margin-bottom:.8rem}
.orca-logo{font-size:1.4rem;font-weight:800;color:#F1F5F9;letter-spacing:-.03em}
.orca-logo span{background:linear-gradient(135deg,#3B82F6,#7C3AED);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
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
@st.cache_data(ttl=30)
def _trades():
    from p4_live.journal import get_all_trades; return get_all_trades()

@st.cache_data(ttl=30)
def _manual_entries():
    from p4_live.journal import get_manual_entries; return get_manual_entries()

@st.cache_data(ttl=30)
def _psychology_all():
    from p4_live.journal import get_all_psychology; return get_all_psychology()

@st.cache_data(ttl=120)
def _assets():
    from core.config import get_all_assets; return get_all_assets()

@st.cache_data(ttl=120)
def _strategies():
    from core.config import get_all_strategies; return get_all_strategies()

@st.cache_data(ttl=120)
def _scanner_assets():
    from p4_live.scanner import ASSETS; return ASSETS

def _clear():
    _trades.clear(); _manual_entries.clear(); _psychology_all.clear()

def _clear_config():
    _assets.clear(); _strategies.clear(); _scanner_assets.clear()
    try:
        from core.config import reload; reload()
    except Exception: pass

# ── Account helpers ───────────────────────────────────────────────────────────
def _acct():
    return {"balance": float(os.getenv("ACCOUNT_BALANCE","7500")),
            "leverage": int(os.getenv("ACCOUNT_LEVERAGE","100")),
            "risk_pct": float(os.getenv("RISK_PER_TRADE_PCT","1.0"))}

def _save_acct(bal, lev, risk):
    e = str(ROOT/".env")
    set_key(e,"ACCOUNT_BALANCE",str(bal)); set_key(e,"ACCOUNT_LEVERAGE",str(lev))
    set_key(e,"RISK_PER_TRADE_PCT",str(risk))
    os.environ.update({"ACCOUNT_BALANCE":str(bal),"ACCOUNT_LEVERAGE":str(lev),
                        "RISK_PER_TRADE_PCT":str(risk)})

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
    rows=[{"Date":t["signal_date"],"Asset":_label(t),
           "Strategy":t.get("strategy","").replace("_"," ").title(),
           "Dir":t["direction"].upper(),
           "Entry":f"{t['entry_low']:,.1f}–{t['entry_high']:,.1f}",
           "SL":f"{t['stop_loss']:,.1f}",
           "Fill":f"{t['fill_price']:,.1f}" if t.get("fill_price") else "—",
           "Exit":f"{t['exit_price']:,.1f}" if t.get("exit_price") else "—",
           "Status":t["status"].upper().replace("_"," "),
           "P&L":f"{t['pnl_r']:+.2f}R" if t.get("pnl_r") is not None else "—",
           "Source":t.get("source","scanner")} for t in tl]
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def _compute_streak(closed):
    if not closed: return 0, "none"
    s = sorted(closed, key=lambda x: x["signal_date"])
    streak, kind = 1, "win" if (s[-1].get("pnl_r") or 0)>0 else "loss"
    for t in reversed(s[:-1]):
        cur_win = (t.get("pnl_r") or 0) > 0
        if (cur_win and kind=="win") or (not cur_win and kind=="loss"): streak+=1
        else: break
    return streak, kind

def _rolling_wr(closed, window=20):
    if len(closed)<3: return [],[]
    s=sorted(closed,key=lambda x:x["signal_date"])
    dates,wrs=[],[]
    for i in range(len(s)):
        w=s[max(0,i-window+1):i+1]
        wrs.append(sum(1 for t in w if (t.get("pnl_r") or 0)>0)/len(w))
        dates.append(s[i]["signal_date"])
    return dates,wrs

# ── Screenshot AI ─────────────────────────────────────────────────────────────
def _analyze_screenshot(img_bytes, media_type, notes):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    b64 = base64.standard_b64encode(img_bytes).decode()
    prompt = (
        "You are an expert trading journal assistant. Analyze this chart screenshot.\n"
        f"User notes: {notes or 'None'}\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"asset":"symbol or Unknown","timeframe":"e.g. 1h","direction":"long/short/unknown",'
        '"entry_price":null,"stop_loss":null,"take_profit":null,'
        '"market_structure":"trend/key level/pattern",'
        '"entry_reason":"setup that triggered entry","trade_quality":7,'
        '"what_worked":"strengths","what_to_improve":"areas to improve",'
        '"lessons":"key lesson","journal_entry":"2-3 paragraph first-person journal entry"}'
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
def page_dashboard():
    st.title("Dashboard")
    all_t   = _trades()
    closed  = [t for t in all_t if t["status"] not in ("pending","filled","expired")]
    open_t  = [t for t in all_t if t["status"]=="filled"]
    pending = [t for t in all_t if t["status"]=="pending"]
    wins    = [t for t in closed if t["status"] in ("win","partial_win")]
    losses  = [t for t in closed if t["status"]=="loss"]
    total_r = sum(t["pnl_r"] for t in closed if t["pnl_r"] is not None)
    wr      = len(wins)/len(closed) if closed else 0.0
    streak, skind = _compute_streak(closed)
    acct    = _acct()
    dollar_risk = acct["balance"]*acct["risk_pct"]/100

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total P&L",       f"{total_r:+.1f}R")
    k2.metric("Win Rate",         f"{wr:.1%}", f"{len(wins)}W/{len(losses)}L")
    k3.metric("Open / Pending",   f"{len(open_t)} / {len(pending)}")
    k4.metric("Account",          f"${acct['balance']:,.0f}", f"{acct['leverage']}x lev")
    k5.metric("Risk / Trade",     f"${dollar_risk:.0f}", f"{acct['risk_pct']:.1f}%")
    streak_delta = f"{'🔥' if skind=='win' else '❄️'} {streak}-trade {skind} streak"
    k6.metric("Streak",           f"{streak}", streak_delta if closed else None)

    st.markdown('<p class="kpi-sub">All metrics from closed trades only · Refresh in sidebar to reload</p>',
                unsafe_allow_html=True)

    # ── Circuit breaker tiles ─────────────────────────────────────────────────
    scanner_assets = _scanner_assets()
    if scanner_assets:
        st.markdown("**Strategy Health**")
        cb_cols = st.columns(len(scanner_assets))
        for col, sa in zip(cb_cols, scanner_assets):
            try:
                from p4_live.risk import check_circuit_breaker
                breached, live_dd, max_dd = check_circuit_breaker(sa["ticker"], "mtf_trend")
                pct = live_dd/max_dd if max_dd else 0
                if breached:
                    status_cls, status_label = "b-tripped", "HALTED"
                elif pct >= 0.75:
                    status_cls, status_label = "b-warning", "WARNING"
                else:
                    status_cls, status_label = "b-active", "ACTIVE"
                col.markdown(
                    f'<div style="padding:.4rem .6rem;background:#101826;border:1px solid #1A2840;'
                    f'border-radius:8px;text-align:center">'
                    f'<div style="font-size:.75rem;font-weight:600;color:#94A3B8">{sa["label"]}</div>'
                    f'<span class="orca-badge {status_cls}" style="margin-top:2px">{status_label}</span>'
                    f'<div style="font-size:.7rem;color:#5A6E85;margin-top:2px">{live_dd:.1f}R / {max_dd:.1f}R</div>'
                    f'</div>', unsafe_allow_html=True)
            except Exception:
                col.markdown(f'<div style="padding:.4rem .6rem;background:#101826;border:1px solid #1A2840;'
                             f'border-radius:8px;text-align:center;color:#5A6E85">{sa["label"]}<br>—</div>',
                             unsafe_allow_html=True)

    st.markdown("---")

    # ── Equity + rolling WR ───────────────────────────────────────────────────
    chart_left, chart_right = st.columns([3, 2])
    with chart_left:
        if closed:
            pts = sorted([(t["signal_date"],t["pnl_r"]) for t in closed if t["pnl_r"] is not None],
                         key=lambda x:x[0])
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
            fig2.update_layout(**_PL,title="Rolling Win Rate (20-trade)",height=240,
                               yaxis=dict(**_PL["yaxis"],ticksuffix="%",range=[0,100]))
            st.plotly_chart(fig2,use_container_width=True)
        else:
            st.caption("Rolling win rate — needs at least 3 closed trades.")

    st.markdown("---")

    # ── Charts row: outcomes + P&L by asset + P&L by strategy ────────────────
    cc1,cc2,cc3 = st.columns(3)
    with cc1:
        st.subheader("Outcomes")
        if closed:
            oc={}
            for t in closed: s=t["status"].replace("_"," ").title(); oc[s]=oc.get(s,0)+1
            clrs={"Win":"#10B981","Partial Win":"#60A5FA","Loss":"#EF4444","Breakeven":"#FBBF24"}
            fig3=go.Figure(go.Pie(labels=list(oc.keys()),values=list(oc.values()),
                marker_colors=[clrs.get(k,"#94A3B8") for k in oc],hole=0.45,
                textinfo="label+percent",hovertemplate="%{label}: %{value}<extra></extra>"))
            fig3.update_layout(**_PL,height=250,showlegend=False,title="Trade Outcomes")
            st.plotly_chart(fig3,use_container_width=True)
            sel=st.selectbox("Drilldown",["All"]+list(oc.keys()),key="d_out")
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
            ap={}
            for t in closed:
                if t["pnl_r"] is not None: ap[_label(t)]=ap.get(_label(t),0)+t["pnl_r"]
            sl=sorted(ap.items(),key=lambda x:x[1],reverse=True)
            lb,vl=[x[0] for x in sl],[x[1] for x in sl]
            fig4=go.Figure(go.Bar(x=[f"{v:+.1f}R" for v in vl],y=lb,orientation="h",
                marker_color=["#10B981" if v>=0 else "#EF4444" for v in vl],
                hovertemplate="%{y}: %{x}<extra></extra>"))
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
            sp={}
            for t in closed:
                if t["pnl_r"] is not None:
                    s=t.get("strategy","unknown"); sp[s]=sp.get(s,0)+t["pnl_r"]
            slb=sorted(sp.items(),key=lambda x:x[1],reverse=True)
            fig5=go.Figure(go.Bar(
                x=[x[0].replace("_"," ").title() for x in slb],
                y=[x[1] for x in slb],
                marker_color=["#10B981" if x[1]>=0 else "#EF4444" for x in slb],
                hovertemplate="%{x}: %{y:+.2f}R<extra></extra>"))
            fig5.update_layout(**_PL,height=250,title="Net R / Strategy",showlegend=False)
            st.plotly_chart(fig5,use_container_width=True)
        else:
            st.markdown('<div style="height:250px;display:flex;align-items:center;'
                        'justify-content:center;color:#5A6E85">No data</div>',unsafe_allow_html=True)

    st.markdown("---")
    lft,rgt=st.columns([3,2])
    with lft:
        combined=sorted(open_t+pending,key=lambda x:x["signal_date"],reverse=True)
        st.subheader(f"Open Trades ({len(combined)})")
        if combined: st.dataframe(_trades_df(combined),use_container_width=True,hide_index=True)
        else: st.info("No open trades or pending signals.")
    with rgt:
        cutoff=(date.today()-timedelta(days=7)).isoformat()
        recent=sorted([t for t in all_t if t["signal_date"]>=cutoff],
                      key=lambda x:x["signal_date"],reverse=True)[:10]
        st.subheader("Recent Signals (7d)")
        if recent:
            for t in recent:
                c2=_SC.get(t["status"],"#94A3B8")
                pnl=f" <b>{t['pnl_r']:+.2f}R</b>" if t.get("pnl_r") is not None else ""
                st.markdown(
                    f'<div class="signal-row"><span style="color:{c2};font-weight:700;'
                    f'font-size:.7rem">{t["status"].replace("_"," ").upper()}</span>'
                    f'<span style="color:#F1F5F9;font-weight:600">{_label(t)}</span>'
                    f'<span style="color:#5A6E85">{t.get("strategy","").replace("_"," ").title()}</span>'
                    f'<span style="color:#60A5FA">{t["direction"].upper()}</span>'
                    f'<span style="margin-left:auto;color:#5A6E85">{t["signal_date"]}{pnl}</span>'
                    f'</div>',unsafe_allow_html=True)
        else:
            st.info("No signals in the past 7 days.")


# =============================================================================
# PAGE 2: MARKET ANALYSIS
# =============================================================================
def page_analysis():
    st.title("Market Analysis")
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

    if st.button("🔍  Analyze Market",type="primary",disabled=not ticker):
        with st.spinner(f"Analyzing {aname} ({ticker}) on {tf}..."):
            try:
                from p1_analysis_engine.engine import analyze_full
                bias,setups=analyze_full(ticker,model=model,interval=tf)
                hist=st.session_state.get("analysis_history",[])
                hist.insert(0,{"bias":bias,"setups":setups,"ticker":ticker,"name":aname,"tf":tf,
                                "ts":datetime.now().strftime("%H:%M")})
                st.session_state["analysis_history"]=hist[:5]
                st.session_state["analysis_result"]=(bias,setups,ticker,aname)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.session_state.pop("analysis_result",None)

    if "analysis_result" not in st.session_state:
        # Show history if available
        hist=st.session_state.get("analysis_history",[])
        if hist:
            st.markdown("**Recent analyses:**")
            hc=st.columns(min(len(hist),5))
            for i,(col,h) in enumerate(zip(hc,hist)):
                if col.button(f"{h['name']} {h['tf']}\n{h['ts']}",key=f"hist_{i}"):
                    st.session_state["analysis_result"]=(h["bias"],h["setups"],h["ticker"],h["name"])
                    st.rerun()
        else:
            st.markdown('<div style="text-align:center;padding:3rem;color:#5A6E85">'
                        '<div style="font-size:3rem">📊</div>'
                        '<div style="font-size:1.1rem;font-weight:600;margin:.5rem 0">Ready to analyze</div>'
                        '<div>Select an asset and click <b>Analyze Market</b></div></div>',
                        unsafe_allow_html=True)
        return

    bias,setups,res_ticker,res_name=st.session_state["analysis_result"]
    b_dir=bias.bias_direction; b_conf=bias.bias_confidence
    ts=getattr(bias,"analysis_timestamp","")

    col_h,col_c=st.columns([8,1])
    with col_h:
        st.markdown(
            f'<h2>{res_name} ({res_ticker}) &nbsp;'
            f'<span class="orca-badge b-{b_dir}">{b_dir.upper()}</span>'
            f'&nbsp;<span style="color:#5A6E85;font-size:.9rem">{b_conf:.0%} confidence · {ts[:10]}</span></h2>',
            unsafe_allow_html=True)
        if getattr(bias,"bias_narrative",None): st.caption(bias.bias_narrative)
    with col_c:
        if st.button("✕",help="Clear result"):
            del st.session_state["analysis_result"]; st.rerun()

    tab_s,tab_t,tab_m,tab_kl,tab_dt,tab_n=st.tabs(
        ["Trade Setups","Technical","Macro & Geo","Key Levels","Decision Tree","News"])

    with tab_s:
        if not setups or not setups.setups:
            st.info("No trade setups generated.")
        else:
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
                    # Plan This Trade button
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
                        st.session_state["page"]="Trade Planner"
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
                       "Action":getattr(e,"action",""),
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
# PAGE 3: TRADE PLANNER
# =============================================================================
def page_planner():
    st.title("Trade Planner")
    st.markdown("Plan a trade, calculate position sizing, visualize R/R, then log it to your journal.")

    acct=_acct()
    assets=_assets()
    prefill=st.session_state.pop("planner_prefill",{})

    source_mode=st.radio("Setup source",["Manual Entry","From Last Analysis"],horizontal=True)
    if source_mode=="From Last Analysis":
        hist=st.session_state.get("analysis_history",[])
        if not hist: st.warning("No analysis in session. Run Market Analysis first."); source_mode="Manual Entry"
        else:
            sel_h=st.selectbox("Select analysis",
                               [f"{h['name']} ({h['tf']}) @ {h['ts']}" for h in hist])
            h_idx=[f"{h['name']} ({h['tf']}) @ {h['ts']}" for h in hist].index(sel_h)
            h=hist[h_idx]
            if h["setups"] and h["setups"].setups:
                sel_s=st.selectbox("Select setup",[f"{s.name} — {getattr(s,'label','')}" for s in h["setups"].setups])
                s_idx=[f"{s.name} — {getattr(s,'label','')}" for s in h["setups"].setups].index(sel_s)
                setup=h["setups"].setups[s_idx]
                prefill={
                    "asset":h["name"],"ticker":h["ticker"],
                    "direction":setup.direction,
                    "entry_low":setup.entry_low,"entry_high":setup.entry_high,
                    "stop_loss":setup.stop_loss,
                    "tp1":setup.targets[0].price if setup.targets else None,
                    "tp1_alloc":setup.targets[0].allocation_pct if setup.targets else 70,
                    "tp2":setup.targets[1].price if len(getattr(setup,"targets",[]))>1 else None,
                    "tp2_alloc":setup.targets[1].allocation_pct if len(getattr(setup,"targets",[]))>1 else 30,
                    "confidence":setup.confidence,"rationale":setup.rationale,
                    "trade_type":setup.trade_type,
                }

    st.markdown("---")
    fc1,fc2=st.columns([2,3])

    with fc1:
        st.subheader("Setup Details")
        asset_labels=[a["label"] for a in assets]+["Custom..."]
        def_asset=prefill.get("asset",asset_labels[0]) if prefill.get("asset") in asset_labels else asset_labels[0]
        sel_asset=st.selectbox("Asset",asset_labels,index=asset_labels.index(def_asset) if def_asset in asset_labels else 0)
        if sel_asset=="Custom...":
            custom_ticker=st.text_input("Custom ticker")
            p_ticker=custom_ticker; p_label=custom_ticker
        else:
            a_cfg=next((a for a in assets if a["label"]==sel_asset),None)
            p_ticker=a_cfg.get("tickers",{}).get("yfinance",sel_asset) if a_cfg else sel_asset
            p_label=sel_asset

        p_dir=st.selectbox("Direction",["long","short"],
                           index=0 if prefill.get("direction","long")=="long" else 1)
        trade_types=["scalp","intraday","swing"]
        p_type=st.selectbox("Trade type",trade_types,
                            index=trade_types.index(prefill.get("trade_type","intraday"))
                            if prefill.get("trade_type") in trade_types else 1)
        strategies_list=list(_strategies().keys())+["manual"]
        p_strat=st.selectbox("Strategy",strategies_list,
                             index=len(strategies_list)-1)

        c_a,c_b=st.columns(2)
        p_el=c_a.number_input("Entry Low",value=float(prefill.get("entry_low") or 0.0),
                               step=0.01,format="%.4f")
        p_eh=c_b.number_input("Entry High",value=float(prefill.get("entry_high") or 0.0),
                               step=0.01,format="%.4f")
        p_sl=st.number_input("Stop Loss",value=float(prefill.get("stop_loss") or 0.0),
                              step=0.01,format="%.4f")

        c_c,c_d=st.columns(2)
        p_tp1=c_c.number_input("TP1 Price",value=float(prefill.get("tp1") or 0.0),
                                step=0.01,format="%.4f")
        p_tp1_alloc=c_d.number_input("TP1 %",value=int(prefill.get("tp1_alloc") or 70),
                                      min_value=1,max_value=100)
        c_e,c_f=st.columns(2)
        p_tp2=c_e.number_input("TP2 Price",value=float(prefill.get("tp2") or 0.0),
                                step=0.01,format="%.4f")
        p_tp2_alloc=c_f.number_input("TP2 %",value=int(prefill.get("tp2_alloc") or 30),
                                      min_value=0,max_value=100)

        p_conf=st.slider("Confidence",0.0,1.0,float(prefill.get("confidence") or 0.75),0.05)
        p_rat=st.text_area("Rationale",value=prefill.get("rationale",""),height=80)

    with fc2:
        st.subheader("Position Sizing & R/R")

        pt_risk=abs(p_eh-p_sl) if p_eh and p_sl else 0
        dollar_risk=acct["balance"]*acct["risk_pct"]/100
        pos_sz=round(dollar_risk/pt_risk,4) if pt_risk>0 else 0
        margin=(pos_sz*p_eh)/acct["leverage"] if pos_sz and p_eh else 0
        rr=(abs(p_tp1-p_eh)/pt_risk) if pt_risk>0 and p_tp1 else 0

        # VIX check
        vix_mult=1.0; vix_note=""
        try:
            import yfinance as yf
            vd=yf.download("^VIX",period="3d",auto_adjust=True,progress=False)
            if not vd.empty:
                vix_val=float(vd["Close"].iloc[-1])
                from p4_live.risk import vix_multiplier
                vix_mult=vix_multiplier(vix_val)
                regime=("CALM" if vix_val<15 else "CAUTION" if vix_val<25 else "STRESS" if vix_val<35 else "CRISIS")
                vix_note=f"VIX {vix_val:.1f} — {regime} regime (size ×{vix_mult:.0%})"
        except Exception: pass

        # Correlation check
        corr_note=""
        try:
            from p4_live.risk import correlation_penalty
            from p4_live.journal import get_active_ticker_set
            open_tickers=list(get_active_ticker_set())
            corr_mult=correlation_penalty(p_ticker,open_tickers)
            if corr_mult<1.0: corr_note=f"Correlated position open — size ×{corr_mult:.0%}"
        except Exception: pass

        adj_risk_pct=acct["risk_pct"]*vix_mult*(corr_mult if corr_note else 1.0)
        adj_dollar=acct["balance"]*adj_risk_pct/100
        adj_pos=round(adj_dollar/pt_risk,4) if pt_risk>0 else 0

        # KPIs
        sz1,sz2,sz3,sz4=st.columns(4)
        sz1.metric("Point Risk",f"{pt_risk:,.4f}" if pt_risk else "—")
        sz2.metric("Dollar Risk",f"${adj_dollar:.2f}" if pt_risk else "—")
        sz3.metric("Position Size",f"{adj_pos:.4f} units" if pt_risk else "—")
        sz4.metric("Margin Needed",f"${margin:.2f}" if margin else "—")

        if vix_note:
            color="#FBBF24" if vix_mult<1.0 else "#10B981"
            st.markdown(f'<div style="padding:.4rem .8rem;background:#1A1A2E;border-left:3px solid {color};'
                        f'border-radius:4px;font-size:.83rem;color:{color};margin:.4rem 0">'
                        f'⚡ {vix_note}</div>',unsafe_allow_html=True)
        if corr_note:
            st.markdown(f'<div style="padding:.4rem .8rem;background:#1A1A2E;border-left:3px solid #FBBF24;'
                        f'border-radius:4px;font-size:.83rem;color:#FBBF24;margin:.4rem 0">'
                        f'🔗 {corr_note}</div>',unsafe_allow_html=True)

        # Scenario table
        st.markdown("**Multi-scenario sizing:**")
        scen_rows=[]
        for r_pct in [0.5,1.0,1.5,2.0,3.0]:
            d=acct["balance"]*r_pct/100
            p=round(d/pt_risk,4) if pt_risk>0 else 0
            m=(p*p_eh)/acct["leverage"] if p and p_eh else 0
            scen_rows.append({"Risk %":f"{r_pct:.1f}%","Dollar Risk":f"${d:.2f}",
                              "Position":f"{p:.4f}","Margin":f"${m:.2f}"})
        st.dataframe(pd.DataFrame(scen_rows),use_container_width=True,hide_index=True)

        # R/R chart
        if p_el and p_eh and p_sl and p_tp1:
            st.markdown("**R/R Visualizer:**")
            mid=(p_el+p_eh)/2
            targets=[("SL",p_sl,"#EF4444"),("Entry",mid,"#94A3B8")]
            if p_tp1: targets.append(("TP1",p_tp1,"#FBBF24"))
            if p_tp2 and p_tp2>0: targets.append(("TP2",p_tp2,"#10B981"))
            fig_rr=go.Figure()
            for lbl,price,color2 in targets:
                r_val=(abs(price-p_eh)/pt_risk) if pt_risk>0 else 0
                sign=1 if price>p_eh else -1
                fig_rr.add_trace(go.Bar(name=lbl,x=[sign*r_val],y=[lbl],
                    orientation="h",marker_color=color2,
                    hovertemplate=f"{lbl}: {price:,.2f} ({sign*r_val:+.2f}R)<extra></extra>"))
            fig_rr.update_layout(**_PL,height=200,title=f"R/R = {rr:.2f}x",
                                 showlegend=False,barmode="relative",
                                 xaxis=dict(**_PL["xaxis"],title="R",zeroline=True,
                                            zerolinecolor="#2D4060"))
            st.plotly_chart(fig_rr,use_container_width=True)

    st.markdown("---")
    btn1,btn2,_=st.columns([2,2,4])
    with btn1:
        if st.button("Log as Pending Signal",type="primary"):
            if not(p_el and p_eh and p_sl and p_tp1 and p_label):
                st.error("Fill in entry zone, stop loss, TP1, and asset.")
            else:
                from p4_live.journal import save_manual_signal
                ok=save_manual_signal({
                    "signal_date":date.today().isoformat(),
                    "ticker":p_ticker,"label":p_label,
                    "strategy":p_strat,"direction":p_dir,
                    "entry_low":p_el,"entry_high":p_eh,
                    "stop_loss":p_sl,"tp1":p_tp1,
                    "tp2":p_tp2 if p_tp2>0 else None,
                    "tp1_alloc":p_tp1_alloc,"tp2_alloc":p_tp2_alloc,
                    "confidence":p_conf,"rationale":p_rat,
                })
                if ok:
                    _clear()
                    st.success(f"Logged {p_label} {p_dir.upper()} signal → Journal")
                else:
                    st.warning("Signal already exists for this date/ticker/strategy.")
    with btn2:
        if st.button("Send Alert"):
            if p_label and p_el and p_eh:
                try:
                    from p4_live.alerts import send
                    subj=f"[ORCA] WATCHLIST — {p_label} {p_dir.upper()}"
                    body=(f"Trade plan logged.\n\nAsset: {p_label} ({p_ticker})\n"
                          f"Direction: {p_dir.upper()}\nEntry: {p_el:,.2f}–{p_eh:,.2f}\n"
                          f"Stop: {p_sl:,.2f}\nTP1: {p_tp1:,.2f}\n"
                          f"Position: {adj_pos:.4f} units | Dollar risk: ${adj_dollar:.2f}")
                    results=send(subj,body)
                    if any(results.values()): st.success("Alert sent.")
                    else: st.warning("No alert channels configured (Settings → Alerts).")
                except Exception as e: st.error(f"Alert failed: {e}")
            else: st.error("Asset and entry zone required for alert.")


# =============================================================================
# PAGE 4: STRATEGIES
# =============================================================================
def page_strategies():
    st.title("Strategies")
    strategies=_strategies(); scanner_assets=_scanner_assets()
    yaml_path=ROOT/"config"/"strategies.yaml"

    t_ov,t_edit,t_bt,t_ft,t_cmp=st.tabs(
        ["Overview","Add / Edit","Backtest","Forward Test","Comparison"])

    # ── Overview ──────────────────────────────────────────────────────────────
    with t_ov:
        cols=st.columns(2)
        for i,(sid,cfg) in enumerate(strategies.items()):
            tf=cfg.get("timeframe","?")
            with cols[i%2]:
                # Live stats
                try:
                    from p4_live.report import compute_forward_stats
                    stats=compute_forward_stats(ticker=None,strategy=sid)
                    live_wr=f"{stats['win_rate']:.0%}" if stats["n_closed"]>0 else "—"
                    live_pf=f"{stats['profit_factor']:.2f}" if stats["n_closed"]>0 else "—"
                    n=stats["n_closed"]
                except Exception:
                    live_wr="—"; live_pf="—"; n=0

                with st.expander(f"**{cfg.get('name',sid)}**  `{tf}`",expanded=True):
                    st.markdown(f"*{cfg.get('description','')}*")
                    oc1,oc2,oc3=st.columns(3)
                    oc1.metric("Timeframe",tf)
                    oc2.metric("Live WR",live_wr,f"{n} trades")
                    oc3.metric("Live PF",live_pf)
                    st.markdown(f"Expiry: `{cfg.get('expiry_days','—')}d` · "
                                f"Lookback: `{cfg.get('lookback_days','?')}d`")

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
                            "timeframe":ns_tf,"lookback_days":int(ns_lb),"expiry_days":int(ns_exp),
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
        _all_intervals=["1m","5m","15m","30m","1h","4h","1d","1wk"]
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
                        stats=compute_forward_stats(ticker=atick,strategy=sid if atick else None)
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
                                st.plotly_chart(fig,use_container_width=True)
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

    # ── Comparison ────────────────────────────────────────────────────────────
    with t_cmp:
        st.markdown("Live performance of each strategy × asset combination.")
        from p4_live.report import compute_forward_stats
        rows=[]
        for sa in scanner_assets:
            for sid in strategies:
                try:
                    st2=compute_forward_stats(ticker=sa["ticker"],strategy=sid)
                    baseline=sa.get("baseline",{})
                    wr_delta=(st2["win_rate"]-baseline.get("win_rate",0)) if st2["n_closed"]>0 else None
                    rows.append({
                        "Asset":sa["label"],"Strategy":sid.replace("_"," ").title(),
                        "Closed":st2["n_closed"],
                        "Win Rate":f"{st2['win_rate']:.1%}" if st2["n_closed"]>0 else "—",
                        "WR vs Base":f"{wr_delta:+.1%}" if wr_delta is not None else "—",
                        "Avg R":f"{st2['avg_r']:+.3f}R" if st2["n_closed"]>0 else "—",
                        "PF":f"{st2['profit_factor']:.2f}" if st2["n_closed"]>0 else "—",
                        "Max DD":f"-{st2['max_drawdown_r']:.1f}R" if st2["n_closed"]>0 else "—",
                        "Total R":f"{st2['total_r']:+.1f}R" if st2["n_closed"]>0 else "—",
                    })
                except Exception:
                    rows.append({"Asset":sa["label"],"Strategy":sid.replace("_"," ").title(),
                                 "Closed":0,"Win Rate":"—","WR vs Base":"—",
                                 "Avg R":"—","PF":"—","Max DD":"—","Total R":"—"})
        if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)


# =============================================================================
# PAGE 5: RISK MANAGER
# =============================================================================
def page_risk():
    st.title("Risk Manager")
    st.markdown("Portfolio exposure, position sizing, circuit breakers, and correlation overview.")

    acct=_acct(); all_t=_trades()
    open_t=[t for t in all_t if t["status"]=="filled"]
    pending=[t for t in all_t if t["status"]=="pending"]
    scanner_assets=_scanner_assets()

    tr1,tr2,tr3,tr4=st.tabs(["Portfolio","Position Sizer","Circuit Breakers","Correlation"])

    # ── Portfolio exposure ─────────────────────────────────────────────────────
    with tr1:
        dollar_risk=acct["balance"]*acct["risk_pct"]/100
        total_trades=open_t+pending
        if total_trades:
            exp_rows=[]
            total_exp=0
            for t in sorted(total_trades,key=lambda x:x["signal_date"],reverse=True):
                pt=abs(t["entry_high"]-t["stop_loss"])
                pos=round(dollar_risk/pt,4) if pt>0 else 0
                mgn=(pos*t["entry_high"])/acct["leverage"] if pos and t["entry_high"] else 0
                pct_acct=(dollar_risk/acct["balance"])*100
                total_exp+=dollar_risk
                exp_rows.append({
                    "Asset":_label(t),"Strategy":t.get("strategy",""),
                    "Dir":t["direction"].upper(),"Status":t["status"].upper(),
                    "Entry":f"{t['entry_high']:,.2f}","SL":f"{t['stop_loss']:,.2f}",
                    "Point Risk":f"{pt:,.2f}","$ Risk":f"${dollar_risk:.2f}",
                    "% Acct":f"{pct_acct:.1f}%","Margin":f"${mgn:.2f}",
                })
            ea,eb,ec=st.columns(3)
            ea.metric("Total $ at Risk",f"${total_exp:.2f}",f"{len(total_trades)} positions")
            eb.metric("% of Account",f"{(total_exp/acct['balance'])*100:.1f}%")
            ec.metric("Avg per Trade",f"${total_exp/len(total_trades):.2f}" if total_trades else "—")
            st.dataframe(pd.DataFrame(exp_rows),use_container_width=True,hide_index=True)
        else:
            st.info("No open or pending trades.")

    # ── Position sizer ────────────────────────────────────────────────────────
    with tr2:
        st.subheader("Position Sizing Calculator")
        ps1,ps2=st.columns(2)
        with ps1:
            ps_bal=st.number_input("Account Balance ($)",value=acct["balance"],step=100.0)
            ps_risk=st.slider("Risk per trade (%)",0.1,5.0,acct["risk_pct"],0.1)
            ps_entry=st.number_input("Entry Price",value=0.0,step=0.01,format="%.4f")
            ps_sl=st.number_input("Stop Loss",value=0.0,step=0.01,format="%.4f")
            ps_lev=st.number_input("Leverage",value=acct["leverage"],min_value=1)
        with ps2:
            ps_pt=abs(ps_entry-ps_sl) if ps_entry and ps_sl else 0
            ps_dr=ps_bal*ps_risk/100
            ps_pos=round(ps_dr/ps_pt,4) if ps_pt>0 else 0
            ps_mgn=(ps_pos*ps_entry)/ps_lev if ps_pos and ps_entry else 0
            st.metric("Dollar Risk",f"${ps_dr:.2f}")
            st.metric("Point Risk",f"{ps_pt:.4f}" if ps_pt else "—")
            st.metric("Position Size",f"{ps_pos:.4f} units" if ps_pt else "—")
            st.metric("Margin Required",f"${ps_mgn:.2f}" if ps_mgn else "—")
            st.metric("Leverage Used",f"{(ps_pos*ps_entry/ps_bal):.1f}×" if ps_pos and ps_entry else "—")
            st.markdown("**Risk scenarios:**")
            scen=[{"Risk %":f"{r:.1f}%","Dollar Risk":f"${ps_bal*r/100:.2f}",
                   "Position":f"{round(ps_bal*r/100/ps_pt,4):.4f}" if ps_pt>0 else "—",
                   "Margin":f"${round(ps_bal*r/100/ps_pt,4)*ps_entry/ps_lev:.2f}" if ps_pt>0 and ps_entry else "—"}
                  for r in [0.5,1.0,1.5,2.0,3.0]]
            st.dataframe(pd.DataFrame(scen),use_container_width=True,hide_index=True)

        # VIX live
        st.markdown("---")
        if st.button("Fetch Live VIX"):
            with st.spinner("Fetching VIX..."):
                try:
                    import yfinance as yf
                    vd=yf.download("^VIX",period="5d",auto_adjust=True,progress=False)
                    vix_val=float(vd["Close"].iloc[-1])
                    from p4_live.risk import vix_multiplier
                    vm=vix_multiplier(vix_val)
                    regime=("CALM" if vix_val<15 else "CAUTION" if vix_val<25 else "STRESS" if vix_val<35 else "CRISIS")
                    cv1,cv2,cv3=st.columns(3)
                    cv1.metric("VIX Level",f"{vix_val:.1f}")
                    cv2.metric("Regime",regime)
                    cv3.metric("Size Multiplier",f"{vm:.0%}")
                    if vm==0: st.error("CRISIS regime — all trend strategies paused by VIX filter.")
                    elif vm<1: st.warning(f"Size reduced to {vm:.0%} due to VIX regime.")
                    else: st.success("VIX regime: full size.")
                except Exception as e: st.error(f"VIX fetch failed: {e}")

    # ── Circuit breakers ──────────────────────────────────────────────────────
    with tr3:
        from p4_live.risk import check_circuit_breaker,check_degradation
        cb_rows=[]
        for sa in scanner_assets:
            for sid in ["mtf_trend","momentum_breakout"]:
                try:
                    breached,live_dd,max_dd=check_circuit_breaker(sa["ticker"],sid)
                    pct=live_dd/max_dd if max_dd and max_dd!=float("inf") else 0
                    if breached: status="TRIPPED"; cls="b-tripped"
                    elif pct>=0.75: status="WARNING"; cls="b-warning"
                    else: status="ACTIVE"; cls="b-active"
                    cb_rows.append({"Asset":sa["label"],"Strategy":sid.replace("_"," ").title(),
                                    "Live DD":f"-{live_dd:.2f}R","Limit":f"-{max_dd:.1f}R",
                                    "Used":f"{pct:.0%}","Status":status})
                except Exception:
                    cb_rows.append({"Asset":sa["label"],"Strategy":sid.replace("_"," ").title(),
                                    "Live DD":"—","Limit":"—","Used":"—","Status":"NO DATA"})
        if cb_rows:
            st.dataframe(pd.DataFrame(cb_rows),use_container_width=True,hide_index=True)
            st.markdown("**TRIPPED** = halt new signals. **WARNING** = >75% of max drawdown used.")

        st.markdown("---")
        st.subheader("Degradation Tests")
        for sa in scanner_assets:
            try:
                deg=check_degradation(sa["ticker"],"mtf_trend")
                if deg.get("enough_data"):
                    col="#EF4444" if deg["degraded"] else "#10B981"
                    st.markdown(
                        f'<span style="color:{col};font-weight:700">{sa["label"]}</span>'
                        f' — Live WR {deg["live_wr"]:.1%} vs Baseline {deg["baseline_wr"]:.1%} '
                        f'| z={deg["z_score"]:.2f} p={deg["p_value"]:.3f} '
                        f'| {"**DEGRADED** ⚠️" if deg["degraded"] else "Consistent ✓"}',
                        unsafe_allow_html=True)
                else:
                    st.caption(f"{sa['label']}: {deg.get('reason','insufficient data')}")
            except Exception: pass

    # ── Correlation ────────────────────────────────────────────────────────────
    with tr4:
        st.markdown("**Correlation groups** — reduce size when multiple assets from same group are open.")
        st.markdown("""
| Group | Assets | Typical Correlation |
|-------|--------|---------------------|
| Equity Indices | SPX500, US30, GER40 | ρ = 0.75–0.95 |
| Precious Metals | Gold, Silver | ρ = 0.75–0.85 |
""")
        st.markdown("**Size reduction rules:**")
        st.markdown("- 1 correlated asset open → 50% of normal size\n"
                    "- 2+ correlated assets open → 25% of normal size")

        open_tickers={_label(t):t["ticker"] for t in open_t+pending}
        if open_tickers:
            st.markdown("**Currently open positions:**")
            for lbl,tick in open_tickers.items():
                try:
                    from p4_live.risk import correlation_penalty
                    others=[t for t in list(open_tickers.values()) if t!=tick]
                    cm=correlation_penalty(tick,others)
                    badge="⚠️ REDUCED" if cm<1.0 else "✅ FULL"
                    st.markdown(f"- **{lbl}** (`{tick}`) — size multiplier `{cm:.0%}` {badge}")
                except Exception: st.markdown(f"- **{lbl}** (`{tick}`)")
        else:
            st.info("No open positions — correlation cap not active.")


# =============================================================================
# PAGE 6: JOURNAL
# =============================================================================
def page_journal():
    st.title("Trade Journal")
    all_t=_trades(); assets=_assets(); strategies=_strategies()

    t_log,t_mgr,t_ss,t_psy,t_met=st.tabs(
        ["Trade Log","Manage Trades","Screenshot Journal","Psychology","Metrics"])

    # ── Trade log ──────────────────────────────────────────────────────────────
    with t_log:
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
        st.caption(f"{len(filt)} trades shown · {len(all_t)} total")
        if filt: st.dataframe(_trades_df(filt),use_container_width=True,hide_index=True)
        else: st.info("No trades match the filters.")
        if filt:
            csv=_trades_df(filt).to_csv(index=False)
            st.download_button("Export filtered to CSV",csv,"trades_filtered.csv","text/csv")

    # ── Manage trades ──────────────────────────────────────────────────────────
    with t_mgr:
        if not all_t: st.info("No trades yet.")
        else:
            sorted_t=sorted(all_t,key=lambda x:x["signal_date"],reverse=True)
            trade_lbl=[f"{t['signal_date']} | {_label(t)} | {t.get('strategy','').replace('_',' ').title()} | {t['status'].upper()}"
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
                            from p4_live.journal import record_fill
                            record_fill(sel["signal_date"],float(fp),fd.isoformat(),
                                       ticker=sel["ticker"],strategy=sel.get("strategy"))
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
                            from p4_live.journal import record_close
                            record_close(sel["signal_date"],out,float(ep),ed.isoformat(),
                                        ticker=sel["ticker"],strategy=sel.get("strategy"))
                            _clear(); st.success(f"Closed as {out}."); st.rerun()
                else: st.info(f"Trade must be FILLED. Status: {sel['status'].upper()}")
            with tt3:
                if sel["status"]=="pending":
                    st.warning(f"Mark {sel['signal_date']} {_label(sel)} as expired?")
                    if st.button("Expire",type="primary"):
                        from p4_live.journal import record_expired
                        record_expired(sel["signal_date"],ticker=sel["ticker"],strategy=sel.get("strategy"))
                        _clear(); st.success("Expired."); st.rerun()
                else: st.info(f"Only PENDING signals can be expired. Status: {sel['status'].upper()}")
            with tt4:
                with st.form("fn"):
                    nt=st.text_area("Note",value=sel.get("notes",""),height=80)
                    if st.form_submit_button("Save Note"):
                        if nt.strip():
                            from p4_live.journal import add_note
                            add_note(sel["signal_date"],nt.strip(),
                                    ticker=sel["ticker"],strategy=sel.get("strategy"))
                            _clear(); st.success("Saved."); st.rerun()

    # ── Screenshot journal ──────────────────────────────────────────────────────
    with t_ss:
        manual=_manual_entries()
        st1,st2=st.tabs([f"New Entry","All Entries ({len(manual)})"])
        with st1:
            sc1,sc2=st.columns([1,1])
            with sc1:
                uploaded=st.file_uploader("Trade screenshot",type=["png","jpg","jpeg","webp"])
                if uploaded: st.image(uploaded,use_container_width=True)
            with sc2:
                j_date=st.date_input("Trade date",value=date.today())
                j_asset=st.text_input("Asset",placeholder="EURUSD, BTC, SPX500...")
                j_dir=st.selectbox("Direction",["long","short","unknown"])
                j_pre=st.selectbox("Pre-trade state",["—","calm","optimal","anxious","overconfident","fomo","fearful","bored"])
                c_e1,c_e2=st.columns(2)
                j_entry=c_e1.number_input("Entry",value=0.0,step=0.01,format="%.4f")
                j_sl=c_e2.number_input("Stop Loss",value=0.0,step=0.01,format="%.4f")
                c_e3,c_e4=st.columns(2)
                j_tp=c_e3.number_input("Take Profit",value=0.0,step=0.01,format="%.4f")
                j_exit=c_e4.number_input("Exit Price",value=0.0,step=0.01,format="%.4f")
                j_pnl=st.number_input("P&L (R)",value=0.0,step=0.01,format="%.2f")
                j_tags=st.text_input("Tags",placeholder="breakout, fomo, patient, revenge...")
                j_notes=st.text_area("Your notes",height=80,placeholder="What did you see? Why did you enter?")
            if st.button("Analyze with AI",type="primary",disabled=not uploaded):
                with st.spinner("AI analyzing screenshot..."):
                    mt=f"image/{uploaded.type.split('/')[-1]}".replace("image/jpg","image/jpeg")
                    analysis=_analyze_screenshot(uploaded.getvalue(),mt,j_notes)
                st.session_state["ss_analysis"]=analysis
                st.success("Done — review and save below.")
            if "ss_analysis" in st.session_state:
                an=st.session_state["ss_analysis"]
                if "error" in an: st.error(an["error"])
                else:
                    st.markdown("---")
                    ac1,ac2=st.columns(2)
                    ac1.metric("Detected Asset",an.get("asset","?"))
                    ac1.metric("Timeframe",an.get("timeframe","?"))
                    ac2.metric("Quality Score",f"{an.get('trade_quality','—')}/10")
                    ac2.metric("Direction",an.get("direction","?").upper())
                    edited=st.text_area("Journal entry (edit before saving)",
                                        value=an.get("journal_entry",""),height=200,key="ss_edited")
                    if an.get("what_worked"): st.success(f"Worked: {an['what_worked']}")
                    if an.get("what_to_improve"): st.warning(f"Improve: {an['what_to_improve']}")
                    if an.get("lessons"): st.info(f"Lesson: {an['lessons']}")
                    if st.button("Save Entry",type="primary"):
                        ss_dir=ROOT/"outputs"/"screenshots"; ss_dir.mkdir(parents=True,exist_ok=True)
                        ts_s=datetime.now().strftime("%Y%m%d_%H%M%S")
                        ss_p=ss_dir/f"{ts_s}_{uploaded.name}"; ss_p.write_bytes(uploaded.getvalue())
                        from p4_live.journal import save_manual_entry,save_psychology
                        eid=save_manual_entry({"entry_date":j_date.isoformat(),"asset":j_asset or an.get("asset",""),
                            "direction":j_dir,"entry_price":j_entry or an.get("entry_price"),
                            "stop_loss":j_sl or an.get("stop_loss"),"take_profit":j_tp or an.get("take_profit"),
                            "exit_price":j_exit or None,"pnl_r":j_pnl or None,
                            "quality_score":an.get("trade_quality"),"notes":edited or j_notes,
                            "ai_analysis":json.dumps(an),"screenshot_path":str(ss_p),"tags":j_tags})
                        if j_pre and j_pre!="—":
                            save_psychology({"trade_ref_type":"manual","trade_ref_id":str(eid),
                                            "pre_state":j_pre,"lesson":an.get("lessons","")})
                        del st.session_state["ss_analysis"]; _clear()
                        st.success(f"Entry #{eid} saved."); st.rerun()
        with st2:
            if not manual: st.info("No entries yet.")
            else:
                for e in manual:
                    ai={}
                    if e.get("ai_analysis"):
                        try: ai=json.loads(e["ai_analysis"])
                        except: pass
                    with st.expander(
                        f"**{e['entry_date']}** · {e.get('asset','—')} · "
                        f"{(e.get('direction') or '').upper()} · "
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
                        from p4_live.journal import save_psychology
                        save_psychology({"trade_ref_type":"live","trade_ref_id":ref_id,
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
                                     height=240,yaxis=dict(**_PL["yaxis"],ticksuffix="%",range=[0,100]),showlegend=False)
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
            if best: st.dataframe(_trades_df(best)[["Date","Asset","Strategy","P&L"]],use_container_width=True,hide_index=True)
        with bc2:
            st.markdown("**Worst 5 trades:**")
            worst=sorted([t for t in closed if t.get("pnl_r")],key=lambda x:x["pnl_r"])[:5]
            if worst: st.dataframe(_trades_df(worst)[["Date","Asset","Strategy","P&L"]],use_container_width=True,hide_index=True)

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
    t1,t2,t3,t4,t5,t6,t7=st.tabs(
        ["Account","Risk Rules","Alerts","Assets","Analysis Engine","API Keys","System"])

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
        with st.form("risk_rules"):
            rr1,rr2=st.columns(2)
            max_trades=rr1.number_input("Max simultaneous open trades",
                                         value=int(os.getenv("MAX_OPEN_TRADES","5")),min_value=1)
            daily_loss=rr2.number_input("Daily loss limit (R) — 0 = disabled",
                                          value=float(os.getenv("DAILY_LOSS_LIMIT_R","0")),min_value=0.0,step=0.5)
            rr3,rr4=st.columns(2)
            weekly_dd=rr3.number_input("Weekly drawdown limit (R) — 0 = disabled",
                                         value=float(os.getenv("WEEKLY_DD_LIMIT_R","0")),min_value=0.0,step=0.5)
            max_pos_risk=rr4.number_input("Max single position risk (%)",
                                            value=float(os.getenv("MAX_POSITION_RISK_PCT","2.0")),min_value=0.1,step=0.1)
            if st.form_submit_button("Save Risk Rules",type="primary"):
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
            tg1,tg2=st.columns(2)
            tg_tok=tg1.text_input("Bot Token",value=os.getenv("TELEGRAM_BOT_TOKEN",""),type="password")
            tg_cid=tg2.text_input("Chat ID",value=os.getenv("TELEGRAM_CHAT_ID",""))
            if st.form_submit_button("Save Telegram"):
                e=str(ROOT/".env"); set_key(e,"TELEGRAM_BOT_TOKEN",tg_tok); set_key(e,"TELEGRAM_CHAT_ID",tg_cid)
                os.environ["TELEGRAM_BOT_TOKEN"]=tg_tok; os.environ["TELEGRAM_CHAT_ID"]=tg_cid
                st.success("Saved. Restart scheduler to apply.")
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
        assets_path=ROOT/"config"/"assets.yaml"
        assets=_assets()
        rows=[{"ID":a["id"],"Label":a["label"],"Category":a.get("category",""),
               "yfinance":a.get("tickers",{}).get("yfinance",""),
               "Strategies":", ".join(a.get("strategies",[]))} for a in assets]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.markdown("---")
        # Edit strategies per asset
        st.subheader("Edit Asset")
        sel_aid=st.selectbox("Asset",[a["id"] for a in assets],
                             format_func=lambda i:next((a["label"] for a in assets if a["id"]==i),i))
        a_cfg=next((a for a in assets if a["id"]==sel_aid),None)
        if a_cfg:
            strat_ids=list(_strategies().keys())
            cur=a_cfg.get("strategies",[])
            new_s=st.multiselect("Enabled strategies",strat_ids,default=cur)
            bc1,bc2=st.columns(2)
            if bc1.button("Save Strategies"):
                d=_read_yaml(assets_path)
                for a in d["assets"]:
                    if a["id"]==sel_aid: a["strategies"]=new_s; break
                _write_yaml(assets_path,d); _clear_config()
                st.success("Saved."); st.rerun()
            if bc2.button("Delete Asset",type="secondary"):
                st.session_state[f"del_asset_{sel_aid}"]=True
        if st.session_state.get(f"del_asset_{sel_aid}"):
            st.warning(f"Delete asset '{sel_aid}'? Existing trades are preserved.")
            if st.button("Confirm Delete",type="primary"):
                d=_read_yaml(assets_path)
                d["assets"]=[a for a in d["assets"] if a["id"]!=sel_aid]
                _write_yaml(assets_path,d); _clear_config()
                st.session_state.pop(f"del_asset_{sel_aid}",None)
                st.success(f"Asset '{sel_aid}' deleted."); st.rerun()
        st.markdown("---")
        st.subheader("Add Asset")
        with st.form("add_asset"):
            na1,na2,na3=st.columns(3)
            na_id=na1.text_input("ID (snake_case)"); na_label=na2.text_input("Label")
            na_cat=na3.selectbox("Category",["index","crypto","commodity","forex","stock"])
            nb1,nb2=st.columns(2)
            na_yf=nb1.text_input("yfinance ticker"); na_strats=nb2.multiselect("Strategies",list(_strategies().keys()),default=["mtf_trend"])
            if st.form_submit_button("Add Asset",type="primary"):
                if na_id and na_label and na_yf:
                    d=_read_yaml(assets_path)
                    d["assets"].append({"id":na_id,"label":na_label,"tickers":{"yfinance":na_yf},"category":na_cat,"strategies":na_strats})
                    _write_yaml(assets_path,d); _clear_config()
                    st.success(f"Added '{na_label}'."); st.rerun()
                else: st.error("ID, label, and yfinance ticker required.")

    # ── Analysis Engine ────────────────────────────────────────────────────────
    with t5:
        st.subheader("Analysis Engine Configuration")
        env=ROOT/".env"

        # API key status
        st.markdown("**Data source status:**")
        ss1,ss2,ss3,ss4=st.columns(4)
        ss1.metric("Anthropic API",  "✅ Set" if os.getenv("ANTHROPIC_API_KEY") else "❌ Missing")
        ss2.metric("FRED API",       "✅ Set" if os.getenv("FRED_API_KEY") else "⚠️ Optional")
        ss3.metric("NewsAPI",        "✅ Set" if os.getenv("NEWSAPI_KEY") else "⚠️ Optional")
        ss4.metric("Data Provider",  os.getenv("DATA_PROVIDER","yfinance").upper())

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
        with st.form("key_form"):
            nk1=st.text_input("Anthropic key",type="password",placeholder="sk-ant-...")
            nk2=st.text_input("FRED key",type="password")
            nk3=st.text_input("NewsAPI key",type="password")
            if st.form_submit_button("Save Keys",type="primary"):
                e=str(ROOT/".env")
                if nk1: set_key(e,"ANTHROPIC_API_KEY",nk1)
                if nk2: set_key(e,"FRED_API_KEY",nk2)
                if nk3: set_key(e,"NEWSAPI_KEY",nk3)
                st.success("Keys saved.")

    # ── System ─────────────────────────────────────────────────────────────────
    with t7:
        # DB stats
        from p4_live.journal import JOURNAL_DB, _conn as _jconn
        if JOURNAL_DB.exists():
            c=_jconn()
            n_trades=c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            n_manual=c.execute("SELECT COUNT(*) FROM manual_entries").fetchone()[0]
            n_psych=c.execute("SELECT COUNT(*) FROM psychology").fetchone()[0]
            c.close()
            db_sz=JOURNAL_DB.stat().st_size/1024
            ds1,ds2,ds3,ds4=st.columns(4)
            ds1.metric("Trades",n_trades); ds2.metric("Manual Entries",n_manual)
            ds3.metric("Psychology Entries",n_psych); ds4.metric("DB Size",f"{db_sz:.1f} KB")

        # Export
        st.markdown("---")
        st.subheader("Export & Backup")
        ec1,ec2,ec3=st.columns(3)
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
        with ec3:
            if st.button("Backup Database",use_container_width=True):
                bk_dir=ROOT/"outputs"/"backups"; bk_dir.mkdir(parents=True,exist_ok=True)
                bk_path=bk_dir/f"live_trades_{date.today().isoformat()}.db"
                import shutil; shutil.copy2(str(JOURNAL_DB),str(bk_path))
                st.success(f"Backed up to {bk_path.name}")

        st.markdown("---")
        st.subheader("Scheduler Log")
        log=ROOT/"logs"/"scheduler.log"
        if log.exists():
            lines=log.read_text(encoding="utf-8",errors="replace").splitlines()
            n=st.slider("Lines",10,200,50,step=10)
            st.code("\n".join(lines[-n:]),language=None)
            st.caption(f"{log} ({len(lines)} lines)")
        else: st.info("No scheduler log. Start: `python run_scheduler.py`")

        st.markdown("---")
        st.code("# Run UI\nstreamlit run ui/app.py\n\n"
                "# Scanner (once)\npython run_scheduler.py --once\n\n"
                "# Continuous scheduler\npython run_scheduler.py\n\n"
                "# Backtest\npython -m p3_backtester\n\n"
                "# Market analysis CLI\npython -m p1_analysis_engine",language="bash")


# =============================================================================
# SIDEBAR + ROUTING
# =============================================================================
PAGES=["Dashboard","Market Analysis","Trade Planner","Strategies","Risk Manager","Journal","Settings"]
ICONS={"Dashboard":"🏠","Market Analysis":"📊","Trade Planner":"📋",
       "Strategies":"⚡","Risk Manager":"🛡️","Journal":"📓","Settings":"⚙️"}

with st.sidebar:
    st.markdown('<div class="orca-logo">Orca<span>trading</span></div>',unsafe_allow_html=True)
    st.markdown('<div style="font-size:.72rem;color:#3B4D61;margin-bottom:1rem">Trading Intelligence Platform</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Navigation
    if "page" not in st.session_state:
        st.session_state["page"]=PAGES[0]
    for p in PAGES:
        selected=st.session_state["page"]==p
        bg="#1A2840" if selected else "transparent"
        color="#F1F5F9" if selected else "#94A3B8"
        if st.sidebar.button(f"{ICONS[p]}  {p}",key=f"nav_{p}",use_container_width=True):
            st.session_state["page"]=p; st.rerun()

    st.markdown("---")
    # Quick stats
    try:
        t=_trades()
        closed_s=[x for x in t if x["status"] not in ("pending","filled","expired")]
        total_r=sum(x["pnl_r"] for x in closed_s if x["pnl_r"] is not None)
        open_s=len([x for x in t if x["status"] in ("pending","filled")])
        streak,skind=_compute_streak(closed_s)
        c_r="#10B981" if total_r>=0 else "#EF4444"
        c_s="#10B981" if skind=="win" else "#EF4444"
        st.markdown(
            f'<div style="font-size:.72rem;color:#5A6E85">TOTAL P&L</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:{c_r}">{total_r:+.2f}R</div>'
            f'<div style="font-size:.72rem;color:#5A6E85;margin-top:.5rem">OPEN / PENDING</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:#F1F5F9">{open_s}</div>'
            f'<div style="font-size:.72rem;color:#5A6E85;margin-top:.5rem">STREAK</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:{c_s}">{streak}-trade {skind}</div>',
            unsafe_allow_html=True)
        st.markdown("")
    except Exception: pass

    if st.button("⟳  Refresh",use_container_width=True): _clear(); st.rerun()
    st.caption(datetime.now().strftime("%H:%M:%S"))

# Route
page=st.session_state.get("page","Dashboard")
if   page=="Dashboard":       page_dashboard()
elif page=="Market Analysis":  page_analysis()
elif page=="Trade Planner":    page_planner()
elif page=="Strategies":       page_strategies()
elif page=="Risk Manager":     page_risk()
elif page=="Journal":          page_journal()
elif page=="Settings":         page_settings()
