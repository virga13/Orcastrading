"""
Generates a self-contained HTML report from a BiasOutput instance.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from p1_analysis_engine.schema import BiasOutput, SetupsOutput, TradingSetup, DecisionTreeEntry


def _bias_css_class(bias: str) -> str:
    return {"bullish": "bull", "bearish": "bear", "neutral": "neutral"}.get(bias, "neutral")


def _risk_level_value(level: str) -> int:
    return {"low": 20, "moderate": 45, "high": 70, "extreme": 95}.get(level, 45)


def _normalize_macro_radar(macro) -> list[float]:
    """Normalize macro fields to 0-10 for radar chart."""
    rate = min((macro.fed_funds_rate or 0) / 8 * 10, 10)
    infl = {"rising": 8, "stable": 5, "falling": 3}.get(macro.cpi_trend, 5)
    usd  = {"strengthening": 7, "stable": 5, "weakening": 3}.get(macro.usd_trend, 5)
    vix  = min((macro.vix_level or 15) / 40 * 10, 10)
    yc   = {"normal": 7, "flat": 5, "inverted": 2}.get(macro.yield_curve, 5)
    bias = {"risk_on": 8, "neutral": 5, "risk_off": 2}.get(macro.macro_bias, 5)
    return [round(rate, 1), round(infl, 1), round(usd, 1), round(vix, 1), round(yc, 1), round(bias, 1)]


def _render_setups(setups: SetupsOutput) -> str:
    dir_color = {"long": "#00d4aa", "short": "#ff4d6d", "no_trade": "#f5c842"}
    priority_badge = {"primary": ("PRIMARY WATCH", "#00d4aa"), "secondary": ("ACTIVE", "#f5c842"), "conditional": ("CONDITIONAL", "#a0a8c0")}
    trade_type_color = {"scalp": "#9c6ef8", "intraday": "#3d8ef8", "swing": "#f5c842"}

    setup_html = ""
    for s in setups.setups:
        dc = dir_color.get(s.direction, "#a0a8c0")
        pb_label, pb_color = priority_badge.get(s.priority, ("WATCH", "#a0a8c0"))
        ttc = trade_type_color.get(s.trade_type, "#a0a8c0")
        rr_pct = min(int((s.rr_ratio / 5) * 100), 100)
        conf_pct = int(s.confidence * 100)
        conf_segs = "".join(
            f'<div style="width:18px;height:8px;border-radius:3px;background:{dc if i < round(s.confidence * 6) else "#1a2540"}"></div>'
            for i in range(6)
        )
        targets_html = "".join(
            f'<tr><td style="color:#4a5578;padding:6px 8px">{tgt.label}</td>'
            f'<td style="text-align:right;padding:6px 8px;color:{dc};font-family:monospace">${tgt.price:,.2f}</td>'
            f'<td style="text-align:right;padding:6px 8px;color:#4a5578">{tgt.allocation_pct}%</td></tr>'
            for tgt in s.targets
        )
        ev_color = "#00d4aa" if s.ev > 0 else "#ff4d6d"

        setup_html += f"""
        <div style="background:#0d1526;border:1px solid #1a2540;border-radius:10px;margin-bottom:16px;overflow:hidden">
          <!-- Setup header -->
          <div style="display:flex;justify-content:space-between;align-items:center;padding:14px 20px;background:color-mix(in srgb,{dc} 8%,#0d1526);border-bottom:1px solid color-mix(in srgb,{dc} 25%,#1a2540)">
            <div style="display:flex;align-items:center;gap:10px">
              <span style="font-size:13px;font-weight:700;color:{dc}">{s.name} &mdash; {s.label}</span>
              <span style="padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;background:color-mix(in srgb,{ttc} 15%,transparent);color:{ttc};border:1px solid {ttc};text-transform:uppercase">{s.trade_type}</span>
            </div>
            <div style="display:flex;gap:10px;align-items:center">
              <span style="font-size:11px;font-weight:700;letter-spacing:1px;color:{pb_color}">{pb_label}</span>
              <span style="padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700;background:color-mix(in srgb,{dc} 20%,transparent);color:{dc};border:1px solid {dc}">{s.direction.upper()}</span>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">
            <!-- Left: details -->
            <div style="padding:16px 20px;border-right:1px solid #1a2540">
              <div style="font-size:11px;color:#4a5578;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Rationale</div>
              <div style="font-size:13px;line-height:1.55;color:#e0e4f0;margin-bottom:14px">{s.rationale}</div>
              <div style="font-size:11px;color:#4a5578;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Trigger</div>
              <div style="font-size:13px;color:{dc};margin-bottom:14px;font-style:italic">{s.trigger}</div>
              <table style="width:100%;border-collapse:collapse;font-size:13px">
                <tr>
                  <td style="color:#4a5578;padding:6px 8px">Entry Zone</td>
                  <td style="text-align:right;padding:6px 8px;font-family:monospace;color:#e0e4f0">${s.entry_low:,.2f} &ndash; ${s.entry_high:,.2f}</td>
                  <td></td>
                </tr>
                <tr style="background:#070c18">
                  <td style="color:#ff4d6d;padding:6px 8px">Stop Loss</td>
                  <td style="text-align:right;padding:6px 8px;color:#ff4d6d;font-family:monospace">${s.stop_loss:,.2f}</td>
                  <td></td>
                </tr>
                <tr>
                  <td style="color:#f5c842;padding:6px 8px">Move SL to B/E at</td>
                  <td style="text-align:right;padding:6px 8px;color:#f5c842;font-family:monospace">${s.trailing_sl_to_breakeven:,.2f}</td>
                  <td style="text-align:right;padding:6px 8px;color:#4a5578;font-size:11px">trailing</td>
                </tr>
                {targets_html}
              </table>
            </div>
            <!-- Right: metrics -->
            <div style="padding:16px 20px">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px">
                <div style="background:#131d35;border-radius:8px;padding:10px;text-align:center">
                  <div style="font-size:10px;color:#4a5578;text-transform:uppercase;letter-spacing:1px">Risk / Reward</div>
                  <div style="font-size:22px;font-weight:700;color:{dc}">1:{s.rr_ratio}</div>
                </div>
                <div style="background:#131d35;border-radius:8px;padding:10px;text-align:center">
                  <div style="font-size:10px;color:#4a5578;text-transform:uppercase;letter-spacing:1px">Win Rate</div>
                  <div style="font-size:22px;font-weight:700;color:#e0e4f0">{int(s.win_rate_estimate*100)}%</div>
                </div>
                <div style="background:#131d35;border-radius:8px;padding:10px;text-align:center">
                  <div style="font-size:10px;color:#4a5578;text-transform:uppercase;letter-spacing:1px">Exp. Value</div>
                  <div style="font-size:22px;font-weight:700;color:{ev_color}">{s.ev:+.2f}</div>
                </div>
                <div style="background:#131d35;border-radius:8px;padding:10px;text-align:center">
                  <div style="font-size:10px;color:#4a5578;text-transform:uppercase;letter-spacing:1px">Profit Factor</div>
                  <div style="font-size:22px;font-weight:700;color:{'#00d4aa' if s.profit_factor >= 1.5 else '#f5c842' if s.profit_factor >= 1.0 else '#ff4d6d'}">{s.profit_factor:.2f}x</div>
                </div>
              </div>
              <!-- R:R bar -->
              <div style="margin-bottom:12px">
                <div style="height:6px;background:#1a2540;border-radius:3px;overflow:hidden">
                  <div style="height:100%;width:{rr_pct}%;background:{dc};border-radius:3px"></div>
                </div>
                <div style="font-size:11px;color:#4a5578;margin-top:4px">R:R {s.rr_ratio}x &middot; {s.trade_duration}</div>
              </div>
              <!-- Confidence -->
              <div style="font-size:10px;color:#4a5578;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Confidence</div>
              <div style="display:flex;gap:4px;margin-bottom:6px">{conf_segs}</div>
              <div style="font-size:12px;color:#a0a8c0;line-height:1.45">{conf_pct}% &mdash; {s.confidence_note}</div>
            </div>
          </div>
        </div>"""

    # Invalidation scenario block
    inv = setups.invalidation
    invalidation_html = f"""
    <div style="background:color-mix(in srgb,#ff4d6d 6%,#0d1526);border:1px solid color-mix(in srgb,#ff4d6d 35%,#1a2540);border-radius:10px;padding:16px 20px;margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#ff4d6d">&#9888; Invalidation Scenario &mdash; Stand Aside</span>
        <span style="font-family:monospace;font-size:16px;font-weight:700;color:#ff4d6d">${inv.price_trigger:,.2f}</span>
      </div>
      <div style="font-size:13px;color:#e0e4f0;margin-bottom:6px"><strong>Condition:</strong> {inv.condition}</div>
      <div style="font-size:13px;color:#a0a8c0;margin-bottom:6px">{inv.description}</div>
      <div style="font-size:12px;color:#ff4d6d;font-style:italic">{inv.action}</div>
    </div>"""

    # Decision tree
    dt_rows = ""
    for i, d in enumerate(setups.decision_tree):
        dc = dir_color.get(d.direction, "#a0a8c0")
        row_bg = "background:#070c18;" if i % 2 == 1 else ""
        dt_rows += f"""
        <tr style="border-bottom:1px solid #1a2540;{row_bg}vertical-align:top">
          <td style="padding:12px 16px;width:55%;font-size:13px;color:#c8cedd;line-height:1.55">{d.scenario}</td>
          <td style="padding:12px 16px;width:45%;font-size:13px;font-weight:600;color:{dc};line-height:1.55">{d.outcome}</td>
        </tr>"""

    return f"""
    <!-- TRADING SETUPS -->
    <div style="margin-bottom:16px">
      <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:#4a5578;margin-bottom:12px">
        Trade Plans &mdash; {setups.asset} at ${setups.current_price:,.2f}
      </div>
      {setup_html}
    </div>

    {invalidation_html}

    <!-- DECISION TREE -->
    <div class="card" style="margin-bottom:16px">
      <div class="card-title">Decision Framework &mdash; What Happens Next</div>
      <table style="width:100%;border-collapse:collapse;table-layout:fixed">
        <thead>
          <tr style="border-bottom:2px solid #1a2540">
            <th style="width:55%;text-align:left;padding:8px 16px;font-size:11px;color:#8a96b8;text-transform:uppercase;letter-spacing:1px;font-weight:600">Scenario</th>
            <th style="width:45%;text-align:left;padding:8px 16px;font-size:11px;color:#8a96b8;text-transform:uppercase;letter-spacing:1px;font-weight:600">Action</th>
          </tr>
        </thead>
        <tbody>{dt_rows}</tbody>
      </table>
      <div style="margin-top:14px;padding:12px 16px;background:#131d35;border-radius:8px;font-size:12px;color:#f5c842;border-left:3px solid #f5c842">
        &#9888; {setups.position_sizing_note}
      </div>
    </div>"""


def generate_html(bias: BiasOutput, setups: Optional[SetupsOutput] = None, interval: str = "1d") -> str:
    b = bias
    t = bias.technical
    m = bias.macro
    g = bias.geopolitical
    cls = _bias_css_class(b.directional_bias)

    bias_color = {"bull": "#00d4aa", "bear": "#ff4d6d", "neutral": "#a0a8c0"}[cls]
    risk_val   = _risk_level_value(g.risk_level)
    risk_color = {"low": "#00d4aa", "moderate": "#f5c842", "high": "#ff8c42", "extreme": "#ff4d6d"}.get(g.risk_level, "#f5c842")
    radar_data = _normalize_macro_radar(m)

    key_levels_json = json.dumps([
        {"x": lv.price, "y": 0, "label": lv.label, "strength": lv.strength}
        for lv in b.key_levels
    ])

    news_cards = ""
    for item in b.recent_news:
        sh = (item.sentiment_hint or "neutral")
        sc = {"positive": "#00d4aa", "negative": "#ff4d6d", "neutral": "#a0a8c0"}.get(sh, "#a0a8c0")
        pub = item.published_at[:10] if item.published_at else ""
        link_open  = f'<a href="{item.url}" target="_blank" rel="noopener" style="text-decoration:none;color:inherit">' if item.url else ""
        link_close = "</a>" if item.url else ""
        news_cards += f"""
        <div class="news-card">
          <div class="news-source">{item.source}</div>
          {link_open}<div class="news-title">{item.title}</div>{link_close}
          <div class="news-footer">
            <span class="news-date">{pub}</span>
            <span class="sentiment-badge" style="color:{sc};border-color:{sc}">{sh}</span>
            {f'<a href="{item.url}" target="_blank" rel="noopener" style="font-size:10px;color:#3d8ef8;text-decoration:none">Read &rsaquo;</a>' if item.url else ""}
          </div>
        </div>"""

    risk_items = ""
    for i, risk in enumerate(b.key_risks, 1):
        risk_items += f'<div class="risk-chip"><span class="risk-num">{i}</span>{risk}</div>\n'

    flag_items = ""
    if b.data_quality_flags:
        for flag in b.data_quality_flags:
            flag_items += f'<div class="flag-chip">⚠ {flag}</div>\n'
    else:
        flag_items = '<div class="flag-ok">✓ All systems nominal</div>'

    geo_factors = "".join(f"<li>{f}</li>" for f in g.key_factors)
    geo_headlines = "".join(f'<div class="geo-headline">{h}</div>' for h in g.relevant_headlines[:5])

    strength_idx = {"weak": 0, "moderate": 1, "strong": 2}.get(b.bias_strength, 1)

    cmf_norm = round((t.cmf_20 + 1) / 2 * 100, 1)
    cmf_color = "#00d4aa" if t.cmf_20 > 0.05 else ("#ff4d6d" if t.cmf_20 < -0.05 else "#a0a8c0")

    support_pct  = round((b.nearest_support  / t.current_price - 1) * 100, 2)
    resist_pct   = round((b.nearest_resistance / t.current_price - 1) * 100, 2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Orcastrading — {b.asset} Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
  :root {{
    --bg:       #070c18;
    --bg-card:  #0d1526;
    --bg-el:    #131d35;
    --border:   #1e2d4a;
    --text:     #f0f2f8;
    --muted:    #7a86a8;
    --bull:     #00d4aa;
    --bear:     #ff4d6d;
    --gold:     #f5c842;
    --blue:     #3d8ef8;
    --bias:     {bias_color};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 15px; }}
  .mono {{ font-family: 'Consolas', 'Courier New', monospace; }}

  /* ── HEADER ── */
  header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 32px; background: var(--bg-card);
    border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10;
  }}
  .wordmark {{ font-size: 20px; font-weight: 700; color: var(--gold); letter-spacing: 2px; }}
  .asset-block {{ display: flex; align-items: center; gap: 12px; }}
  .asset-ticker {{ font-size: 28px; font-weight: 700; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
  .badge-class {{ background: var(--bg-el); color: var(--muted); border: 1px solid var(--border); }}
  .badge-bias {{ background: color-mix(in srgb, var(--bias) 20%, transparent); color: var(--bias); border: 1px solid var(--bias); font-size: 13px; padding: 5px 16px; }}
  .timestamp {{ color: var(--muted); font-size: 12px; }}

  /* ── GRID ── */
  .page {{ max-width: 1600px; margin: 0 auto; padding: 24px 24px; }}
  .grid {{ display: grid; gap: 16px; }}
  .g12 {{ grid-template-columns: 1fr; }}
  .g5  {{ grid-template-columns: repeat(5, 1fr); }}
  .g84 {{ grid-template-columns: 2fr 1fr; }}
  .g3  {{ grid-template-columns: repeat(3, 1fr); }}
  .g64 {{ grid-template-columns: 3fr 2fr; }}
  .g48 {{ grid-template-columns: 2fr 3fr; }}

  /* ── CARD ── */
  .card {{
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px;
  }}
  .card-title {{ font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.5px; color: #8a96b8; margin-bottom: 12px; }}
  .card-elevated {{ background: var(--bg-el); border-radius: 8px; padding: 12px; }}

  /* ── HERO STATS ── */
  .big-num {{ font-size: 32px; font-weight: 700; line-height: 1; }}
  .sub-stat {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .price-support {{ color: var(--bull); font-size: 12px; margin-top: 6px; }}
  .price-resist  {{ color: var(--bear); font-size: 12px; }}

  .strength-bar {{ display: flex; gap: 4px; margin-top: 10px; }}
  .strength-seg {{
    flex: 1; height: 8px; border-radius: 4px; background: var(--bg-el);
    transition: background 0.3s;
  }}
  .strength-seg.active {{ background: var(--bias); }}

  .timeframe-badge {{
    margin-top: 10px; display: inline-block; padding: 4px 12px;
    background: var(--bg-el); border-radius: 4px; font-size: 11px;
    color: var(--blue); border: 1px solid var(--blue);
  }}

  /* ── GAUGE CANVAS ── */
  .gauge-wrap {{ position: relative; width: 160px; height: 90px; margin: 0 auto; }}
  .gauge-label {{
    position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
    text-align: center;
  }}
  .gauge-val {{ font-size: 22px; font-weight: 700; }}
  .gauge-sub {{ font-size: 10px; color: var(--muted); }}

  /* ── THESIS ── */
  .thesis-block {{
    border-left: 3px solid var(--bias); padding-left: 16px;
    font-size: 15px; line-height: 1.6; color: var(--text); margin-bottom: 16px;
  }}
  .case-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .bull-case {{ border-left: 3px solid var(--bull); padding: 12px 12px 12px 14px; background: color-mix(in srgb, var(--bull) 6%, var(--bg-el)); border-radius: 0 6px 6px 0; font-size: 13px; line-height: 1.55; }}
  .bear-case {{ border-left: 3px solid var(--bear); padding: 12px 12px 12px 14px; background: color-mix(in srgb, var(--bear) 6%, var(--bg-el)); border-radius: 0 6px 6px 0; font-size: 13px; line-height: 1.55; }}
  .case-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}

  /* ── SIGNAL WEIGHTS ── */
  .weight-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .weight-label {{ width: 100px; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .weight-bar-bg {{ flex: 1; height: 6px; background: var(--bg-el); border-radius: 3px; overflow: hidden; }}
  .weight-bar {{ height: 100%; border-radius: 3px; }}
  .weight-pct {{ width: 36px; text-align: right; font-size: 12px; color: var(--text); }}
  .weight-note {{ font-size: 12px; color: var(--muted); font-style: italic; margin-top: 12px; line-height: 1.5; }}

  /* ── TECH INDICATORS ── */
  .indicator-row {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); }}
  .indicator-row:last-child {{ border-bottom: none; }}
  .ind-label {{ color: #8a96b8; font-size: 13px; }}
  .ind-val {{ font-size: 14px; font-weight: 600; }}
  .tag {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .tag-bull {{ background: color-mix(in srgb, var(--bull) 20%, transparent); color: var(--bull); }}
  .tag-bear {{ background: color-mix(in srgb, var(--bear) 20%, transparent); color: var(--bear); }}
  .tag-neutral {{ background: color-mix(in srgb, var(--muted) 20%, transparent); color: var(--muted); }}

  .adx-bar-bg {{ width: 100px; height: 5px; background: var(--bg-el); border-radius: 3px; display: inline-block; vertical-align: middle; }}
  .adx-bar    {{ height: 100%; border-radius: 3px; }}

  .ema-chip {{ display: inline-block; padding: 2px 8px; background: var(--bg-el); border-radius: 4px; font-size: 11px; margin: 2px; }}
  .ema-label {{ color: var(--muted); }}

  /* ── KEY LEVELS ── */
  canvas.levels-chart {{ width: 100% !important; }}

  /* ── MACRO GRID ── */
  .macro-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
  .macro-cell {{ background: var(--bg-el); border-radius: 8px; padding: 12px; }}
  .macro-cell-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 6px; }}
  .macro-cell-val {{ font-size: 18px; font-weight: 700; }}

  /* ── GEO ── */
  .geo-factor {{ padding: 4px 0; font-size: 13px; color: var(--text); }}
  .geo-factor::before {{ content: "•"; color: var(--bias); margin-right: 8px; }}
  .geo-headline {{ font-size: 12px; color: var(--muted); padding: 5px 0; border-bottom: 1px solid var(--border); line-height: 1.4; }}
  .geo-headline:last-child {{ border-bottom: none; }}

  /* ── NEWS RAIL ── */
  .news-rail {{ display: flex; gap: 14px; overflow-x: auto; padding-bottom: 8px; scroll-behavior: smooth; }}
  .news-rail::-webkit-scrollbar {{ height: 4px; }}
  .news-rail::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
  .news-card {{
    min-width: 240px; max-width: 240px; background: var(--bg-el);
    border: 1px solid var(--border); border-radius: 8px; padding: 14px;
    display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;
  }}
  .news-source {{ font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--blue); }}
  .news-title {{ font-size: 13px; line-height: 1.45; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }}
  .news-footer {{ display: flex; justify-content: space-between; align-items: center; margin-top: auto; }}
  .news-date {{ font-size: 11px; color: var(--muted); }}
  .sentiment-badge {{ font-size: 10px; padding: 2px 7px; border-radius: 10px; border: 1px solid; text-transform: uppercase; font-weight: 600; }}

  /* ── RISKS ── */
  .risks-grid {{ display: flex; flex-direction: column; gap: 10px; }}
  .risk-chip {{ display: flex; align-items: flex-start; gap: 12px; background: color-mix(in srgb, var(--bear) 8%, var(--bg-el)); border: 1px solid color-mix(in srgb, var(--bear) 30%, transparent); border-radius: 8px; padding: 12px 16px; font-size: 13px; line-height: 1.5; }}
  .risk-num {{ background: color-mix(in srgb, var(--bear) 25%, transparent); color: var(--bear); border-radius: 50%; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex-shrink: 0; margin-top: 1px; }}

  /* ── FLAGS ── */
  .flags-wrap {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .flag-chip {{ background: color-mix(in srgb, var(--gold) 10%, var(--bg-el)); border: 1px solid color-mix(in srgb, var(--gold) 40%, transparent); color: var(--gold); border-radius: 6px; padding: 6px 12px; font-size: 12px; line-height: 1.4; }}
  .flag-ok {{ color: var(--bull); font-size: 13px; }}

  /* ── RESPONSIVE ── */
  @media (max-width: 1100px) {{
    .g5  {{ grid-template-columns: repeat(3, 1fr); }}
    .g3  {{ grid-template-columns: 1fr; }}
    .g84, .g64, .g48 {{ grid-template-columns: 1fr; }}
  }}
  @media (max-width: 700px) {{
    .g5  {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="wordmark">ORCASTRADING</div>
  <div class="asset-block">
    <span class="asset-ticker mono">{b.asset}</span>
    <span class="badge badge-class">{b.asset_class}</span>
    <span class="badge" style="background:#0d1526;color:#3d8ef8;border:1px solid #3d8ef8">{interval}</span>
  </div>
  <span class="badge badge-bias">{b.directional_bias.upper()} &middot; {b.bias_strength}</span>
  <span class="timestamp">{b.analysis_timestamp}</span>
</header>

<div class="page">

  <!-- HERO ROW -->
  <div class="grid g5" style="margin-bottom:16px">

    <!-- Confidence Gauge -->
    <div class="card" style="text-align:center">
      <div class="card-title">Confidence</div>
      <div class="gauge-wrap">
        <canvas id="gaugeConf" width="160" height="90"></canvas>
        <div class="gauge-label">
          <div class="gauge-val" style="color:var(--bias)">{round(b.confidence_score*100)}%</div>
          <div class="gauge-sub">score</div>
        </div>
      </div>
    </div>

    <!-- Current Price -->
    <div class="card">
      <div class="card-title">Price</div>
      <div class="big-num mono">{t.current_price:,.2f}</div>
      <div class="sub-stat" style="margin-top:8px">
        <div class="price-support">▲ Support &nbsp; {b.nearest_support:,.2f} &nbsp;({support_pct:+.2f}%)</div>
        <div class="price-resist"> ▼ Resist &nbsp;&nbsp; {b.nearest_resistance:,.2f} &nbsp;({resist_pct:+.2f}%)</div>
      </div>
      <div style="margin-top:8px">
        <span class="badge" style="background:color-mix(in srgb,var(--bias) 15%,transparent);color:var(--bias);border:1px solid var(--bias)">{t.trend}</span>
      </div>
    </div>

    <!-- Bias Strength -->
    <div class="card" style="text-align:center">
      <div class="card-title">Bias Strength</div>
      <div style="font-size:20px;font-weight:700;color:var(--bias);margin-top:8px">{b.bias_strength.upper()}</div>
      <div class="strength-bar" style="margin-top:14px">
        <div class="strength-seg {'active' if strength_idx >= 0 else ''}"></div>
        <div class="strength-seg {'active' if strength_idx >= 1 else ''}"></div>
        <div class="strength-seg {'active' if strength_idx >= 2 else ''}"></div>
      </div>
      <div class="timeframe-badge">{b.suggested_timeframe.replace('_', ' ')}</div>
    </div>

    <!-- RSI Gauge -->
    <div class="card" style="text-align:center">
      <div class="card-title">RSI (14)</div>
      <div class="gauge-wrap">
        <canvas id="gaugeRSI" width="160" height="90"></canvas>
        <div class="gauge-label">
          <div class="gauge-val" id="rsiLabel">{t.rsi_14}</div>
          <div class="gauge-sub">{'oversold' if t.rsi_14 < 30 else 'overbought' if t.rsi_14 > 70 else 'neutral'}</div>
        </div>
      </div>
    </div>

    <!-- VIX Card -->
    <div class="card" style="text-align:center">
      <div class="card-title">Volatility (VIX)</div>
      <div class="big-num mono" style="color:{'var(--bull)' if m.vix_regime=='low' else 'var(--bear)' if m.vix_regime=='high' else 'var(--gold)'}">{m.vix_level}</div>
      <div style="margin-top:8px">
        <span class="badge" style="background:color-mix(in srgb,{'var(--bull)' if m.vix_regime=='low' else 'var(--bear)' if m.vix_regime=='high' else 'var(--gold)'} 20%,transparent);color:{'var(--bull)' if m.vix_regime=='low' else 'var(--bear)' if m.vix_regime=='high' else 'var(--gold)'};">{m.vix_regime.upper()}</span>
      </div>
      <div class="sub-stat" style="margin-top:10px">USD: {m.usd_trend}</div>
      <div class="sub-stat">Macro: {m.macro_bias.replace('_',' ')}</div>
    </div>
  </div>

  <!-- THESIS + CONFIDENCE BREAKDOWN -->
  <div class="grid g84" style="margin-bottom:16px">
    <div class="card">
      <div class="card-title">Primary Thesis</div>
      <div class="thesis-block">{b.primary_thesis}</div>
      <div class="case-grid">
        <div class="bull-case">
          <div class="case-label" style="color:var(--bull)">▲ Bull Case</div>
          {b.bull_case}
        </div>
        <div class="bear-case">
          <div class="case-label" style="color:var(--bear)">▼ Bear Case</div>
          {b.bear_case}
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Signal Weights</div>
      <div class="weight-row">
        <div class="weight-label">Technical</div>
        <div class="weight-bar-bg"><div class="weight-bar" style="width:{round(b.confidence_breakdown.technical_weight*100)}%;background:var(--blue)"></div></div>
        <div class="weight-pct">{round(b.confidence_breakdown.technical_weight*100)}%</div>
      </div>
      <div class="weight-row">
        <div class="weight-label">Fundamental</div>
        <div class="weight-bar-bg"><div class="weight-bar" style="width:{round(b.confidence_breakdown.fundamental_weight*100)}%;background:var(--gold)"></div></div>
        <div class="weight-pct">{round(b.confidence_breakdown.fundamental_weight*100)}%</div>
      </div>
      <div class="weight-row">
        <div class="weight-label">Geopolitical</div>
        <div class="weight-bar-bg"><div class="weight-bar" style="width:{round(b.confidence_breakdown.geopolitical_weight*100)}%;background:#9c6ef8"></div></div>
        <div class="weight-pct">{round(b.confidence_breakdown.geopolitical_weight*100)}%</div>
      </div>
      <div class="weight-note">{b.confidence_breakdown.note}</div>
    </div>
  </div>

  <!-- TECHNICAL INDICATORS -->
  <div class="grid g3" style="margin-bottom:16px">

    <!-- Momentum -->
    <div class="card">
      <div class="card-title">Momentum</div>
      <div class="indicator-row">
        <span class="ind-label">MACD</span>
        <span class="tag tag-{'bull' if t.macd_signal=='bullish' else 'bear' if t.macd_signal=='bearish' else 'neutral'}">{t.macd_signal}</span>
      </div>
      <div class="indicator-row">
        <span class="ind-label">Stoch %K / %D</span>
        <span class="ind-val mono">{t.stoch_k} / {t.stoch_d}</span>
      </div>
      <div class="indicator-row">
        <span class="ind-label">Stoch Signal</span>
        <span class="tag tag-{'bear' if t.stoch_signal=='overbought' else 'bull' if t.stoch_signal=='oversold' else 'neutral'}">{t.stoch_signal}</span>
      </div>
      <div class="indicator-row">
        <span class="ind-label">ADX (14)</span>
        <span>
          <div class="adx-bar-bg"><div class="adx-bar" style="width:{min(t.adx_14,100)}%;background:{'var(--gold)' if t.adx_14>25 else 'var(--muted)'}"></div></div>
          <span class="ind-val mono" style="margin-left:8px">{t.adx_14}</span>
        </span>
      </div>
      <div class="indicator-row">
        <span class="ind-label">EMA Align</span>
        <span class="tag tag-{'bull' if t.ema_alignment=='bullish' else 'bear' if t.ema_alignment=='bearish' else 'neutral'}">{t.ema_alignment}</span>
      </div>
      <div style="margin-top:10px">
        <span class="ema-chip"><span class="ema-label">EMA20 </span>{'N/A' if t.ema20 is None else f'{t.ema20:,.2f}'}</span>
        <span class="ema-chip"><span class="ema-label">EMA50 </span>{'N/A' if t.ema50 is None else f'{t.ema50:,.2f}'}</span>
        <span class="ema-chip"><span class="ema-label">EMA200 </span>{'N/A' if t.ema200 is None else f'{t.ema200:,.2f}'}</span>
      </div>
    </div>

    <!-- Bollinger Bands -->
    <div class="card">
      <div class="card-title">Bollinger Bands (20,2)</div>
      <div style="position:relative;height:100px"><canvas id="chartBB"></canvas></div>
      <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center">
        <div class="card-elevated">
          <div style="font-size:10px;color:var(--muted)">UPPER</div>
          <div class="mono" style="font-size:13px">{t.bb_upper:,.2f}</div>
        </div>
        <div class="card-elevated" style="border:1px solid var(--bias)">
          <div style="font-size:10px;color:var(--muted)">POSITION</div>
          <div style="font-size:12px;color:var(--bias)">{t.bb_position.replace('_',' ')}</div>
        </div>
        <div class="card-elevated">
          <div style="font-size:10px;color:var(--muted)">LOWER</div>
          <div class="mono" style="font-size:13px">{t.bb_lower:,.2f}</div>
        </div>
      </div>
    </div>

    <!-- Volume & CMF -->
    <div class="card">
      <div class="card-title">Volume & Money Flow</div>
      <div style="display:flex;gap:16px;align-items:center;margin-bottom:16px">
        <div class="gauge-wrap" style="width:130px;height:75px">
          <canvas id="gaugeCMF" width="130" height="75"></canvas>
          <div class="gauge-label">
            <div class="gauge-val" style="font-size:16px;color:{cmf_color}">{t.cmf_20:+.4f}</div>
            <div class="gauge-sub">CMF 20</div>
          </div>
        </div>
        <div>
          <div class="card-elevated" style="margin-bottom:8px;text-align:center">
            <div style="font-size:10px;color:var(--muted)">VOLUME TREND</div>
            <div style="font-size:16px;font-weight:700;color:{'var(--bull)' if t.volume_trend=='increasing' else 'var(--bear)' if t.volume_trend=='decreasing' else 'var(--muted)'}"">
              {'▲' if t.volume_trend=='increasing' else '▼' if t.volume_trend=='decreasing' else '—'} {t.volume_trend}
            </div>
          </div>
          <div class="card-elevated">
            <div style="font-size:10px;color:var(--muted)">SIGNAL</div>
            <div class="tag tag-{'bull' if t.cmf_signal=='accumulation' else 'bear' if t.cmf_signal=='distribution' else 'neutral'}" style="margin-top:4px">{t.cmf_signal}</div>
          </div>
        </div>
      </div>
      <div class="indicator-row">
        <span class="ind-label">ATR (14)</span>
        <span class="ind-val mono">{t.atr_14:,.4f}</span>
      </div>
      <div class="indicator-row">
        <span class="ind-label">ATR %</span>
        <span class="ind-val mono">{t.atr_pct}%</span>
      </div>
      <div class="indicator-row">
        <span class="ind-label">BB Position</span>
        <span class="tag tag-neutral">{t.bb_position.replace('_',' ')}</span>
      </div>
    </div>
  </div>

  <!-- KEY LEVELS CHART -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Key Price Levels</div>
    <div style="position:relative;height:260px"><canvas id="chartLevels"></canvas></div>
  </div>

  <!-- MACRO -->
  <div class="grid g64" style="margin-bottom:16px">
    <div class="card">
      <div class="card-title">Macro Environment</div>
      <div class="macro-grid">
        <div class="macro-cell">
          <div class="macro-cell-label">Fed Funds</div>
          <div class="macro-cell-val mono">{m.fed_funds_rate}%</div>
        </div>
        <div class="macro-cell">
          <div class="macro-cell-label">Yield Curve</div>
          <div class="macro-cell-val" style="color:{'var(--bull)' if m.yield_curve=='normal' else 'var(--bear)' if m.yield_curve=='inverted' else 'var(--gold)'}">{m.yield_curve}</div>
        </div>
        <div class="macro-cell">
          <div class="macro-cell-label">CPI Trend</div>
          <div class="macro-cell-val" style="color:{'var(--bear)' if m.cpi_trend=='rising' else 'var(--bull)' if m.cpi_trend=='falling' else 'var(--gold)'}">{m.cpi_trend}</div>
        </div>
        <div class="macro-cell">
          <div class="macro-cell-label">USD Trend</div>
          <div class="macro-cell-val" style="font-size:14px">{m.usd_trend}</div>
        </div>
        <div class="macro-cell">
          <div class="macro-cell-label">Macro Bias</div>
          <div class="macro-cell-val" style="color:{'var(--bull)' if m.macro_bias=='risk_on' else 'var(--bear)' if m.macro_bias=='risk_off' else 'var(--gold)'};">{m.macro_bias.replace('_',' ')}</div>
        </div>
        <div class="macro-cell">
          <div class="macro-cell-label">Unemployment</div>
          <div class="macro-cell-val mono">{m.unemployment_rate}%</div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Macro Radar</div>
      <div style="position:relative;height:220px"><canvas id="chartRadar"></canvas></div>
    </div>
  </div>

  <!-- GEOPOLITICAL -->
  <div class="grid g48" style="margin-bottom:16px">
    <div class="card" style="text-align:center">
      <div class="card-title">Geopolitical Risk</div>
      <div class="gauge-wrap" style="width:180px;height:100px;margin:0 auto">
        <canvas id="gaugeGeo" width="180" height="100"></canvas>
        <div class="gauge-label">
          <div class="gauge-val" style="color:{risk_color}">{g.risk_level.upper()}</div>
          <div class="gauge-sub">risk level</div>
        </div>
      </div>
      <div style="margin-top:16px">
        <ul style="list-style:none;text-align:left">
          {geo_factors}
        </ul>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Geopolitical Headlines</div>
      {geo_headlines}
    </div>
  </div>

  <!-- NEWS -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Recent News</div>
    <div class="news-rail">
      {news_cards}
    </div>
  </div>

  {_render_setups(setups) if setups else ''}

  <!-- KEY RISKS -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Key Risks</div>
    <div class="risks-grid">
      {risk_items}
    </div>
  </div>

  <!-- FLAGS -->
  <div class="card" style="margin-bottom:24px">
    <div class="card-title">Data Quality</div>
    <div class="flags-wrap">
      {flag_items}
    </div>
  </div>

</div><!-- /page -->

<script>
// Safe annotation plugin registration — auto-registered by CDN but guard against undefined
try {{ if (typeof ChartAnnotation !== 'undefined') Chart.register(ChartAnnotation); }} catch(e) {{}}

const BIAS   = '{bias_color}';
const BULL   = '#00d4aa';
const BEAR   = '#ff4d6d';
const GOLD   = '#f5c842';
const MUTED  = '#4a5578';
const BG_EL  = '#131d35';

// ── HALF-RING GAUGE HELPER ──
function halfGauge(id, value, max, colors, segments) {{
  const ctx = document.getElementById(id).getContext('2d');
  const data = segments
    ? segments.map(s => s.val)
    : [value, max - value];

  const bgColors = segments
    ? segments.map(s => s.color)
    : [colors[0], BG_EL];

  return new Chart(ctx, {{
    type: 'doughnut',
    data: {{ datasets: [{{ data, backgroundColor: bgColors, borderWidth: 0, hoverOffset: 0 }}] }},
    options: {{
      rotation: -90, circumference: 180,
      cutout: '72%', responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }},
      animation: {{ duration: 800, easing: 'easeInOutQuart' }},
    }}
  }});
}}

// Confidence
halfGauge('gaugeConf', {round(b.confidence_score*100)}, 100,
  [BIAS], null);

// RSI — three-zone segments
const rsiVal = {t.rsi_14};
halfGauge('gaugeRSI', rsiVal, 100, null, [
  {{ val: Math.min(rsiVal, 30),           color: rsiVal <= 30 ? BULL : MUTED }},
  {{ val: Math.max(0, Math.min(rsiVal-30, 40)), color: (rsiVal>30&&rsiVal<70) ? GOLD : MUTED }},
  {{ val: Math.max(0, rsiVal-70),         color: rsiVal >= 70 ? BEAR : MUTED }},
  {{ val: 100 - rsiVal,                   color: BG_EL }},
]);

// CMF
const cmfNorm = {cmf_norm};
halfGauge('gaugeCMF', cmfNorm, 100,
  ['{cmf_color}'], null);

// Geopolitical
halfGauge('gaugeGeo', {risk_val}, 100,
  ['{risk_color}'], null);

// ── BOLLINGER BANDS ──
(function() {{
  const ctx = document.getElementById('chartBB').getContext('2d');
  const lo   = {t.bb_lower};
  const hi   = {t.bb_upper};
  const curr = {t.current_price};
  const pad  = (hi - lo) * 0.3;

  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: [''],
      datasets: [
        {{ label: 'range', data: [[lo - pad, lo]], backgroundColor: 'transparent', borderWidth: 0 }},
        {{ label: 'BB Band', data: [[lo, hi]], backgroundColor: 'rgba(61,142,248,0.15)', borderColor: '#3d8ef8', borderWidth: 1, borderSkipped: false }},
        {{ label: 'gap',    data: [[hi, hi + pad]], backgroundColor: 'transparent', borderWidth: 0 }},
      ]
    }},
    options: {{
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ enabled: false }},
        annotation: {{
          annotations: {{
            price: {{
              type: 'line', scaleID: 'x', value: curr,
              borderColor: BIAS, borderWidth: 2,
              label: {{ content: 'Price ' + curr.toFixed(2), display: true, position: 'start', backgroundColor: BIAS, color: '#000', font: {{ size: 11 }} }}
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ grid: {{ color: '#1e2d4a' }}, ticks: {{ color: '#7a86a8', maxTicksLimit: 5 }} }},
        y: {{ display: false }},
      }}
    }}
  }});
}})();

// ── KEY LEVELS ──
(function() {{
  const ctx = document.getElementById('chartLevels').getContext('2d');
  const levels = {key_levels_json};
  const curr   = {t.current_price};
  const support = {b.nearest_support};
  const resist  = {b.nearest_resistance};

  const annotations = {{}};
  const colors = {{ strong: GOLD, moderate: '#3d8ef8', weak: MUTED }};

  levels.forEach((lv, i) => {{
    annotations['lv' + i] = {{
      type: 'line', scaleID: 'y', value: lv.x,
      borderColor: colors[lv.strength] || MUTED, borderWidth: 1,
      borderDash: [4, 4],
      label: {{ content: lv.label + ' ' + lv.x.toFixed(2), display: true, position: 'start', backgroundColor: 'rgba(13,21,38,0.85)', color: colors[lv.strength] || MUTED, font: {{ size: 10 }} }}
    }};
  }});

  annotations.support = {{
    type: 'line', scaleID: 'y', value: support,
    borderColor: BULL, borderWidth: 2,
    label: {{ content: 'Support ' + support.toFixed(2), display: true, position: 'end', backgroundColor: BULL, color: '#000', font: {{ size: 11, weight: 'bold' }} }}
  }};
  annotations.resist = {{
    type: 'line', scaleID: 'y', value: resist,
    borderColor: BEAR, borderWidth: 2,
    label: {{ content: 'Resistance ' + resist.toFixed(2), display: true, position: 'end', backgroundColor: BEAR, color: '#fff', font: {{ size: 11, weight: 'bold' }} }}
  }};
  annotations.curr = {{
    type: 'line', scaleID: 'y', value: curr,
    borderColor: BIAS, borderWidth: 2,
    label: {{ content: '▶ ' + curr.toFixed(2), display: true, position: 'start', backgroundColor: BIAS, color: '#000', font: {{ size: 12, weight: 'bold' }} }}
  }};

  const allPrices = levels.map(l => l.x).concat([curr, support, resist]);
  const minP = Math.min(...allPrices) * 0.995;
  const maxP = Math.max(...allPrices) * 1.005;

  // Anchor points force Y axis to render even with no real data
  const anchorData = [{{ x: 0, y: minP }}, {{ x: 0, y: maxP }}];

  new Chart(ctx, {{
    type: 'scatter',
    data: {{ datasets: [{{ data: anchorData, pointRadius: 0, pointHoverRadius: 0, label: '' }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ enabled: false }},
        annotation: {{ annotations }}
      }},
      scales: {{
        x: {{ display: false }},
        y: {{ min: minP, max: maxP, grid: {{ color: '#1e2d4a' }}, ticks: {{ color: '#7a86a8', font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}})();

// ── MACRO RADAR ──
(function() {{
  const ctx = document.getElementById('chartRadar').getContext('2d');
  new Chart(ctx, {{
    type: 'radar',
    data: {{
      labels: ['Rate Env', 'Inflation', 'USD', 'Volatility', 'Yield Curve', 'Macro Sent'],
      datasets: [{{
        label: 'Macro',
        data: {radar_data},
        fill: true,
        backgroundColor: BIAS + '30',
        borderColor: BIAS,
        pointBackgroundColor: BIAS,
        pointRadius: 4,
        borderWidth: 2,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: true }} }},
      scales: {{
        r: {{
          min: 0, max: 10,
          grid: {{ color: '#1e2d4a' }},
          angleLines: {{ color: '#1e2d4a' }},
          ticks: {{ color: '#7a86a8', backdropColor: 'transparent', stepSize: 2 }},
          pointLabels: {{ color: '#c0c8e0', font: {{ size: 12 }} }},
        }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""


def save_report(bias: BiasOutput, setups: Optional[SetupsOutput] = None, output_dir: str = "outputs", interval: str = "1d") -> str:
    """Generate and save the HTML report. Returns the file path."""
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_ticker = bias.asset.replace("=", "").replace("-", "")
    path = Path(output_dir) / f"{safe_ticker}_{ts}_report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_html(bias, setups, interval=interval), encoding="utf-8")
    return str(path)
