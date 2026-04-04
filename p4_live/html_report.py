"""
p4_live/html_report.py — Generate a self-contained HTML forward-test report.

Covers all registered assets, with per-asset performance tables,
equity curves, and full trade logs.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from p4_live.report import compute_forward_stats
from p4_live.scanner import ASSETS


# ── Palette ───────────────────────────────────────────────────────────────────
_COLORS = ["#4f8ef7", "#f7a44f", "#4fcea2", "#f75f5f", "#b44ff7", "#f7e24f"]


def _equity_curve(trades: list[dict]) -> list[float]:
    closed = sorted(
        [t for t in trades if t["pnl_r"] is not None],
        key=lambda t: t["exit_date"] or t["signal_date"],
    )
    eq, curve = 0.0, [0.0]
    for t in closed:
        eq += t["pnl_r"]
        curve.append(round(eq, 3))
    return curve


def _status_badge(status: str) -> str:
    colors = {
        "win":         ("#1a7a4a", "#d4f5e2"),
        "partial_win": ("#0e6680", "#cdf0fa"),
        "loss":        ("#8b1a1a", "#fde8e8"),
        "breakeven":   ("#7a6f00", "#fef9c3"),
        "expired":     ("#555", "#eee"),
        "filled":      ("#1a4a8b", "#d8e8fd"),
        "pending":     ("#555", "#eee"),
    }
    fg, bg = colors.get(status, ("#333", "#f5f5f5"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:10px;font-size:0.78em;font-weight:600;'
            f'white-space:nowrap">{status.upper().replace("_", " ")}</span>')


def _pnl_cell(pnl_r) -> str:
    if pnl_r is None:
        return '<td style="color:#999">—</td>'
    color = "#1a7a4a" if pnl_r > 0 else ("#8b1a1a" if pnl_r < 0 else "#555")
    sign  = "+" if pnl_r > 0 else ""
    return f'<td style="color:{color};font-weight:600">{sign}{pnl_r:.2f}R</td>'


def _metric_row(label, live_val, base_val, fmt, higher_is_better=True):
    live_s = fmt.format(live_val)
    base_s = fmt.format(base_val)
    ratio  = live_val / base_val if base_val else 0
    if higher_is_better:
        ok = ratio >= 0.90
        warn = ratio >= 0.70
    else:
        ok = ratio <= 1.10
        warn = ratio <= 1.30
    if ok:
        color, icon = "#1a7a4a", "&#10003;"
    elif warn:
        color, icon = "#8b6f00", "~"
    else:
        color, icon = "#8b1a1a", "&#10007;"
    return (
        f"<tr>"
        f"<td>{label}</td>"
        f"<td style='font-weight:700'>{live_s}</td>"
        f"<td style='color:#888'>{base_s}</td>"
        f"<td style='color:{color};font-weight:700;font-size:1.1em'>{icon}</td>"
        f"</tr>"
    )


def generate_html(output_path: str | None = None) -> str:
    """Generate the report and return the file path."""
    from p4_live.journal import get_all_trades

    all_trades = get_all_trades()
    generated  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Build per-asset sections ──────────────────────────────────────────────
    asset_sections_html = ""
    summary_rows_html   = ""
    chart_datasets      = []

    for idx, asset in enumerate(ASSETS):
        ticker = asset["ticker"]
        label  = asset["label"]
        base   = asset["baseline"]
        color  = _COLORS[idx % len(_COLORS)]
        stats  = compute_forward_stats(ticker=ticker)
        s      = stats
        trades = sorted(
            [t for t in all_trades if t.get("ticker") == ticker],
            key=lambda t: t["signal_date"],
        )
        curve  = _equity_curve(trades)

        # Verdict flag for summary
        if s["n_closed"] < 10:
            verdict_badge = '<span style="background:#f5f5f5;color:#888;padding:2px 8px;border-radius:10px;font-size:0.78em">INSUFFICIENT DATA</span>'
        elif s["win_rate"] >= base["win_rate"] * 0.85 and s["profit_factor"] >= 1.0:
            verdict_badge = '<span style="background:#d4f5e2;color:#1a7a4a;padding:2px 8px;border-radius:10px;font-size:0.78em;font-weight:600">ON TRACK</span>'
        else:
            verdict_badge = '<span style="background:#fde8e8;color:#8b1a1a;padding:2px 8px;border-radius:10px;font-size:0.78em;font-weight:600">UNDERPERFORMING</span>'

        # Summary table row
        pf_color = "#1a7a4a" if s["profit_factor"] >= 1.0 else "#8b1a1a"
        wr_color = "#1a7a4a" if s["win_rate"] >= base["win_rate"] * 0.85 else "#8b1a1a"
        summary_rows_html += f"""
        <tr>
          <td style="font-weight:600;color:{color}">{label}</td>
          <td>{s['n_signals']}</td>
          <td>{s['n_fills']}</td>
          <td>{s['n_closed']}</td>
          <td style="color:{wr_color};font-weight:600">{s['win_rate']:.1%}</td>
          <td style="font-weight:600">{s['avg_r']:+.3f}R</td>
          <td style="color:{pf_color};font-weight:600">{s['profit_factor']:.2f}</td>
          <td style="color:#8b1a1a">-{s['max_drawdown_r']:.1f}R</td>
          <td style="font-weight:600">{s['total_r']:+.1f}R</td>
          <td>{verdict_badge}</td>
        </tr>"""

        # Equity curve dataset
        chart_datasets.append({
            "label":           label,
            "data":            curve,
            "borderColor":     color,
            "backgroundColor": color + "22",
            "tension":         0.3,
            "fill":            False,
            "pointRadius":     2,
        })

        # Metric comparison table
        if s["n_closed"] > 0:
            metric_rows = (
                _metric_row("Win Rate",      s["win_rate"],       base["win_rate"],     "{:.1%}")
                + _metric_row("Avg R/trade", s["avg_r"],          base["avg_net_pnl_r"],"{:+.3f}R")
                + _metric_row("Profit Factor",s["profit_factor"], base["net_pf"],       "{:.2f}")
                + _metric_row("Max Drawdown", s["max_drawdown_r"],base["max_drawdown_r"],"{:.1f}R", higher_is_better=False)
            )
            # Trade quality rows (no baseline comparison — purely informational)
            quality_rows = ""
            if s.get("avg_actual_win_r") is not None:
                quality_rows += f'<tr><td>Avg Win (actual)</td><td>{s["avg_actual_win_r"]:+.2f}R</td><td>—</td><td>—</td></tr>'
            if s.get("avg_actual_loss_r") is not None:
                quality_rows += f'<tr><td>Avg Loss (actual)</td><td>{s["avg_actual_loss_r"]:+.2f}R</td><td>—</td><td>—</td></tr>'
            if s.get("actual_rr") is not None:
                quality_rows += f'<tr><td>Actual R/R</td><td>{s["actual_rr"]:.2f}×</td><td>—</td><td>—</td></tr>'
            if s.get("avg_duration_days") is not None:
                quality_rows += f'<tr><td>Avg Hold (days)</td><td>{s["avg_duration_days"]:.1f}</td><td>—</td><td>—</td></tr>'
            metrics_html = f"""
            <table class="metrics-table">
              <thead><tr><th>Metric</th><th>Live</th><th>Baseline</th><th>Status</th></tr></thead>
              <tbody>{metric_rows}{quality_rows}</tbody>
            </table>"""
        else:
            metrics_html = '<p style="color:#888;margin:8px 0">No closed trades yet.</p>'

        # Trade log rows
        trade_rows = ""
        for t in reversed(trades):
            sc   = _status_badge(t["status"])
            fill = f"{t['fill_price']:,.2f}" if t["fill_price"] else "—"
            exit_p = f"{t['exit_price']:,.2f}" if t["exit_price"] else "—"
            fill_d = t["fill_date"] or "—"
            exit_d = t["exit_date"] or "—"
            pnl_td = _pnl_cell(t["pnl_r"])
            trade_rows += f"""
            <tr>
              <td>{t['signal_date']}</td>
              <td>{t['direction'].upper()}</td>
              <td>{t['entry_low']:,.2f} – {t['entry_high']:,.2f}</td>
              <td>{t['stop_loss']:,.2f}</td>
              <td>{t['tp1']:,.2f}</td>
              <td>{fill}</td>
              <td>{fill_d}</td>
              <td>{exit_p}</td>
              <td>{exit_d}</td>
              <td>{sc}</td>
              {pnl_td}
            </tr>"""

        # Stat cards
        total_r_color = "#1a7a4a" if s["total_r"] >= 0 else "#8b1a1a"
        asset_sections_html += f"""
        <div class="asset-section" id="asset-{ticker.replace('^','').replace('=','')}">
          <div class="asset-header">
            <div class="asset-dot" style="background:{color}"></div>
            <h2>{label} <span class="ticker-tag">{ticker}</span></h2>
            <div style="margin-left:auto">{verdict_badge}</div>
          </div>

          <div class="stat-cards">
            <div class="stat-card">
              <div class="stat-val">{s['n_signals']}</div>
              <div class="stat-lbl">Signals</div>
            </div>
            <div class="stat-card">
              <div class="stat-val">{s['n_fills']}</div>
              <div class="stat-lbl">Fills</div>
            </div>
            <div class="stat-card">
              <div class="stat-val" style="color:{wr_color}">{s['win_rate']:.1%}</div>
              <div class="stat-lbl">Win Rate</div>
              <div class="stat-sub">base {base['win_rate']:.1%}</div>
            </div>
            <div class="stat-card">
              <div class="stat-val" style="color:{pf_color}">{s['profit_factor']:.2f}x</div>
              <div class="stat-lbl">Profit Factor</div>
              <div class="stat-sub">base {base['net_pf']:.2f}x</div>
            </div>
            <div class="stat-card">
              <div class="stat-val" style="color:{total_r_color}">{s['total_r']:+.1f}R</div>
              <div class="stat-lbl">Total P&L</div>
            </div>
            <div class="stat-card">
              <div class="stat-val" style="color:#8b1a1a">-{s['max_drawdown_r']:.1f}R</div>
              <div class="stat-lbl">Max Drawdown</div>
              <div class="stat-sub">base -{base['max_drawdown_r']:.1f}R</div>
            </div>
            <div class="stat-card">
              <div class="stat-val">{s['sharpe']:.2f}</div>
              <div class="stat-lbl">Sharpe</div>
            </div>
            <div class="stat-card">
              <div class="stat-val">{base['kelly_25pct']:.1%}</div>
              <div class="stat-lbl">Kelly 25% Size</div>
              <div class="stat-sub">{base['period']}</div>
            </div>
          </div>

          <div class="two-col">
            <div>{metrics_html}</div>
            <div>
              <canvas id="chart-{idx}" height="160"></canvas>
            </div>
          </div>

          <details>
            <summary style="cursor:pointer;font-weight:600;margin:16px 0 8px;color:#555">
              Trade Log ({len(trades)} signals, {s['n_fills']} filled)
            </summary>
            <div style="overflow-x:auto">
            <table class="trade-table">
              <thead>
                <tr>
                  <th>Signal Date</th><th>Dir</th><th>Entry Zone</th>
                  <th>Stop Loss</th><th>TP1</th>
                  <th>Fill</th><th>Fill Date</th>
                  <th>Exit</th><th>Exit Date</th>
                  <th>Status</th><th>P&L (R)</th>
                </tr>
              </thead>
              <tbody>{trade_rows}</tbody>
            </table>
            </div>
          </details>
        </div>"""

    # ── Chart init JS ─────────────────────────────────────────────────────────
    chart_js = ""
    for idx, (asset, ds) in enumerate(zip(ASSETS, chart_datasets)):
        label = asset["label"]
        chart_js += f"""
    new Chart(document.getElementById('chart-{idx}'), {{
      type: 'line',
      data: {{
        labels: Array.from({{length: {len(ds['data'])}}}, (_, i) => i),
        datasets: [{{
          label: '{label} Equity (R)',
          data: {json.dumps(ds['data'])},
          borderColor: '{ds['borderColor']}',
          backgroundColor: '{ds['backgroundColor']}',
          tension: {ds['tension']},
          fill: true,
          pointRadius: {ds['pointRadius']},
        }}]
      }},
      options: {{
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ display: false }},
          y: {{ grid: {{ color: '#f0f0f0' }}, ticks: {{ callback: v => v + 'R' }} }}
        }},
        animation: false,
      }}
    }});"""

    # ── Full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orcastrading P4 — Forward-Test Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f7f8fa; color: #1a1a2e; font-size: 14px; }}
  .page {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px; }}

  /* Header */
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
             color: #fff; border-radius: 12px; padding: 28px 32px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 1.6em; font-weight: 700; letter-spacing: 2px; }}
  .header .subtitle {{ color: #aab; margin-top: 4px; font-size: 0.9em; }}
  .header .meta {{ color: #778; font-size: 0.82em; margin-top: 12px; }}

  /* Nav pills */
  .nav {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }}
  .nav a {{ padding: 6px 14px; border-radius: 20px; background: #fff;
             border: 1px solid #ddd; text-decoration: none; color: #444;
             font-size: 0.85em; font-weight: 500; transition: all .15s; }}
  .nav a:hover {{ background: #1a1a2e; color: #fff; border-color: #1a1a2e; }}

  /* Summary table */
  .summary-card {{ background: #fff; border-radius: 10px; padding: 20px;
                   box-shadow: 0 1px 4px rgba(0,0,0,.06); margin-bottom: 24px; }}
  .summary-card h3 {{ font-size: 1em; color: #555; margin-bottom: 14px; font-weight: 600; }}
  .summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  .summary-table th {{ text-align: left; color: #888; font-weight: 500;
                        border-bottom: 2px solid #f0f0f0; padding: 6px 10px; }}
  .summary-table td {{ padding: 8px 10px; border-bottom: 1px solid #f5f5f5; }}
  .summary-table tr:hover td {{ background: #fafafa; }}

  /* Asset sections */
  .asset-section {{ background: #fff; border-radius: 10px; padding: 22px 24px;
                    box-shadow: 0 1px 4px rgba(0,0,0,.06); margin-bottom: 20px; }}
  .asset-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }}
  .asset-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .asset-header h2 {{ font-size: 1.15em; font-weight: 700; }}
  .ticker-tag {{ font-size: 0.72em; color: #888; font-weight: 400;
                 background: #f0f0f0; padding: 2px 8px; border-radius: 10px;
                 margin-left: 6px; vertical-align: middle; }}

  /* Stat cards */
  .stat-cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }}
  .stat-card {{ background: #f8f9fc; border-radius: 8px; padding: 12px 16px;
                min-width: 100px; flex: 1; }}
  .stat-val {{ font-size: 1.3em; font-weight: 700; color: #1a1a2e; }}
  .stat-lbl {{ font-size: 0.75em; color: #888; margin-top: 2px; font-weight: 500; }}
  .stat-sub {{ font-size: 0.7em; color: #aaa; margin-top: 1px; }}

  /* Two-column layout */
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
              margin-bottom: 16px; align-items: start; }}
  @media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

  /* Metrics table */
  .metrics-table {{ width: 100%; border-collapse: collapse; font-size: 0.87em; }}
  .metrics-table th {{ text-align: left; color: #888; font-weight: 500; padding: 5px 8px;
                        border-bottom: 2px solid #f0f0f0; }}
  .metrics-table td {{ padding: 7px 8px; border-bottom: 1px solid #f5f5f5; }}

  /* Trade table */
  .trade-table {{ width: 100%; border-collapse: collapse; font-size: 0.82em;
                  white-space: nowrap; }}
  .trade-table th {{ background: #f5f7fa; color: #666; font-weight: 600;
                      padding: 7px 10px; text-align: left;
                      border-bottom: 2px solid #e8eaf0; }}
  .trade-table td {{ padding: 6px 10px; border-bottom: 1px solid #f0f2f5; }}
  .trade-table tr:hover td {{ background: #f9fafc; }}

  details summary::-webkit-details-marker {{ color: #888; }}
  details[open] summary {{ margin-bottom: 12px; }}

  .disclaimer {{ background: #fffbeb; border: 1px solid #f0d060; border-radius: 8px;
                 padding: 12px 16px; margin-bottom: 20px; font-size: 0.82em; color: #7a6000; }}
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <div style="display:flex;align-items:center;gap:16px">
      <div>
        <h1>ORCASTRADING &mdash; P4 Forward-Test</h1>
        <div class="subtitle">MTF Trend Strategy &middot; Daily entry / Weekly bias &middot; Long-only &middot; ADX &ge; 15</div>
        <div class="meta">Generated {generated} &middot; Simulated from 2025-01-01 &middot; 6 assets</div>
      </div>
    </div>
  </div>

  <div class="disclaimer">
    &#9888; <strong>Simulated forward test.</strong> Trades from 2025-01-01 replayed bar-by-bar
    with zero lookahead using the locked strategy config. Fills use pessimistic entry (top of zone).
    Not live trading results.
  </div>

  <nav class="nav">
    {''.join(f'<a href="#asset-{a["ticker"].replace(chr(94),"").replace("=","")}">&#9679; {a["label"]}</a>' for a in ASSETS)}
  </nav>

  <div class="summary-card">
    <h3>Portfolio Summary</h3>
    <table class="summary-table">
      <thead>
        <tr>
          <th>Asset</th><th>Signals</th><th>Fills</th><th>Closed</th>
          <th>Win Rate</th><th>Avg R</th><th>Profit Factor</th>
          <th>Max DD</th><th>Total R</th><th>Verdict</th>
        </tr>
      </thead>
      <tbody>{summary_rows_html}</tbody>
    </table>
  </div>

  {asset_sections_html}

</div>
<script>
{chart_js}
</script>
</body>
</html>"""

    # Write file
    out = Path(output_path) if output_path else (
        Path(__file__).parent.parent / "outputs" /
        f"p4_forward_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)
