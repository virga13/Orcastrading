"""
report.py — generates a self-contained HTML backtest report.
Uses the same dark-theme design language as p1_analysis_engine/utils/report.py.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from p3_backtester.schema import RunStats, TradeRecord
from p3_backtester.aggregator import compute_equity_curve, compute_max_drawdown, RESULTS_DIR


def save_report(
    stats: RunStats,
    trades: list[TradeRecord],
    db_path: Path,
) -> str:
    path = db_path.with_suffix(".html")
    path.write_text(_generate_html(stats, trades), encoding="utf-8")
    return str(path)


def _outcome_color(outcome: str) -> str:
    return {
        "WIN": "#00d4aa", "PARTIAL_WIN": "#7ec8a0",
        "BREAKEVEN": "#f5c842", "LOSS": "#ff4d6d", "EXPIRED": "#4a5578",
    }.get(outcome, "#a0a8c0")


def _generate_html(stats: RunStats, trades: list[TradeRecord]) -> str:
    filled  = [t for t in trades if t.outcome != "EXPIRED"]
    equity  = compute_equity_curve(filled)
    equity_js = json.dumps(equity)

    # Outcome distribution in R-buckets
    buckets: dict[str, int] = {}
    for t in filled:
        bucket = f"{t.pnl_r:.1f}R"
        buckets[bucket] = buckets.get(bucket, 0) + 1
    dist_labels = json.dumps(sorted(buckets.keys(), key=lambda x: float(x[:-1])))
    dist_data   = json.dumps([buckets[k] for k in sorted(buckets.keys(), key=lambda x: float(x[:-1]))])

    # Trade log rows
    trade_rows = ""
    for t in sorted(filled, key=lambda x: x.signal_bar_time):
        oc = _outcome_color(t.outcome)
        pnl_col     = "var(--bull)" if t.pnl_r > 0 else ("var(--bear)" if t.pnl_r < 0 else "var(--gold)")
        net_pnl_col = "var(--bull)" if t.net_pnl_r > 0 else ("var(--bear)" if t.net_pnl_r < 0 else "var(--gold)")
        trade_rows += f"""
        <tr>
          <td>{t.signal_bar_time[:16]}</td>
          <td>{t.setup_name}</td>
          <td style="color:{'var(--bull)' if t.direction=='long' else 'var(--bear)'}">{t.direction.upper()}</td>
          <td>{t.trade_type}</td>
          <td style="font-family:monospace">{f'{t.fill_price:,.4f}' if t.fill_price is not None else 'N/A'}</td>
          <td style="color:{oc};font-weight:600">{t.outcome}</td>
          <td style="color:{pnl_col};font-family:monospace">{t.pnl_r:+.2f}R</td>
          <td style="color:var(--muted);font-family:monospace">-{t.cost_r:.3f}R</td>
          <td style="color:{net_pnl_col};font-family:monospace;font-weight:600">{t.net_pnl_r:+.2f}R</td>
          <td>{t.bars_held or '—'}</td>
        </tr>"""

    wr_delta   = stats.actual_win_rate - stats.avg_claude_win_rate
    ev_delta   = stats.actual_avg_pnl_r - stats.avg_claude_ev
    wr_color   = "#00d4aa" if wr_delta >= 0 else "#ff4d6d"
    ev_color   = "#00d4aa" if ev_delta >= 0 else "#ff4d6d"
    gen_ts     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    net_pf_color = "#00d4aa" if stats.actual_net_profit_factor >= 1.0 else "#ff4d6d"

    # Walk-forward section
    wf_rows = ""
    if stats.walk_forward:
        for w in stats.walk_forward:
            is_test = w.name == "test"
            style   = "font-weight:700" if is_test else ""
            rc      = "#00d4aa" if w.avg_net_pnl_r > 0 else "#ff4d6d"
            pfc     = "#00d4aa" if w.net_profit_factor >= 1.0 else "#ff4d6d"
            wf_rows += f"""
            <tr style="{style}">
              <td>{w.name.upper()}</td>
              <td>{w.start_date} &rarr; {w.end_date}</td>
              <td style="text-align:right">{w.n_fills}</td>
              <td style="text-align:right">{w.win_rate:.1%}</td>
              <td style="text-align:right;color:{rc}">{w.avg_net_pnl_r:+.3f}R</td>
              <td style="text-align:right;color:{pfc}">{w.net_profit_factor:.2f}x</td>
              <td style="text-align:right;color:var(--muted)">{w.kelly_25pct*100:.1f}%</td>
            </tr>"""

    # Category rows for by_trade_type
    type_rows = ""
    for tt, cs in stats.by_trade_type.items():
        if cs.n_trades == 0:
            continue
        wr_c = "#00d4aa" if cs.win_rate >= 0.5 else "#ff4d6d"
        pf_c = "#00d4aa" if cs.profit_factor >= 1.0 else "#ff4d6d"
        type_rows += f"""
        <tr>
          <td style="font-weight:600">{tt}</td>
          <td style="text-align:right">{cs.n_trades}</td>
          <td style="text-align:right;color:{wr_c}">{cs.win_rate:.1%}</td>
          <td style="text-align:right;color:{'var(--bull)' if cs.avg_pnl_r>0 else 'var(--bear)'}">{cs.avg_pnl_r:+.2f}R</td>
          <td style="text-align:right;color:{pf_c}">{cs.profit_factor:.2f}x</td>
          <td style="text-align:right;color:var(--muted)">{cs.avg_claude_win_rate:.1%}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Orcastrading P3 — {stats.ticker} Backtest</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#070c18; --bg-card:#0d1526; --bg-el:#131d35;
    --border:#1e2d4a; --text:#f0f2f8; --muted:#7a86a8;
    --bull:#00d4aa; --bear:#ff4d6d; --gold:#f5c842; --blue:#3d8ef8;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:14px; }}
  header {{ display:flex; align-items:center; justify-content:space-between; padding:14px 32px;
            background:var(--bg-card); border-bottom:1px solid var(--border); position:sticky; top:0; z-index:10; }}
  .wordmark {{ font-size:18px; font-weight:700; color:var(--gold); letter-spacing:2px; }}
  .page {{ max-width:1400px; margin:0 auto; padding:24px; }}
  .card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:20px; margin-bottom:16px; }}
  .card-title {{ font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; color:#8a96b8; margin-bottom:14px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:16px; }}
  .stat-card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:18px 20px; }}
  .stat-val {{ font-size:28px; font-weight:700; line-height:1; margin-bottom:4px; }}
  .stat-label {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:left; padding:8px 12px; font-size:11px; color:#8a96b8; text-transform:uppercase;
        letter-spacing:1px; border-bottom:2px solid var(--border); }}
  td {{ padding:8px 12px; border-bottom:1px solid var(--border); font-size:13px; }}
  tr:hover td {{ background:var(--bg-el); }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:700; text-transform:uppercase; }}
  .chart-wrap {{ position:relative; height:200px; }}
  .chart-wrap-lg {{ position:relative; height:260px; }}
  .timestamp {{ font-size:11px; color:var(--muted); }}
</style>
</head>
<body>

<header>
  <div class="wordmark">ORCASTRADING</div>
  <div style="display:flex;align-items:center;gap:12px">
    <span style="font-size:22px;font-weight:700">{stats.ticker}</span>
    <span class="badge" style="background:#131d35;color:#3d8ef8;border:1px solid #3d8ef8">{stats.interval}</span>
    <span style="color:var(--muted);font-size:13px">{stats.start_date} &rarr; {stats.end_date}</span>
  </div>
  <span class="timestamp">Generated {gen_ts}</span>
</header>

<div class="page">

  <!-- HERO STATS -->
  <div class="grid4">
    <div class="stat-card">
      <div class="stat-val" style="color:{'var(--bull)' if stats.actual_win_rate>=0.5 else 'var(--bear)'}">{stats.actual_win_rate:.1%}</div>
      <div class="stat-label">Actual Win Rate</div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Claude est. {stats.avg_claude_win_rate:.1%}
        <span style="color:{wr_color}">({wr_delta:+.1%})</span></div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:{'var(--bull)' if stats.actual_avg_pnl_r>0 else 'var(--bear)'}">{stats.actual_avg_pnl_r:+.2f}R</div>
      <div class="stat-label">Avg EV per Trade</div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Claude est. {stats.avg_claude_ev:+.2f}R
        <span style="color:{ev_color}">({ev_delta:+.2f})</span></div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:{'var(--bull)' if stats.actual_profit_factor>=1 else 'var(--bear)'}">{stats.actual_profit_factor:.2f}x</div>
      <div class="stat-label">Profit Factor (Gross)</div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Net PF: <span style="color:{net_pf_color}">{stats.actual_net_profit_factor:.2f}x</span> &middot; MaxDD: <span style="color:var(--bear)">-{stats.max_drawdown_r:.2f}R</span></div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{stats.total_trades_filled}</div>
      <div class="stat-label">Trades Filled</div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">{stats.total_signals} signals &middot; {stats.total_trades_expired} expired &middot; Sharpe {stats.sharpe_r:.2f}</div>
    </div>
  </div>

  <!-- RISK METRICS ROW -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:16px">
    <div class="stat-card">
      <div class="stat-val" style="color:{'var(--bull)' if stats.calmar_ratio >= 2.0 else ('var(--gold)' if stats.calmar_ratio >= 1.0 else 'var(--bear)')}">{stats.calmar_ratio:.2f}x</div>
      <div class="stat-label">Calmar Ratio</div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Total net R / Max drawdown &middot; &ge;2.0 target</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:var(--bear)">-{stats.max_drawdown_r:.2f}R</div>
      <div class="stat-label">Max Drawdown</div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Duration: {stats.max_drawdown_duration} trades in drawdown</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:var(--bull)">{stats.max_win_streak}</div>
      <div class="stat-label">Max Win Streak</div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Consecutive winning trades</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:var(--bear)">{stats.max_loss_streak}</div>
      <div class="stat-label">Max Loss Streak</div>
      <div style="font-size:11px;color:var(--muted);margin-top:6px">Consecutive losing trades &middot; Kelly: {stats.kelly_25pct*100:.1f}%</div>
    </div>
  </div>

  <!-- CHARTS ROW -->
  <div class="grid2">
    <div class="card">
      <div class="card-title">Equity Curve (Cumulative R)</div>
      <div class="chart-wrap-lg"><canvas id="chartEquity"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Trade Outcome Distribution (R-multiples)</div>
      <div class="chart-wrap-lg"><canvas id="chartDist"></canvas></div>
    </div>
  </div>

  <!-- BY TYPE TABLE -->
  <div class="card">
    <div class="card-title">Performance by Trade Type</div>
    <table>
      <thead><tr>
        <th>Type</th><th style="text-align:right">Trades</th>
        <th style="text-align:right">Win %</th><th style="text-align:right">Avg R</th>
        <th style="text-align:right">Profit Factor</th><th style="text-align:right">Claude Win % Est</th>
      </tr></thead>
      <tbody>{type_rows}</tbody>
    </table>
  </div>

  <!-- WALK-FORWARD -->
  {'<div class="card"><div class="card-title">Walk-Forward Validation (Train 60% / Val 20% / Test 20%)</div><div style="overflow-x:auto"><table><thead><tr><th>Window</th><th>Period</th><th style="text-align:right">Fills</th><th style="text-align:right">Win %</th><th style="text-align:right">Net Avg R</th><th style="text-align:right">Net PF</th><th style="text-align:right">Kelly 25%</th></tr></thead><tbody>' + wf_rows + '</tbody></table></div><p style="font-size:11px;color:var(--muted);margin-top:12px">TEST window is held-out — never used during optimization. This is the honest performance estimate.</p></div>' if wf_rows else ''}

  <!-- TRADE LOG -->
  <div class="card">
    <div class="card-title">Trade Log ({len(filled)} filled trades) — Strategy: {stats.strategy_name}</div>
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Signal Time</th><th>Setup</th><th>Dir</th><th>Type</th>
          <th>Fill</th><th>Outcome</th><th>Gross R</th><th>Cost</th><th>Net R</th><th>Bars</th>
        </tr></thead>
        <tbody>{trade_rows}</tbody>
      </table>
    </div>
  </div>

</div>

<script>
const BULL = '#00d4aa', BEAR = '#ff4d6d', GOLD = '#f5c842', MUTED = '#4a5578';

// Equity curve
(function() {{
  const eq = {equity_js};
  if (!eq.length) return;
  new Chart(document.getElementById('chartEquity').getContext('2d'), {{
    type: 'line',
    data: {{
      labels: eq.map((_,i) => i+1),
      datasets: [{{
        data: eq,
        borderColor: eq[eq.length-1] >= 0 ? BULL : BEAR,
        borderWidth: 2,
        fill: true,
        backgroundColor: (eq[eq.length-1] >= 0 ? '#00d4aa' : '#ff4d6d') + '18',
        pointRadius: 0,
        tension: 0.3,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
        label: ctx => ctx.parsed.y.toFixed(2) + 'R'
      }} }} }},
      scales: {{
        x: {{ display: false }},
        y: {{ grid: {{ color: '#1e2d4a' }}, ticks: {{ color: '#7a86a8',
             callback: v => v.toFixed(1)+'R' }} }}
      }}
    }}
  }});
}})();

// Distribution
(function() {{
  const labels = {dist_labels};
  const data   = {dist_data};
  const colors = labels.map(l => parseFloat(l) > 0 ? BULL : parseFloat(l) < 0 ? BEAR : GOLD);
  new Chart(document.getElementById('chartDist').getContext('2d'), {{
    type: 'bar',
    data: {{ labels, datasets: [{{ data, backgroundColor: colors, borderRadius: 4 }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ display: false }}, ticks: {{ color: '#7a86a8' }} }},
        y: {{ grid: {{ color: '#1e2d4a' }}, ticks: {{ color: '#7a86a8' }} }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""
