# Orcastrading

A modular trading intelligence platform built in sequential phases, each feeding into the next — AI-driven asset analysis, a historical strategy backtester, a live multi-asset scanner with MT5 execution, an insider/momentum tracker, and a Streamlit dashboard tying it all together.

---

## Quickstart (dashboard, no API keys needed)

The Streamlit UI (`ui/app.py`) has a built-in **dev mode** that skips the Supabase login and runs entirely against a local SQLite journal, so you can explore the full dashboard without provisioning any backend.

```bash
git clone <this-repo>
cd Orcastrading
pip install -r requirements.txt

# create a .env with just this line:
echo "ORCA_DEV_MODE=1" > .env

streamlit run ui/app.py
```

This gets you the Dashboard, Strategies, Journal, Insider, Momentum, and Settings pages backed by sample/local data. Live scanning, MT5 execution, and the P1 AI analysis CLI need real API keys (see each phase's Setup section below) — everything else works out of the box.

---

## Architecture Overview

```
P1 → P2 → P3 → P4 → P5
                ↓
         Streamlit UI (ui/app.py)
```

| Phase | What it does | Status |
|-------|-------------|--------|
| P1 | Claude-driven asset analysis → bias + trade setups | Complete |
| P2 | Trade setup decision tree | Integrated into P1 |
| P3 | Historical strategy backtester (signal gen + simulation) | Complete |
| P4 | Live multi-asset scanner, MT5 execution, trade journal, alerts | Complete |
| P5 | Insider/momentum tracker (correlator, alerts, analytics) | Complete |

---

## Phases

### P1 — Asset Analysis Engine `[COMPLETE]`
**Technical indicators + macro + news + geopolitics → structured bias + trade setups**

Aggregates multi-source data at a user-selected candlestick timeframe and synthesizes it via Claude AI into a structured directional bias and concrete trade plans for any asset.

#### Features
- **Any asset class** — equities, crypto, forex, commodities
- **Common-name aliases** — type `SILVER`, `GOLD`, `BTC`, `EURUSD` without needing yfinance ticker syntax
- **Candlestick timeframe selection** — `1m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1wk`; OHLCV data and all indicators are computed at the chosen resolution
- **Technical indicators** — RSI(14), MACD, EMA 20/50/200, Bollinger Bands(20,2), ATR(14), ADX(14), Stochastic(14,3), CMF(20), volume trend, Fibonacci retracements, key support/resistance levels
- **Macroeconomic data** — Fed Funds Rate, yield curve, CPI trend, VIX, USD trend, unemployment (via FRED)
- **News sentiment** — ticker-specific recent headlines via yfinance + NewsAPI fallback
- **Geopolitical risk** — asset-class-aware risk scoring from live news
- **Claude AI synthesis** — bias output and trade setups generated via structured tool use (validated Pydantic schemas, no hallucinated JSON)
- **Extended thinking** — optional two-pass Claude reasoning for deeper analysis (`--extended-thinking`)
- **Trade setups** — 2–4 setups per analysis with entry zone, SL (structural), trailing SL to breakeven, 2–3 targets, R:R (min 1.5), win rate estimate, EV, profit factor, trade duration, invalidation scenario, and a 4–6 scenario decision tree
- **Rich HTML report** — dark-theme dashboard with Chart.js gauges, Bollinger Band chart, key levels chart, macro radar, news cards with links, and full trade plan cards
- **Interactive CLI** — prompts for ticker and timeframe when run without arguments

#### Usage

```bash
# Interactive mode — prompts for ticker and candlestick timeframe, opens HTML report
python -m p1_analysis_engine

# Direct CLI
python -m p1_analysis_engine AAPL --timeframe 1h --report
python -m p1_analysis_engine BTC --timeframe 15m --report
python -m p1_analysis_engine SILVER --timeframe 1d --report
python -m p1_analysis_engine EURUSD --timeframe 4h --report

# Extended thinking (two-pass Claude reasoning — slower, higher quality setups)
python -m p1_analysis_engine AAPL --timeframe 1h --report --extended-thinking

# JSON output only
python -m p1_analysis_engine AAPL --json

# Skip saving output file
python -m p1_analysis_engine AAPL --no-save
```

#### Supported Timeframes

| Flag | Interval | Max lookback | Use case |
|------|----------|-------------|----------|
| `1m` | 1 minute | 6 days | Scalping |
| `5m` | 5 minutes | 58 days | Scalping / intraday |
| `15m` | 15 minutes | 58 days | Intraday / short swing |
| `30m` | 30 minutes | 58 days | Intraday |
| `1h` | 1 hour | 720 days | Swing / intraday |
| `1d` | Daily | 730 days | Swing / positional (default) |
| `1wk` | Weekly | 10 years | Macro / positional |

#### Asset Aliases

Common names are automatically resolved to the correct yfinance ticker:

| Input | Resolves to |
|-------|------------|
| `GOLD`, `SILVER`, `OIL`, `COPPER` | `GC=F`, `SI=F`, `CL=F`, `HG=F` |
| `BTC`, `ETH`, `SOL`, `XRP` | `BTC-USD`, `ETH-USD`, `SOL-USD`, `XRP-USD` |
| `EURUSD`, `GBPUSD`, `USDJPY` | `EURUSD=X`, `GBPUSD=X`, `USDJPY=X` |

#### Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
FRED_API_KEY=...
NEWS_API_KEY=...
```

---

### P2 — Trading Setups Decision Tree `[INTEGRATED INTO P1]`
**Scenario branches: direction, entry, SL/TP, RR, probability, EV, trade type, duration**

Trade setups and decision trees are currently generated as part of the P1 HTML report. A standalone P2 module with additional filtering, backtesting hooks, and setup persistence is planned.

---

### P3 — Historical Strategy Backtester `[COMPLETE]`
**Validates P1 trade setups against historical OHLCV data — actual vs Claude-estimated win rate, EV, profit factor, drawdown**

Two-pass architecture that separates expensive signal generation (Pass 1) from free, re-runnable trade simulation (Pass 2).

#### How it works

**Pass 1 — Signal Generation** (runs Claude API once per signal bar)
- Iterates over historical bars at the selected signal mode
- At each signal bar, slices the OHLCV data to that point only (zero lookahead)
- Recomputes all technical indicators on the slice (same logic as P1 fetcher)
- Calls `synthesize_full` to generate bias + trade setups
- Caches the full output as JSON — skipped on re-runs unless `--force-regenerate`

**Pass 2 — Trade Simulation** (no API calls — free to re-run)
- Loads cached signals, simulates each setup forward through the full OHLCV history
- Pessimistic fill: entry at worst end of the zone (entry_high for longs, entry_low for shorts)
- SL checked before TP on same-candle conflicts (conservative)
- Trailing SL to breakeven: moves SL to fill price once `trailing_sl_to_breakeven` level is touched
- Partial target tracking: allocation closed progressively at each TP level
- Results written to SQLite

#### Features
- **Signal modes**: `session-open` (default — first bar of each day), `every-n-bars`, `key-level-touch`
- **Comparison table**: Claude's estimated win rate, EV, and profit factor vs actual results
- **Breakdowns**: by trade type (scalp / intraday / swing), direction (long / short), priority (primary / secondary / conditional)
- **Statistics**: win rate, avg R, profit factor, max drawdown (R), Sharpe (R)
- **HTML report**: equity curve, outcome distribution chart, trade log table
- **SQLite storage**: all signals, trades, and run metadata persisted for later querying
- **Known limitation**: macro and news data during Pass 1 reflects current state, not historical — flagged in each signal record

#### Usage

```bash
# Interactive — prompts for ticker, interval, date range
python -m p3_backtester

# Direct
python -m p3_backtester AAPL --interval 1d --start 2024-01-01 --end 2024-12-31 --report
python -m p3_backtester BTC --interval 1h --start 2025-01-01 --end 2025-06-30 --signal-mode every-n-bars --every-n 24

# Re-run simulation without new API calls (uses existing cache)
python -m p3_backtester AAPL --interval 1d --start 2024-01-01 --end 2024-12-31 --simulate-only --report

# Force re-generation of all signals (re-runs Pass 1)
python -m p3_backtester AAPL --interval 1d --start 2024-01-01 --end 2024-12-31 --force-regenerate

# List cached runs
python -m p3_backtester --list-runs

# JSON output
python -m p3_backtester AAPL --interval 1d --start 2024-01-01 --end 2024-12-31 --json
```

#### Signal Modes

| Mode | Behavior |
|------|----------|
| `session-open` | One signal per trading day — first bar after midnight/open |
| `every-n-bars` | Signal every N bars (use `--every-n N`) |
| `key-level-touch` | Signal when price crosses a rolling 20-bar S/R level |

#### File Structure

```
p3_backtester/
├── schema.py            — BacktestConfig, SignalRecord, TradeRecord, RunStats
├── bar_slicer.py        — recompute_technical() on sliced df (zero lookahead guarantee)
├── market_data.py       — yfinance fetch with warmup window calculation
├── signal_scheduler.py  — determines which bar indices generate signals
├── trade_simulator.py   — fill, SL, TP, trailing breakeven simulation
├── pass1_signal_gen.py  — Pass 1 loop with JSON caching and progress bar
├── pass2_simulator.py   — Pass 2 loop, no API calls
├── aggregator.py        — statistics, SQLite persistence, equity curve, drawdown
├── report.py            — dark-theme HTML report
├── cli.py               — argparse + interactive prompt + rich console output
├── cache/               — JSON signal cache (gitignored)
└── results/             — SQLite databases + HTML reports (gitignored)
```

#### Cost

Pass 1 calls Claude once per signal bar. With `session-open` mode on daily candles over a 1-year backtest (~252 trading days), that is ~252 API calls at ~$0.025–0.035 each ≈ **$6–9 total**. Pass 2 is free. The cache means you only pay for Pass 1 once per date range.

---

### P4 — Live Scanner, Journal & MT5 Execution `[COMPLETE]`
**Runs the validated P3 strategies against live market data, journals every signal, and (optionally) executes on a MetaTrader 5 demo account**

#### Components
- `p4_live/scanner.py` — scans the configured asset/strategy/timeframe combos (`config/assets.yaml`, `config/strategies.yaml`) for live signals, builds an HTML forward-test report
- `p4_live/mt5_monitor.py` — places and monitors trades on a MetaTrader 5 terminal (Windows only, demo account recommended); guarded by `MT5_ENABLED` in `.env` so it's safe to leave off
- `p4_live/journal.py` / `p4_live/journal_supabase.py` — SQLite-backed trade journal (single-user/dev mode) and a Supabase-backed equivalent for multi-user deployments, with RLS-enforced isolation
- `run_scheduler.py` — orchestrates recurring scans, MT5 order retries, risk gate checks (daily/weekly loss limits), and stale-signal detection
- Alerts via Telegram and/or email (`config/assets.yaml`-driven, see `.env.example`)

#### Usage

```bash
# One-off scan / CLI
python -m p4_live

# Continuous scheduler (scans, journals, optionally executes on MT5)
python run_scheduler.py
```

#### Setup

MT5 execution requires `pip install MetaTrader5` (commented out in `requirements.txt` since it's Windows-only) and a running MT5 terminal logged into a demo/live account. Leave `MT5_ENABLED=false` to scan and journal without placing any trades.

---

### Multi-user / SaaS mode `[COMPLETE]`

The Streamlit UI supports two backends:

- **Dev mode** (`ORCA_DEV_MODE=1`, no Supabase env vars) — single local user, SQLite journal, no login screen. This is the default for running the project locally.
- **Supabase mode** (`SUPABASE_URL` + `SUPABASE_ANON_KEY` set) — full auth (login/register/reset password), per-user data isolation via Postgres RLS (`supabase/schema.sql`), and `journal_supabase.py` instead of the local journal. Intended for a hosted multi-user deployment.

---

### P5 — Insider & Momentum Tracker `[COMPLETE]`
**Correlates insider-style signals and momentum spikes across assets, with its own alerting and analytics**

- `p5_insider/correlator.py` — cross-asset signal correlation
- `p5_insider/momentum_scanner.py` — momentum spike detection
- `p5_insider/analytics.py`, `p5_insider/db.py`, `p5_insider/alerts.py` — supporting analytics, persistence, and Telegram alerts

Surfaced in the Streamlit UI under the **Insider** and **Momentum** pages.
