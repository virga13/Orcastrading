# Orcastrading

A modular trading intelligence system built in sequential phases, each feeding into the next.

---

## Architecture Overview

```
P1 → P2 → P3 → P4 → P5 → P6
```

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

### P3 — Historical Strategy Backtester `[PLANNED]`
**Historical EV per setup, win rate, profit factor, drawdown, strategy comparison**

Validates setups generated in P1/P2 against historical data. Outputs expected value per setup, win rate, profit factor, drawdown metrics, and side-by-side strategy comparisons.

---

### P4 — Trader Personality Profiler `[PLANNED]`
**Behavioral modeling → risk tolerance mapping → strategy templates per archetype**

Models trader behavior to map individual risk tolerance and psychological tendencies to specific trader archetypes, then surfaces strategy templates suited to each archetype.

---

### P5 — Lock-in Trade Journal `[PLANNED]`
**Trade entry locked once opened — discipline enforcement, statistical data integrity**

A write-once trade journal where entries are locked upon creation. Enforces trading discipline and ensures statistical integrity of the performance dataset.

---

### P6 — Parallel Trading Profiles `[PLANNED]`
**Shadow strategies running in parallel — alternative exits, regime adaptations, A/B testing**

Runs shadow strategies alongside live trades to test alternative exits, adapt to changing market regimes, and A/B test strategy variants without affecting the live position.
