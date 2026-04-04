# Orcastrading — Project Documentation

## Overview

Orcastrading is an algorithmic trading research platform with four modules:

| Module | Purpose |
|--------|---------|
| **P1** | Daily market analysis — Claude AI generates bias + setup analysis per asset |
| **P3** | Backtester — walk-forward strategy validation over historical data |
| **P4** | Live forward-test — scan for real setups daily, track them in a journal |
| **core** | Shared infrastructure — asset config, data providers, strategy registry |

---

## Project Structure

```
Orcastrading/
├── config/
│   ├── assets.yaml          # Central asset registry — add/remove assets here
│   └── strategies.yaml      # Strategy default parameters — tune params here
│
├── core/
│   ├── config.py            # Loads YAML config, provides get_asset(), get_ticker() etc.
│   └── data.py              # Data provider factory (yfinance / EODHD / CCXT)
│
├── p3_backtester/
│   ├── strategies/
│   │   ├── base.py               # StrategyBase ABC
│   │   ├── registry.py           # build_from_config("orb") factory
│   │   ├── mtf_trend.py          # Multi-Timeframe Trend (validated, locked)
│   │   ├── orb.py                # Opening Range Breakout (fully implemented)
│   │   ├── momentum_breakout.py  # Daily momentum breakout (fully implemented)
│   │   ├── mean_reversion.py     # Intraday mean reversion (stub — needs EODHD)
│   │   └── continuation.py       # Intraday continuation (stub — needs EODHD)
│   ├── market_data.py       # Multi-source OHLCV fetcher (stooq / yfinance / CCXT)
│   └── walk_forward.py      # Walk-forward validation pipeline
│
├── p4_live/
│   ├── scanner.py           # scan_strategy(asset_id, strategy_id) — core scan function
│   ├── journal.py           # SQLite journal — records every signal, fill, close
│   ├── alerts.py            # Telegram + Email alert system
│   ├── report.py            # CLI performance report vs backtest baseline
│   ├── html_report.py       # Self-contained HTML report generator
│   ├── backfill.py          # Historical simulation tool (research only, not live data)
│   └── cli.py               # python -m p4_live <command>
│
├── p1_analysis_engine/      # Claude AI market analysis (P1 module)
│
├── run_scheduler.py         # Continuous scheduler — runs 24/7, scans every 15 min
├── run_daily.bat            # Windows Task Scheduler alternative (runs once at 22:00)
├── .env                     # API keys and config (never commit this)
└── .env.example             # Template for .env
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure .env
Copy `.env.example` to `.env` and fill in your keys:
```
ANTHROPIC_API_KEY=...
TELEGRAM_BOT_TOKEN=...      # get from @BotFather on Telegram
TELEGRAM_CHAT_ID=...        # message your bot, then check /getUpdates
EODHD_API_KEY=...           # eodhd.com — needed for intraday history
DATA_PROVIDER=yfinance      # switch to "eodhd" after upgrading plan
```

### 3. Test alerts
```bash
python -m p4_live alert-test
```

### 4. Start the scheduler
```bash
python run_scheduler.py
```

---

## Running the System

### Option A — Continuous scheduler (recommended)
```bash
python run_scheduler.py
```
- Runs until you stop it (Ctrl+C)
- Scans ORB every 15 min during 16:30–18:30 local time (US market open)
- Runs full daily scan at 22:00
- Sends Telegram startup alert so you know it's running
- Logs everything to `logs/scheduler.log`

To start automatically at Windows boot:
```powershell
schtasks /create /tn "Orcastrading" ^
  /tr "\"C:\Users\admin\AppData\Local\Python\pythoncore-3.14-64\python.exe\" \"C:\Users\admin\Desktop\Orcastrading\run_scheduler.py\"" ^
  /sc onstart /ru SYSTEM /f
```

### Option B — Task Scheduler (daily only)
```powershell
schtasks /create /tn "Orcastrading Daily" ^
  /tr "\"C:\Users\admin\Desktop\Orcastrading\run_daily.bat\"" ^
  /sc weekly /d MON,TUE,WED,THU,FRI /st 22:00 /f
```

---

## CLI Commands

All commands: `python -m p4_live <command>`

| Command | Description |
|---------|-------------|
| `scan` | Scan all assets/strategies for today's signals |
| `scan --ticker ^GSPC` | Scan one asset only |
| `scan --strategy orb` | Scan one strategy only |
| `check` | Check open trades vs latest prices, flag SL/TP hits |
| `daily --html` | Full end-of-day: scan + check + report + HTML |
| `fill DATE PRICE --ticker GC=F` | Record a trade fill |
| `close DATE win EXIT --ticker GC=F` | Record a trade close |
| `expire DATE --ticker GC=F` | Mark a pending signal as expired |
| `log` | Show all trades |
| `log --open` | Show only open/pending trades |
| `log --ticker ^GSPC` | Filter by asset |
| `report` | CLI performance report |
| `report --html` | Generate HTML report and open in browser |
| `alert-test` | Send test alert on all configured channels |
| `backfill 2025-01-01` | Historical simulation (research only) |

---

## Adding Assets

Edit `config/assets.yaml`:

```yaml
- id: nasdaq
  label: NASDAQ
  tickers:
    yfinance: "^NDX"
    eodhd: "NDX.INDX"
  category: index
  currency: USD
  session:
    open: "09:30"
    close: "16:00"
    timezone: "America/New_York"
  orb:
    opening_range_start: "09:30"
    opening_range_end: "10:00"
    trade_window_end: "11:00"
  strategies:
    - mtf_trend
    - orb
```

No Python changes needed. The scanner picks it up automatically.

---

## Adding Strategies

1. Create `p3_backtester/strategies/your_strategy.py` — extend `StrategyBase`
2. Add it to `registry.py` `_MAP` dict and `build_from_config`
3. Add default params to `config/strategies.yaml`
4. Add strategy ID to asset entries in `config/assets.yaml`

---

## Switching Data Providers

In `.env`:
```
DATA_PROVIDER=eodhd    # full intraday history (requires paid plan — $29.99/mo)
DATA_PROVIDER=yfinance # free, 60-day intraday limit
```

When using EODHD, update tickers in `config/assets.yaml` to EODHD format:
- Stocks: `AAPL.US`
- US Indices (via ETF): `SPY.US`, `DIA.US`, `QQQ.US`
- Gold/Silver (via ETF): `GLD.US`, `SLV.US`
- Crypto: `BTC-USD.CC`
- GER40: `GDAXI.INDX` (verify symbol with EODHD support)

---

## Alert Channels

Configure in `.env`. Both channels are optional and independent.

**Telegram** (recommended):
1. Message @BotFather → `/newbot` → copy the token
2. Message your new bot once, then visit:
   `https://api.telegram.org/bot8774721747:AAGHbC12Qwb7HZoP-ynnxoFyG2CZ6QZJ0hQ/getUpdates` → copy `chat.id`
    Docs / https://core.telegram.org/bots/api
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

**Email** (Gmail):
```
ALERT_EMAIL_FROM=you@gmail.com
ALERT_EMAIL_TO=you@gmail.com
ALERT_EMAIL_PASSWORD=your_app_password   # Gmail App Password, not your main password
```

Alert types:
- `signal` — new setup detected
- `entry` — pending trade's entry zone touched
- `sl` — stop loss breached
- `tp` — TP1 or TP2 hit
- `startup` — scheduler started (so you know it's running)
- `daily_summary` — end-of-day scan results even when no signals fire

---

## Strategy Status

| Strategy | Timeframe | Status | Backtest baseline |
|----------|-----------|--------|-------------------|
| MTF Trend | 1D | Production | SPX500 WR 42.9% PF 2.43 / US30 WR 51.3% PF 2.05 |
| ORB | 15m | Production | Needs EODHD for proper backtest (yfinance = 60 days) |
| Momentum Breakout | 1D | Production | Not yet backtested — validate before trading |
| Mean Reversion | 15m | Stub | Not yet implemented |
| Continuation | 15m | Stub | Not yet implemented |

---

## Logs

| File | Contents |
|------|---------|
| `logs/scheduler.log` | Every scan, signal, alert — timestamped |
| `logs/daily.log` | Output from run_daily.bat (Task Scheduler runs) |

---

## How to Know the System is Running

1. **Startup Telegram alert** — when `run_scheduler.py` starts, it sends "Scheduler started" with the list of assets and strategies being monitored
2. **Daily summary alert** — every day at 22:00, sends scan results even if no signals fired
3. **`logs/scheduler.log`** — shows every scan with timestamp; if the last entry is recent, the system is running
4. **`python -m p4_live log`** — shows the live trade journal; signals recorded here are real
