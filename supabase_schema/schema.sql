-- Orcastrading SaaS — Supabase schema
-- Run this in the Supabase SQL editor after creating a new project.
-- Auth is handled by Supabase Auth (email/password). Enable it in:
--   Dashboard → Authentication → Providers → Email

-- ── Per-user watchlist ────────────────────────────────────────────────────────
-- Stores which assets/strategies/timeframes each user has enabled.
-- The global catalog (available assets and strategies) lives in config/assets.yaml
-- and config/strategies.yaml — users pick from that catalog here.

CREATE TABLE IF NOT EXISTS user_watchlist (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id     UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    asset_id    TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    params      JSONB DEFAULT '{}',
    enabled     BOOLEAN DEFAULT true,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, asset_id, strategy_id, timeframe)
);

-- ── Per-user trade journal ────────────────────────────────────────────────────
-- Mirrors the SQLite trades table in p4_live/journal.py.
-- RLS ensures each user only sees/writes their own rows.

CREATE TABLE IF NOT EXISTS trades (
    id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id             UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    signal_date         TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    strategy            TEXT NOT NULL DEFAULT 'mtf_trend',
    timeframe           TEXT NOT NULL DEFAULT '1d',
    label               TEXT NOT NULL,
    direction           TEXT NOT NULL,
    entry_low           REAL NOT NULL,
    entry_high          REAL NOT NULL,
    stop_loss           REAL NOT NULL,
    tp1                 REAL NOT NULL,
    tp2                 REAL,
    tp1_alloc           INTEGER,
    tp2_alloc           INTEGER,
    risk_pts            REAL,
    rr                  REAL,
    confidence          REAL,
    rationale           TEXT,
    price_at_signal     REAL,
    atr_at_signal       REAL,
    rsi_at_signal       REAL,
    regime_note         TEXT,
    status              TEXT DEFAULT 'pending',
    fill_price          REAL,
    fill_date           TEXT,
    exit_price          REAL,
    exit_date           TEXT,
    pnl_r               REAL,
    notes               TEXT,
    source              TEXT DEFAULT 'scanner',
    signal_bar_ts       TEXT,
    fill_bar_ts         TEXT,
    chart_screenshot_path TEXT,
    mt5_ticket          TEXT,
    partial_close_pnl_r REAL,
    recorded_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, signal_date, ticker, strategy, timeframe)
);

-- ── Per-user manual journal entries (screenshot-based trading log) ────────────

CREATE TABLE IF NOT EXISTS manual_entries (
    id                   UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id              UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    entry_date           TEXT NOT NULL,
    asset                TEXT,
    direction            TEXT,
    entry_price          REAL,
    stop_loss            REAL,
    original_sl          REAL,
    take_profit          REAL,
    exit_price           REAL,
    pnl_r                REAL,
    pnl_dollars          REAL,
    quality_score        INTEGER,
    notes                TEXT,
    ai_analysis          TEXT,
    screenshot_path      TEXT,
    chart_screenshot_path TEXT,
    tags                 TEXT,
    recorded_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── Per-user psychology journal ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS psychology (
    id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id           UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    trade_ref_type    TEXT NOT NULL,
    trade_ref_id      TEXT NOT NULL,
    pre_state         TEXT,
    pre_confidence    INTEGER,
    pre_notes         TEXT,
    post_state        TEXT,
    post_notes        TEXT,
    execution_quality INTEGER,
    mistakes          TEXT,
    lesson            TEXT,
    recorded_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── Per-user settings ─────────────────────────────────────────────────────────
-- Telegram config, risk gates, account sizing — replaces .env per user.

CREATE TABLE IF NOT EXISTS user_settings (
    user_id              UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    telegram_bot_token   TEXT,
    telegram_chat_id     TEXT,
    max_open_trades      INTEGER DEFAULT 5,
    daily_loss_limit_r   REAL DEFAULT 0,
    weekly_dd_limit_r    REAL DEFAULT 0,
    account_balance      REAL DEFAULT 0,
    risk_per_trade_pct   REAL DEFAULT 1.0,
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ── Row Level Security ────────────────────────────────────────────────────────
-- Each user can only SELECT / INSERT / UPDATE / DELETE their own rows.
-- The Supabase client must be authenticated with the user's JWT for RLS to work.

ALTER TABLE user_watchlist  ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades          ENABLE ROW LEVEL SECURITY;
ALTER TABLE manual_entries  ENABLE ROW LEVEL SECURITY;
ALTER TABLE psychology      ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_settings   ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users_own_rows" ON user_watchlist  FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "users_own_rows" ON trades          FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "users_own_rows" ON manual_entries  FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "users_own_rows" ON psychology      FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "users_own_rows" ON user_settings   FOR ALL USING (auth.uid() = user_id);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_trades_user_status    ON trades(user_id, status);
CREATE INDEX IF NOT EXISTS idx_trades_user_date      ON trades(user_id, signal_date DESC);
CREATE INDEX IF NOT EXISTS idx_manual_user_date      ON manual_entries(user_id, entry_date DESC);
CREATE INDEX IF NOT EXISTS idx_psychology_user       ON psychology(user_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_user        ON user_watchlist(user_id, enabled);
