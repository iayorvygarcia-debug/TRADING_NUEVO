-- ===========================================================
-- AI QUANT TRADING PLATFORM — Database Initialization
-- ===========================================================
-- This script runs on first database creation (docker-compose).
-- It enables TimescaleDB and creates the full schema.
-- ===========================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================
-- CORE TABLES
-- ============================================

CREATE TABLE IF NOT EXISTS instruments (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) UNIQUE NOT NULL,
    name            VARCHAR(255),
    exchange        VARCHAR(20),
    asset_type      VARCHAR(20),
    sector          VARCHAR(100),
    industry        VARCHAR(100),
    market_cap      BIGINT,
    is_active       BOOLEAN DEFAULT true,
    is_tradeable    BOOLEAN DEFAULT true,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_instruments_symbol ON instruments(symbol);
CREATE INDEX IF NOT EXISTS idx_instruments_sector ON instruments(sector);
CREATE INDEX IF NOT EXISTS idx_instruments_active ON instruments(is_active) WHERE is_active = true;

-- ============================================
-- MARKET DATA (TimescaleDB Hypertables)
-- ============================================

CREATE TABLE IF NOT EXISTS bars (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    open            NUMERIC(12,4) NOT NULL,
    high            NUMERIC(12,4) NOT NULL,
    low             NUMERIC(12,4) NOT NULL,
    close           NUMERIC(12,4) NOT NULL,
    volume          BIGINT NOT NULL,
    vwap            NUMERIC(12,4),
    trade_count     INT,
    source          VARCHAR(50),
    quality_status  VARCHAR(20) DEFAULT 'valid'
);

SELECT create_hypertable('bars', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_bars_symbol_tf ON bars (symbol, timeframe, time DESC);

CREATE TABLE IF NOT EXISTS quotes (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    bid             NUMERIC(12,4),
    ask             NUMERIC(12,4),
    bid_size        INT,
    ask_size        INT,
    spread          NUMERIC(8,4),
    source          VARCHAR(50)
);

SELECT create_hypertable('quotes', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_quotes_symbol ON quotes (symbol, time DESC);

CREATE TABLE IF NOT EXISTS trades_raw (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    price           NUMERIC(12,4) NOT NULL,
    size            INT NOT NULL,
    exchange        VARCHAR(10),
    conditions      VARCHAR(50),
    source          VARCHAR(50)
);

SELECT create_hypertable('trades_raw', 'time', if_not_exists => TRUE);

-- ============================================
-- NEWS & EVENTS
-- ============================================

CREATE TABLE IF NOT EXISTS news_articles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id     VARCHAR(255),
    headline        TEXT NOT NULL,
    summary         TEXT,
    content         TEXT,
    source          VARCHAR(100) NOT NULL,
    source_trust    NUMERIC(3,2),
    url             TEXT,
    published_at    TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    is_duplicate    BOOLEAN DEFAULT false,
    dedup_hash      VARCHAR(64)
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_source ON news_articles(source);

CREATE TABLE IF NOT EXISTS news_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id      UUID REFERENCES news_articles(id),
    symbol          VARCHAR(20),
    event_type      VARCHAR(50) NOT NULL,
    sentiment       NUMERIC(4,3),
    importance      NUMERIC(3,2),
    novelty         NUMERIC(3,2),
    confidence      NUMERIC(3,2),
    expected_impact VARCHAR(20),
    detected_entities JSONB,
    model_version   VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_events_symbol ON news_events(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_events_type ON news_events(event_type);

-- ============================================
-- SEC / FUNDAMENTALS
-- ============================================

CREATE TABLE IF NOT EXISTS sec_filings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cik             VARCHAR(20) NOT NULL,
    symbol          VARCHAR(20),
    form_type       VARCHAR(20) NOT NULL,
    filing_date     DATE NOT NULL,
    accepted_date   TIMESTAMPTZ,
    accession_number VARCHAR(50) UNIQUE,
    primary_doc_url TEXT,
    data_extracted  JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sec_filings_symbol ON sec_filings(symbol, filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_sec_filings_form ON sec_filings(form_type);

CREATE TABLE IF NOT EXISTS fundamentals (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,
    period          VARCHAR(10),
    period_end      DATE NOT NULL,
    revenue         NUMERIC(18,2),
    net_income      NUMERIC(18,2),
    eps             NUMERIC(8,4),
    eps_diluted     NUMERIC(8,4),
    total_assets    NUMERIC(18,2),
    total_liabilities NUMERIC(18,2),
    cash            NUMERIC(18,2),
    total_debt      NUMERIC(18,2),
    operating_income NUMERIC(18,2),
    free_cash_flow  NUMERIC(18,2),
    shares_outstanding BIGINT,
    source          VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, period, period_end)
);

-- ============================================
-- MACRO DATA
-- ============================================

CREATE TABLE IF NOT EXISTS macro_data (
    time            TIMESTAMPTZ NOT NULL,
    series_id       VARCHAR(50) NOT NULL,
    value           NUMERIC(18,4) NOT NULL,
    source          VARCHAR(50) DEFAULT 'FRED',
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('macro_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_macro_series ON macro_data(series_id, time DESC);

CREATE TABLE IF NOT EXISTS macro_regime (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    regime          VARCHAR(50) NOT NULL,
    confidence      NUMERIC(3,2),
    components      JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- FEATURES
-- ============================================

CREATE TABLE IF NOT EXISTS features (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    feature_set     VARCHAR(50) NOT NULL,
    features        JSONB NOT NULL,
    version         VARCHAR(20),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('features', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_features_symbol_tf ON features(symbol, timeframe, time DESC);

-- ============================================
-- SIGNALS & STRATEGIES
-- ============================================

CREATE TABLE IF NOT EXISTS signals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          VARCHAR(20) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    direction       VARCHAR(10) NOT NULL,
    strategy        VARCHAR(50) NOT NULL,
    score           NUMERIC(4,3),
    confidence      NUMERIC(4,3),
    expected_return NUMERIC(6,4),
    risk_estimate   NUMERIC(6,4),
    timeframe       VARCHAR(10),
    reasons         JSONB,
    market_regime   VARCHAR(50),
    status          VARCHAR(20) DEFAULT 'generated',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);

CREATE TABLE IF NOT EXISTS opportunity_scores (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          VARCHAR(20) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    composite_score NUMERIC(4,3),
    technical_score NUMERIC(4,3),
    momentum_score  NUMERIC(4,3),
    volume_score    NUMERIC(4,3),
    news_score      NUMERIC(4,3),
    fundamental_score NUMERIC(4,3),
    macro_score     NUMERIC(4,3),
    regime_score    NUMERIC(4,3),
    ml_score        NUMERIC(4,3),
    weights         JSONB,
    rank            INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- ML MODELS
-- ============================================

CREATE TABLE IF NOT EXISTS model_registry (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name      VARCHAR(100) NOT NULL,
    version         VARCHAR(20) NOT NULL,
    model_type      VARCHAR(50),
    training_date   TIMESTAMPTZ,
    dataset_version VARCHAR(50),
    features_version VARCHAR(50),
    hyperparameters JSONB,
    metrics         JSONB,
    regime          VARCHAR(50),
    status          VARCHAR(20) DEFAULT 'training',
    artifact_path   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_name, version)
);

CREATE TABLE IF NOT EXISTS predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        UUID REFERENCES model_registry(id),
    symbol          VARCHAR(20) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    prediction      JSONB NOT NULL,
    features_snapshot JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- ORDERS & EXECUTION
-- ============================================

CREATE TABLE IF NOT EXISTS orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    broker_order_id VARCHAR(100),
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    order_type      VARCHAR(20) NOT NULL,
    order_class     VARCHAR(20),
    quantity        NUMERIC(12,4) NOT NULL,
    limit_price     NUMERIC(12,4),
    stop_price      NUMERIC(12,4),
    time_in_force   VARCHAR(10) DEFAULT 'day',
    status          VARCHAR(30) NOT NULL DEFAULT 'created',
    signal_id       UUID REFERENCES signals(id),
    strategy        VARCHAR(50),
    filled_qty      NUMERIC(12,4) DEFAULT 0,
    filled_avg_price NUMERIC(12,4),
    expected_slippage NUMERIC(8,4),
    actual_slippage NUMERIC(8,4),
    commission      NUMERIC(8,4),
    fees            NUMERIC(8,4),
    submitted_at    TIMESTAMPTZ,
    filled_at       TIMESTAMPTZ,
    cancelled_at    TIMESTAMPTZ,
    trading_mode    VARCHAR(10) NOT NULL DEFAULT 'paper',
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_broker_id ON orders(broker_order_id);

CREATE TABLE IF NOT EXISTS order_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id        UUID REFERENCES orders(id),
    event_type      VARCHAR(30) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    details         JSONB,
    broker_response JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_order_events_order ON order_events(order_id, timestamp DESC);

-- ============================================
-- POSITIONS & PORTFOLIO
-- ============================================

CREATE TABLE IF NOT EXISTS positions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    quantity        NUMERIC(12,4) NOT NULL,
    avg_entry_price NUMERIC(12,4) NOT NULL,
    current_price   NUMERIC(12,4),
    market_value    NUMERIC(14,2),
    unrealized_pnl  NUMERIC(14,2),
    realized_pnl    NUMERIC(14,2) DEFAULT 0,
    stop_loss       NUMERIC(12,4),
    take_profit     NUMERIC(12,4),
    strategy        VARCHAR(50),
    entry_signal_id UUID,
    trading_mode    VARCHAR(10) NOT NULL DEFAULT 'paper',
    opened_at       TIMESTAMPTZ DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    status          VARCHAR(20) DEFAULT 'open',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    total_equity    NUMERIC(14,2),
    cash            NUMERIC(14,2),
    market_value    NUMERIC(14,2),
    unrealized_pnl  NUMERIC(14,2),
    realized_pnl    NUMERIC(14,2),
    daily_pnl       NUMERIC(14,2),
    drawdown        NUMERIC(6,4),
    max_drawdown    NUMERIC(6,4),
    num_positions   INT,
    exposure        NUMERIC(6,4),
    sector_exposure JSONB,
    trading_mode    VARCHAR(10)
);

SELECT create_hypertable('portfolio_snapshots', 'time', if_not_exists => TRUE);

-- ============================================
-- RISK & SYSTEM
-- ============================================

CREATE TABLE IF NOT EXISTS risk_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ NOT NULL,
    event_type      VARCHAR(50) NOT NULL,
    level           VARCHAR(20),
    details         JSONB NOT NULL,
    action_taken    VARCHAR(100),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_events_time ON risk_events(timestamp DESC);

CREATE TABLE IF NOT EXISTS system_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ NOT NULL,
    service         VARCHAR(50) NOT NULL,
    event_type      VARCHAR(50) NOT NULL,
    severity        VARCHAR(20),
    message         TEXT,
    details         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_events_time ON system_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_events_service ON system_events(service);

CREATE TABLE IF NOT EXISTS audit_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ NOT NULL,
    action          VARCHAR(50) NOT NULL,
    symbol          VARCHAR(20),
    strategy        VARCHAR(50),
    signal_data     JSONB,
    market_data     JSONB,
    technical_data  JSONB,
    news_data       JSONB,
    fundamental_data JSONB,
    macro_data      JSONB,
    ml_data         JSONB,
    risk_data       JSONB,
    order_data      JSONB,
    result          VARCHAR(50),
    model_version   VARCHAR(50),
    feature_version VARCHAR(50),
    trading_mode    VARCHAR(10),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_symbol ON audit_logs(symbol);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);

-- ============================================
-- BACKTESTING
-- ============================================

CREATE TABLE IF NOT EXISTS backtest_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255),
    strategy        VARCHAR(50),
    config          JSONB NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    initial_capital NUMERIC(14,2),
    final_equity    NUMERIC(14,2),
    metrics         JSONB,
    status          VARCHAR(20),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id     UUID REFERENCES backtest_runs(id),
    symbol          VARCHAR(20),
    side            VARCHAR(10),
    entry_time      TIMESTAMPTZ,
    exit_time       TIMESTAMPTZ,
    entry_price     NUMERIC(12,4),
    exit_price      NUMERIC(12,4),
    quantity        NUMERIC(12,4),
    pnl             NUMERIC(14,2),
    pnl_pct         NUMERIC(8,4),
    commission      NUMERIC(8,4),
    slippage        NUMERIC(8,4),
    strategy        VARCHAR(50),
    signal_data     JSONB
);

-- ============================================
-- CONFIGURATION
-- ============================================

CREATE TABLE IF NOT EXISTS system_config (
    key             VARCHAR(100) PRIMARY KEY,
    value           JSONB NOT NULL,
    description     TEXT,
    category        VARCHAR(50),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_by      VARCHAR(100)
);

-- Default risk configuration
INSERT INTO system_config (key, value, category, description) VALUES
    ('trading_mode', '"paper"', 'system', 'Current trading mode: paper or live'),
    ('risk_per_trade', '0.01', 'risk', 'Max risk per trade as fraction of equity'),
    ('max_daily_loss', '0.03', 'risk', 'Max daily loss as fraction of equity'),
    ('max_weekly_loss', '0.05', 'risk', 'Max weekly loss as fraction of equity'),
    ('max_drawdown', '0.10', 'risk', 'Max portfolio drawdown'),
    ('max_position_size', '0.05', 'risk', 'Max single position as fraction of equity'),
    ('max_sector_exposure', '0.25', 'risk', 'Max sector exposure as fraction of equity'),
    ('max_trades_per_day', '20', 'risk', 'Max trades per day'),
    ('max_orders_per_minute', '5', 'risk', 'Max orders per minute'),
    ('min_signal_score', '0.65', 'strategy', 'Min composite signal score to consider'),
    ('min_probability', '0.55', 'strategy', 'Min ML probability to consider'),
    ('max_spread_pct', '0.005', 'execution', 'Max spread as fraction of price'),
    ('strategies_enabled', '["trend","momentum","breakout"]', 'strategy', 'Enabled strategies')
ON CONFLICT (key) DO NOTHING;

-- ============================================
-- TimescaleDB Retention Policies
-- ============================================
-- Keep raw ticks for 30 days, bars for 5 years, snapshots forever

-- Note: Uncomment when data volume justifies it
-- SELECT add_retention_policy('trades_raw', INTERVAL '30 days');
-- SELECT add_retention_policy('quotes', INTERVAL '90 days');

-- ============================================
-- Continuous Aggregates (example: 1m bars from trades)
-- ============================================
-- Note: Implement when ingest pipeline is ready
-- These auto-refresh materialized views for downsampling

-- ============================================
-- COMPLETE ✅
-- ============================================
-- Schema version: 1.0.0
-- TimescaleDB hypertables: bars, quotes, trades_raw, macro_data,
--                          features, portfolio_snapshots
-- Total tables: 25
