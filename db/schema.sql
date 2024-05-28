-- =============================================================
-- AI-ML Data Science Simulation Project
-- Database Schema
-- Author: Deshraj Jogiya
-- Date: 2024-04-01
-- =============================================================

-- ---------------------------------------------------------------
-- Table: branch_sales
-- Stores daily sales transactions from all 5 branch stores
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS branch_sales (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id       TEXT NOT NULL,          -- e.g. 'BR001'
    city            TEXT NOT NULL,          -- e.g. 'Chicago'
    product_id      TEXT NOT NULL,          -- e.g. 'PROD_042'
    category        TEXT NOT NULL,          -- Electronics, Clothing, etc.
    units_sold      INTEGER NOT NULL,
    revenue         REAL NOT NULL,
    stock_remaining INTEGER NOT NULL,
    sale_date       TEXT NOT NULL,          -- ISO format: YYYY-MM-DD
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Index for fast date-range queries
CREATE INDEX IF NOT EXISTS idx_branch_sales_date
    ON branch_sales(sale_date);

CREATE INDEX IF NOT EXISTS idx_branch_sales_branch
    ON branch_sales(branch_id);

CREATE INDEX IF NOT EXISTS idx_branch_sales_category
    ON branch_sales(category);

-- ---------------------------------------------------------------
-- Table: inventory_summary
-- Centralized snapshot of current stock levels per branch/product
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inventory_summary (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id           TEXT NOT NULL,
    city                TEXT NOT NULL,
    product_id          TEXT NOT NULL,
    category            TEXT NOT NULL,
    total_units_sold    INTEGER NOT NULL,
    total_revenue       REAL NOT NULL,
    avg_daily_units     REAL NOT NULL,
    current_stock       INTEGER NOT NULL,
    stock_out_risk      TEXT NOT NULL,      -- 'HIGH', 'MEDIUM', 'LOW'
    last_updated        TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_inventory_branch
    ON inventory_summary(branch_id);

CREATE INDEX IF NOT EXISTS idx_inventory_risk
    ON inventory_summary(stock_out_risk);

-- ---------------------------------------------------------------
-- Table: weekly_aggregates
-- Pre-computed weekly revenue and units by branch
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weekly_aggregates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start      TEXT NOT NULL,          -- Monday of the week
    week_end        TEXT NOT NULL,          -- Sunday of the week
    week_number     INTEGER NOT NULL,
    year            INTEGER NOT NULL,
    branch_id       TEXT NOT NULL,
    city            TEXT NOT NULL,
    category        TEXT,                   -- NULL = all categories combined
    total_units     INTEGER NOT NULL,
    total_revenue   REAL NOT NULL,
    avg_stock       REAL NOT NULL,
    transaction_count INTEGER NOT NULL,
    computed_at     TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_weekly_branch
    ON weekly_aggregates(branch_id, year, week_number);

-- ---------------------------------------------------------------
-- Table: monthly_aggregates
-- Pre-computed monthly summaries per branch
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS monthly_aggregates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,       -- 1-12
    month_name      TEXT NOT NULL,
    branch_id       TEXT NOT NULL,
    city            TEXT NOT NULL,
    category        TEXT,
    total_units     INTEGER NOT NULL,
    total_revenue   REAL NOT NULL,
    avg_daily_units REAL NOT NULL,
    avg_stock       REAL NOT NULL,
    peak_day        TEXT,                   -- Date with highest revenue
    peak_revenue    REAL,
    computed_at     TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_monthly_branch
    ON monthly_aggregates(branch_id, year, month);

-- ---------------------------------------------------------------
-- View: sales_summary
-- Unified view joining branch_sales with aggregations
-- ---------------------------------------------------------------
CREATE VIEW IF NOT EXISTS sales_summary AS
SELECT
    bs.branch_id,
    bs.city,
    bs.product_id,
    bs.category,
    bs.sale_date,
    bs.units_sold,
    bs.revenue,
    bs.stock_remaining,
    strftime('%W', bs.sale_date) AS week_number,
    strftime('%m', bs.sale_date) AS month_number,
    strftime('%Y', bs.sale_date) AS year
FROM branch_sales bs;

-- ---------------------------------------------------------------
-- View: stock_alert_view
-- Products at high stock-out risk (< 10 units remaining)
-- ---------------------------------------------------------------
CREATE VIEW IF NOT EXISTS stock_alert_view AS
SELECT
    branch_id,
    city,
    product_id,
    category,
    stock_remaining,
    sale_date,
    CASE
        WHEN stock_remaining < 10  THEN 'CRITICAL'
        WHEN stock_remaining < 25  THEN 'HIGH'
        WHEN stock_remaining < 50  THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_level
FROM branch_sales
WHERE sale_date = (
    SELECT MAX(sale_date) FROM branch_sales
);
