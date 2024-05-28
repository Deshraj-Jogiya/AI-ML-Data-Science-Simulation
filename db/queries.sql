-- =============================================================
-- AI-ML Data Science Simulation Project
-- Analytical SQL Queries
-- Author: Deshraj Jogiya
-- Database: SQLite (data/sales.db)
-- =============================================================

-- ---------------------------------------------------------------
-- QUERY 1: Weekly Revenue by Branch
-- Shows total revenue per branch, per week, ordered chronologically
-- ---------------------------------------------------------------
SELECT
    strftime('%Y', sale_date)            AS year,
    strftime('%W', sale_date)            AS week_number,
    branch_id,
    city,
    COUNT(*)                             AS transactions,
    SUM(units_sold)                      AS total_units,
    ROUND(SUM(revenue), 2)               AS total_revenue,
    ROUND(AVG(revenue), 2)               AS avg_daily_revenue
FROM branch_sales
GROUP BY
    strftime('%Y', sale_date),
    strftime('%W', sale_date),
    branch_id
ORDER BY
    year ASC,
    week_number ASC,
    total_revenue DESC;


-- ---------------------------------------------------------------
-- QUERY 2: Top 10 Products by Units Sold (All Branches Combined)
-- Identifies best-selling products across the entire chain
-- ---------------------------------------------------------------
SELECT
    product_id,
    category,
    SUM(units_sold)          AS total_units_sold,
    ROUND(SUM(revenue), 2)   AS total_revenue,
    COUNT(DISTINCT branch_id) AS branches_sold_in,
    ROUND(AVG(units_sold), 2) AS avg_daily_units
FROM branch_sales
GROUP BY product_id, category
ORDER BY total_units_sold DESC
LIMIT 10;


-- ---------------------------------------------------------------
-- QUERY 3: Stock-Out Risk Detection
-- Flags products with fewer than 25 units remaining at any branch
-- ---------------------------------------------------------------
SELECT
    branch_id,
    city,
    product_id,
    category,
    stock_remaining,
    sale_date,
    CASE
        WHEN stock_remaining < 10  THEN '🔴 CRITICAL'
        WHEN stock_remaining < 25  THEN '🟠 HIGH'
        WHEN stock_remaining < 50  THEN '🟡 MEDIUM'
        ELSE '🟢 LOW'
    END AS risk_level,
    ROUND(stock_remaining * 1.0 / NULLIF(units_sold, 0), 1) AS days_of_stock_left
FROM branch_sales
WHERE stock_remaining < 25
ORDER BY stock_remaining ASC
LIMIT 50;


-- ---------------------------------------------------------------
-- QUERY 4: Monthly Trend Comparison Across Branches
-- Shows revenue growth month-over-month per branch
-- ---------------------------------------------------------------
WITH monthly_data AS (
    SELECT
        branch_id,
        city,
        strftime('%Y', sale_date)  AS year,
        strftime('%m', sale_date)  AS month,
        SUM(units_sold)            AS total_units,
        ROUND(SUM(revenue), 2)     AS total_revenue
    FROM branch_sales
    GROUP BY branch_id, year, month
),
lagged AS (
    SELECT
        *,
        LAG(total_revenue) OVER (
            PARTITION BY branch_id ORDER BY year, month
        ) AS prev_month_revenue
    FROM monthly_data
)
SELECT
    branch_id,
    city,
    year,
    month,
    total_units,
    total_revenue,
    prev_month_revenue,
    CASE
        WHEN prev_month_revenue IS NULL THEN NULL
        ELSE ROUND(
            (total_revenue - prev_month_revenue) / prev_month_revenue * 100, 2
        )
    END AS pct_change_vs_prev_month
FROM lagged
ORDER BY branch_id, year, month;


-- ---------------------------------------------------------------
-- QUERY 5: Category Revenue Breakdown per Branch
-- Pie-chart-ready data showing which category drives the most revenue
-- ---------------------------------------------------------------
SELECT
    branch_id,
    city,
    category,
    SUM(units_sold)                              AS total_units,
    ROUND(SUM(revenue), 2)                       AS total_revenue,
    ROUND(
        SUM(revenue) * 100.0 / SUM(SUM(revenue)) OVER (PARTITION BY branch_id),
        2
    )                                            AS revenue_pct_of_branch
FROM branch_sales
GROUP BY branch_id, city, category
ORDER BY branch_id, total_revenue DESC;


-- ---------------------------------------------------------------
-- QUERY 6: Top Performing Branch Leaderboard (All-Time)
-- Ranks branches by total revenue, units sold, and avg order value
-- ---------------------------------------------------------------
SELECT
    branch_id,
    city,
    COUNT(*)                             AS total_transactions,
    SUM(units_sold)                      AS total_units_sold,
    ROUND(SUM(revenue), 2)               AS total_revenue,
    ROUND(AVG(revenue), 2)               AS avg_transaction_revenue,
    ROUND(AVG(units_sold), 2)            AS avg_units_per_transaction,
    ROUND(AVG(stock_remaining), 1)       AS avg_stock_remaining,
    MIN(sale_date)                       AS first_sale_date,
    MAX(sale_date)                       AS last_sale_date,
    RANK() OVER (ORDER BY SUM(revenue) DESC) AS revenue_rank
FROM branch_sales
GROUP BY branch_id, city
ORDER BY total_revenue DESC;


-- ---------------------------------------------------------------
-- QUERY 7: Daily Sales Velocity (7-Day Rolling Average)
-- Detects acceleration or deceleration in sales per branch
-- ---------------------------------------------------------------
SELECT
    branch_id,
    city,
    sale_date,
    SUM(units_sold)  AS daily_units,
    ROUND(SUM(revenue), 2) AS daily_revenue,
    ROUND(
        AVG(SUM(units_sold)) OVER (
            PARTITION BY branch_id
            ORDER BY sale_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2
    ) AS rolling_7day_avg_units,
    ROUND(
        AVG(SUM(revenue)) OVER (
            PARTITION BY branch_id
            ORDER BY sale_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2
    ) AS rolling_7day_avg_revenue
FROM branch_sales
GROUP BY branch_id, city, sale_date
ORDER BY branch_id, sale_date;
