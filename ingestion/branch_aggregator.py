"""
branch_aggregator.py
---------------------
AI-ML Data Science Simulation Project
Reads from the unified branch_sales table, merges all branches,
computes weekly/monthly aggregations, identifies top performers,
and populates the inventory_summary, weekly_aggregates, and
monthly_aggregates tables.

Author: Deshraj Jogiya
Date: April 2024
"""

import sys
import sqlite3
import logging
from pathlib import Path
from calendar import month_name

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "sales.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def ensure_aggregation_tables(conn: sqlite3.Connection):
    """Create aggregation tables if they don't exist."""
    conn.executescript("""
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
            stock_out_risk      TEXT NOT NULL,
            last_updated        TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS weekly_aggregates (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start        TEXT NOT NULL,
            week_end          TEXT NOT NULL,
            week_number       INTEGER NOT NULL,
            year              INTEGER NOT NULL,
            branch_id         TEXT NOT NULL,
            city              TEXT NOT NULL,
            category          TEXT,
            total_units       INTEGER NOT NULL,
            total_revenue     REAL NOT NULL,
            avg_stock         REAL NOT NULL,
            transaction_count INTEGER NOT NULL,
            computed_at       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS monthly_aggregates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            year            INTEGER NOT NULL,
            month           INTEGER NOT NULL,
            month_name      TEXT NOT NULL,
            branch_id       TEXT NOT NULL,
            city            TEXT NOT NULL,
            category        TEXT,
            total_units     INTEGER NOT NULL,
            total_revenue   REAL NOT NULL,
            avg_daily_units REAL NOT NULL,
            avg_stock       REAL NOT NULL,
            peak_day        TEXT,
            peak_revenue    REAL,
            computed_at     TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sales_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all branch sales into a single DataFrame."""
    df = pd.read_sql_query(
        "SELECT * FROM branch_sales ORDER BY sale_date ASC",
        conn,
        parse_dates=["sale_date"]
    )
    log.info(f"Loaded {len(df):,} records from branch_sales")
    return df


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def compute_inventory_summary(df: pd.DataFrame, conn: sqlite3.Connection):
    """
    Compute per-branch, per-product inventory summary.
    Flags stock-out risk based on average daily velocity.
    """
    log.info("Computing inventory_summary ...")

    # Latest stock per branch+product
    latest_stock = (
        df.sort_values("sale_date")
        .groupby(["branch_id", "city", "product_id", "category"])
        .agg(current_stock=("stock_remaining", "last"))
        .reset_index()
    )

    # Aggregate metrics
    agg = (
        df.groupby(["branch_id", "city", "product_id", "category"])
        .agg(
            total_units_sold =("units_sold",      "sum"),
            total_revenue    =("revenue",          "sum"),
            num_days         =("sale_date",        "nunique"),
        )
        .reset_index()
    )
    agg["avg_daily_units"] = (agg["total_units_sold"] / agg["num_days"]).round(2)

    summary = agg.merge(latest_stock, on=["branch_id", "city", "product_id", "category"])

    # Stock-out risk classification
    def risk_label(row):
        days_left = row["current_stock"] / max(row["avg_daily_units"], 0.1)
        if days_left < 3:
            return "HIGH"
        elif days_left < 7:
            return "MEDIUM"
        else:
            return "LOW"

    summary["stock_out_risk"] = summary.apply(risk_label, axis=1)
    summary["total_revenue"]  = summary["total_revenue"].round(2)
    summary.drop(columns=["num_days"], inplace=True)

    # Persist
    conn.execute("DELETE FROM inventory_summary")
    summary.to_sql("inventory_summary", conn, if_exists="append", index=False)
    conn.commit()
    log.info(f"  → {len(summary):,} inventory records written")
    return summary


def compute_weekly_aggregates(df: pd.DataFrame, conn: sqlite3.Connection):
    """Compute weekly aggregations per branch × category."""
    log.info("Computing weekly_aggregates ...")

    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["year"]        = df["sale_date"].dt.year
    df["week_number"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["week_start"]  = df["sale_date"] - pd.to_timedelta(df["sale_date"].dt.dayofweek, unit="d")
    df["week_end"]    = df["week_start"] + pd.Timedelta(days=6)

    weekly = (
        df.groupby(["year", "week_number", "week_start", "week_end", "branch_id", "city", "category"])
        .agg(
            total_units      =("units_sold",      "sum"),
            total_revenue    =("revenue",          "sum"),
            avg_stock        =("stock_remaining", "mean"),
            transaction_count=("id",              "count"),
        )
        .reset_index()
    )
    weekly["week_start"]   = weekly["week_start"].dt.strftime("%Y-%m-%d")
    weekly["week_end"]     = weekly["week_end"].dt.strftime("%Y-%m-%d")
    weekly["total_revenue"] = weekly["total_revenue"].round(2)
    weekly["avg_stock"]    = weekly["avg_stock"].round(1)

    conn.execute("DELETE FROM weekly_aggregates")
    weekly.to_sql("weekly_aggregates", conn, if_exists="append", index=False)
    conn.commit()
    log.info(f"  → {len(weekly):,} weekly aggregate rows written")
    return weekly


def compute_monthly_aggregates(df: pd.DataFrame, conn: sqlite3.Connection):
    """Compute monthly aggregations with peak day detection."""
    log.info("Computing monthly_aggregates ...")

    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["year"]  = df["sale_date"].dt.year
    df["month"] = df["sale_date"].dt.month

    # Peak day per branch + month
    daily = (
        df.groupby(["branch_id", "year", "month", "sale_date"])
        .agg(daily_revenue=("revenue", "sum"))
        .reset_index()
    )
    peak_days = (
        daily.loc[daily.groupby(["branch_id", "year", "month"])["daily_revenue"].idxmax()]
        [["branch_id", "year", "month", "sale_date", "daily_revenue"]]
        .rename(columns={"sale_date": "peak_day", "daily_revenue": "peak_revenue"})
    )
    peak_days["peak_day"] = peak_days["peak_day"].dt.strftime("%Y-%m-%d")

    # Monthly metrics
    monthly = (
        df.groupby(["branch_id", "city", "year", "month", "category"])
        .agg(
            total_units     =("units_sold",      "sum"),
            total_revenue   =("revenue",          "sum"),
            num_days        =("sale_date",        "nunique"),
            avg_stock       =("stock_remaining", "mean"),
        )
        .reset_index()
    )
    monthly["avg_daily_units"] = (monthly["total_units"] / monthly["num_days"]).round(2)
    monthly["month_name"] = monthly["month"].apply(lambda m: month_name[m])
    monthly["total_revenue"] = monthly["total_revenue"].round(2)
    monthly["avg_stock"]     = monthly["avg_stock"].round(1)
    monthly.drop(columns=["num_days"], inplace=True)

    monthly = monthly.merge(peak_days, on=["branch_id", "year", "month"], how="left")

    conn.execute("DELETE FROM monthly_aggregates")
    monthly.to_sql("monthly_aggregates", conn, if_exists="append", index=False)
    conn.commit()
    log.info(f"  → {len(monthly):,} monthly aggregate rows written")
    return monthly


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_top_branches(weekly: pd.DataFrame):
    """Print a ranked leaderboard of top-performing branches."""
    log.info("")
    log.info("=" * 60)
    log.info("  TOP-PERFORMING BRANCHES (All-Time Weekly Aggregates)")
    log.info("=" * 60)

    ranked = (
        weekly.groupby(["branch_id", "city"])
        .agg(
            total_revenue=("total_revenue", "sum"),
            total_units  =("total_units",   "sum"),
            weeks        =("week_number",   "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.index + 1

    log.info(f"  {'Rank':<5} {'Branch':<8} {'City':<12} {'Revenue':>14} {'Units':>8} {'Weeks':>6}")
    log.info(f"  {'-'*5} {'-'*8} {'-'*12} {'-'*14} {'-'*8} {'-'*6}")
    for _, row in ranked.iterrows():
        log.info(
            f"  #{int(row['rank']):<4} {row.branch_id:<8} {row.city:<12} "
            f"${row.total_revenue:>13,.2f} {int(row.total_units):>8,} {int(row.weeks):>6}"
        )


def print_top_categories(df: pd.DataFrame):
    """Print category revenue ranking."""
    log.info("")
    log.info("  CATEGORY REVENUE RANKING")
    log.info("-" * 50)
    cat_rank = (
        df.groupby("category")
        .agg(revenue=("revenue", "sum"), units=("units_sold", "sum"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    for i, row in cat_rank.iterrows():
        bar_len = int(row["revenue"] / cat_rank["revenue"].max() * 30)
        bar = "█" * bar_len
        log.info(f"  {row.category:<18} {bar:<30}  ${row.revenue:,.2f}")


def print_stock_alert_summary(inventory: pd.DataFrame):
    """Print count of HIGH/MEDIUM/LOW risk items."""
    log.info("")
    log.info("  STOCK-OUT RISK SUMMARY")
    log.info("-" * 40)
    risk_counts = inventory["stock_out_risk"].value_counts()
    for risk_level in ["HIGH", "MEDIUM", "LOW"]:
        count = risk_counts.get(risk_level, 0)
        icon  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk_level, "")
        log.info(f"  {icon}  {risk_level:<8}: {count:>4} products")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_aggregation():
    if not DB_PATH.exists():
        log.error(f"Database not found: {DB_PATH}")
        log.error("Please run ingestion/daily_sales_pipeline.py first.")
        sys.exit(1)

    conn = get_connection()
    ensure_aggregation_tables(conn)

    log.info("=" * 60)
    log.info("  AI-ML Data Science Simulation – Branch Aggregator")
    log.info("=" * 60)

    df = load_sales_data(conn)

    inventory = compute_inventory_summary(df, conn)
    weekly    = compute_weekly_aggregates(df, conn)
    monthly   = compute_monthly_aggregates(df, conn)

    print_top_branches(weekly)
    print_top_categories(df)
    print_stock_alert_summary(inventory)

    log.info("")
    log.info("  ✅  AGGREGATION COMPLETE")
    log.info(f"  Database: {DB_PATH}")
    log.info("=" * 60)

    conn.close()
    return weekly, monthly, inventory


if __name__ == "__main__":
    run_aggregation()
