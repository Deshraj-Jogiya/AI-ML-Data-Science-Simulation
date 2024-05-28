"""
daily_sales_pipeline.py
-----------------------
AI-ML Data Science Simulation Project
ETL Pipeline: Generates synthetic daily sales data for 5 branch stores
and ingests into a centralized SQLite database.

Author: Deshraj Jogiya
Date: April 2024
"""

import os
import sys
import sqlite3
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
DB_PATH     = BASE_DIR / "data" / "sales.db"
SCHEMA_PATH = BASE_DIR / "db" / "schema.sql"

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Simulation period: 90 days ending yesterday
SIM_END_DATE   = datetime(2024, 6, 30)
SIM_START_DATE = SIM_END_DATE - timedelta(days=89)  # 90 days total
RECORDS_PER_BRANCH_PER_DAY = 20  # 20 product transactions per branch per day

# ---------------------------------------------------------------------------
# Branch Definitions (5 stores across the U.S.)
# ---------------------------------------------------------------------------
BRANCHES = [
    {"branch_id": "BR001", "city": "Chicago",  "state": "Illinois",      "region": "Midwest",    "base_revenue_multiplier": 1.10},
    {"branch_id": "BR002", "city": "Dallas",   "state": "Texas",         "region": "South",      "base_revenue_multiplier": 1.05},
    {"branch_id": "BR003", "city": "Seattle",  "state": "Washington",    "region": "Northwest",  "base_revenue_multiplier": 0.95},
    {"branch_id": "BR004", "city": "Miami",    "state": "Florida",       "region": "Southeast",  "base_revenue_multiplier": 1.00},
    {"branch_id": "BR005", "city": "Boston",   "state": "Massachusetts", "region": "Northeast",  "base_revenue_multiplier": 1.08},
]

# ---------------------------------------------------------------------------
# Product Catalog (30 products across 5 categories)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Electronics":    {"price_range": (49.99, 999.99), "avg_units": 8,  "stock_range": (20, 200)},
    "Clothing":       {"price_range": (14.99, 149.99), "avg_units": 15, "stock_range": (30, 300)},
    "Groceries":      {"price_range": (2.99,  49.99),  "avg_units": 40, "stock_range": (50, 500)},
    "Home & Garden":  {"price_range": (9.99,  299.99), "avg_units": 12, "stock_range": (25, 250)},
    "Sports":         {"price_range": (19.99, 399.99), "avg_units": 10, "stock_range": (20, 200)},
}

PRODUCTS = []
pid = 1
for cat, cfg in CATEGORIES.items():
    for i in range(6):  # 6 products per category = 30 total
        PRODUCTS.append({
            "product_id":   f"PROD_{pid:03d}",
            "category":     cat,
            "unit_price":   round(random.uniform(*cfg["price_range"]), 2),
            "avg_units":    cfg["avg_units"],
            "stock_range":  cfg["stock_range"],
        })
        pid += 1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Create required directories if they don't exist."""
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "output").mkdir(parents=True, exist_ok=True)
    log.info("Directories ensured: data/, output/")


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the project database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def apply_schema(conn: sqlite3.Connection):
    """Apply schema.sql to initialise tables/views if not already present."""
    if SCHEMA_PATH.exists():
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        # Execute each statement individually (split on ';')
        statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # View / index may already exist
        conn.commit()
        log.info("Schema applied from db/schema.sql")
    else:
        # Fallback: create minimal table inline
        conn.execute("""
            CREATE TABLE IF NOT EXISTS branch_sales (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                branch_id       TEXT NOT NULL,
                city            TEXT NOT NULL,
                product_id      TEXT NOT NULL,
                category        TEXT NOT NULL,
                units_sold      INTEGER NOT NULL,
                revenue         REAL NOT NULL,
                stock_remaining INTEGER NOT NULL,
                sale_date       TEXT NOT NULL,
                created_at      TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        log.warning("schema.sql not found – created minimal table inline.")

# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def seasonal_multiplier(date: datetime) -> float:
    """Return a demand multiplier based on month (seasonal pattern)."""
    month = date.month
    # Higher demand in Nov-Dec (holiday), lower in Jan-Feb (post-holiday)
    seasonal = {
        1: 0.80, 2: 0.78, 3: 0.85, 4: 0.92,
        5: 1.00, 6: 1.05, 7: 1.08, 8: 1.06,
        9: 0.95, 10: 0.98, 11: 1.20, 12: 1.35,
    }
    return seasonal.get(month, 1.0)


def day_of_week_multiplier(date: datetime) -> float:
    """Weekends see higher foot traffic."""
    dow = date.weekday()  # 0=Mon, 6=Sun
    return 1.20 if dow >= 5 else 1.0


def generate_daily_records(branch: dict, date: datetime) -> list[dict]:
    """
    Generate RECORDS_PER_BRANCH_PER_DAY synthetic sales records
    for a given branch on a given date.
    """
    records = []
    s_mult = seasonal_multiplier(date)
    d_mult = day_of_week_multiplier(date)
    br_mult = branch["base_revenue_multiplier"]

    # Sample products for this day (with replacement for variety)
    day_products = random.choices(PRODUCTS, k=RECORDS_PER_BRANCH_PER_DAY)

    for prod in day_products:
        cat_cfg   = CATEGORIES[prod["category"]]
        base_units = prod["avg_units"]

        # Gaussian noise around the mean, floored at 1
        units_sold = max(
            1,
            int(np.random.normal(
                loc=base_units * s_mult * d_mult * br_mult,
                scale=base_units * 0.3
            ))
        )

        # Stock remaining: random snapshot (in real pipeline this would be live)
        stock_remaining = random.randint(*prod["stock_range"])
        # Simulate depletion: stock decreases as units sold increases
        stock_remaining = max(0, stock_remaining - units_sold)

        revenue = round(units_sold * prod["unit_price"] * random.uniform(0.95, 1.05), 2)

        records.append({
            "branch_id":       branch["branch_id"],
            "city":            branch["city"],
            "product_id":      prod["product_id"],
            "category":        prod["category"],
            "units_sold":      units_sold,
            "revenue":         revenue,
            "stock_remaining": stock_remaining,
            "sale_date":       date.strftime("%Y-%m-%d"),
        })
    return records

# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def run_pipeline():
    """
    Full ETL pipeline:
    1. Generate 90 days × 5 branches × 20 records of synthetic sales
    2. Load into SQLite branch_sales table
    3. Print ingestion summary
    """
    ensure_dirs()
    conn = get_connection()
    apply_schema(conn)

    log.info("=" * 60)
    log.info("  AI-ML Data Science Simulation – Daily Sales Pipeline")
    log.info("=" * 60)
    log.info(f"  Simulation period : {SIM_START_DATE.date()} → {SIM_END_DATE.date()}")
    log.info(f"  Branches          : {len(BRANCHES)}")
    log.info(f"  Products          : {len(PRODUCTS)}")
    log.info(f"  Records/branch/day: {RECORDS_PER_BRANCH_PER_DAY}")
    log.info("=" * 60)

    all_records = []
    date_cursor = SIM_START_DATE

    while date_cursor <= SIM_END_DATE:
        for branch in BRANCHES:
            day_records = generate_daily_records(branch, date_cursor)
            all_records.extend(day_records)
        date_cursor += timedelta(days=1)

    df = pd.DataFrame(all_records)

    # --- Clear existing data and reload (idempotent re-runs) ---
    conn.execute("DELETE FROM branch_sales")
    conn.commit()

    df.to_sql("branch_sales", conn, if_exists="append", index=False)
    conn.commit()

    # --- Ingestion Summary ---
    total_records = len(df)
    total_revenue = df["revenue"].sum()
    total_units   = df["units_sold"].sum()

    log.info("")
    log.info("  ✅  INGESTION COMPLETE")
    log.info("-" * 60)
    log.info(f"  Total records ingested : {total_records:,}")
    log.info(f"  Total units sold       : {total_units:,}")
    log.info(f"  Total revenue          : ${total_revenue:,.2f}")
    log.info("-" * 60)

    # Per-branch summary
    branch_summary = (
        df.groupby(["branch_id", "city"])
        .agg(
            records      =("revenue", "count"),
            units_sold   =("units_sold", "sum"),
            revenue      =("revenue", "sum"),
            avg_stock    =("stock_remaining", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    log.info("\n  BRANCH INGESTION SUMMARY:")
    log.info(f"  {'Branch':<8} {'City':<12} {'Records':>8} {'Units':>8} {'Revenue':>14} {'Avg Stock':>10}")
    log.info(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*14} {'-'*10}")
    for _, row in branch_summary.iterrows():
        log.info(
            f"  {row.branch_id:<8} {row.city:<12} "
            f"{int(row.records):>8,} {int(row.units_sold):>8,} "
            f"${row.revenue:>13,.2f} {row.avg_stock:>10.1f}"
        )

    # Per-category summary
    cat_summary = (
        df.groupby("category")
        .agg(records=("revenue", "count"), units=("units_sold", "sum"), revenue=("revenue", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    log.info("\n  CATEGORY SUMMARY:")
    log.info(f"  {'Category':<18} {'Records':>8} {'Units':>8} {'Revenue':>14}")
    log.info(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*14}")
    for _, row in cat_summary.iterrows():
        log.info(
            f"  {row.category:<18} {int(row.records):>8,} "
            f"{int(row.units):>8,} ${row.revenue:>13,.2f}"
        )

    log.info("")
    log.info(f"  Database saved to: {DB_PATH}")
    log.info("=" * 60)

    conn.close()
    return df


if __name__ == "__main__":
    run_pipeline()
