"""
trend_charts.py
---------------
AI-ML Data Science Simulation Project
Generates 3 Matplotlib/Seaborn trend charts (the interactive version is the Tableau Public dashboard):
  (a) Weekly revenue trends by branch
  (b) Product category breakdown pie chart
  (c) Forecast vs Actual line chart

All charts are saved as high-resolution PNGs in output/.

Author: Deshraj Jogiya
Date: June 2024
"""

import sys
import sqlite3
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
DB_PATH    = BASE_DIR / "data" / "sales.db"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (Tableau-inspired)
# ---------------------------------------------------------------------------
BRANCH_COLORS = {
    "Chicago":  "#4E79A7",
    "Dallas":   "#F28E2B",
    "Seattle":  "#E15759",
    "Miami":    "#76B7B2",
    "Boston":   "#59A14F",
}

CATEGORY_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
]

DARK_BG   = "#0F1B2D"
PANEL_BG  = "#162032"
TEXT_MAIN = "#EAEAEA"
TEXT_SUB  = "#9AABB8"
GRID_COL  = "#2A3A50"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        log.error("Database not found. Run daily_sales_pipeline.py first.")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def load_sales(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM branch_sales ORDER BY sale_date",
        conn,
        parse_dates=["sale_date"]
    )
    log.info(f"Loaded {len(df):,} records for visualisation")
    return df

# ---------------------------------------------------------------------------
# Chart 1 – Weekly Revenue Trends by Branch
# ---------------------------------------------------------------------------

def chart_weekly_revenue_trends(df: pd.DataFrame):
    """
    Line chart showing total weekly revenue for each of the 5 branches.
    Styled to mimic a Tableau dashboard export.
    """
    log.info("Generating Chart 1: Weekly Revenue Trends by Branch ...")

    df = df.copy()
    df["week_start"] = df["sale_date"] - pd.to_timedelta(df["sale_date"].dt.dayofweek, unit="d")

    weekly = (
        df.groupby(["week_start", "city"])["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_revenue"})
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for city, color in BRANCH_COLORS.items():
        city_data = weekly[weekly["city"] == city].sort_values("week_start")
        ax.plot(
            city_data["week_start"],
            city_data["total_revenue"],
            marker="o", markersize=5, linewidth=2.2,
            color=color, label=city, alpha=0.9
        )
        # Subtle area fill
        ax.fill_between(
            city_data["week_start"],
            city_data["total_revenue"],
            alpha=0.07, color=color
        )

    # Formatting
    ax.set_title(
        "Weekly Revenue Trends by Branch\nApril – June 2024",
        color=TEXT_MAIN, fontsize=16, fontweight="bold", pad=18
    )
    ax.set_xlabel("Week", color=TEXT_SUB, fontsize=12, labelpad=8)
    ax.set_ylabel("Total Revenue (USD)", color=TEXT_SUB, fontsize=12, labelpad=8)
    ax.tick_params(colors=TEXT_SUB, labelsize=10)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=0, interval=2))
    plt.xticks(rotation=30, ha="right")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="y", color=GRID_COL, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(axis="x", color=GRID_COL, linestyle=":",  linewidth=0.5, alpha=0.4)

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

    legend = ax.legend(
        title="Branch City", title_fontsize=11,
        fontsize=10, loc="upper left",
        framealpha=0.25, facecolor=PANEL_BG,
        labelcolor=TEXT_MAIN, edgecolor=GRID_COL
    )
    legend.get_title().set_color(TEXT_SUB)

    # Annotation: peak week
    peak_row = weekly.loc[weekly["total_revenue"].idxmax()]
    ax.annotate(
        f"  Peak: ${peak_row.total_revenue:,.0f}\n  {peak_row.city}",
        xy=(peak_row.week_start, peak_row.total_revenue),
        xytext=(30, 20), textcoords="offset points",
        fontsize=9, color="#FFD700",
        arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.2),
    )

    out_path = OUTPUT_DIR / "weekly_revenue_trends.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Chart 2 – Product Category Revenue Breakdown (Pie / Donut)
# ---------------------------------------------------------------------------

def chart_category_breakdown(df: pd.DataFrame):
    """
    Donut chart showing each product category's share of total revenue,
    with an inner breakdown by branch. Simulates a Tableau pie view.
    """
    log.info("Generating Chart 2: Category Revenue Breakdown ...")

    cat_rev = (
        df.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total_rev = cat_rev["revenue"].sum()

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(DARK_BG)

    # --- Donut Chart (left) ---
    ax_pie.set_facecolor(DARK_BG)
    wedges, texts, autotexts = ax_pie.pie(
        cat_rev["revenue"],
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=CATEGORY_COLORS,
        pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor=DARK_BG, linewidth=2),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(11)
        at.set_fontweight("bold")

    # Center text
    ax_pie.text(0, 0.08, "Total Revenue", ha="center", va="center",
                color=TEXT_SUB, fontsize=11)
    ax_pie.text(0, -0.18, f"${total_rev:,.0f}", ha="center", va="center",
                color=TEXT_MAIN, fontsize=14, fontweight="bold")

    ax_pie.set_title("Revenue Share by Category", color=TEXT_MAIN, fontsize=14,
                     fontweight="bold", pad=20)

    legend_labels = [
        f"{row.category}  (${row.revenue:,.0f})"
        for _, row in cat_rev.iterrows()
    ]
    ax_pie.legend(
        wedges, legend_labels, loc="lower center",
        bbox_to_anchor=(0.5, -0.12), ncol=2,
        fontsize=9, framealpha=0.2, facecolor=PANEL_BG,
        labelcolor=TEXT_MAIN, edgecolor=GRID_COL
    )

    # --- Grouped bar (right) – revenue by branch × category ---
    ax_bar.set_facecolor(PANEL_BG)
    branch_cat = (
        df.groupby(["city", "category"])["revenue"]
        .sum()
        .unstack(fill_value=0)
    )
    branch_cat = branch_cat[cat_rev["category"].tolist()]  # consistent order
    branch_cat.plot(
        kind="bar", stacked=True, ax=ax_bar,
        color=CATEGORY_COLORS, edgecolor="none", width=0.65,
    )

    ax_bar.set_title("Category Revenue by Branch", color=TEXT_MAIN, fontsize=14,
                     fontweight="bold", pad=15)
    ax_bar.set_xlabel("Branch City", color=TEXT_SUB, fontsize=11, labelpad=8)
    ax_bar.set_ylabel("Revenue (USD)", color=TEXT_SUB, fontsize=11, labelpad=8)
    ax_bar.tick_params(colors=TEXT_SUB, labelsize=10)
    plt.setp(ax_bar.get_xticklabels(), rotation=30, ha="right")
    ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_bar.grid(axis="y", color=GRID_COL, linestyle="--", alpha=0.6)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor(GRID_COL)

    leg = ax_bar.legend(
        title="Category", title_fontsize=10,
        fontsize=9, loc="upper right",
        framealpha=0.25, facecolor=PANEL_BG,
        labelcolor=TEXT_MAIN, edgecolor=GRID_COL
    )
    leg.get_title().set_color(TEXT_SUB)

    out_path = OUTPUT_DIR / "category_breakdown_pie.png"
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Chart 3 – Forecast vs Actual (Line Chart)
# ---------------------------------------------------------------------------

def chart_forecast_vs_actual(df: pd.DataFrame):
    """
    Actual daily units (all branches and categories) against the Random Forest's held-out forecasts and the
    last-week baseline, read from exports/forecast_vs_actual.csv written by models/forecast_models.py.
    """
    log.info("Generating Chart 3: Forecast vs Actual ...")
    exports = BASE_DIR / "exports"
    fc_path, metrics_path = exports / "forecast_vs_actual.csv", exports / "model_metrics.csv"
    if not fc_path.exists():
        log.error(f"{fc_path} not found. Run models/forecast_models.py first.")
        return
    fc = pd.read_csv(fc_path, parse_dates=["sale_date"])
    daily = fc.groupby("sale_date")[["units_sold", "rf_forecast", "naive_forecast"]].sum().reset_index()
    metrics = pd.read_csv(metrics_path).set_index("model")

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.plot(daily["sale_date"], daily["units_sold"], color="#2ECC71", linewidth=2.5, marker="o", markersize=4,
            label="Actual units")
    ax.plot(daily["sale_date"], daily["rf_forecast"], color="#E87C4C", linewidth=2.2, linestyle="--",
            label="Random Forest forecast")
    ax.plot(daily["sale_date"], daily["naive_forecast"], color="#8899AA", linewidth=1.5, linestyle=":",
            label="Same weekday last week")
    ax.set_title("Forecast vs Actual on the four held-out weeks (daily units, all branches)",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Units sold", color="#CCCCCC")
    ax.tick_params(colors="#CCCCCC")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3A50")
    ax.grid(color="#2A3A50", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, framealpha=0.3, labelcolor="white", loc="upper left")
    rf, naive = metrics.loc["RandomForest"], metrics.loc["NaiveLastWeek"]
    ax.text(0.99, 0.04,
            f"Forecast accuracy (1 - WAPE): Random Forest {rf['forecast_accuracy']:.1%}, last-week baseline "
            f"{naive['forecast_accuracy']:.1%}\nComputed per branch, category and day; the chart sums them",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color="#CCCCCC")

    out_path = OUTPUT_DIR / "forecast_vs_actual.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_visualisations():
    log.info("=" * 60)
    log.info("  AI-ML Data Science Simulation – Trend Charts")
    log.info("=" * 60)

    conn = get_conn()
    df   = load_sales(conn)
    conn.close()

    chart_weekly_revenue_trends(df)
    chart_category_breakdown(df)
    chart_forecast_vs_actual(df)

    log.info("")
    log.info("  ✅  ALL 3 CHARTS GENERATED")
    log.info(f"  Outputs saved to: {OUTPUT_DIR}")
    log.info("  Files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        log.info(f"    • {f.name}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_visualisations()
