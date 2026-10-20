"""
forecast_models.py
-------------------
AI-ML Data Science Simulation Project
Trains Linear Regression AND Random Forest on synthetic sales data.
Forecasts daily demand per branch and category from data at least a week old, scores Linear Regression and
Random Forest on held-out weeks against a naive last-week baseline (R2, RMSE, forecast accuracy = 1 - WAPE),
simulates weekly reordering to count stock-outs, plots feature importance and saves both models.

Author: Deshraj Jogiya
Date: May 2024
"""

import os
import sys
import logging
import sqlite3
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import r2_score, mean_squared_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
DB_PATH    = BASE_DIR / "data" / "sales.db"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load branch_sales and roll it up to one row per branch, category and day (missing days count as zero sales)."""
    if not DB_PATH.exists():
        log.error(f"Database not found: {DB_PATH}. Run daily_sales_pipeline.py first.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT branch_id, category, sale_date, units_sold FROM branch_sales", conn,
                           parse_dates=["sale_date"])
    conn.close()
    log.info(f"Loaded {len(df):,} records from sales.db")

    daily = df.groupby(["branch_id", "category", "sale_date"], as_index=False)["units_sold"].sum()
    dates = pd.date_range(daily["sale_date"].min(), daily["sale_date"].max(), freq="D")
    full = pd.MultiIndex.from_product([daily["branch_id"].unique(), daily["category"].unique(), dates],
                                      names=["branch_id", "category", "sale_date"])
    daily = daily.set_index(["branch_id", "category", "sale_date"]).reindex(full, fill_value=0).reset_index()
    log.info(f"Daily demand series: {daily.groupby(['branch_id', 'category']).ngroups} branch-category pairs x {len(dates)} days")
    return daily


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features known 7 days before the day being forecast, so a weekly reorder decision can use them:
      - day_of_week, is_weekend, day_of_month, month (calendar, always known)
      - branch and category codes
      - lag_7_units, lag_14_units: demand on the same weekday one and two weeks earlier
      - rolling_7_mean, rolling_14_mean, rolling_7_std: demand over the 7 / 14 days ending 7 days before
    No feature includes the target day or the 6 days before it.
    """
    df = df.copy().sort_values(["branch_id", "category", "sale_date"])
    df["day_of_week"]  = df["sale_date"].dt.dayofweek
    df["month"]        = df["sale_date"].dt.month
    df["day_of_month"] = df["sale_date"].dt.day
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    df["branch_encoded"]   = LabelEncoder().fit_transform(df["branch_id"])
    df["category_encoded"] = LabelEncoder().fit_transform(df["category"])

    grp = df.groupby(["branch_id", "category"])["units_sold"]
    df["lag_7_units"]     = grp.shift(7)
    df["lag_14_units"]    = grp.shift(14)
    df["rolling_7_mean"]  = grp.transform(lambda x: x.shift(7).rolling(7).mean())
    df["rolling_14_mean"] = grp.transform(lambda x: x.shift(7).rolling(14).mean())
    df["rolling_7_std"]   = grp.transform(lambda x: x.shift(7).rolling(7).std())

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    log.info(f"Feature-engineered dataset: {len(df):,} rows")
    return df


FEATURE_COLS = [
    "day_of_week", "month", "day_of_month", "is_weekend",
    "branch_encoded", "category_encoded",
    "lag_7_units", "lag_14_units",
    "rolling_7_mean", "rolling_14_mean", "rolling_7_std",
]
TARGET_COL = "units_sold"
TEST_DAYS = 28          # the last four weeks are held out; the models never see them in training
SAFETY_FACTOR = 1.10    # both reorder policies order 10% above their demand estimate


def forecast_accuracy(y_true, y_pred) -> float:
    """1 - WAPE: one minus total absolute error over total actual demand."""
    return 1.0 - np.abs(np.asarray(y_true) - np.asarray(y_pred)).sum() / np.asarray(y_true).sum()

# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train LR and RF on all but the last TEST_DAYS days and score them on those days, next to a naive baseline."""
    cutoff = df["sale_date"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train, test = df[df["sale_date"] < cutoff], df[df["sale_date"] >= cutoff]
    log.info(f"Train: {train['sale_date'].min().date()} to {train['sale_date'].max().date()} ({len(train):,} rows) | "
             f"Test: {test['sale_date'].min().date()} to {test['sale_date'].max().date()} ({len(test):,} rows)")
    y_test = test[TARGET_COL].values

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                              random_state=RANDOM_SEED, n_jobs=-1),
    }
    results = {}
    for name, model in models.items():
        model.fit(train[FEATURE_COLS].values, train[TARGET_COL].values)
        y_pred = np.maximum(model.predict(test[FEATURE_COLS].values), 0)
        results[name] = {"model": model, "y_test": y_test, "y_pred": y_pred}
    results["RandomForest"]["feature_importances"] = models["RandomForest"].feature_importances_
    # Naive baseline: demand on the same weekday last week
    results["NaiveLastWeek"] = {"model": None, "y_test": y_test, "y_pred": test["lag_7_units"].values}

    for name, res in results.items():
        res["r2"] = r2_score(y_test, res["y_pred"])
        res["rmse"] = float(np.sqrt(mean_squared_error(y_test, res["y_pred"])))
        res["accuracy"] = forecast_accuracy(y_test, res["y_pred"])
        log.info(f"  {name:<17} R2 {res['r2']:.4f}  RMSE {res['rmse']:.2f}  forecast accuracy {res['accuracy']:.1%}")
    results["_test"] = test.assign(rf_forecast=results["RandomForest"]["y_pred"],
                                   naive_forecast=results["NaiveLastWeek"]["y_pred"])
    return results


def simulate_stockouts(test: pd.DataFrame, forecast_col: str, safety: float = SAFETY_FACTOR) -> dict:
    """
    Weekly review per branch and category over the test weeks: at the start of each week stock is topped up to
    (forecast demand for the coming 7 days) x safety, then that week's real demand is served day by day.
    A stock-out day is a day on which demand exceeds the stock left.
    """
    stockout_days, lost_units, served, leftover, weeks = 0, 0.0, 0.0, 0.0, 0
    for _, g in test.sort_values("sale_date").groupby(["branch_id", "category"]):
        g = g.reset_index(drop=True)
        for start in range(0, len(g), 7):
            week = g.iloc[start:start + 7]
            stock = week[forecast_col].sum() * safety
            weeks += 1
            for demand in week["units_sold"]:
                if demand > stock:
                    stockout_days += 1
                    lost_units += demand - stock
                    served += stock
                    stock = 0.0
                else:
                    stock -= demand
                    served += demand
            leftover += stock
    total = test["units_sold"].sum()
    return {"stockout_days": int(stockout_days), "lost_units": round(lost_units, 1),
            "fill_rate": served / total if total else 1.0,
            "avg_unsold_per_week": round(leftover / weeks, 1) if weeks else 0.0}


def compare_reorder_policies(test: pd.DataFrame) -> dict:
    naive = simulate_stockouts(test, "naive_forecast")
    model = simulate_stockouts(test, "rf_forecast")
    reduction = 1 - model["stockout_days"] / naive["stockout_days"] if naive["stockout_days"] else 0.0
    log.info("")
    log.info("  STOCK-OUT SIMULATION (test weeks, weekly top-up to forecast x %.2f)" % SAFETY_FACTOR)
    for label, r in (("Last-week policy", naive), ("Forecast policy ", model)):
        log.info(f"  {label} : {r['stockout_days']} stock-out days, {r['lost_units']} units short, "
                 f"fill rate {r['fill_rate']:.2%}, {r['avg_unsold_per_week']} units left over per series-week")
    log.info(f"  Stock-out days reduced by {reduction:.1%}")
    return {"naive": naive, "forecast": model, "stockout_reduction": reduction}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict):
    """Print a side-by-side model comparison table."""
    log.info("")
    log.info("=" * 62)
    log.info("  MODEL PERFORMANCE ON THE HELD-OUT WEEKS")
    log.info("=" * 62)
    log.info(f"  {'Model':<17} {'R2':>8} {'RMSE':>8} {'Forecast accuracy (1-WAPE)':>28}")
    for name, res in results.items():
        if name.startswith("_"):
            continue
        log.info(f"  {name:<17} {res['r2']:>8.4f} {res['rmse']:>8.2f} {res['accuracy']:>27.1%}")
    log.info("=" * 62)
    best = max(((n, r) for n, r in results.items() if not n.startswith("_")), key=lambda kv: kv[1]["accuracy"])
    log.info(f"  Best model: {best[0]} (forecast accuracy {best[1]['accuracy']:.1%})")
    log.info("")


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

PALETTE = {
    "LinearRegression": "#4C9BE8",
    "RandomForest":     "#E87C4C",
    "actual":           "#2ECC71",
}


def plot_feature_importance(results: dict):
    """Bar chart of Random Forest feature importances."""
    rf_res = results["RandomForest"]
    importances = rf_res["feature_importances"]

    fi_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0F1B2D")
    ax.set_facecolor("#162032")

    colors = plt.cm.plasma(np.linspace(0.25, 0.9, len(fi_df)))
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors, edgecolor="none", height=0.65)

    # Value labels
    for bar, val in zip(bars, fi_df["Importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", color="#E0E0E0", fontsize=9)

    ax.set_xlabel("Feature Importance", color="#CCCCCC", fontsize=11)
    ax.set_title("Random Forest – Feature Importance", color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="#CCCCCC", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3A50")
    ax.grid(axis="x", color="#2A3A50", linestyle="--", alpha=0.6)

    out_path = OUTPUT_DIR / "feature_importance.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  Saved: {out_path}")


def plot_actual_vs_predicted(results: dict):
    """Scatter plots – Actual vs Predicted for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0F1B2D")
    fig.suptitle("Actual vs Predicted – Units Sold", color="white", fontsize=15, fontweight="bold", y=1.01)

    model_colors = {"LinearRegression": "#4C9BE8", "RandomForest": "#E87C4C"}

    for ax, name in zip(axes, ["LinearRegression", "RandomForest"]):
        res = results[name]
        ax.set_facecolor("#162032")
        y_test, y_pred = res["y_test"], res["y_pred"]
        max_val = max(y_test.max(), y_pred.max())

        ax.scatter(y_test, y_pred, alpha=0.25, s=12,
                   color=model_colors[name], edgecolors="none", rasterized=True)
        ax.plot([0, max_val], [0, max_val], "--", color="#FFFFFF", linewidth=1.5, label="Perfect fit")

        ax.set_xlabel("Actual Units", color="#CCCCCC", fontsize=11)
        ax.set_ylabel("Predicted Units", color="#CCCCCC", fontsize=11)
        ax.set_title(f"{name}\nR²={res['r2']:.4f}  RMSE={res['rmse']:.2f}",
                     color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="#CCCCCC")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A3A50")
        ax.grid(color="#2A3A50", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, framealpha=0.3, labelcolor="white")

    out_path = OUTPUT_DIR / "actual_vs_predicted.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  Saved: {out_path}")


def export_results(results: dict):
    """Write the held-out forecasts and the policy comparison for the Tableau dashboard."""
    out = OUTPUT_DIR.parent / "exports"
    out.mkdir(exist_ok=True)
    cols = ["sale_date", "branch_id", "category", "units_sold", "rf_forecast", "naive_forecast"]
    results["_test"][cols].round(2).to_csv(out / "forecast_vs_actual.csv", index=False)
    rows = [{"model": n, "r2": round(r["r2"], 4), "rmse": round(r["rmse"], 2), "forecast_accuracy": round(r["accuracy"], 4)}
            for n, r in results.items() if not n.startswith("_")]
    pd.DataFrame(rows).to_csv(out / "model_metrics.csv", index=False)
    pol = results["_policies"]
    pd.DataFrame([{"policy": "Last-week sales", **pol["naive"]}, {"policy": "Random Forest forecast", **pol["forecast"]}]
                 ).to_csv(out / "reorder_policy_comparison.csv", index=False)
    log.info(f"  Exports written to {out}")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_models(results: dict):
    """Serialize trained models to .pkl files."""
    for name in ["LinearRegression", "RandomForest"]:
        res = results[name]
        fname = name.lower().replace(" ", "_") + "_model.pkl"
        path  = OUTPUT_DIR / fname
        joblib.dump(res["model"], path)
        log.info(f"  Model saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_forecasting():
    log.info("=" * 60)
    log.info("  AI-ML Data Science Simulation – Forecast Models")
    log.info("=" * 60)

    raw_df  = load_data()
    feat_df = engineer_features(raw_df)
    results = train_and_evaluate(feat_df)

    print_comparison_table(results)
    results["_policies"] = compare_reorder_policies(results["_test"])
    export_results(results)

    log.info("Generating plots ...")
    plot_feature_importance(results)
    plot_actual_vs_predicted(results)

    log.info("Saving models ...")
    save_models(results)

    log.info("")
    log.info("  ✅  FORECASTING COMPLETE")
    log.info(f"  Outputs saved to: {OUTPUT_DIR}")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    run_forecasting()
