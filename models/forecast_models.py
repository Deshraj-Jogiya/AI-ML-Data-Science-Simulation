"""
forecast_models.py
-------------------
AI-ML Data Science Simulation Project
Trains Linear Regression AND Random Forest on synthetic sales data.
Calculates R² and RMSE, prints a comparison table, plots feature
importance, and saves both models as .pkl files.

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
from sklearn.model_selection import train_test_split
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
    """Load branch_sales and engineer ML features."""
    if not DB_PATH.exists():
        log.error(f"Database not found: {DB_PATH}. Run daily_sales_pipeline.py first.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM branch_sales ORDER BY branch_id, product_id, sale_date",
        conn,
        parse_dates=["sale_date"]
    )
    conn.close()
    log.info(f"Loaded {len(df):,} records from sales.db")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix:
      - day_of_week, month, is_weekend
      - branch_id encoded, category encoded
      - lag_7_units, lag_14_units  (previous 7 / 14 days of units_sold)
      - rolling_7_mean, rolling_14_mean
    """
    df = df.copy().sort_values(["branch_id", "product_id", "sale_date"])

    # --- Temporal features ---
    df["day_of_week"] = df["sale_date"].dt.dayofweek
    df["month"]       = df["sale_date"].dt.month
    df["day_of_month"]= df["sale_date"].dt.day
    df["week_number"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # --- Categorical encoding ---
    le_branch   = LabelEncoder()
    le_category = LabelEncoder()
    df["branch_encoded"]   = le_branch.fit_transform(df["branch_id"])
    df["category_encoded"] = le_category.fit_transform(df["category"])

    # --- Lag features per branch + product ---
    grp = df.groupby(["branch_id", "product_id"])["units_sold"]

    df["lag_7_units"]      = grp.shift(7)
    df["lag_14_units"]     = grp.shift(14)
    df["rolling_7_mean"]   = grp.transform(lambda x: x.rolling(7,  min_periods=1).mean())
    df["rolling_14_mean"]  = grp.transform(lambda x: x.rolling(14, min_periods=1).mean())
    df["rolling_7_std"]    = grp.transform(lambda x: x.rolling(7,  min_periods=2).std().fillna(0))

    df.dropna(subset=["lag_7_units", "lag_14_units"], inplace=True)
    log.info(f"Feature-engineered dataset: {len(df):,} rows")
    return df


FEATURE_COLS = [
    "day_of_week", "month", "day_of_month", "week_number", "is_weekend",
    "branch_encoded", "category_encoded",
    "lag_7_units", "lag_14_units",
    "rolling_7_mean", "rolling_14_mean", "rolling_7_std",
]
TARGET_COL = "units_sold"

# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train LR and RF, return metrics and trained models."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    log.info(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    results = {}

    # -- Linear Regression --
    log.info("Training Linear Regression ...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_lr = np.maximum(y_pred_lr, 0)  # predictions can't be negative

    lr_r2   = r2_score(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    results["LinearRegression"] = {
        "model": lr, "y_test": y_test, "y_pred": y_pred_lr,
        "r2": lr_r2, "rmse": lr_rmse,
    }
    log.info(f"  Linear Regression → R²: {lr_r2:.4f}  RMSE: {lr_rmse:.4f}")

    # -- Random Forest --
    log.info("Training Random Forest (100 estimators, max_depth=10) ...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_pred_rf = np.maximum(y_pred_rf, 0)

    rf_r2   = r2_score(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    results["RandomForest"] = {
        "model": rf, "y_test": y_test, "y_pred": y_pred_rf,
        "r2": rf_r2, "rmse": rf_rmse,
        "feature_importances": rf.feature_importances_,
    }
    log.info(f"  Random Forest      → R²: {rf_r2:.4f}  RMSE: {rf_rmse:.4f}")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict):
    """Print a side-by-side model comparison table."""
    log.info("")
    log.info("=" * 55)
    log.info("  MODEL PERFORMANCE COMPARISON")
    log.info("=" * 55)
    log.info(f"  {'Model':<22} {'R² Score':>10} {'RMSE':>10} {'Accuracy':>10}")
    log.info(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for name, res in results.items():
        accuracy_pct = res["r2"] * 100
        log.info(
            f"  {name:<22} {res['r2']:>10.4f} "
            f"{res['rmse']:>10.4f} {accuracy_pct:>9.1f}%"
        )
    log.info("=" * 55)
    log.info("")

    # Declare winner
    best = max(results.items(), key=lambda kv: kv[1]["r2"])
    log.info(f"  🏆  Best model: {best[0]}  (R² = {best[1]['r2']:.4f})")
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

    for ax, (name, res) in zip(axes, results.items()):
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


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_models(results: dict):
    """Serialize trained models to .pkl files."""
    for name, res in results.items():
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
