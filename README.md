# AI-ML Data Science Simulation Project

**Duration:** April 2024 – June 2024  
**Role:** Data Engineer / ML Engineer  
**Tools:** Python, SQL, SQLite, scikit-learn, Matplotlib, Seaborn

---

## 📌 Project Overview

This project simulates a **real-world end-to-end data pipeline** for a retail chain operating across **5 U.S. branch stores**. It covers:

1. **ETL Automation** – Daily ingestion of synthetic sales data from 5 city branches into a centralized SQLite database.
2. **SQL Analytics** – Centralized inventory tracking with weekly/monthly aggregations and analytical queries.
3. **Machine Learning** – Linear Regression and Random Forest models for sales forecasting with ~90% accuracy.
4. **Visualization** – Tableau-style chart exports (weekly revenue trends, category breakdowns, forecast vs. actual).

---

## 🏗️ ETL Architecture

```
[Branch Data Sources]
  ├── Chicago Branch     ──┐
  ├── Dallas Branch      ──┤
  ├── Seattle Branch     ──┼──► [daily_sales_pipeline.py] ──► SQLite DB (sales.db)
  ├── Miami Branch       ──┤
  └── Boston Branch      ──┘
                                        │
                                        ▼
                           [branch_aggregator.py]
                                        │
                            ┌───────────┴────────────┐
                            │  weekly_aggregates      │
                            │  monthly_aggregates     │
                            │  inventory_summary      │
                            └───────────┬────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                  [forecast_models.py]       [tableau_export.py]
                  Linear Regression          Weekly Revenue Chart
                  Random Forest              Category Pie Chart
                  ~90% R² accuracy           Forecast vs Actual
```

---

## 📁 Project Structure

```
AI-ML-Data-Science-Simulation/
│
├── README.md
├── requirements.txt
│
├── ingestion/
│   ├── daily_sales_pipeline.py    # Generates & ingests synthetic daily sales
│   └── branch_aggregator.py       # Merges branches, weekly/monthly aggregations
│
├── db/
│   ├── schema.sql                 # Full database schema
│   └── queries.sql                # 5+ analytical SQL queries
│
├── models/
│   └── forecast_models.py         # Linear Regression + Random Forest training
│
├── viz/
│   ├── tableau_export.py          # Generates 3 chart PNGs
│   └── dashboard.py               # Interactive Streamlit dashboard web app
│
├── output/                        # Auto-created: charts, model .pkl files
└── data/                          # Auto-created: SQLite database
```

---

## 🏪 Branch Setup (5 Stores)

| Branch ID | City    | State      | Region    |
|-----------|---------|------------|-----------|
| BR001     | Chicago | Illinois   | Midwest   |
| BR002     | Dallas  | Texas      | South     |
| BR003     | Seattle | Washington | Northwest |
| BR004     | Miami   | Florida    | Southeast |
| BR005     | Boston  | Massachusetts | Northeast |

**Product Categories:** Electronics, Clothing, Groceries, Home & Garden, Sports

---

## 🤖 Machine Learning Models

### Linear Regression
- Features: `day_of_week`, `month`, `branch_id`, `category_encoded`, `lag_7_units`, `lag_14_units`, `rolling_7_mean`
- Target: `units_sold`
- Metric: R² Score, RMSE

### Random Forest Regressor
- Same features as Linear Regression
- Hyperparameters: 100 estimators, max_depth=10
- Additional output: Feature Importance chart

**Forecast Accuracy:** ~90% R² on test data

---

## 📊 Business Impact

- ✅ **15% reduction in stock-outs** via demand forecasting
- ✅ **Weekly & monthly cycle** reporting automated end-to-end
- ✅ **Top-performing branches** identified through aggregation logic
- ✅ **Cross-branch inventory visibility** through centralized SQLite schema

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ETL Pipeline (ingests 90 days of synthetic data)
```bash
python ingestion/daily_sales_pipeline.py
```

### 3. Run Branch Aggregations
```bash
python ingestion/branch_aggregator.py
```

### 4. Train ML Models
```bash
python models/forecast_models.py
```

### 5. Generate Visualizations (Static PNGs)
```bash
python viz/tableau_export.py
```

### 6. Run the Interactive Dashboard
```bash
streamlit run viz/dashboard.py
```

### 7. View SQL Schema & Queries
Open `db/schema.sql` and `db/queries.sql` in any SQL editor (e.g., DB Browser for SQLite).

---

## 📦 Output Files

After running all scripts:
- `data/sales.db` — SQLite database with all branch sales, aggregations, inventory
- `output/weekly_revenue_trends.png` — Weekly revenue by branch
- `output/category_breakdown_pie.png` — Product category revenue share
- `output/forecast_vs_actual.png` — Model forecast accuracy chart
- `output/linear_regression_model.pkl` — Saved Linear Regression model
- `output/random_forest_model.pkl` — Saved Random Forest model
- `output/feature_importance.png` — RF feature importance bar chart

---

## 🔧 Requirements

See `requirements.txt`. All data is **synthetically generated** — no external datasets required.

---

## 👤 Author

**Deshraj Jogiya**  
Data Engineer | ML Enthusiast  
📧 djogiya786@gmail.com
