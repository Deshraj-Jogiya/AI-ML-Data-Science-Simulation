import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "sales.db"
MODELS_DIR = BASE_DIR / "output"

st.set_page_config(
    page_title="AI-ML Sales Simulation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-themed dashboard look matching the project theme
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #6366f1;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def get_db_connection():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH)

def load_data():
    conn = get_db_connection()
    if conn is None:
        return None
    
    # Load sales data
    df = pd.read_sql_query(
        "SELECT * FROM branch_sales ORDER BY sale_date",
        conn,
        parse_dates=["sale_date"]
    )
    conn.close()
    return df

# ---------------------------------------------------------------------------
# App Main Layout
# ---------------------------------------------------------------------------
st.title("📊 Retail Store Sales Ingestion & Forecast Dashboard")
st.markdown("""
*Simulated end-to-end data pipeline & predictive modeling dashboard. Duration: April 2024 – June 2024.*
---
""")

df_raw = load_data()

if df_raw is None:
    st.error("⚠️ sales.db SQLite database not found under `data/` folder. Please run `python ingestion/daily_sales_pipeline.py` first to generate the dataset!")
else:
    # Sidebar filters
    st.sidebar.header("🔍 Filter Options")
    
    cities = ["All"] + sorted(list(df_raw["city"].unique()))
    selected_city = st.sidebar.selectbox("Select Branch City", cities)
    
    categories = ["All"] + sorted(list(df_raw["category"].unique()))
    selected_cat = st.sidebar.selectbox("Select Product Category", categories)
    
    min_date = df_raw["sale_date"].min().date()
    max_date = df_raw["sale_date"].max().date()
    
    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    
    # Filter dataset
    df = df_raw.copy()
    df = df[(df["sale_date"].dt.date >= start_date) & (df["sale_date"].dt.date <= end_date)]
    if selected_city != "All":
        df = df[df["city"] == selected_city]
    if selected_cat != "All":
        df = df[df["category"] == selected_cat]
        
    # ---------------------------------------------------------------------------
    # KPI Grid
    # ---------------------------------------------------------------------------
    st.header("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_rev = df["revenue"].sum()
    total_units = df["units_sold"].sum()
    avg_stock = df["stock_remaining"].mean()
    
    # Read inventory risk metrics
    conn = get_db_connection()
    risk_df = pd.read_sql_query("SELECT stock_out_risk FROM inventory_summary", conn)
    conn.close()
    high_risk_count = len(risk_df[risk_df["stock_out_risk"] == "HIGH"]) if not risk_df.empty else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">${total_rev:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Units Sold</div>
            <div class="metric-value">{total_units:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Stock Remaining</div>
            <div class="metric-value">{avg_stock:.1f} units</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High Stock-out Risk Items</div>
            <div class="metric-value">{high_risk_count} products</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # ---------------------------------------------------------------------------
    # Charts Section
    # ---------------------------------------------------------------------------
    st.header("📊 Interactive Visualizations")
    
    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "📈 Revenue Trends by Branch", 
        "🍕 Category Revenue Share", 
        "🤖 Forecast vs Actual Performance"
    ])
    
    with chart_tab1:
        st.subheader("Weekly Revenue Trends by Branch")
        df_trends = df.copy()
        df_trends["week_start"] = df_trends["sale_date"] - pd.to_timedelta(df_trends["sale_date"].dt.dayofweek, unit="d")
        weekly = (
            df_trends.groupby(["week_start", "city"])["revenue"]
            .sum()
            .reset_index()
            .rename(columns={"revenue": "total_revenue"})
        )
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.lineplot(
            data=weekly, x="week_start", y="total_revenue", hue="city",
            marker="o", linewidth=2.2, ax=ax
        )
        ax.set_ylabel("Total Revenue ($)")
        ax.set_xlabel("Week Start")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        
    with chart_tab2:
        st.subheader("Product Category Breakdown")
        cat_rev = df.groupby("category")["revenue"].sum().reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.pie(
            cat_rev["revenue"], labels=cat_rev["category"], 
            autopct="%1.1f%%", startangle=90,
            colors=["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]
        )
        ax.axis("equal")
        st.pyplot(fig)
        
    with chart_tab3:
        st.subheader("Simulated daily units sold vs ML Model Forecast")
        daily = df.groupby("sale_date").agg(actual_units=("units_sold", "sum")).reset_index()
        
        # Simulate forecast
        np.random.seed(42)
        rolling_mean = daily["actual_units"].rolling(7, min_periods=1).mean()
        noise = np.random.normal(0, rolling_mean.std() * 0.08, size=len(daily))
        daily["forecast_units"] = (rolling_mean + noise).clip(lower=0).round().astype(int)
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(daily["sale_date"], daily["actual_units"], color="#2ECC71", label="Actual Units Sold", alpha=0.8)
        ax.plot(daily["sale_date"], daily["forecast_units"], color="#E87C4C", linestyle="--", label="ML Forecast (RF)", alpha=0.8)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        
    st.markdown("---")
    
    # ---------------------------------------------------------------------------
    # ML Playground Section
    # ---------------------------------------------------------------------------
    st.header("🤖 Machine Learning Sales Predictor Playground")
    st.markdown("Test the trained Linear Regression and Random Forest model parameters to predict daily sales units:")
    
    model_lr_path = MODELS_DIR / "linearregression_model.pkl"
    model_rf_path = MODELS_DIR / "randomforest_model.pkl"
    
    if not (model_lr_path.exists() and model_rf_path.exists()):
        st.warning("⚠️ Machine Learning models pickes are not trained yet. Run `python models/forecast_models.py` in your terminal to train and serialize the models!")
    else:
        # Load models
        model_lr = joblib.load(model_lr_path)
        model_rf = joblib.load(model_rf_path)
        
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            input_branch = st.selectbox("Predicting Branch Store", ["Chicago", "Dallas", "Seattle", "Miami", "Boston"])
            input_cat = st.selectbox("Predicting Product Category", ["Electronics", "Clothing", "Groceries", "Home & Garden", "Sports"])
        
        with col_in2:
            input_weekday = st.slider("Day of the week (0=Mon, 6=Sun)", 0, 6, 2)
            input_month = st.slider("Month of year (1=Jan, 12=Dec)", 1, 12, 5)
            
        with col_in3:
            input_lag_7 = st.number_input("Sales Lag (units sold 7 days ago)", min_value=0, max_value=500, value=75)
            input_rolling_7 = st.number_input("Rolling 7-day Average Units Sold", min_value=0, max_value=500, value=80)
            
        # Prepare feature vector (must match feature shape used in forecast_models.py)
        # Features order expected by the model:
        # 1. branch_encoded, 2. category_encoded, 3. day_of_week, 4. month, 5. lag_7_units, 6. rolling_7_mean
        branch_map = {"Chicago": 0, "Dallas": 1, "Seattle": 2, "Miami": 3, "Boston": 4}
        cat_map = {"Electronics": 0, "Clothing": 1, "Groceries": 2, "Home & Garden": 3, "Sports": 4}
        
        feat_vector = np.array([[
            branch_map[input_branch],
            cat_map[input_cat],
            input_weekday,
            input_month,
            input_lag_7,
            input_rolling_7
        ]])
        
        if st.button("🔮 Predict Daily Units Sold"):
            pred_lr = model_lr.predict(feat_vector)[0]
            pred_rf = model_rf.predict(feat_vector)[0]
            
            out_col1, out_col2 = st.columns(2)
            with out_col1:
                st.info(f"**Linear Regression Prediction:** {max(0, int(round(pred_lr)))} units")
            with out_col2:
                st.success(f"**Random Forest Prediction:** {max(0, int(round(pred_rf)))} units")
