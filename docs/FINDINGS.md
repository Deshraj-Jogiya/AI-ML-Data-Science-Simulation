# Findings

Measured results from this pipeline. All sales, branches and promotions are simulated with a fixed seed; the results describe the
method, not a real retailer.

## Which model forecasts daily demand best?

**How it was tested.** Daily demand per branch and category was forecast from data at least a week old plus the promotion
calendar, and scored on four held-out weeks (the split is by date, not random). Forecast accuracy is 1 − WAPE. The baseline repeats
the same weekday of the previous week.

| Model | R² | RMSE (units) | Forecast accuracy |
|---|---|---|---|
| Random Forest | 0.9591 | 19.28 | 0.9005 |
| Linear Regression | 0.9066 | 29.13 | 0.8591 |
| Same weekday last week | 0.7847 | 44.22 | 0.7837 |

**Decision.** The Random Forest is the forecast used for reordering. A test (`test_features_never_see_the_target_day_or_the_six_days_before_it`)
checks that no feature can see the target day or the six days before it, so the held-out score is not inflated by leakage.

**Source.** `exports/model_metrics.csv` (written by `models/forecast_models.py`), `tests/test_forecast_models.py`.

**Skills shown.** scikit-learn (RandomForestRegressor, LinearRegression), pandas, time-based train/test split, pytest.

## Does reordering from the forecast beat reordering from last week's sales?

**How it was tested.** A weekly reorder was simulated over the held-out weeks: stock is topped up each week to the Random Forest
forecast plus 10%, or to last week's sales plus 10%, and each day's demand is drawn from that stock.

| Reorder policy | Stock-out days | Units short | Fill rate | Unsold units per branch-category week |
|---|---|---|---|---|
| Last week's sales | 51 | 7038.2 | 92.1% | 105.4 |
| Random Forest forecast | 20 | 419.1 | 99.5% | 57.5 |

**Decision.** Reordering from the forecast cut stock-out days by 60.8% and unsold stock from 105.4 to 57.5 units per branch-category week. The gain comes from
promotion weeks, which last week's sales cannot anticipate but the published promotion calendar can.

**Source.** `exports/reorder_policy_comparison.csv` (written by `models/forecast_models.py`); fill rates as percentages and the 60.8%
reduction from `README.md`.

**Skills shown.** pandas, inventory simulation in Python, scikit-learn forecasts, pytest (`test_stockout_simulation_counts_days_demand_exceeds_stock`).
