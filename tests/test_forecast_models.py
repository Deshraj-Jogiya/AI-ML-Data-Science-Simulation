import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
import forecast_models as fm  # noqa: E402


def _daily(units, branch="BR001", category="Groceries"):
    dates = pd.date_range("2024-04-01", periods=len(units), freq="D")
    return pd.DataFrame({"branch_id": branch, "category": category, "sale_date": dates, "units_sold": units})


def test_features_never_see_the_target_day_or_the_six_days_before_it():
    units = list(np.arange(40, dtype=float))
    base = fm.engineer_features(_daily(units))
    changed = units.copy()
    for i in range(33, 40):  # change the last 7 days, including the last target day
        changed[i] += 1000
    after = fm.engineer_features(_daily(changed))
    last = base["sale_date"].max()
    cols = fm.FEATURE_COLS
    pd.testing.assert_frame_equal(base.loc[base["sale_date"] == last, cols].reset_index(drop=True),
                                  after.loc[after["sale_date"] == last, cols].reset_index(drop=True))


def test_forecast_accuracy_is_one_minus_wape():
    assert fm.forecast_accuracy([100, 100], [90, 110]) == 0.9
    assert fm.forecast_accuracy([50, 50], [50, 50]) == 1.0


def test_stockout_simulation_counts_days_demand_exceeds_stock():
    test = _daily([10] * 7).assign(perfect=10.0, short=5.0)
    perfect = fm.simulate_stockouts(test, "perfect", safety=1.0)
    short = fm.simulate_stockouts(test, "short", safety=1.0)
    assert perfect["stockout_days"] == 0 and perfect["fill_rate"] == 1.0
    # 35 units of stock against 7 x 10 demand: days 4 to 7 run short (day 4 is half served)
    assert short["stockout_days"] == 4 and round(short["fill_rate"], 2) == 0.5
