import pandas as pd

from src import config
from src.feature_engineering import add_features


def _row(**kw):
    base = {"tenure": 10, "MonthlyCharges": 50.0, **{c: "No" for c in config.ADDON_COLUMNS}}
    base.update(kw)
    return pd.DataFrame([base])


def test_tenure_group_boundaries():
    expected = {0: "0-6m", 6: "0-6m", 7: "7-12m", 12: "7-12m", 13: "13-24m", 24: "13-24m", 25: "25m+", 72: "25m+"}
    for t, label in expected.items():
        assert add_features(_row(tenure=t)).loc[0, "tenure_group"] == label, t


def test_charge_category_boundaries():
    expected = {18.25: "Low", 34.99: "Low", 35.0: "Medium", 89.99: "Medium", 90.0: "High", 118.75: "High"}
    for m, label in expected.items():
        assert add_features(_row(MonthlyCharges=m)).loc[0, "charge_category"] == label, m


def test_n_addons_counts_only_yes():
    row = _row(OnlineSecurity="Yes", TechSupport="Yes", StreamingTV="No internet service")
    assert add_features(row).loc[0, "n_addons"] == 2
