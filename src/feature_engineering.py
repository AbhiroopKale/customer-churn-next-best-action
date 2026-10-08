"""Feature engineering shared by training, batch scoring and the Streamlit app.
Every feature uses fixed bins/rules (nothing is learned from data), so there is no leakage."""
import numpy as np
import pandas as pd

from src import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_group"] = pd.cut(df["tenure"], bins=config.TENURE_BINS,
                                labels=config.TENURE_LABELS).astype(str)
    df["charge_category"] = pd.cut(
        df["MonthlyCharges"], bins=[-np.inf, config.CHARGE_LOW_MAX, config.CHARGE_HIGH_MIN, np.inf],
        labels=["Low", "Medium", "High"], right=False).astype(str)
    df["n_addons"] = (df[config.ADDON_COLUMNS] == "Yes").sum(axis=1).astype(int)
    return df
