"""Loading, cleaning and splitting. Nothing here learns from data (no fitting),
except the split itself, which happens BEFORE any scaler/encoder is fitted."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.feature_engineering import add_features


def load_raw(path=None) -> pd.DataFrame:
    path = path or config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' "
            "from Kaggle and place it in data/raw/.")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    # TotalCharges is read as text because 11 rows contain a blank string.
    # In the Telco data those rows all have tenure == 0 (brand-new customers).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[(df["tenure"] == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    df = df.drop_duplicates(subset=config.ID_COL).reset_index(drop=True)
    df[config.TARGET] = df[config.TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def prepare() -> pd.DataFrame:
    """Raw CSV -> cleaned data with engineered features."""
    return add_features(clean(load_raw()))


def get_xy(df: pd.DataFrame):
    return df[config.FEATURES], df[config.TARGET]


def split(df: pd.DataFrame):
    X, y = get_xy(df)
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)
