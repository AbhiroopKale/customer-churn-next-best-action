"""Central configuration. Change thresholds here, not inside other modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
SQL_DIR = ROOT / "sql"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

ID_COL = "customerID"
TARGET = "Churn"

ADDON_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "n_addons"]
CATEGORICAL_FEATURES = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                        "PaperlessBilling", "PaymentMethod", "tenure_group", "charge_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fixed (not learned) bins -> no leakage. Charge cut-offs are roughly the 25th/75th
# percentiles of MonthlyCharges in the Telco data (35.5 / 89.85).
TENURE_BINS = [-1, 6, 12, 24, float("inf")]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25m+"]
CHARGE_LOW_MAX = 35.0     # Low:    MonthlyCharges < 35
CHARGE_HIGH_MIN = 90.0    # High:   MonthlyCharges >= 90

# Risk bands applied to the churn probability (configurable, validate on your data)
RISK_LOW_BELOW = 0.30
RISK_HIGH_FROM = 0.70

# Recommendation-rule thresholds
HIGH_CHARGE_MIN = CHARGE_HIGH_MIN
LOW_TENURE_MAX = 12
LOYAL_TENURE_MIN = 24
