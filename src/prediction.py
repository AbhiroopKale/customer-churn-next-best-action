"""Single-customer prediction used by the Streamlit app (and tests).
Uses the SAME saved Pipeline as training, so preprocessing is identical."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config
from src.explainability import explain, humanize_factor
from src.feature_engineering import add_features
from src.recommendation_engine import recommend, risk_level

RAW_INPUTS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
              "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
              "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
              "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]


@lru_cache(maxsize=1)
def load_artifacts():
    path = config.MODELS / "final_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Model not found. Run:  python -m src.train_models")
    pipe = joblib.load(path)
    meta = json.loads((config.MODELS / "meta.json").read_text())
    background = pd.read_csv(config.MODELS / "background.csv")
    return pipe, meta, background


def normalize_inputs(raw: dict) -> dict:
    """Enforce the same logical consistency that exists in the training data."""
    c = dict(raw)
    if c["PhoneService"] == "No":
        c["MultipleLines"] = "No phone service"
    if c["InternetService"] == "No":
        for col in config.ADDON_COLUMNS:
            c[col] = "No internet service"
    if c.get("TotalCharges") in (None, ""):
        # Not provided: ESTIMATE as tenure x monthly charges (flagged in the UI)
        c["TotalCharges"] = float(c["tenure"]) * float(c["MonthlyCharges"])
    return c


def validate_inputs(c: dict, meta: dict):
    for col, allowed in meta["categories"].items():
        if col in c and c[col] not in allowed:
            raise ValueError(f"Invalid value {c[col]!r} for {col}. Allowed: {allowed}")


def predict_customer(raw: dict, k: int = 5) -> dict:
    pipe, meta, background = load_artifacts()
    c = normalize_inputs(raw)
    row = add_features(pd.DataFrame([{col: c[col] for col in RAW_INPUTS}]))
    validate_inputs(row.iloc[0].to_dict(), meta)
    proba = float(pipe.predict_proba(row[config.FEATURES])[0, 1])
    factors = explain(pipe, row[config.FEATURES], background[config.FEATURES], k=k)
    factors["label"] = factors["feature"].map(humanize_factor)
    cust = row.iloc[0].to_dict()
    rec = recommend(cust, proba, factors.to_dict("records"))
    top_risk = factors[factors["direction"] == "increases risk"].head(3)
    return {
        "customer": cust, "probability": proba, "risk_level": risk_level(proba),
        "predicted_churn": proba >= meta["decision_threshold"],
        "threshold": meta["decision_threshold"], "factors": factors,
        "risk_factors_text": "; ".join(top_risk["label"]),
        "recommendation": rec, "model_version": meta["model_version"],
        "model_name": meta["model_name"], "meta": meta,
    }
