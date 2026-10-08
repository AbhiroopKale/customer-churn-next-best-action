"""SHAP explanations. IMPORTANT: SHAP values describe how the MODEL uses each feature
for a prediction. They do not prove that a feature causes churn in the real world.
Units are the model's native scale (log-odds for Logistic Regression / XGBoost,
probability for tree ensembles), so use them to rank factors, not as exact percentages."""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src import config


def _feature_names(prep):
    """Readable name, is-one-hot flag and base column for every transformed feature."""
    cat_cols = sorted(config.CATEGORICAL_FEATURES, key=len, reverse=True)
    names, is_cat, base = [], [], []
    for n in prep.get_feature_names_out():
        raw = n.split("__", 1)[1]
        if n.startswith("num__"):
            names.append(raw); is_cat.append(False); base.append(raw)
        else:
            for c in cat_cols:
                if raw.startswith(c + "_"):
                    names.append(f"{c} = {raw[len(c) + 1:]}")
                    is_cat.append(True); base.append(c)
                    break
            else:
                raise ValueError(f"Cannot map feature {n}")
    return names, np.array(is_cat), base


def shap_matrix(pipe, X: pd.DataFrame, background: pd.DataFrame):
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = prep.transform(X)
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, prep.transform(background))
    else:
        explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    names, is_cat, base = _feature_names(prep)
    return np.asarray(sv), Xt, names, is_cat, base


def explain(pipe, X: pd.DataFrame, background: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Top-k contributions per row (long format, column 'row' = position in X)."""
    X = X.reset_index(drop=True)
    sv, Xt, names, is_cat, base = shap_matrix(pipe, X, background)
    # For one-hot features keep only the category the customer actually has,
    # otherwise we would show e.g. "InternetService = DSL" for a fiber customer.
    active = np.where(is_cat[None, :], Xt > 0.5, True)
    values = {b: X[b].to_numpy() for b in set(base)}
    rows = []
    for i in range(sv.shape[0]):
        contrib = np.where(active[i], sv[i], 0.0)
        for j in np.argsort(-np.abs(contrib))[:k]:
            label = names[j] if is_cat[j] else f"{base[j]} = {values[base[j]][i]:g}"
            rows.append({"row": i, "feature": label, "contribution": float(contrib[j]),
                         "direction": "increases risk" if contrib[j] > 0 else "decreases risk"})
    return pd.DataFrame(rows)


def humanize_factor(label: str) -> str:
    """Turn 'Contract = Month-to-month' into business wording."""
    feat, _, val = label.partition(" = ")
    special = {
        ("Contract", "Month-to-month"): "Month-to-month contract",
        ("Contract", "One year"): "One-year contract",
        ("Contract", "Two year"): "Two-year contract",
        ("InternetService", "Fiber optic"): "Fiber optic internet service",
        ("TechSupport", "No"): "No tech-support add-on",
        ("OnlineSecurity", "No"): "No online-security add-on",
        ("PaymentMethod", "Electronic check"): "Pays by electronic check",
        ("PaperlessBilling", "Yes"): "Paperless billing",
        ("SeniorCitizen", "1"): "Senior citizen",
        ("SeniorCitizen", "0"): "Not a senior citizen",
        ("Partner", "No"): "No partner",
        ("Dependents", "No"): "No dependents",
    }
    if (feat, val) in special:
        return special[(feat, val)]
    try:
        num = float(val)
    except ValueError:
        num = None
    if feat == "tenure" and num is not None:
        unit = "month" if num == 1 else "months"
        return f"{'Low tenure' if num <= config.LOW_TENURE_MAX else 'Tenure'} ({num:g} {unit})"
    if feat == "TotalCharges" and num is not None:
        return f"Total billed to date ({num:g})"
    if feat == "MonthlyCharges" and num is not None:
        return f"{'High m' if num >= config.HIGH_CHARGE_MIN else 'M'}onthly charges ({num:g})"
    if feat == "tenure_group":
        return f"Tenure band {val}"
    if feat == "charge_category":
        return f"{val} monthly-charge band"
    if feat == "n_addons" and num is not None:
        return f"{num:g} add-on service(s) subscribed"
    return f"{feat}: {val}"
