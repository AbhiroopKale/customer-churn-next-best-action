"""Score ALL customers -> probability, risk level, SHAP factors, Next Best Action.
Writes CSVs to data/processed/ (loaded into PostgreSQL by scripts/load_to_postgres.py).

Run:  python -m scripts.run_batch_scoring
"""
import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config
from src.explainability import explain, humanize_factor
from src.recommendation_engine import recommend, risk_level


def main():
    df = pd.read_csv(config.PROCESSED / "clean_customers.csv")
    pipe = joblib.load(config.MODELS / "final_model.joblib")
    meta = json.loads((config.MODELS / "meta.json").read_text())
    background = pd.read_csv(config.MODELS / "background.csv")
    X, y = df[config.FEATURES], df[config.TARGET]

    # Out-of-fold probabilities: every customer is scored by a model that did NOT see them,
    # so dashboard numbers are not inflated by the model memorising its training rows.
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    proba = cross_val_predict(clone(pipe), X, y, cv=skf, method="predict_proba")[:, 1]

    # Explanations come from the final model (trained on the training split)
    exp = explain(pipe, X, background[config.FEATURES], k=5)
    exp["label"] = exp["feature"].map(humanize_factor)
    exp_by_row = {i: g for i, g in exp.groupby("row")}

    preds = pd.DataFrame({
        "customer_id": df[config.ID_COL],
        "churn_probability": proba.round(4),
        "risk_level": [risk_level(p) for p in proba],
        "predicted_churn": proba >= meta["decision_threshold"],
    })

    def top_text(i):
        g = exp_by_row[i]
        return "; ".join(g[g["direction"] == "increases risk"]["label"].head(3))

    # Retention actions only make sense for CURRENT customers (Churn == No).
    # Customers with Churn == Yes have already left; their scores only validate the model.
    recs = []
    records = X.to_dict("records")
    for i in df.index[df[config.TARGET] == 0]:
        factors = exp_by_row[i][["feature", "contribution"]].to_dict("records")
        r = recommend(records[i], float(proba[i]), factors)
        recs.append({
            "customer_id": df.at[i, config.ID_COL], "risk_level": r["risk_level"],
            "segment": r["segment"], "risk_factors": top_text(i),
            "recommended_action": r["next_best_action"],
            "secondary_actions": "; ".join(r["secondary_actions"]),
            "rule_ids": ",".join(r["rule_ids"]), "method": r["method"],
            "revenue_at_risk_proxy": r["revenue_at_risk_proxy"],
        })
    recs = pd.DataFrame(recs)

    exp_out = exp[["row", "feature", "label", "contribution", "direction"]].rename(columns={"label": "feature_label"})
    exp_out["customer_id"] = df[config.ID_COL].iloc[exp_out["row"]].values
    exp_out = exp_out.drop(columns="row")[["customer_id", "feature", "feature_label", "contribution", "direction"]]

    out = config.PROCESSED
    preds.to_csv(out / "predictions.csv", index=False)
    exp_out.to_csv(out / "explanations.csv", index=False)
    recs.to_csv(out / "recommendations.csv", index=False)

    # Flat file for Power BI if you prefer CSV over PostgreSQL
    flat = (df.rename(columns={config.ID_COL: "customer_id", config.TARGET: "churn_actual"})
              .merge(preds, on="customer_id")
              .merge(recs.drop(columns=["risk_level"]), on="customer_id", how="left"))
    flat["is_active"] = flat["churn_actual"] == 0
    flat.to_csv(out / "powerbi_customer_risk.csv", index=False)

    print("Risk level counts (all customers):")
    print(preds["risk_level"].value_counts().to_string())
    print(f"\nFlagged at threshold {meta['decision_threshold']}: {int(preds['predicted_churn'].sum())}")
    act = recs
    print(f"\nActive customers with recommendations: {len(act)}")
    print(act["risk_level"].value_counts().to_string())
    print("\nSample HIGH-risk recommendations:")
    print(act[act.risk_level == "HIGH"].head(5)[
        ["customer_id", "segment", "risk_factors", "recommended_action"]].to_string(index=False))
    print("\nMean probability  churned:", round(preds.loc[y == 1, "churn_probability"].mean(), 3),
          "| stayed:", round(preds.loc[y == 0, "churn_probability"].mean(), 3))


if __name__ == "__main__":
    main()
