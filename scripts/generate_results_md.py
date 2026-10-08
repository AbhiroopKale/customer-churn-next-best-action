"""Collect REAL numbers produced by the pipeline into reports/RESULTS.md (paste into your report).
Run:  python -m scripts.generate_results_md"""
import json

import pandas as pd

from src import config


def main():
    meta = json.loads((config.MODELS / "meta.json").read_text())
    comp = pd.read_csv(config.MODELS / "model_comparison.csv")
    preds = pd.read_csv(config.PROCESSED / "predictions.csv")
    recs = pd.read_csv(config.PROCESSED / "recommendations.csv")
    out = ["# Results (auto-generated from the actual pipeline run)\n"]
    out.append(f"Selected model: **{meta['model_name']}** (`{meta['model_version']}`), decision threshold "
               f"{meta['decision_threshold']:.2f}. Rule: {meta['selection_rule']}.\n")
    cols = ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "CV PR-AUC"]
    out.append("## Model comparison (held-out test set, each model at its own tuned threshold)\n")
    out.append(comp[cols].to_string(index=False) + "\n")
    out.append("## Same models at the default 0.5 threshold\n")
    out.append(comp[["Model", "Precision@0.5", "Recall@0.5", "F1@0.5"]].to_string(index=False) + "\n")
    f = config.MODELS / "imbalance_experiment.csv"
    if f.exists():
        out.append("## Imbalance strategies (5-fold CV on training data)\n")
        out.append(pd.read_csv(f).to_string(index=False) + "\n")
    out.append("## Risk levels, all customers\n")
    out.append(preds["risk_level"].value_counts().to_string() + "\n")
    out.append("## Risk levels, active customers (retention targets)\n")
    out.append(recs["risk_level"].value_counts().to_string() + "\n")
    out.append("## Next Best Action counts (HIGH risk)\n")
    out.append(recs[recs.risk_level == "HIGH"]["recommended_action"].value_counts().to_string() + "\n")
    eda = config.ROOT / "reports" / "eda_summary.txt"
    if eda.exists():
        out.append("## EDA findings (training split)\n```\n" + eda.read_text() + "\n```\n")
    (config.ROOT / "reports" / "RESULTS.md").write_text("\n".join(out))
    print("Wrote reports/RESULTS.md")


if __name__ == "__main__":
    main()
