"""Train and compare 5 models, select the deployable one, save everything.

Run from the project root:   python -m src.train_models
"""
import json
import sys
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import prepare, split
from src.evaluate_models import best_f1_threshold, compute_metrics

ANN_NAME = "ANN (Keras MLP)"


def build_preprocessor() -> ColumnTransformer:
    """Fitted only on training data (inside the Pipeline, also per CV fold)."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", numeric, config.NUMERIC_FEATURES),
                              ("cat", categorical, config.CATEGORICAL_FEATURES)])


def get_models() -> dict:
    """No class_weight here on purpose: re-weighting classes distorts the predicted
    probabilities, and we need honest probabilities for risk bands. Imbalance is handled
    by threshold tuning + PR-based metrics. See scripts/imbalance_experiment.py for the
    comparison with class weights and SMOTE."""
    rs = config.RANDOM_STATE
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                n_jobs=-1, random_state=rs),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=rs),
    }
    try:
        from src.ann_model import KerasANN
        import tensorflow  # noqa: F401
        models[ANN_NAME] = KerasANN(random_state=rs)
    except Exception as e:  # TensorFlow missing: continue without ANN, but say so
        print(f"[WARN] ANN skipped (TensorFlow not available): {e}")
    return models


def main():
    config.MODELS.mkdir(exist_ok=True)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)

    df = prepare()
    X_train, X_test, y_train, y_test = split(df)
    print(f"Rows: {len(df)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Churn rate  full: {df[config.TARGET].mean():.3f} | "
          f"train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows, fitted, test_probs, thresholds = [], {}, {"y_true": y_test.values}, {}

    for name, clf in get_models().items():
        print(f"\n--- {name} ---")
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
        # Out-of-fold probabilities on TRAIN only -> model selection + threshold tuning
        oof = cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
        cv = compute_metrics(y_train, oof, 0.5)
        thr = best_f1_threshold(y_train, oof)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        tuned = compute_metrics(y_test, proba, thr)
        default = compute_metrics(y_test, proba, 0.5)
        print(f"CV PR-AUC {cv['PR-AUC']:.4f} | tuned threshold {thr:.2f} | "
              f"test F1 {tuned['F1']:.4f}, recall {tuned['Recall']:.4f}")
        rows.append({
            "Model": name, "Threshold": thr,
            "Accuracy": tuned["Accuracy"], "Precision": tuned["Precision"],
            "Recall": tuned["Recall"], "F1": tuned["F1"],
            "ROC-AUC": tuned["ROC-AUC"], "PR-AUC": tuned["PR-AUC"], "Brier": tuned["Brier"],
            "TP": tuned["TP"], "FP": tuned["FP"], "FN": tuned["FN"], "TN": tuned["TN"],
            "Precision@0.5": default["Precision"], "Recall@0.5": default["Recall"],
            "F1@0.5": default["F1"],
            "CV ROC-AUC": cv["ROC-AUC"], "CV PR-AUC": cv["PR-AUC"],
        })
        test_probs[name] = proba
        thresholds[name] = thr
        if name != ANN_NAME:
            fitted[name] = pipe

    results = pd.DataFrame(rows).round(4)
    results.to_csv(config.MODELS / "model_comparison.csv", index=False)
    pd.DataFrame(test_probs).to_csv(config.MODELS / "test_probabilities.csv", index=False)

    show = ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print("\n=== MODEL COMPARISON (test set, at each model's tuned threshold) ===")
    print(results[show].to_string(index=False))
    print("\n=== Same models at the default 0.5 threshold ===")
    print(results[["Model", "Precision@0.5", "Recall@0.5", "F1@0.5"]].to_string(index=False))

    # Selection among DEPLOYABLE models (the ANN is compared but not deployed because
    # SHAP explanations + a single saved Pipeline are required for the app).
    deployable = results[results["Model"] != ANN_NAME].sort_values("CV PR-AUC", ascending=False)
    best = deployable.iloc[0]
    best_name = best["Model"]
    print(f"\nSELECTED: {best_name} (highest cross-validated PR-AUC on training data "
          f"among deployable models)")
    if ANN_NAME in results["Model"].values:
        ann = results[results["Model"] == ANN_NAME].iloc[0]
        print(f"ANN CV PR-AUC {ann['CV PR-AUC']:.4f} vs selected {best['CV PR-AUC']:.4f}")

    final = fitted[best_name]
    version = f"v1-{best_name.lower().replace(' ', '-')}-{datetime.now():%Y%m%d}"
    meta = {
        "model_version": version, "model_name": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "decision_threshold": float(best["Threshold"]),
        "test_metrics": {k: float(best[k]) for k in
                         ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier"]},
        "cv_pr_auc": float(best["CV PR-AUC"]),
        "selection_rule": "highest 5-fold CV PR-AUC on training data among deployable models; "
                          "threshold = F1-optimal on out-of-fold training predictions",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "churn_rate_train": float(y_train.mean()),
        "feature_ranges": {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max())}
                           for c in ["tenure", "MonthlyCharges", "TotalCharges"]},
        "categories": {c: sorted(X_train[c].unique().tolist()) for c in config.CATEGORICAL_FEATURES},
        "ann_included": ANN_NAME in results["Model"].values,
        "sklearn_version": sklearn.__version__, "python_version": sys.version.split()[0],
    }
    (config.MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    joblib.dump(final, config.MODELS / "final_model.joblib")
    joblib.dump(fitted, config.MODELS / "all_models.joblib")
    X_train.sample(200, random_state=config.RANDOM_STATE).to_csv(config.MODELS / "background.csv", index=False)
    X_test.assign(Churn=y_test.values).to_csv(config.PROCESSED / "test_set.csv", index=False)
    df.to_csv(config.PROCESSED / "clean_customers.csv", index=False)
    print(f"\nSaved final model ({version}) and metadata to {config.MODELS}")


if __name__ == "__main__":
    main()
