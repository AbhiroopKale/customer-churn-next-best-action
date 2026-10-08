"""Compare imbalance strategies with cross-validation on the TRAINING split:
none vs class_weight vs SMOTE (SMOTE is applied INSIDE each training fold only).
Run:  python -m scripts.imbalance_experiment"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import prepare, split
from src.evaluate_models import compute_metrics
from src.train_models import build_preprocessor


def main():
    X_train, _, y_train, _ = split(prepare())
    pos_w = float((y_train == 0).sum() / (y_train == 1).sum())
    rs = config.RANDOM_STATE
    makers = {
        "Logistic Regression": lambda cw, w: LogisticRegression(max_iter=2000, class_weight=cw),
        "Random Forest": lambda cw, w: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                              class_weight=cw, random_state=rs),
        "XGBoost": lambda cw, w: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, eval_metric="logloss",
                                               scale_pos_weight=w, random_state=rs),
    }
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=rs)
    rows = []
    for name, mk in makers.items():
        for strat in ["none", "class_weight", "smote"]:
            cw = "balanced" if strat == "class_weight" else None
            w = pos_w if strat == "class_weight" else 1.0
            steps = [("prep", build_preprocessor())]
            if strat == "smote":
                steps.append(("smote", SMOTE(random_state=rs)))
            steps.append(("model", mk(cw, w)))
            oof = cross_val_predict(ImbPipeline(steps), X_train, y_train, cv=skf, method="predict_proba")[:, 1]
            m = compute_metrics(y_train, oof, 0.5)
            rows.append({"Model": name, "Strategy": strat, "ROC-AUC": m["ROC-AUC"], "PR-AUC": m["PR-AUC"],
                         "Precision@0.5": m["Precision"], "Recall@0.5": m["Recall"], "F1@0.5": m["F1"],
                         "Brier (lower=better calibrated)": m["Brier"]})
            print(rows[-1])
    out = pd.DataFrame(rows).round(4)
    out.to_csv(config.MODELS / "imbalance_experiment.csv", index=False)
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
