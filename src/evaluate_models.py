"""Metrics, threshold tuning and evaluation plots."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, proba, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Pick the threshold that maximises F1. Call this ONLY on training-set
    out-of-fold probabilities, never on the test set."""
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def plot_roc_pr(test_probs: pd.DataFrame, path):
    import matplotlib.pyplot as plt
    y = test_probs["y_true"].values
    models = [c for c in test_probs.columns if c != "y_true"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for m in models:
        fpr, tpr, _ = roc_curve(y, test_probs[m])
        ax[0].plot(fpr, tpr, label=f"{m} (AUC={roc_auc_score(y, test_probs[m]):.3f})")
        p, r, _ = precision_recall_curve(y, test_probs[m])
        ax[1].plot(r, p, label=f"{m} (AP={average_precision_score(y, test_probs[m]):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(title="ROC curves (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(y.mean(), color="k", ls="--", lw=1, label=f"No-skill ({y.mean():.2f})")
    ax[1].set(title="Precision-Recall curves (test set)", xlabel="Recall", ylabel="Precision")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_confusion_matrices(test_probs: pd.DataFrame, thresholds: dict, path):
    import matplotlib.pyplot as plt
    y = test_probs["y_true"].values
    models = [c for c in test_probs.columns if c != "y_true"]
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.6))
    axes = np.atleast_1d(axes)
    for a, m in zip(axes, models):
        t = thresholds[m]
        cm = confusion_matrix(y, (test_probs[m] >= t).astype(int), labels=[0, 1])
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, v, ha="center", va="center",
                   color="white" if v > cm.max() / 2 else "black")
        a.set(title=f"{m}\nthreshold={t:.2f}", xticks=[0, 1], yticks=[0, 1],
              xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"],
              xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_calibration(test_probs: pd.DataFrame, path, bins=10):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    y = test_probs["y_true"].values
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in [c for c in test_probs.columns if c != "y_true"]:
        frac, mean_pred = calibration_curve(y, test_probs[m], n_bins=bins, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=m)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set(title="Calibration (do predicted probabilities match reality?)",
           xlabel="Mean predicted probability", ylabel="Observed churn rate")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
