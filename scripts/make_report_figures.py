"""Evaluation + SHAP figures for the report.  Run:  python -m scripts.make_report_figures"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src import config
from src.evaluate_models import plot_calibration, plot_confusion_matrices, plot_roc_pr
from src.explainability import shap_matrix


def main():
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    probs = pd.read_csv(config.MODELS / "test_probabilities.csv")
    comp = pd.read_csv(config.MODELS / "model_comparison.csv")
    thr = dict(zip(comp["Model"], comp["Threshold"]))
    plot_roc_pr(probs, config.FIGURES / "11_roc_pr_curves.png")
    plot_confusion_matrices(probs, thr, config.FIGURES / "12_confusion_matrices.png")
    plot_calibration(probs, config.FIGURES / "13_calibration.png")

    pipe = joblib.load(config.MODELS / "final_model.joblib")
    test = pd.read_csv(config.PROCESSED / "test_set.csv")
    background = pd.read_csv(config.MODELS / "background.csv")
    X = test[config.FEATURES]
    sv, Xt, names, is_cat, base = shap_matrix(pipe, X, background[config.FEATURES])

    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=names, plot_type="bar", max_display=15, show=False)
    plt.title("Global importance: mean |SHAP| (model behaviour, not causation)")
    plt.tight_layout(); plt.savefig(config.FIGURES / "14_shap_global_bar.png", dpi=140, bbox_inches="tight"); plt.close()
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=names, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(config.FIGURES / "15_shap_beeswarm.png", dpi=140, bbox_inches="tight"); plt.close()
    print("Figures saved to", config.FIGURES)


if __name__ == "__main__":
    main()
