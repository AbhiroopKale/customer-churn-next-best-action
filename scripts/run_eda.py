"""EDA on the TRAINING split only (so rule thresholds chosen from EDA never peek at test data).
Each figure answers one business question.  Run:  python -m scripts.run_eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.data_preprocessing import prepare, split

sns.set_theme(style="whitegrid")
OUT = config.FIGURES
OUT.mkdir(parents=True, exist_ok=True)
lines = []


def note(s):
    print(s); lines.append(s)


def rate_bar(df, col, title, fname, order=None, rotate=0):
    g = df.groupby(col)[config.TARGET].agg(["mean", "count"]).mul([100, 1])
    if order:
        g = g.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(g.index.astype(str), g["mean"], color=sns.color_palette("Blues_d", len(g)))
    for b, (_, r) in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{r['mean']:.1f}%\n(n={int(r['count'])})", ha="center", fontsize=8)
    ax.axhline(df[config.TARGET].mean() * 100, color="red", ls="--", lw=1, label="Overall churn rate")
    ax.set(title=title, ylabel="Churn rate (%)", xlabel="", ylim=(0, g["mean"].max() * 1.25))
    ax.legend(); plt.xticks(rotation=rotate)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=140); plt.close(fig)
    note(f"{col}: " + ", ".join(f"{i}={r['mean']:.1f}% (n={int(r['count'])})" for i, r in g.iterrows()))


def main():
    df = prepare()
    X_train, _, y_train, _ = split(df)
    tr = X_train.assign(**{config.TARGET: y_train})
    note(f"EDA on training split: {len(tr)} customers")

    # Q1 How big is the churn problem?
    rate = tr[config.TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Stayed", "Churned"], [100 - rate, rate], color=["#4C72B0", "#C44E52"])
    for i, v in enumerate([100 - rate, rate]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    ax.set(title="Overall churn rate (imbalanced classes)", ylabel="% of customers", ylim=(0, 100))
    fig.tight_layout(); fig.savefig(OUT / "01_churn_overall.png", dpi=140); plt.close(fig)
    note(f"Overall churn rate: {rate:.1f}%  -> imbalanced, accuracy alone would mislead")

    # Q2 Does contract type matter?
    rate_bar(tr, "Contract", "Churn rate by contract type", "02_churn_by_contract.png",
             ["Month-to-month", "One year", "Two year"])
    # Q3 Do new customers leave more?
    rate_bar(tr, "tenure_group", "Churn rate by tenure group", "03_churn_by_tenure_group.png",
             config.TENURE_LABELS)
    # Q4 Are churners paying more?
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(data=tr, x="MonthlyCharges", hue=tr[config.TARGET].map({0: "Stayed", 1: "Churned"}),
                 element="step", stat="density", common_norm=False, ax=ax)
    ax.set(title="Monthly charges: churned vs stayed", xlabel="Monthly charges")
    fig.tight_layout(); fig.savefig(OUT / "04_monthly_charges_by_churn.png", dpi=140); plt.close(fig)
    note("Median MonthlyCharges  churned: %.2f | stayed: %.2f" % (
        tr.loc[tr.Churn == 1, "MonthlyCharges"].median(), tr.loc[tr.Churn == 0, "MonthlyCharges"].median()))
    rate_bar(tr, "charge_category", "Churn rate by monthly-charge band (<35 / 35-90 / >=90)",
             "04b_churn_by_charge_band.png", ["Low", "Medium", "High"])
    # Q5 Payment method / internet service
    rate_bar(tr, "PaymentMethod", "Churn rate by payment method", "05_churn_by_payment_method.png", rotate=15)
    rate_bar(tr, "InternetService", "Churn rate by internet service", "06_churn_by_internet_service.png")

    # Q6 Demographics
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, col in zip(axes, ["gender", "SeniorCitizen", "Partner", "Dependents"]):
        g = tr.groupby(col)[config.TARGET].mean().mul(100)
        ax.bar(g.index.astype(str), g.values, color="#4C72B0")
        for i, v in enumerate(g.values):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set(title=col, ylabel="Churn rate (%)", ylim=(0, 50))
    fig.suptitle("Churn rate by demographics"); fig.tight_layout()
    fig.savefig(OUT / "07_churn_by_demographics.png", dpi=140); plt.close(fig)
    for col in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
        g = tr.groupby(col)[config.TARGET].mean().mul(100)
        note(f"{col}: " + ", ".join(f"{i}={v:.1f}%" for i, v in g.items()))

    # Q7 Do add-on services relate to churn? (internet customers only)
    net = tr[tr["InternetService"] != "No"]
    res = {c: net[net[c] == v][config.TARGET].mean() * 100 for c in config.ADDON_COLUMNS for v in ["Yes"]}
    res_no = {c: net[net[c] == "No"][config.TARGET].mean() * 100 for c in config.ADDON_COLUMNS}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(config.ADDON_COLUMNS))
    ax.bar(x - 0.2, [res[c] for c in config.ADDON_COLUMNS], 0.4, label="Subscribed", color="#4C72B0")
    ax.bar(x + 0.2, [res_no[c] for c in config.ADDON_COLUMNS], 0.4, label="Not subscribed", color="#C44E52")
    ax.set_xticks(x, config.ADDON_COLUMNS, rotation=20)
    ax.set(title="Churn rate by add-on service (internet customers only)", ylabel="Churn rate (%)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "08_churn_by_addon_services.png", dpi=140); plt.close(fig)
    note("Add-on churn rate (subscribed vs not): " + ", ".join(
        f"{c}={res[c]:.1f}% vs {res_no[c]:.1f}%" for c in config.ADDON_COLUMNS))

    # Q8 Relationships between key features
    pivot = tr.pivot_table(index="Contract", columns="tenure_group", values=config.TARGET,
                           aggfunc="mean").reindex(columns=config.TENURE_LABELS) * 100
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds", ax=ax, cbar_kws={"label": "Churn rate (%)"})
    ax.set(title="Churn rate (%): contract x tenure group")
    fig.tight_layout(); fig.savefig(OUT / "09_contract_x_tenure_heatmap.png", dpi=140); plt.close(fig)
    corr = tr[["tenure", "MonthlyCharges", "TotalCharges", "n_addons", config.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Correlation of numeric features with churn")
    fig.tight_layout(); fig.savefig(OUT / "10_numeric_correlation.png", dpi=140); plt.close(fig)
    note("corr(TotalCharges, tenure) = %.2f  -> TotalCharges is largely a function of tenure x charges" %
         corr.loc["TotalCharges", "tenure"])

    note("Distribution percentiles of MonthlyCharges (train): " +
         str(tr["MonthlyCharges"].quantile([.25, .5, .75]).round(2).to_dict()))
    (config.ROOT / "reports" / "eda_summary.txt").write_text("\n".join(lines))
    print(f"\nFigures saved to {OUT}")


if __name__ == "__main__":
    main()
