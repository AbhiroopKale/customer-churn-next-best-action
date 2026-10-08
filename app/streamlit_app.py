"""Streamlit demo:  Predict -> Explain -> Recommend -> Act
Run from the project root:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config, database
from src.prediction import RAW_INPUTS, load_artifacts, predict_customer

st.set_page_config(page_title="Churn Prediction & Next Best Action", page_icon="📉", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_engine():
    return database.get_engine()   # creating an engine does not connect yet


def db_status(engine):
    try:
        database.ping(engine)
        return True
    except Exception:
        return False


try:
    pipe, meta, _ = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

cats = meta["categories"]
rng = meta["feature_ranges"]
engine = get_engine()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📉 Churn & NBA")
    st.caption("Predict → Explain → Recommend → Act")
    st.markdown(f"**Model:** {meta['model_name']}  \n**Version:** `{meta['model_version']}`  \n"
                f"**Decision threshold:** {meta['decision_threshold']:.2f}")
    db_ok = db_status(engine)
    save_to_db = st.checkbox("Save prediction to PostgreSQL", value=db_ok, disabled=not db_ok)
    st.caption("🟢 PostgreSQL connected" if db_ok else
               "🔴 PostgreSQL not reachable (check .env, run scripts.setup_database). "
               "Predictions still work; they just won't be saved.")
    st.divider()
    st.caption(f"Risk bands: LOW < {config.RISK_LOW_BELOW:.0%} ≤ MEDIUM < {config.RISK_HIGH_FROM:.0%} ≤ HIGH "
               "(configurable in src/config.py)")

PRESETS = {
    "Custom (defaults)": dict(tenure=24, Contract="One year", MonthlyCharges=65.0, InternetService="DSL",
                              PhoneService="Yes", MultipleLines="No", OnlineSecurity="No", OnlineBackup="No",
                              DeviceProtection="No", TechSupport="No", StreamingTV="No", StreamingMovies="No",
                              PaymentMethod="Mailed check", PaperlessBilling="No", gender="Female",
                              SeniorCitizen=0, Partner="Yes", Dependents="No"),
    "Example: new fiber customer, month-to-month": dict(
        tenure=4, Contract="Month-to-month", MonthlyCharges=95.0, InternetService="Fiber optic",
        PhoneService="Yes", MultipleLines="No", OnlineSecurity="No", OnlineBackup="No",
        DeviceProtection="No", TechSupport="No", StreamingTV="Yes", StreamingMovies="Yes",
        PaymentMethod="Electronic check", PaperlessBilling="Yes", gender="Male",
        SeniorCitizen=0, Partner="No", Dependents="No"),
    "Example: long-term two-year customer": dict(
        tenure=60, Contract="Two year", MonthlyCharges=55.0, InternetService="DSL",
        PhoneService="Yes", MultipleLines="No", OnlineSecurity="Yes", OnlineBackup="Yes",
        DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No", StreamingMovies="No",
        PaymentMethod="Credit card (automatic)", PaperlessBilling="No", gender="Female",
        SeniorCitizen=0, Partner="Yes", Dependents="Yes"),
}

tab_predict, tab_models, tab_recent = st.tabs(["🔮 Predict", "📊 Model comparison", "🗄️ Saved predictions"])

# ---------------------------------------------------------------- predict tab
with tab_predict:
    st.header("Customer churn prediction & Next Best Action")
    preset_name = st.selectbox("Start from a preset (optional)", list(PRESETS))
    d = PRESETS[preset_name]
    k = lambda name: f"{name}|{preset_name}"          # new widget keys when preset changes

    def pick(label, col, default, options=None):
        options = options or cats[col]
        return st.selectbox(label, options, index=options.index(default), key=k(col))

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Account")
            tenure = st.number_input("Tenure (months)", 0, int(rng["tenure"]["max"]), d["tenure"], key=k("tenure"))
            contract = pick("Contract", "Contract", d["Contract"])
            monthly = st.number_input(
                f"Monthly charges (dataset range {rng['MonthlyCharges']['min']:.0f}-{rng['MonthlyCharges']['max']:.0f})",
                0.0, 300.0, float(d["MonthlyCharges"]), step=0.5, key=k("MonthlyCharges"))
            payment = pick("Payment method", "PaymentMethod", d["PaymentMethod"])
            paperless = pick("Paperless billing", "PaperlessBilling", d["PaperlessBilling"])
        with c2:
            st.subheader("Services")
            internet = pick("Internet service", "InternetService", d["InternetService"])
            phone = pick("Phone service", "PhoneService", d["PhoneService"])
            lines = pick("Multiple lines", "MultipleLines", d["MultipleLines"], ["No", "Yes"])
            sec = pick("Online security", "OnlineSecurity", d["OnlineSecurity"], ["No", "Yes"])
            backup = pick("Online backup", "OnlineBackup", d["OnlineBackup"], ["No", "Yes"])
        with c3:
            st.subheader("Add-ons & demographics")
            dev = pick("Device protection", "DeviceProtection", d["DeviceProtection"], ["No", "Yes"])
            tech = pick("Tech support", "TechSupport", d["TechSupport"], ["No", "Yes"])
            stv = pick("Streaming TV", "StreamingTV", d["StreamingTV"], ["No", "Yes"])
            smov = pick("Streaming movies", "StreamingMovies", d["StreamingMovies"], ["No", "Yes"])
            gender = pick("Gender", "gender", d["gender"])
            senior = st.selectbox("Senior citizen", [0, 1], index=d["SeniorCitizen"], key=k("SeniorCitizen"),
                                  format_func=lambda v: "Yes" if v else "No")
            partner = pick("Partner", "Partner", d["Partner"])
            deps = pick("Dependents", "Dependents", d["Dependents"])
        st.caption("Add-ons are ignored when there is no internet service, and 'multiple lines' when there is "
                   "no phone service. Total charges are ESTIMATED as tenure × monthly charges.")
        submitted = st.form_submit_button("PREDICT CHURN", type="primary", width="stretch")

    if submitted:
        raw = dict(gender=gender, SeniorCitizen=int(senior), Partner=partner, Dependents=deps, tenure=int(tenure),
                   PhoneService=phone, MultipleLines=lines, InternetService=internet, OnlineSecurity=sec,
                   OnlineBackup=backup, DeviceProtection=dev, TechSupport=tech, StreamingTV=stv,
                   StreamingMovies=smov, Contract=contract, PaperlessBilling=paperless, PaymentMethod=payment,
                   MonthlyCharges=float(monthly), TotalCharges=None)
        lo, hi = rng["MonthlyCharges"]["min"], rng["MonthlyCharges"]["max"]
        if not lo <= monthly <= hi:
            st.warning(f"Monthly charges {monthly:g} is outside the range the model was trained on "
                       f"({lo:g}-{hi:g}). The prediction is an extrapolation and should not be trusted.")
        try:
            res = predict_customer(raw)
        except ValueError as e:
            st.error(str(e)); st.stop()

        p, level, rec = res["probability"], res["risk_level"], res["recommendation"]
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{p:.0%}")
        m2.metric("Risk level", level)
        m3.metric("Flagged as likely churner", "Yes" if res["predicted_churn"] else "No",
                  help=f"Probability ≥ decision threshold {res['threshold']:.2f}")
        st.progress(min(max(p, 0.0), 1.0))
        {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.success}[level](
            f"{p:.0%} probability of churn → **{level}** risk")

        left, right = st.columns(2)
        with left:
            st.subheader("WHY? Main factors in the model's prediction")
            f = res["factors"]
            fig, ax = plt.subplots(figsize=(6, 3.2))
            f_plot = f.iloc[::-1]
            ax.barh(f_plot["label"], f_plot["contribution"],
                    color=["#C44E52" if v > 0 else "#4C72B0" for v in f_plot["contribution"]])
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("← lowers risk | raises risk →  (SHAP contribution)")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            for _, r in f[f["direction"] == "increases risk"].head(3).iterrows():
                st.markdown(f"- 🔺 {r['label']}")
            st.caption("SHAP shows how the MODEL uses each feature for this customer. It explains the model, "
                       "it does not prove these factors cause churn.")
        with right:
            st.subheader("NEXT BEST ACTION")
            st.info(f"**{rec['next_best_action']}**")
            if rec["secondary_actions"]:
                st.markdown("**Also consider:**")
                for a in rec["secondary_actions"]:
                    st.markdown(f"- {a}")
            st.markdown(f"**Segment:** {rec['segment']}  \n"
                        f"**Expected monthly revenue at risk (proxy = probability × monthly charges):** "
                        f"{rec['revenue_at_risk_proxy']:.2f}  \n"
                        f"**Rules fired:** {', '.join(rec['rule_ids']) or 'none'}")
            st.caption(f"Method: {rec['method']}. This is a recommendation, not a guarantee of retention; "
                       "its effect on churn has not been measured.")

        if save_to_db:
            try:
                cid = database.save_live_prediction(engine, res)
                st.success(f"Saved to PostgreSQL as customer **{cid}**")
            except Exception as e:
                st.error(f"Could not save to PostgreSQL: {e}")

# ---------------------------------------------------------------- model comparison tab
with tab_models:
    st.header("Model comparison (held-out test set)")
    comp_path = config.MODELS / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        cols = ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "CV PR-AUC"]
        st.dataframe(comp[cols].style.format({c: "{:.3f}" for c in cols[1:]}), width="stretch", hide_index=True)
        st.caption(f"Selected: **{meta['model_name']}** — {meta['selection_rule']}. Metrics are at each model's "
                   "own tuned threshold (tuned on training data only).")
        for img, cap in [("11_roc_pr_curves.png", "ROC and Precision-Recall curves"),
                         ("12_confusion_matrices.png", "Confusion matrices"),
                         ("14_shap_global_bar.png", "Global SHAP importance")]:
            if (config.FIGURES / img).exists():
                st.image(str(config.FIGURES / img), caption=cap)
    else:
        st.info("Run `python -m src.train_models` first.")

# ---------------------------------------------------------------- saved predictions tab
with tab_recent:
    st.header("Predictions saved by this app (PostgreSQL)")
    if db_ok:
        df = database.recent_live_predictions(engine)
        st.dataframe(df, width="stretch", hide_index=True) if len(df) else st.info("Nothing saved yet.")
    else:
        st.info("PostgreSQL is not connected.")
