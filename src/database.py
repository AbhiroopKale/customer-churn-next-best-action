"""All PostgreSQL code lives here. Credentials come from .env (never hard-code them)."""
import os
import uuid

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from src import config

load_dotenv(config.ROOT / ".env")

# CSV/feature column -> database column
CUSTOMER_MAP = {
    "gender": "gender", "SeniorCitizen": "senior_citizen", "Partner": "partner",
    "Dependents": "dependents", "tenure": "tenure", "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines", "InternetService": "internet_service",
    "OnlineSecurity": "online_security", "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection", "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv", "StreamingMovies": "streaming_movies",
    "Contract": "contract", "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method", "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges", "tenure_group": "tenure_group",
    "charge_category": "charge_category", "n_addons": "n_addons",
}


def _url(database=None):
    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("PGUSER", "postgres"), password=os.getenv("PGPASSWORD", ""),
        host=os.getenv("PGHOST", "localhost"), port=int(os.getenv("PGPORT", "5432")),
        database=database or os.getenv("PGDATABASE", "churn_nba"))


def get_engine():
    return create_engine(_url(), pool_pre_ping=True)


def ping(engine):
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def create_database_if_missing():
    name = os.getenv("PGDATABASE", "churn_nba")
    admin = create_engine(_url("postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
            print(f"Created database {name}")
        else:
            print(f"Database {name} already exists")
    admin.dispose()


def init_schema(engine):
    with engine.begin() as conn:
        for f in ("schema.sql", "views.sql"):
            conn.exec_driver_sql((config.SQL_DIR / f).read_text())


def _py(v):
    """numpy -> plain Python (psycopg2 cannot adapt numpy.int64)."""
    if hasattr(v, "item"):
        return v.item()
    return None if (isinstance(v, float) and v != v) else v


def _register_model(conn, meta):
    m = meta["test_metrics"]
    conn.execute(text("""
        INSERT INTO model_registry (model_version, model_name, trained_at, decision_threshold,
                                    roc_auc, pr_auc, recall_score, precision_score, f1_score, notes)
        VALUES (:v, :n, :t, :thr, :roc, :pr, :rec, :prec, :f1, :notes)
        ON CONFLICT (model_version) DO NOTHING"""),
        {"v": meta["model_version"], "n": meta["model_name"], "t": meta["trained_at"],
         "thr": meta["decision_threshold"], "roc": m["ROC-AUC"], "pr": m["PR-AUC"],
         "rec": m["Recall"], "prec": m["Precision"], "f1": m["F1"], "notes": meta["selection_rule"]})


def load_batch(engine, customers, preds, expl, recs, meta):
    """Replace all 'batch' rows (live-app rows are kept). Safe to re-run."""
    cust = customers.rename(columns={config.ID_COL: "customer_id", config.TARGET: "churn_actual",
                                     **CUSTOMER_MAP})
    cust = cust[["customer_id"] + list(CUSTOMER_MAP.values()) + ["churn_actual"]].copy()
    cust["source"] = "batch"
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM churn_predictions WHERE source = 'batch'"))  # cascades
        conn.execute(text("DELETE FROM customers WHERE source = 'batch'"))
        _register_model(conn, meta)
        cust.to_sql("customers", conn, if_exists="append", index=False, method="multi", chunksize=1000)

        p = preds.copy()
        p["model_version"] = meta["model_version"]
        p["source"] = "batch"
        p.to_sql("churn_predictions", conn, if_exists="append", index=False, method="multi", chunksize=1000)
        ids = pd.read_sql(text("SELECT prediction_id, customer_id FROM churn_predictions "
                               "WHERE source = 'batch' AND model_version = :v"), conn,
                          params={"v": meta["model_version"]})

        e = expl.merge(ids, on="customer_id")[["prediction_id", "feature", "feature_label", "contribution", "direction"]]
        e.to_sql("explanations", conn, if_exists="append", index=False, method="multi", chunksize=1000)
        r = recs.merge(ids, on="customer_id")
        r = r[["prediction_id", "customer_id", "risk_level", "segment", "risk_factors", "recommended_action",
               "secondary_actions", "rule_ids", "method", "revenue_at_risk_proxy"]]
        r.to_sql("recommendations", conn, if_exists="append", index=False, method="multi", chunksize=1000)
    return {"customers": len(cust), "predictions": len(p), "explanations": len(e), "recommendations": len(r)}


def save_live_prediction(engine, result: dict) -> str:
    """Store one Streamlit prediction (customer, prediction, explanations, recommendation)."""
    c, rec = result["customer"], result["recommendation"]
    cid = "LIVE-" + uuid.uuid4().hex[:8].upper()
    cust = {db: _py(c[col]) for col, db in CUSTOMER_MAP.items()}
    cust.update(customer_id=cid, churn_actual=None, source="live")
    cols = ", ".join(cust)
    marks = ", ".join(f":{k}" for k in cust)
    with engine.begin() as conn:
        _register_model(conn, result["meta"])
        conn.execute(text(f"INSERT INTO customers ({cols}) VALUES ({marks})"), cust)
        pid = conn.execute(text("""
            INSERT INTO churn_predictions (customer_id, churn_probability, risk_level, predicted_churn,
                                           model_version, source)
            VALUES (:c, :p, :r, :pc, :mv, 'live') RETURNING prediction_id"""),
            {"c": cid, "p": round(result["probability"], 4), "r": result["risk_level"],
             "pc": bool(result["predicted_churn"]), "mv": result["model_version"]}).scalar_one()
        for _, f in result["factors"].iterrows():
            conn.execute(text("""INSERT INTO explanations (prediction_id, feature, feature_label, contribution, direction)
                                 VALUES (:p, :f, :l, :c, :d)"""),
                         {"p": pid, "f": f["feature"], "l": f["label"], "c": float(f["contribution"]),
                          "d": f["direction"]})
        conn.execute(text("""
            INSERT INTO recommendations (prediction_id, customer_id, risk_level, segment, risk_factors,
                                         recommended_action, secondary_actions, rule_ids, method,
                                         revenue_at_risk_proxy)
            VALUES (:p, :c, :r, :s, :rf, :a, :sa, :ri, :m, :rv)"""),
            {"p": pid, "c": cid, "r": rec["risk_level"], "s": rec["segment"],
             "rf": result["risk_factors_text"], "a": rec["next_best_action"],
             "sa": "; ".join(rec["secondary_actions"]), "ri": ",".join(rec["rule_ids"]),
             "m": rec["method"], "rv": rec["revenue_at_risk_proxy"]})
    return cid


def recent_live_predictions(engine, limit=15) -> pd.DataFrame:
    q = text("""SELECT p.customer_id, p.churn_probability, p.risk_level, r.recommended_action,
                       p.prediction_date
                FROM churn_predictions p LEFT JOIN recommendations r ON r.prediction_id = p.prediction_id
                WHERE p.source = 'live' ORDER BY p.prediction_id DESC LIMIT :n""")
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"n": limit})
