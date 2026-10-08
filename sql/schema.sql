-- PostgreSQL schema for the Churn Prediction & Next Best Action system
-- Run via:  python -m scripts.setup_database   (or paste into pgAdmin's Query Tool)

CREATE TABLE IF NOT EXISTS customers (
    customer_id        VARCHAR(30) PRIMARY KEY,
    gender             VARCHAR(10),
    senior_citizen     SMALLINT,
    partner            VARCHAR(3),
    dependents         VARCHAR(3),
    tenure             INTEGER NOT NULL,
    phone_service      VARCHAR(3),
    multiple_lines     VARCHAR(20),
    internet_service   VARCHAR(20),
    online_security    VARCHAR(25),
    online_backup      VARCHAR(25),
    device_protection  VARCHAR(25),
    tech_support       VARCHAR(25),
    streaming_tv       VARCHAR(25),
    streaming_movies   VARCHAR(25),
    contract           VARCHAR(20),
    paperless_billing  VARCHAR(3),
    payment_method     VARCHAR(30),
    monthly_charges    NUMERIC(8,2) NOT NULL,
    total_charges      NUMERIC(10,2),
    tenure_group       VARCHAR(10),
    charge_category    VARCHAR(10),
    n_addons           SMALLINT,
    churn_actual       SMALLINT CHECK (churn_actual IN (0,1)),   -- NULL for live app customers
    source             VARCHAR(10) NOT NULL DEFAULT 'batch',     -- 'batch' or 'live'
    created_at         TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version      VARCHAR(80) PRIMARY KEY,
    model_name         VARCHAR(50),
    trained_at         TIMESTAMP,
    decision_threshold NUMERIC(5,4),
    roc_auc            NUMERIC(6,4),
    pr_auc             NUMERIC(6,4),
    recall_score       NUMERIC(6,4),
    precision_score    NUMERIC(6,4),
    f1_score           NUMERIC(6,4),
    notes              TEXT
);

CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id      SERIAL PRIMARY KEY,
    customer_id        VARCHAR(30) NOT NULL REFERENCES customers(customer_id),
    churn_probability  NUMERIC(6,4) NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    risk_level         VARCHAR(6) NOT NULL CHECK (risk_level IN ('LOW','MEDIUM','HIGH')),
    predicted_churn    BOOLEAN,
    model_version      VARCHAR(80) REFERENCES model_registry(model_version),
    prediction_date    TIMESTAMP DEFAULT now(),
    source             VARCHAR(10) NOT NULL DEFAULT 'batch'
);

CREATE TABLE IF NOT EXISTS explanations (
    explanation_id     SERIAL PRIMARY KEY,
    prediction_id      INTEGER NOT NULL REFERENCES churn_predictions(prediction_id) ON DELETE CASCADE,
    feature            VARCHAR(120),
    feature_label      VARCHAR(160),         -- business wording, e.g. 'Month-to-month contract'
    contribution       DOUBLE PRECISION,     -- SHAP value on the model's native scale (ranking only)
    direction          VARCHAR(20)           -- 'increases risk' / 'decreases risk'
);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id     SERIAL PRIMARY KEY,
    prediction_id         INTEGER NOT NULL REFERENCES churn_predictions(prediction_id) ON DELETE CASCADE,
    customer_id           VARCHAR(30) NOT NULL,
    risk_level            VARCHAR(6),
    segment               VARCHAR(40),
    risk_factors          TEXT,
    recommended_action    TEXT,
    secondary_actions     TEXT,
    rule_ids              VARCHAR(40),
    method                TEXT,              -- documents that this is rule-based, not learned
    revenue_at_risk_proxy NUMERIC(10,2),     -- probability x monthly charges (a PROXY, not CLV)
    recommendation_date   TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pred_customer ON churn_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_pred_risk     ON churn_predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_expl_pred     ON explanations(prediction_id);
CREATE INDEX IF NOT EXISTS idx_rec_pred      ON recommendations(prediction_id);
