-- Flat, dashboard-ready views for Power BI

CREATE OR REPLACE VIEW vw_customer_risk AS
SELECT p.prediction_id, p.customer_id, p.churn_probability, p.risk_level, p.predicted_churn,
       p.source, p.prediction_date, p.model_version,
       c.gender, c.senior_citizen, c.partner, c.dependents,
       c.contract, c.tenure, c.tenure_group, c.monthly_charges, c.total_charges,
       c.internet_service, c.payment_method, c.paperless_billing,
       c.online_security, c.online_backup, c.device_protection, c.tech_support,
       c.streaming_tv, c.streaming_movies, c.phone_service, c.multiple_lines,
       c.churn_actual,
       (c.churn_actual IS DISTINCT FROM 1) AS is_active,
       f.risk_factors
FROM churn_predictions p
JOIN customers c ON c.customer_id = p.customer_id
LEFT JOIN LATERAL (
    SELECT string_agg(t.lbl, '; ' ORDER BY t.contribution DESC) AS risk_factors
    FROM (SELECT COALESCE(e.feature_label, e.feature) AS lbl, e.contribution
          FROM explanations e
          WHERE e.prediction_id = p.prediction_id AND e.direction = 'increases risk'
          ORDER BY e.contribution DESC
          LIMIT 3) t
) f ON TRUE;

CREATE OR REPLACE VIEW vw_retention_actions AS
SELECT r.recommendation_id, r.customer_id, p.churn_probability, r.risk_level, r.segment,
       r.risk_factors, r.recommended_action, r.secondary_actions, r.rule_ids, r.method,
       r.revenue_at_risk_proxy, c.contract, c.tenure, c.monthly_charges, c.payment_method,
       c.internet_service, p.source, p.prediction_date
FROM recommendations r
JOIN churn_predictions p ON p.prediction_id = r.prediction_id
JOIN customers c ON c.customer_id = r.customer_id;

-- One row per (service, status) so Power BI can chart "churn by service" directly
CREATE OR REPLACE VIEW vw_churn_by_service AS
SELECT service, status, COUNT(*) AS customers,
       SUM(CASE WHEN churn_actual = 1 THEN 1 ELSE 0 END) AS churned
FROM (
    SELECT 'OnlineSecurity'   AS service, online_security   AS status, churn_actual FROM customers WHERE source = 'batch'
    UNION ALL SELECT 'OnlineBackup',     online_backup,     churn_actual FROM customers WHERE source = 'batch'
    UNION ALL SELECT 'DeviceProtection', device_protection, churn_actual FROM customers WHERE source = 'batch'
    UNION ALL SELECT 'TechSupport',      tech_support,      churn_actual FROM customers WHERE source = 'batch'
    UNION ALL SELECT 'StreamingTV',      streaming_tv,      churn_actual FROM customers WHERE source = 'batch'
    UNION ALL SELECT 'StreamingMovies',  streaming_movies,  churn_actual FROM customers WHERE source = 'batch'
) s
GROUP BY service, status;
