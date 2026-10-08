# Power BI dashboard guide (3 pages)

Power BI files cannot be generated from code here, so build the `.pbix` by following this guide (about 60-90 min).
Save it as `dashboard/churn_dashboard.pbix` and put screenshots in `dashboard/screenshots/`.

## 1. Connect

**Option A - PostgreSQL (recommended):** Home → Get Data → PostgreSQL database → Server `localhost`,
Database `churn_nba`, **Import** → Database credentials (user `postgres`, your password).
Select: `vw_customer_risk`, `vw_retention_actions`, `vw_churn_by_service`.
(If Power BI asks for the Npgsql provider, install it from its GitHub releases and restart Power BI.)

**Option B - CSV fallback:** Get Data → Text/CSV → `data/processed/powerbi_customer_risk.csv`
(one flat table with the same columns; replace table names below with this table).

**Relationship (Model view):** `vw_customer_risk[customer_id]` → `vw_retention_actions[customer_id]` (one to one).

## 2. Calculated columns (vw_customer_risk)

```
Churn Status = IF(vw_customer_risk[churn_actual] = 1, "Churned", IF(vw_customer_risk[churn_actual] = 0, "Stayed", "Live (unknown)"))
Tenure Order = SWITCH(vw_customer_risk[tenure_group], "0-6m", 1, "7-12m", 2, "13-24m", 3, "25m+", 4)
```
Select `tenure_group` → Column tools → **Sort by column → Tenure Order**.

## 3. Measures (create in vw_customer_risk)

```
Total Customers      = CALCULATE(COUNTROWS(vw_customer_risk), vw_customer_risk[source] = "batch")
Churned Customers    = CALCULATE(COUNTROWS(vw_customer_risk), vw_customer_risk[churn_actual] = 1)
Churn Rate           = DIVIDE([Churned Customers], [Total Customers])
High-Risk Customers  = CALCULATE(COUNTROWS(vw_customer_risk), vw_customer_risk[risk_level] = "HIGH", vw_customer_risk[is_active] = TRUE())
Avg Churn Probability= CALCULATE(AVERAGE(vw_customer_risk[churn_probability]), vw_customer_risk[is_active] = TRUE())
Customers at Risk    = CALCULATE(COUNTROWS(vw_customer_risk), vw_customer_risk[predicted_churn] = TRUE(), vw_customer_risk[is_active] = TRUE())
Revenue at Risk (proxy) = SUM(vw_retention_actions[revenue_at_risk_proxy])
Service Churn Rate   = DIVIDE(SUM(vw_churn_by_service[churned]), SUM(vw_churn_by_service[customers]))
```
Definitions to state on the dashboard: *High-Risk* = probability ≥ 70% (configurable); *Customers at Risk* =
active customers whose probability is at or above the model's decision threshold (see `models/meta.json`);
*Active* = customers who have not churned (retention targets).

## 4. Pages

**Page 1 - Executive Overview** (page filter: `source = batch`)
- KPI cards: Total Customers, Churned Customers, Churn Rate, High-Risk Customers, Avg Churn Probability, Customers at Risk.
- Churn by Contract: clustered column, axis `contract`, value `Churn Rate`.
- Churn by Tenure: column, axis `tenure_group`, value `Churn Rate`.
- Churn by Service: clustered bar from `vw_churn_by_service`, axis `service`, legend `status`, value `Service Churn Rate`.
- Churn by Payment Method: bar, axis `payment_method`, value `Churn Rate`.
- Churn distribution: donut, legend `Churn Status`, values count of `customer_id`.

**Page 2 - Customer Risk Analysis** (page filter: `is_active = TRUE`)
- Table: `customer_id`, `churn_probability` (data bars), `risk_level` (conditional colours: HIGH red, MEDIUM amber, LOW green),
  `risk_factors`, `contract`, `tenure`, `monthly_charges`.
- Slicers: `risk_level`, `contract`, `tenure_group` (or tenure range), `payment_method`, `internet_service`
  (and `tech_support` / `online_security` for the service filter).

**Page 3 - Retention Recommendations** (table from `vw_retention_actions`, filter `source = batch` or leave all)
- Table: `customer_id`, `churn_probability`, `risk_level`, `segment`, `risk_factors`, `recommended_action`, `revenue_at_risk_proxy`.
- Visual-level Top N filter: top 50 customers by `revenue_at_risk_proxy`; keep a slicer on `risk_level` (default HIGH).
- Bar: count of customers by `recommended_action`; matrix or donut: customers by `segment`.
- Text box (keep it!): *"Recommendations are rule-based suggestions. Their effect on retention has not been measured. Revenue at risk is a proxy (probability × monthly charges), not CLV."*

## 5. Refresh
After retraining: `run_pipeline.bat` → `run_database.bat` → Power BI **Refresh**.
Customers predicted live in the Streamlit app appear with `source = live` (they have no actual churn label).
