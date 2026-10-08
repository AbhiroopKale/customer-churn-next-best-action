# Final demo script (about 6-8 minutes)

Before presenting: `run_pipeline.bat`, `run_database.bat`, open Power BI, start the app with `run_app.bat`.
Keep a backup: screenshots of each step, in case of a live failure.

1. **Story (30 s):** "Companies often find out about churn after the customer has left. This system predicts who is likely to
   leave, explains why, and recommends what to do: Predict → Explain → Recommend → Act."
2. **Streamlit:** choose preset *"new fiber customer, month-to-month"* (or enter your own values within the dataset's
   charge range ~18-119), click **PREDICT CHURN**.
3. **Result:** show probability, risk level, then the **WHY** chart (top factors) and say SHAP explains the model, not causes.
4. **Next Best Action:** primary action, secondary actions, segment. Say it is rule-based and unproven until tested.
5. **Change one input** (Contract → Two year, tenure → 60) and predict again: probability falls, action changes.
6. **Saved predictions tab:** show the row stored in PostgreSQL (open pgAdmin and run
   `SELECT * FROM vw_retention_actions WHERE source='live';`).
7. **Model comparison tab:** 5 models, why accuracy is not enough, threshold tuning, why XGBoost was selected and that
   the margins between top models are small.
8. **Power BI:** Page 1 (business view), Page 2 (filter HIGH risk, month-to-month), Page 3 (action list sorted by revenue at risk).
9. **Limitations (30 s):** no complaint data, no CLV, rule-based actions, single dataset.
