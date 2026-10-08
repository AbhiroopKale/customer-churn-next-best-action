# AI-Powered Customer Churn Prediction & Next Best Action Recommendation System

**Predict → Explain → Recommend → Act**

A complete, working system built on the public IBM *Telco Customer Churn* dataset:

| Question | Module | Output |
|---|---|---|
| Who will churn? How likely? | 5 compared ML models, threshold tuning | churn probability + LOW/MEDIUM/HIGH risk |
| Why? | SHAP (global + per customer) | top factors behind each prediction |
| What should we do? | rule-based **Next Best Action engine** | primary + secondary retention actions, segment |
| Where does it live? | PostgreSQL, Power BI, Streamlit | stored results, dashboards, interactive demo |

## Quick start (Windows + VS Code)

1. Install **Python 3.11 or 3.12**, **PostgreSQL**, **Git**, (Power BI Desktop for the dashboard).
2. `setup_windows.bat` – creates `venv`, installs requirements, copies `.env.example` to `.env`.
3. Open `.env` and set `PGPASSWORD` to your PostgreSQL password.
4. Dataset: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (included; the Kaggle copy with the same file name works too).
5. `run_pipeline.bat` – trains + compares models, EDA, figures, batch scoring, results summary (a few minutes).
6. `run_database.bat` – creates database `churn_nba`, tables, views and loads results.
7. `run_app.bat` – opens the Streamlit app.

Manual equivalents (run from the project root with the venv active):

```
python -m src.train_models            # train 5 models, select, save to models/
python -m scripts.run_eda             # EDA figures -> reports/figures
python -m scripts.make_report_figures # ROC/PR, confusion matrices, calibration, SHAP
python -m scripts.run_batch_scoring   # probabilities, SHAP factors, recommendations -> data/processed
python -m scripts.imbalance_experiment# none vs class_weight vs SMOTE (CV, train only)
python -m scripts.generate_results_md # real numbers -> reports/RESULTS.md
python -m scripts.setup_database      # create DB + schema + views
python -m scripts.load_to_postgres    # load batch results
streamlit run app/streamlit_app.py
python -m pytest                      # 20 tests
```

If PostgreSQL is unavailable, everything except the DB steps still works, and Power BI can read
`data/processed/powerbi_customer_risk.csv` instead (see `docs/POWERBI_GUIDE.md`).

## Architecture

```
Telco CSV -> clean -> EDA -> features -> split -> sklearn Pipeline (impute/scale/one-hot + model)
   -> 5 models compared (CV on train) -> final model + threshold saved (joblib)
   -> probability -> risk band -> SHAP -> recommendation engine -> Next Best Action
   -> PostgreSQL (customers, churn_predictions, explanations, recommendations, model_registry + views)
   -> Power BI (3 pages)            and          final model + engine -> Streamlit app (-> PostgreSQL)
```

## Key design decisions (and why)

- **No data leakage.** Train/test split happens first; imputer, scaler and encoder live inside one
  `Pipeline` and are fitted on training data only (and only on training folds during CV). Engineered
  features use fixed bins. Model selection and threshold tuning use cross-validated out-of-fold
  predictions on the *training* set; the test set is only used for the final report.
  Test: `tests/test_data_and_leakage.py`.
- **Imbalance** (~26.5% churn): threshold tuning + PR-AUC/recall/F1 instead of accuracy.
  `scripts/imbalance_experiment.py` compares no weighting vs `class_weight` vs SMOTE (SMOTE only inside
  training folds). Re-weighting/oversampling distorts predicted probabilities (worse Brier score), and the
  risk bands need honest probabilities, so the deployed models are unweighted + tuned threshold.
- **Model selection:** highest 5-fold CV PR-AUC among the *deployable* models (LR, DT, RF, XGBoost).
  Differences between the top models are small; see `models/model_comparison.csv` and treat near-ties honestly.
- **ANN** is trained and compared fairly (same pipeline, same CV) but not deployed: the app needs one
  saved Pipeline with SHAP explanations. The comparison table shows whether it adds predictive value.
- **Retention actions only for current customers.** In this dataset `Churn = Yes` customers have already
  left, so recommendations are generated for `Churn = No` customers. Scores for churned customers only
  validate the model. Views expose `is_active`.
- **Dashboard probabilities are out-of-fold** (each customer scored by a model that did not train on them).

## Honesty statements (put these in your report)

- Recommendations are **rule-based / heuristic**. There is no history of retention offers or outcomes, so
  nothing is learned or proven to reduce churn. Effect would need A/B tests.
- **SHAP explains model behaviour, not causation.** Contribution units are the model's native scale
  (log-odds for XGBoost/LR) and are for ranking factors.
- The dataset has **no complaint / support-ticket / service-issue data.** `TechSupport`/`OnlineSecurity`
  are add-on subscriptions. Rule R4 is therefore an *add-on adoption gap*, not a complaint signal.
- There is **no true CLV.** `revenue_at_risk_proxy = probability × MonthlyCharges` is a labelled proxy.
  Segments are business rules, not clustering.
- Dataset currency/ranges: `MonthlyCharges` ≈ 18–119 (dataset units). The app warns when an input is outside
  the training range.
- `gender` and `SeniorCitizen` are in the data; they are not used by any recommendation rule. In a real
  deployment, review fairness before using demographic attributes.
- Single dataset, single split. Production would need monitoring, retraining and calibration checks.

## Project layout

```
data/raw, data/processed     dataset and generated CSVs
notebooks/                   01 exploration, 02 EDA, 03 modeling results
src/                         config, preprocessing, features, training, evaluation, explainability,
                             recommendation_engine, prediction, database, ann_model
scripts/                     runnable steps (EDA, figures, batch scoring, DB setup/load, experiments)
sql/                         schema.sql, views.sql
app/streamlit_app.py         interactive app
models/                      saved pipeline, metadata, comparison tables (generated)
reports/                     figures, eda_summary.txt, RESULTS.md (generated)
dashboard/                   put your .pbix + screenshots here
docs/                        Power BI guide, report outline, slides outline, demo script, viva Q&A
tests/                       pytest suite
```

## Tested with

Python 3.12, pandas 3.0, scikit-learn 1.8, xgboost 3.4, shap 0.52, tensorflow-cpu 2.21, streamlit 1.64,
SQLAlchemy 2.1, PostgreSQL 16. `requirements.txt` uses minimum versions; if a newer library breaks something,
paste the exact error and fix that specific issue.

## Future scope

Real support/complaint features, true CLV, historical campaign data → uplift modelling / ML-based ranking of
actions, A/B testing, probability calibration monitoring, automated retraining, real-time scoring, cloud deployment.
