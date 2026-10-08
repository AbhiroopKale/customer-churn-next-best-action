# Project report outline (fill using `reports/RESULTS.md`, `reports/figures/`, `models/`)

1. **Abstract** - problem, approach (Predict → Explain → Recommend → Act), main results (copy from RESULTS.md), limitations.
2. **Introduction** - churn cost, why prediction alone is not actionable, objectives (the four questions).
3. **Literature / background** - churn prediction, class imbalance, SHAP, next-best-action systems (cite sources you actually read).
4. **Dataset** - IBM Telco Customer Churn: size, columns, target, class balance; data-quality findings (TotalCharges blanks,
   no duplicates); what the dataset lacks (complaints, true CLV). Source: notebook 01.
5. **Methodology**
   5.1 Preprocessing and leakage control (split first, Pipeline/ColumnTransformer, fixed-bin features)
   5.2 EDA findings (figures 01-10, `reports/eda_summary.txt`)
   5.3 Feature engineering (tenure_group, charge_category, n_addons: justify each with EDA)
   5.4 Models and why each was chosen (LR baseline, DT, RF, XGBoost, ANN)
   5.5 Imbalance handling (experiment table, calibration/Brier argument)
   5.6 Evaluation protocol (CV on train, test untouched, metrics, threshold tuning on OOF predictions)
6. **Results** - comparison table, ROC/PR curves, confusion matrices, calibration, final model justification
   (be honest about near-ties; ANN result), risk-band distribution.
7. **Explainability** - global SHAP, example customers, caveat that SHAP is not causation.
8. **Recommendation engine** - rules R1-R5, how SHAP ranks the matching rules, segments, revenue-at-risk proxy,
   explicit statement that it is rule-based and unvalidated.
9. **System implementation** - architecture diagram, PostgreSQL schema (ER diagram from `sql/schema.sql`), Streamlit app, Power BI dashboard (screenshots).
10. **Testing** - pytest suite (20 tests), leakage test, app smoke test.
11. **Limitations and ethics** - see README honesty statements; fairness of demographic features.
12. **Future scope** - real support data, CLV, campaign history → uplift modelling, A/B testing, monitoring, MLOps.
13. **References**, **Appendix** (requirements, how to run, code listing links).
