# Presentation outline (12 slides)

1. Title + one-line story (Predict → Explain → Recommend → Act)
2. Problem: churn is noticed too late; prediction alone is not actionable
3. Objectives: the four questions
4. Dataset: Telco (size, churn rate, what it lacks) - `01_churn_overall.png`
5. Key EDA insights: contract, tenure, payment, internet service - `02`, `03`, `05`, `06`, `09`
6. Pipeline & leakage control (diagram from README)
7. Five models, imbalance handling, metrics - comparison table + `11_roc_pr_curves.png`
8. Model selection and honest interpretation (threshold tuning, near-ties, ANN result)
9. Explainability: global SHAP + one customer's WHY - `14_shap_global_bar.png`
10. Recommendation engine & Next Best Action (rules, ranking by SHAP, rule-based disclaimer)
11. System: PostgreSQL → Power BI → Streamlit (screenshots) + live demo
12. Limitations, future scope, conclusion
