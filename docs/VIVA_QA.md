# Likely questions from a mentor / interviewer, with honest answers

**Why not just report accuracy?** About 26.5% of customers churn, so predicting "stay" for everyone gives ~73% accuracy. We use
recall, precision, F1, ROC-AUC and PR-AUC, and tune the decision threshold.

**Why was XGBoost selected? Is it really better?** It had the highest cross-validated PR-AUC among deployable models, but the top
models are close (see `model_comparison.csv`). Logistic Regression is a strong, simpler alternative. Selection used training-set CV, not the test set, so the test numbers are an unbiased report card.

**Did the ANN win?** Check your table: it is compared on the same pipeline and CV. It is not deployed because the app needs SHAP explanations from one saved pipeline; the comparison shows whether it adds predictive value.

**How did you avoid data leakage?** Split first; all fitted preprocessing is inside a Pipeline; CV refits it per fold; threshold tuned on out-of-fold training predictions; features use fixed bins; no outcome columns (churn reason/score) are used; SMOTE only appears inside the training folds of an experiment.

**Why not class weights / SMOTE?** The experiment (`imbalance_experiment.csv`) shows ranking quality (PR-AUC) is similar, but probabilities become less well calibrated (higher Brier). We need honest probabilities for risk bands, so we tune the threshold instead.

**What does the probability mean?** Estimated churn probability from the model; calibration plot (`13_calibration.png`) shows how well it matches observed rates. Risk bands (30% / 70%) are configurable.

**Does SHAP prove why customers churn?** No. It explains how the model uses features. Month-to-month contract being influential means the model relies on it, not that changing contract type will cause retention.

**Is the recommender machine learning?** No: it is rule-based. Rules are ranked per customer using the SHAP contributions of related features. There is no data on past offers or outcomes, so we cannot claim any action works. A/B tests would be needed; with campaign data one could build uplift models.

**Why do you only recommend actions for non-churned customers?** Customers with Churn = Yes have already left; scoring them only validates the model.

**What is `revenue_at_risk_proxy`? Is it CLV?** No. It is probability × monthly charges, a labelled proxy for prioritisation.

**What features would a real system add?** Support tickets, complaints, outages, usage, billing disputes, offer history and responses, contract end dates.

**Fairness?** Gender and senior-citizen are in the data; no recommendation rule uses them. A production system should audit performance across groups before using demographic features.

**Why is an input of 899 rejected/warned in the app?** The dataset's monthly charges range roughly 18-119 (dataset units). Outside that, the model extrapolates; the app warns.

**What would you do next?** Calibration monitoring, retraining pipeline, richer features, campaign A/B testing, cloud deployment.
