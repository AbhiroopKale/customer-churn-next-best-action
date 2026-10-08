# Results (auto-generated from the actual pipeline run)

Selected model: **XGBoost** (`v1-xgboost-20261001`), decision threshold 0.32. Rule: highest 5-fold CV PR-AUC on training data among deployable models; threshold = F1-optimal on out-of-fold training predictions.

## Model comparison (held-out test set, each model at its own tuned threshold)

              Model  Threshold  Accuracy  Precision  Recall     F1  ROC-AUC  PR-AUC  CV PR-AUC
Logistic Regression       0.34    0.7750     0.5585  0.7273 0.6318   0.8451  0.6476     0.6612
      Decision Tree       0.35    0.7551     0.5276  0.7406 0.6162   0.8281  0.6203     0.6239
      Random Forest       0.35    0.7750     0.5610  0.7005 0.6231   0.8400  0.6449     0.6537
            XGBoost       0.32    0.7622     0.5373  0.7513 0.6265   0.8440  0.6612     0.6644
    ANN (Keras MLP)       0.38    0.7793     0.5695  0.6898 0.6239   0.8452  0.6395     0.6509

## Same models at the default 0.5 threshold

              Model  Precision@0.5  Recall@0.5  F1@0.5
Logistic Regression         0.6632      0.5160  0.5805
      Decision Tree         0.6347      0.5668  0.5989
      Random Forest         0.6564      0.5107  0.5744
            XGBoost         0.6678      0.5321  0.5923
    ANN (Keras MLP)         0.6554      0.5187  0.5791

## Imbalance strategies (5-fold CV on training data)

              Model     Strategy  ROC-AUC  PR-AUC  Precision@0.5  Recall@0.5  F1@0.5  Brier (lower=better calibrated)
Logistic Regression         none   0.8476  0.6612         0.6709      0.5344  0.5949                           0.1341
Logistic Regression class_weight   0.8475  0.6596         0.5226      0.7953  0.6308                           0.1630
Logistic Regression        smote   0.8460  0.6581         0.5280      0.7819  0.6304                           0.1607
      Random Forest         none   0.8442  0.6537         0.6750      0.5043  0.5773                           0.1355
      Random Forest class_weight   0.8445  0.6513         0.5357      0.7579  0.6277                           0.1563
      Random Forest        smote   0.8402  0.6346         0.5948      0.6589  0.6252                           0.1451
            XGBoost         none   0.8460  0.6644         0.6636      0.5224  0.5846                           0.1347
            XGBoost class_weight   0.8462  0.6639         0.5190      0.7873  0.6256                           0.1601
            XGBoost        smote   0.8449  0.6573         0.6070      0.6261  0.6164                           0.1387

## Risk levels, all customers

risk_level
LOW       4394
MEDIUM    2036
HIGH       613

## Risk levels, active customers (retention targets)

risk_level
LOW       3947
MEDIUM    1097
HIGH       130

## Next Best Action counts (HIGH risk)

recommended_action
Offer an incentive to move to a 1- or 2-year plan                                                                                       83
Early-retention offer with a welcome / onboarding call                                                                                  41
Review plan fit and offer a lower-cost / discounted plan                                                                                 5
Offer a trial of TechSupport / OnlineSecurity add-ons (a service-adoption gap; this dataset has NO complaint or support-ticket data)     1

## EDA findings (training split)
```
EDA on training split: 5634 customers
Overall churn rate: 26.5%  -> imbalanced, accuracy alone would mislead
Contract: Month-to-month=42.7% (n=3102), One year=11.1% (n=1173), Two year=2.9% (n=1359)
tenure_group: 0-6m=52.2% (n=1170), 7-12m=37.2% (n=567), 13-24m=28.6% (n=818), 25m+=14.3% (n=3079)
Median MonthlyCharges  churned: 79.95 | stayed: 64.40
charge_category: Low=10.6% (n=1383), Medium=30.9% (n=2840), High=33.4% (n=1411)
PaymentMethod: Bank transfer (automatic)=16.2% (n=1244), Credit card (automatic)=14.9% (n=1213), Electronic check=45.7% (n=1891), Mailed check=19.3% (n=1286)
InternetService: DSL=18.7% (n=1937), Fiber optic=42.1% (n=2483), No=7.2% (n=1214)
gender: Female=26.6%, Male=26.4%
SeniorCitizen: 0=23.7%, 1=41.1%
Partner: No=32.8%, Yes=19.9%
Dependents: No=31.4%, Yes=15.2%
Add-on churn rate (subscribed vs not): OnlineSecurity=14.4% vs 41.9%, OnlineBackup=21.6% vs 40.1%, DeviceProtection=22.8% vs 39.0%, TechSupport=15.2% vs 41.8%, StreamingTV=30.5% vs 33.1%, StreamingMovies=30.0% vs 33.6%
corr(TotalCharges, tenure) = 0.83  -> TotalCharges is largely a function of tenure x charges
Distribution percentiles of MonthlyCharges (train): {0.25: 35.66, 0.5: 70.5, 0.75: 90.0}
```
