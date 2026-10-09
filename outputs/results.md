## Results (held-out test set)

- Customers: 30,000 (test set: 6,000), overall default rate 22.1%
- XGBoost AUC-ROC: 0.783 (5-fold CV on train: 0.787 +/- 0.005)
- Logistic regression baseline AUC-ROC: 0.742
- At threshold 0.30: F1 0.543, precision 0.536, recall 0.549

| Risk segment | Customers | Share of customers | Actual default rate | Share of all defaults |
|---|---|---|---|---|
| Low | 3,100 | 51.7% | 9.0% | 20.9% |
| Medium | 1,540 | 25.7% | 20.8% | 24.1% |
| High | 1,360 | 22.7% | 53.6% | 54.9% |

Top risk drivers (mean |SHAP|): max_delay, PAY_1, months_delayed, utilization_latest, BILL_AMT1.
