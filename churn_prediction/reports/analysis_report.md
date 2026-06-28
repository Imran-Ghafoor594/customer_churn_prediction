# Customer Churn Prediction System
### Teyzix Core Internship — Task ML-2
**Date:** June 2026 | **Domain:** Machine Learning

---

## 1. Executive Summary
| Metric | Value |
|---|---|
| Best Model | **Random Forest** |
| Test Accuracy | **91.00%** |
| ROC-AUC | **0.9813** |
| F1 Score | **0.9126** |
| Dataset | 1,500 records, ~50% churn (balanced) |

---

## 2. Dataset Description
- **Source:** Self-generated synthetic dataset (customer_churn_dataset.csv)
- **Records:** 1,500 rows (after deduplication)
- **Target:** Churn Status (0=Retained, 1=Churned)
- **Balance:** ~50/50 — no SMOTE required
- **Missing Values:** None

---

## 3. Data Preparation
- Removed duplicates | No missing values found
- Last Activity Date → Days Since Last Activity 
- Dropped Customer ID (non-predictive)
- Encoded: Gender, City, Subscription Type, Device Type, Tenure Bucket
- Scaled: StandardScaler on 12 numeric columns
- Split: 80/20 stratified | Random State: 42

---

## 4. Feature Engineering
| Feature | Formula |
|---|---|
| Days Since Last Activity | Reference Date − Last Activity Date |
| Spending Per Tenure | Monthly Spending / (Tenure + 1) |
| Engagement Score | Login×0.5 + Purchases×0.3 − DaysSince×0.2 |
| Support Burden | Support Requests / (Satisfaction + 0.1) |
| Tenure Bucket | Binned: New/Short/Mid/Long |
| High Value Flag | Spending>$80 AND Purchases>8 |

---

## 5. Model Performance

|                     |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |   CV-F1(Train) |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|---------------:|
| Logistic Regression |     0.8967 |      0.9178 |   0.8758 | 0.8963 |    0.9747 |         0.8887 |
| Random Forest       |     0.91   |      0.9038 |   0.9216 | 0.9126 |    0.9813 |         0.8974 |
| XGBoost             |     0.8933 |      0.906  |   0.8824 | 0.894  |    0.9799 |         0.913  |
| SVM                 |     0.8733 |      0.9078 |   0.8366 | 0.8707 |    0.9685 |         0.8879 |

**Best Model:** Random Forest (selected by ROC-AUC | CV Folds: 5)

---

## 6. Key EDA Findings
- Satisfaction Score is the single strongest churn predictor (negative correlation)
- High Support Requests → higher churn probability
- Low Login Frequency → higher churn risk
- New customers (≤6 months tenure) churn most frequently
- Basic plan has highest churn rate among subscription types

---

## 7. Business Recommendations
| Priority | Trigger | Action |
|---|---|---|
| CRITICAL | Satisfaction ≤ 2 | Assign dedicated account manager |
| HIGH | Support Requests ≥ 5 | Escalate tickets, priority SLA |
| HIGH | Login Frequency ≤ 3 | Re-engagement campaign |
| MEDIUM | Inactive ≥ 30 days | Loyalty discount or upgrade offer |
| MEDIUM | Tenure ≤ 6 months | Onboarding specialist |
| LOW | All customers | Monthly NPS surveys |

---

## 8. Configuration
All paths and settings managed centrally via `config.py`:
- Churn Threshold: 0.5
- CV Folds: 5
- Test Size: 0.2
- Random State: 42

---
*Report auto-generated — Teyzix Core Internship Task ML-2 | June 2026*
