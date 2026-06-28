
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import joblib

from config import (
    MODEL_COMPARISON_PATH, ANALYSIS_REPORT_PATH,
    BEST_MODEL_NAME_PATH, FEATURE_NAMES_PATH,
    CHURN_THRESHOLD, CV_FOLDS, TEST_SIZE, RANDOM_STATE,
    REPORTS_DIR
)

metrics_df = pd.read_csv(MODEL_COMPARISON_PATH, index_col=0)
best_name  = joblib.load(BEST_MODEL_NAME_PATH)
best_acc   = metrics_df.loc[best_name, "Accuracy"]
best_auc   = metrics_df.loc[best_name, "ROC-AUC"]
best_f1    = metrics_df.loc[best_name, "F1"]

report = f"""# Customer Churn Prediction System
### Teyzix Core Internship — Task ML-2
**Date:** June 2026 | **Domain:** Machine Learning

---

## 1. Executive Summary
| Metric | Value |
|---|---|
| Best Model | **{best_name}** |
| Test Accuracy | **{best_acc*100:.2f}%** |
| ROC-AUC | **{best_auc:.4f}** |
| F1 Score | **{best_f1:.4f}** |
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
- Last Activity Date → Days Since Last Activity )
- Dropped Customer ID (non-predictive)
- Encoded: Gender, City, Subscription Type, Device Type, Tenure Bucket
- Scaled: StandardScaler on {len(['Age','Monthly Spending','Tenure','Number of Purchases','Customer Support Requests','Login Frequency','Satisfaction Score','Total Spending','Days Since Last Activity','Spending Per Tenure','Engagement Score','Support Burden'])} numeric columns
- Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} stratified | Random State: {RANDOM_STATE}

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

{metrics_df[['Accuracy','Precision','Recall','F1','ROC-AUC','CV-F1(Train)']].round(4).to_markdown()}

**Best Model:** {best_name} (selected by ROC-AUC | CV Folds: {CV_FOLDS})

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
- Churn Threshold: {CHURN_THRESHOLD}
- CV Folds: {CV_FOLDS}
- Test Size: {TEST_SIZE}
- Random State: {RANDOM_STATE}

---
*Report auto-generated — Teyzix Core Internship Task ML-2 | June 2026*
"""

# Fix placeholder
report = report.replace("REFERENCE_DATE_PLACEHOLDER", "2026-06-20")

with open(ANALYSIS_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print("=" * 60)
print("  MODULE 6 — REPORT GENERATION")
print("=" * 60)
print(f"\n  Best Model : {best_name}")
print(f"  Accuracy   : {best_acc*100:.2f}%")
print(f"  ROC-AUC    : {best_auc:.4f}")
print(f"\n  Report saved → {ANALYSIS_REPORT_PATH}")
