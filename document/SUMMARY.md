# Project Summary — Customer Behavior Analytics & Churn Prediction
**Teyzix Core Internship | June Batch 2026**

---

## 1. Problem Statement

Companies lose significant revenue when customers stop using their services (churn). The challenge is identifying *which* customers are likely to churn *before* they actually do, so the business can take action in time.

This project builds an end-to-end system that:
- Analyzes customer behavior patterns
- Segments customers by value
- Predicts churn probability using Machine Learning
- Presents results through an interactive dashboard

---

## 2. Dataset

- **Source:** Telco Customer Churn Dataset (Kaggle)
- **Size:** ~7,000 customers, 21 features
- **Target Variable:** `Churn` (Yes / No)
- **Key Features:** tenure, MonthlyCharges, TotalCharges, contract type, payment method, internet service, support interactions

---

## 3. Approach

### Data Cleaning
- Filled missing numeric values with median
- Filled missing categorical values with mode
- Removed outliers using IQR method
- Encoded categorical columns using Label Encoding

### Feature Engineering
New features created to improve model performance:

| Feature | Description |
|---------|-------------|
| `AvgMonthlySpend` | TotalCharges / (tenure + 1) |
| `TenureGroup` | Tenure grouped into 5 bands (0–12, 12–24, etc.) |
| `SeniorTenureInteraction` | SeniorCitizen × tenure |

### Customer Segmentation
Customers divided into 3 value segments based on MonthlyCharges:

- 🟡 **High Value** — top 33% spenders
- 🟣 **Medium Value** — middle 33%
- 🟤 **Low Value** — bottom 33%

---

## 4. Model Results

Two models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | **0.8357** | **0.7033** | **0.5269** | **0.6025** | **0.8517** |
| Random Forest | 0.8180 | 0.6684 | 0.4552 | 0.5416 | 0.8247 |

**Winner: Logistic Regression** — outperformed Random Forest on all metrics for this dataset.

A ROC-AUC of **0.85** means the model correctly identifies a churner over a non-churner 85% of the time.

---

## 5. Churn Risk Categories

Based on predicted churn probability:

| Risk Level | Probability | Action Recommended |
|------------|-------------|-------------------|
| 🔴 High Risk | ≥ 70% | Immediate retention offer |
| 🟡 Medium Risk | 40% – 70% | Monitor closely, send engagement campaign |
| 🟢 Low Risk | < 40% | No action needed |

---

## 6. Business Insights

### Top Churn Indicators
Based on feature correlation and model importance:
- **Short tenure** — new customers churn more than long-term ones
- **Month-to-month contracts** — far higher churn than annual contracts
- **High monthly charges** — customers paying more are more likely to leave
- **No tech support / online security** — customers without add-on services churn more
- **Electronic check payment** — higher churn than auto-pay methods

### High-Risk Customer Profile
A typical high-risk customer:
- Tenure less than 12 months
- On a month-to-month contract
- Paying via electronic check
- No online security or tech support subscription
- Higher-than-average monthly charges

### Revenue Impact
High-risk customers represent a significant portion of monthly revenue. Retaining even 30–40% of them through targeted campaigns could save thousands in monthly recurring revenue.

---

## 7. Dashboard Features

An interactive Streamlit dashboard was built with:
- **Overview page** — risk summary, segmentation charts, revenue at risk
- **Visualizations page** — heatmap, revenue trend, feature importance
- **Predict Customers page** — bulk CSV prediction + single customer manual prediction

---

## 8. Deliverables

| Deliverable | Status |
|-------------|--------|
| Jupyter Notebook (`Churn_Prediction.ipynb`) | ✅ |
| Dataset (`customer_data.csv`) | ✅ |
| Figures (`figures/`) | ✅ |
| Streamlit Dashboard (`app.py`) | ✅ |
| Summary Document (`SUMMARY.md`) | ✅ |
| README (`README.md`) | ✅ |

---

## 9. Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| Pandas & NumPy | Data analysis |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | ML models & evaluation |
| Joblib | Model saving |
| Streamlit | Interactive dashboard |

---

*Submitted by: Imran Ghafoor | Teyzix Core Internship — June Batch 2026*