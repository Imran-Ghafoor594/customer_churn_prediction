# Customer Churn Prediction System
**Teyzix Core Internship — Task ML-2 | June 2026**

---

## Project Overview
End-to-end Machine Learning system that predicts whether a customer
will churn (leave) based on behavioral and demographic data.

**Dataset:** 1,500 records | 14 features | ~50% churn rate (balanced)
**Models:** Logistic Regression, Random Forest, XGBoost, SVM

---

## Setup Instructions

### Step 1 — Install Requirements
```bash
pip install pandas numpy scikit-learn plotly xgboost matplotlib seaborn joblib
```

### Step 2 — Place Your Dataset
Put your CSV file in the `data/` folder:
```
churn_project/
└── data/
    └── customer_churn_dataset.csv   ← your file here
```

### Step 3 — Run the Pipeline (in order)
```bash
# From inside churn_project/ folder:

python src/data_preprocessing.py   # Clean + encode + split data
python src/eda.py                   # EDA plots (saved to plots/)
python src/model_training.py        # Train 4 models + evaluate
python src/generate_report.py       # Generate analysis report
```


## Project Structure
```
churn_project/
├── data/
│   ├── customer_churn_dataset.csv     <- PUT YOUR CSV HERE
│   ├── customer_churn_clean.csv       <- Generated after preprocessing
│   ├── X_train.csv / X_test.csv
│   └── y_train.csv / y_test.csv
│
├── models/                            <- Auto-created during training
│   ├── best_model.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── scaler.pkl
│   └── le_gender/city/sub/device/tenure.pkl
│
├── plots/                             <- 7 charts auto-saved here
│   ├── eda_01_demographics.png
│   ├── eda_02_feature_distributions.png
│   ├── eda_03_correlation.png
│   ├── eda_04_behavior_trends.png
│   ├── model_01_roc_comparison.png
│   ├── model_02_confusion_matrices.png
│   └── model_03_feature_importance.png
│
├── reports/
│   ├── model_comparison.csv           <- All model metrics
│   └── analysis_report.md            <- Full analysis report
│
├── src/
    ├── generate_data.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   └── generate_report.py
│
├── app.py
└── README.md                          <- This file
```

---

## Features Used
| # | Feature | Type |
|---|---|---|
| 1 | Age | Numeric |
| 2 | Gender | Categorical |
| 3 | City | Categorical |
| 4 | Subscription Type | Categorical |
| 5 | Monthly Spending | Numeric |
| 6 | Tenure | Numeric |
| 7 | Number of Purchases | Numeric |
| 8 | Customer Support Requests | Numeric |
| 9 | Login Frequency | Numeric |
| 10 | Satisfaction Score | Numeric |
| 11 | Device Type | Categorical |
| 12 | Total Spending | Numeric |
| 13 | Days Since Last Activity | Numeric (engineered from date) |
| 14 | Spending Per Tenure | Engineered |
| 15 | Engagement Score | Engineered |
| 16 | Support Burden | Engineered |
| 17 | Tenure Bucket | Engineered |
| 18 | High Value Flag | Engineered |

---

## Models & Expected Performance
| Model | Expected Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~75-80% | Linear baseline |
| Random Forest | ~85-90% | Best overall performer |
| XGBoost | ~85-92% | Highest ROC-AUC |
| SVM (RBF) | ~80-85% | Good on balanced data |

Best model is auto-selected by ROC-AUC and saved as `models/best_model.pkl`.

---

## Output Files After Running
- `plots/` — 7 visualization charts (PNG)
- `reports/model_comparison.csv` — All 4 models x 6 metrics
- `reports/analysis_report.md` — Full submission-ready report
- `models/best_model.pkl` — Trained best model ready for deployment

---

## Technology Stack
- **Python 3.x**
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — ML models, preprocessing, evaluation
- **XGBoost** — Gradient boosting
- **Matplotlib & Seaborn** — Visualizations
- **Joblib** — Model persistence

---

*Teyzix Core Internship | Task ML-2 | Customer Churn Prediction*
