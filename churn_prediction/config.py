
import os

# ══════════════════════════════════════════════════════════════
#  BASE DIRECTORY  
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════
#  FOLDER PATHS
# ══════════════════════════════════════════════════════════════
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SRC_DIR     = os.path.join(BASE_DIR, "src")

# ══════════════════════════════════════════════════════════════
#  DATA FILE PATHS
# ══════════════════════════════════════════════════════════════
RAW_DATA_PATH       = os.path.join(DATA_DIR, "customer_churn_dataset.csv")
CLEAN_DATA_PATH     = os.path.join(DATA_DIR, "customer_churn_clean.csv")
X_TRAIN_PATH        = os.path.join(DATA_DIR, "X_train.csv")
X_TEST_PATH         = os.path.join(DATA_DIR, "X_test.csv")
Y_TRAIN_PATH        = os.path.join(DATA_DIR, "y_train.csv")
Y_TEST_PATH         = os.path.join(DATA_DIR, "y_test.csv")

# ══════════════════════════════════════════════════════════════
#  MODEL FILE PATHS
# ══════════════════════════════════════════════════════════════
BEST_MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.pkl")
BEST_MODEL_NAME_PATH = os.path.join(MODELS_DIR, "best_model_name.pkl")
FEATURE_NAMES_PATH   = os.path.join(MODELS_DIR, "feature_names.pkl")
SCALER_PATH          = os.path.join(MODELS_DIR, "scaler.pkl")

# Encoder paths
LE_GENDER_PATH  = os.path.join(MODELS_DIR, "le_gender.pkl")
LE_CITY_PATH    = os.path.join(MODELS_DIR, "le_city.pkl")
LE_SUB_PATH     = os.path.join(MODELS_DIR, "le_sub.pkl")
LE_DEVICE_PATH  = os.path.join(MODELS_DIR, "le_device.pkl")
LE_TENURE_PATH  = os.path.join(MODELS_DIR, "le_tenure.pkl")

# Individual model paths
MODEL_RF_PATH   = os.path.join(MODELS_DIR, "random_forest.pkl")
MODEL_XGB_PATH  = os.path.join(MODELS_DIR, "xgboost.pkl")
MODEL_LR_PATH   = os.path.join(MODELS_DIR, "logistic_regression.pkl")
MODEL_SVM_PATH  = os.path.join(MODELS_DIR, "svm.pkl")

# ══════════════════════════════════════════════════════════════
#  REPORT FILE PATHS
# ══════════════════════════════════════════════════════════════
MODEL_COMPARISON_PATH = os.path.join(REPORTS_DIR, "model_comparison.csv")
ANALYSIS_REPORT_PATH  = os.path.join(REPORTS_DIR, "analysis_report.md")

# ══════════════════════════════════════════════════════════════
#  PLOT FILE PATHS
# ══════════════════════════════════════════════════════════════
PLOT_EDA_DEMOGRAPHICS    = os.path.join(PLOTS_DIR, "eda_01_demographics.png")
PLOT_EDA_DISTRIBUTIONS   = os.path.join(PLOTS_DIR, "eda_02_feature_distributions.png")
PLOT_EDA_CORRELATION     = os.path.join(PLOTS_DIR, "eda_03_correlation.png")
PLOT_EDA_BEHAVIOR        = os.path.join(PLOTS_DIR, "eda_04_behavior_trends.png")
PLOT_MODEL_ROC           = os.path.join(PLOTS_DIR, "model_01_roc_comparison.png")
PLOT_MODEL_CONFUSION     = os.path.join(PLOTS_DIR, "model_02_confusion_matrices.png")
PLOT_MODEL_IMPORTANCE    = os.path.join(PLOTS_DIR, "model_03_feature_importance.png")

# ══════════════════════════════════════════════════════════════
#  DATASET SETTINGS
# ══════════════════════════════════════════════════════════════
TARGET_COLUMN   = "Churn Status"
CUSTOMER_ID_COL = "Customer ID"
DATE_COLUMN     = "Last Activity Date"
REFERENCE_DATE  = "2026-06-20"
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

# ══════════════════════════════════════════════════════════════
#  FEATURE LISTS
# ══════════════════════════════════════════════════════════════
CATEGORICAL_COLS = [
    "Gender",
    "City",
    "Subscription Type",
    "Device Type",
    "Tenure Bucket",
]

NUMERIC_COLS = [
    "Age",
    "Monthly Spending",
    "Tenure",
    "Number of Purchases",
    "Customer Support Requests",
    "Login Frequency",
    "Satisfaction Score",
    "Total Spending",
    "Days Since Last Activity",
    "Spending Per Tenure",
    "Engagement Score",
    "Support Burden",
]

ENGINEERED_COLS = [
    "Days Since Last Activity",
    "Spending Per Tenure",
    "Engagement Score",
    "Support Burden",
    "Tenure Bucket",
    "High Value",
]

# ══════════════════════════════════════════════════════════════
#  MODEL SETTINGS
# ══════════════════════════════════════════════════════════════
CHURN_THRESHOLD    = 0.50      # probability >= this → predicted churn
CV_FOLDS           = 5
BEST_MODEL_METRIC  = "ROC-AUC" # metric used to pick best model

# Risk level thresholds
RISK_LOW_MAX    = 0.35
RISK_MEDIUM_MAX = 0.60
# >= 0.60 → HIGH

# ══════════════════════════════════════════════════════════════
#  CATEGORICAL OPTION LISTS  (for UI dropdowns)
# ══════════════════════════════════════════════════════════════
GENDER_OPTIONS       = ["Male", "Female", "Other"]
CITY_OPTIONS         = ["Chicago", "Houston", "Los Angeles", "Miami", "New York"]
SUBSCRIPTION_OPTIONS = ["Basic", "Standard", "Premium"]
DEVICE_OPTIONS       = ["Mobile", "Desktop", "Tablet"]
SATISFACTION_OPTIONS = [1, 2, 3, 4, 5]

# ══════════════════════════════════════════════════════════════
#  AUTO-CREATE ALL DIRECTORIES ON IMPORT
# ══════════════════════════════════════════════════════════════
for _dir in [DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, SRC_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  QUICK SELF-TEST  
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  CONFIG.PY — Path Verification")
    print("=" * 55)

    print(f"\n  BASE_DIR    : {BASE_DIR}")
    print(f"\n  FOLDERS:")
    for name, path in [("data", DATA_DIR), ("models", MODELS_DIR),
                        ("plots", PLOTS_DIR), ("reports", REPORTS_DIR)]:
        exists = "EXISTS" if os.path.isdir(path) else "will be created"
        print(f"    {name:<10} → {path}  [{exists}]")

    print(f"\n  DATA FILES:")
    for name, path in [("Raw CSV", RAW_DATA_PATH), ("Clean CSV", CLEAN_DATA_PATH),
                        ("X_train", X_TRAIN_PATH), ("X_test", X_TEST_PATH)]:
        exists = "✓ found" if os.path.isfile(path) else "✗ not yet"
        print(f"    {name:<12} → {os.path.basename(path)}  [{exists}]")

    print(f"\n  MODEL FILES:")
    for name, path in [("Best Model", BEST_MODEL_PATH), ("RF", MODEL_RF_PATH),
                        ("XGBoost", MODEL_XGB_PATH), ("Scaler", SCALER_PATH)]:
        exists = "✓ found" if os.path.isfile(path) else "✗ not yet (run training first)"
        print(f"    {name:<12} → {os.path.basename(path)}  [{exists}]")

    print(f"\n  MODEL COMPARISON FILES:")
    for name, path in [("Model Comparison", MODEL_COMPARISON_PATH), ("Analysis Report", ANALYSIS_REPORT_PATH )]:
        exists = "✓ found" if os.path.isfile(path) else "✗ not yet (run training first)"
        print(f"    {name:<12} → {os.path.basename(path)}  [{exists}]")
    print(f"\n  SETTINGS:")
    print(f"    Target Column  : {TARGET_COLUMN}")
    print(f"    Test Size      : {TEST_SIZE*100:.0f}%")
    print(f"    Random State   : {RANDOM_STATE}")
    print(f"    Churn Threshold: {CHURN_THRESHOLD}")
    print(f"    CV Folds       : {CV_FOLDS}")
    print(f"    Best Metric    : {BEST_MODEL_METRIC}")
    print(f"\n  Config loaded successfully!")
