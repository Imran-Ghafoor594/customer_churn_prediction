
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ── Import all paths & settings from config ──
from config import (
    RAW_DATA_PATH, CLEAN_DATA_PATH,
    X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
    MODELS_DIR, SCALER_PATH, FEATURE_NAMES_PATH,
    LE_GENDER_PATH, LE_CITY_PATH, LE_SUB_PATH, LE_DEVICE_PATH, LE_TENURE_PATH,
    TARGET_COLUMN, CUSTOMER_ID_COL, DATE_COLUMN, REFERENCE_DATE,
    TEST_SIZE, RANDOM_STATE, NUMERIC_COLS, CATEGORICAL_COLS
)



# ── 1. Load ───────────────────────────────────
df = pd.read_csv(RAW_DATA_PATH)
print(f"\n[1] Raw Shape  : {df.shape}")
print(f"    Churn Rate : {df[TARGET_COLUMN].mean():.2%}")

# ── 2. Remove Duplicates ──────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"\n[2] Duplicates Removed : {before - len(df)}  →  Shape {df.shape}")

# ── 3. Missing Values ─────────────────────────
missing = df.isnull().sum()
print(f"\n[3] Missing Values : {missing.sum()} total")
if missing.sum() == 0:
    print("    None found — dataset is clean!")
else:
    print(missing[missing > 0])

# ── 4. Drop ID Column ─────────────────────────
df.drop(columns=[CUSTOMER_ID_COL], inplace=True)
print(f"\n[4] Dropped '{CUSTOMER_ID_COL}' column")

# ── 5. Feature Engineering ────────────────────
print("\n[5] Feature Engineering ...")

df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df["Days Since Last Activity"] = (pd.Timestamp(REFERENCE_DATE) - df[DATE_COLUMN]).dt.days
df.drop(columns=[DATE_COLUMN], inplace=True)
print("    Done: Last Activity Date → Days Since Last Activity")

df["Spending Per Tenure"] = df["Monthly Spending"] / (df["Tenure"] + 1)
print("    Done: Spending Per Tenure")

df["Engagement Score"] = (
    df["Login Frequency"] * 0.5
    + df["Number of Purchases"] * 0.3
    - df["Days Since Last Activity"] * 0.2
)
print("    Done: Engagement Score")

df["Support Burden"] = df["Customer Support Requests"] / (df["Satisfaction Score"] + 0.1)
print("    Done: Support Burden")

df["Tenure Bucket"] = pd.cut(
    df["Tenure"],
    bins=[0, 6, 12, 24, 48],
    labels=["New(0-6m)", "Short(6-12m)", "Mid(12-24m)", "Long(24m+)"]
)
print("    Done: Tenure Bucket")

df["High Value"] = ((df["Monthly Spending"] > 80) & (df["Number of Purchases"] > 8)).astype(int)
print("    Done: High Value Flag")

# ── 6. Encode Categoricals ────────────────────
print("\n[6] Encoding Categorical Variables ...")

le_gender = LabelEncoder(); df["Gender"]            = le_gender.fit_transform(df["Gender"])
le_city   = LabelEncoder(); df["City"]              = le_city.fit_transform(df["City"])
le_sub    = LabelEncoder(); df["Subscription Type"] = le_sub.fit_transform(df["Subscription Type"])
le_device = LabelEncoder(); df["Device Type"]       = le_device.fit_transform(df["Device Type"])
le_tenure = LabelEncoder(); df["Tenure Bucket"]     = le_tenure.fit_transform(df["Tenure Bucket"].astype(str))

joblib.dump(le_gender, LE_GENDER_PATH)
joblib.dump(le_city,   LE_CITY_PATH)
joblib.dump(le_sub,    LE_SUB_PATH)
joblib.dump(le_device, LE_DEVICE_PATH)
joblib.dump(le_tenure, LE_TENURE_PATH)
print(f"    Encoders saved → {MODELS_DIR}/")

# ── 7. Split X / y ────────────────────────────
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
print(f"\n[7] Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
print(f"    Churn Rate: {y.mean():.2%}  (Balanced — No SMOTE needed)")

# ── 8. Scale Numeric Features ─────────────────
scaler = StandardScaler()
X = X.copy()
X[NUMERIC_COLS] = scaler.fit_transform(X[NUMERIC_COLS])
joblib.dump(scaler, SCALER_PATH)
print(f"\n[8] StandardScaler applied to {len(NUMERIC_COLS)} numeric features → saved")

# ── 9. Train-Test Split ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n[9] Train-Test Split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} Stratified):")
print(f"    Train : {X_train.shape}  →  Churn: {y_train.mean():.2%}")
print(f"    Test  : {X_test.shape}   →  Churn: {y_test.mean():.2%}")

# ── Save Splits & Clean Data ──────────────────
X_train.to_csv(X_TRAIN_PATH, index=False)
X_test.to_csv(X_TEST_PATH,   index=False)
y_train.to_csv(Y_TRAIN_PATH, index=False)
y_test.to_csv(Y_TEST_PATH,   index=False)
df.to_csv(CLEAN_DATA_PATH,   index=False)
joblib.dump(list(X.columns), FEATURE_NAMES_PATH)

print(f"\n{'='*60}")
print(f"  Preprocessing Complete!")
print(f"     Clean CSV   → {CLEAN_DATA_PATH}")
print(f"     Splits      → {X_TRAIN_PATH}")
print(f"     Models      → {MODELS_DIR}/")
print(f"     Features ({X.shape[1]}): {list(X.columns)}")
