import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     roc_curve, classification_report)
from xgboost import XGBClassifier
import joblib

from config import (
    X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
    BEST_MODEL_PATH, BEST_MODEL_NAME_PATH, FEATURE_NAMES_PATH,
    MODEL_RF_PATH, MODEL_XGB_PATH, MODEL_LR_PATH, MODEL_SVM_PATH,
    MODEL_COMPARISON_PATH,
    PLOT_MODEL_ROC, PLOT_MODEL_CONFUSION, PLOT_MODEL_IMPORTANCE,
    RANDOM_STATE, CV_FOLDS, BEST_MODEL_METRIC, MODELS_DIR, PLOTS_DIR
)



X_train = pd.read_csv(X_TRAIN_PATH)
X_test  = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_test  = pd.read_csv(Y_TEST_PATH).squeeze()

print(f"\nTrain : {X_train.shape}  →  Churn: {y_train.mean():.2%}")
print(f"Test  : {X_test.shape}   →  Churn: {y_test.mean():.2%}")
print(f"Note  : Balanced dataset — SMOTE not required.\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=12,
                                                   min_samples_leaf=3, max_features="sqrt",
                                                   random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8,
                                         use_label_encoder=False, eval_metric="logloss",
                                         random_state=RANDOM_STATE, verbosity=0),
    "SVM":                 SVC(kernel="rbf", C=1.5, gamma="scale",
                               probability=True, random_state=RANDOM_STATE),
}

model_save_paths = {
    "Logistic Regression": MODEL_LR_PATH,
    "Random Forest":       MODEL_RF_PATH,
    "XGBoost":             MODEL_XGB_PATH,
    "SVM":                 MODEL_SVM_PATH,
}

cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, model in models.items():
    print(f"  Training: {name} ...")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy":      accuracy_score(y_test, y_pred),
        "Precision":     precision_score(y_test, y_pred),
        "Recall":        recall_score(y_test, y_pred),
        "F1":            f1_score(y_test, y_pred),
        "ROC-AUC":       roc_auc_score(y_test, y_proba),
        "CV-F1(Train)":  cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean(),
        "model": model, "y_pred": y_pred, "y_proba": y_proba
    }

    print(f"    Accuracy={results[name]['Accuracy']:.4f}  "
          f"F1={results[name]['F1']:.4f}  "
          f"ROC-AUC={results[name]['ROC-AUC']:.4f}\n")

    joblib.dump(model, model_save_paths[name])

# Best model by config metric
best_name = max(results, key=lambda k: results[k][BEST_MODEL_METRIC])
joblib.dump(results[best_name]["model"], BEST_MODEL_PATH)
joblib.dump(best_name,                   BEST_MODEL_NAME_PATH)

print(f"{'='*60}")
print(f"  BEST MODEL ({BEST_MODEL_METRIC}): {best_name}")
print(f"  Accuracy : {results[best_name]['Accuracy']:.4f}")
print(f"  ROC-AUC  : {results[best_name]['ROC-AUC']:.4f}")
print(f"{'='*60}")

# Save comparison
metrics_df = pd.DataFrame({
    k: {m: v for m, v in v.items() if m not in ("model","y_pred","y_proba")}
    for k, v in results.items()
}).T.round(4)
metrics_df.to_csv(MODEL_COMPARISON_PATH)
print(f"\n── Model Comparison ──\n{metrics_df.to_string()}")
print(f"\n── Classification Report: {best_name} ──")
print(classification_report(y_test, results[best_name]["y_pred"],
                             target_names=["Retained","Churned"]))

# ── Plots ─────────────────────────────────────
colors = ["#3498db","#e74c3c","#2ecc71","#9b59b6"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['ROC-AUC']:.3f})", color=color, lw=2)
axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
axes[0].set_title("ROC Curves"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].legend(fontsize=9)
x = np.arange(5); width = 0.18
for i,(name,res) in enumerate(results.items()):
    vals = [res[m] for m in ["Accuracy","Precision","Recall","F1","ROC-AUC"]]
    axes[1].bar(x+i*width, vals, width, label=name, color=colors[i], alpha=0.85)
axes[1].set_xticks(x+width*1.5)
axes[1].set_xticklabels(["Accuracy","Precision","Recall","F1","ROC-AUC"], rotation=15)
axes[1].set_title("Metric Comparison"); axes[1].set_ylim(0,1.15); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_MODEL_ROC, dpi=130, bbox_inches="tight"); plt.close()
print(f"\n[Saved] {PLOT_MODEL_ROC}")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"])
    ax.set_title(name, fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(PLOT_MODEL_CONFUSION, dpi=130, bbox_inches="tight"); plt.close()
print(f"[Saved] {PLOT_MODEL_CONFUSION}")

feat_names = joblib.load(FEATURE_NAMES_PATH)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Feature Importance Analysis", fontsize=14, fontweight="bold")
for ax, (mname, color) in zip(axes, [("Random Forest","#3498db"),("XGBoost","#e74c3c")]):
    imps = results[mname]["model"].feature_importances_
    idx  = np.argsort(imps)
    ax.barh([feat_names[i] for i in idx], imps[idx], color=color, alpha=0.8)
    ax.set_title(f"{mname} — Feature Importance"); ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_MODEL_IMPORTANCE, dpi=130, bbox_inches="tight"); plt.close()
print(f"[Saved] {PLOT_MODEL_IMPORTANCE}")

print(f"\nTraining Complete! Models saved → {MODELS_DIR}/")
