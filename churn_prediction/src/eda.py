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

from config import (
    RAW_DATA_PATH, REFERENCE_DATE,
    PLOT_EDA_DEMOGRAPHICS, PLOT_EDA_DISTRIBUTIONS,
    PLOT_EDA_CORRELATION, PLOT_EDA_BEHAVIOR,
    TARGET_COLUMN, DATE_COLUMN, PLOTS_DIR
)

sns.set_style("whitegrid")

df = pd.read_csv(RAW_DATA_PATH)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df["Days Since Last Activity"] = (pd.Timestamp(REFERENCE_DATE) - df[DATE_COLUMN]).dt.days
df["ChurnLabel"] = df[TARGET_COLUMN].map({0: "Retained", 1: "Churned"})

print(f"\nDataset Shape : {df.shape}")
print(f"Churn Rate    : {df[TARGET_COLUMN].mean():.2%}")
print(f"\n── Descriptive Statistics ──")
print(df.describe().round(2).to_string())

# ── FIGURE 1: Demographics ──
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Churn Distribution & Customer Demographics", fontsize=15, fontweight="bold")

counts = df[TARGET_COLUMN].value_counts()
axes[0,0].pie(counts, labels=["Retained","Churned"], autopct="%1.1f%%",
              colors=["#2ecc71","#e74c3c"], startangle=90, explode=(0,0.05))
axes[0,0].set_title("Overall Churn Distribution")

g_churn = df.groupby(["Gender","ChurnLabel"]).size().unstack(fill_value=0)
g_churn.plot(kind="bar", ax=axes[0,1], color=["#2ecc71","#e74c3c"], rot=0, edgecolor="white")
axes[0,1].set_title("Churn by Gender"); axes[0,1].legend(title="Status")

s_churn = df.groupby(["Subscription Type","ChurnLabel"]).size().unstack(fill_value=0)
s_churn.plot(kind="bar", ax=axes[0,2], color=["#2ecc71","#e74c3c"], rot=0, edgecolor="white")
axes[0,2].set_title("Churn by Subscription Type"); axes[0,2].legend(title="Status")

for label, grp in df.groupby("ChurnLabel"):
    axes[1,0].hist(grp["Age"], bins=20, alpha=0.65, label=label,
                   color="#e74c3c" if label=="Churned" else "#2ecc71", edgecolor="white")
axes[1,0].set_title("Age Distribution by Churn"); axes[1,0].legend()

df.boxplot(column="Monthly Spending", by="ChurnLabel", ax=axes[1,1],
           boxprops=dict(color="#3498db", linewidth=2),
           medianprops=dict(color="#e74c3c", linewidth=2))
plt.sca(axes[1,1]); plt.title("Monthly Spending by Churn")

d_rate = df.groupby("Device Type")[TARGET_COLUMN].mean().sort_values(ascending=False)
bars = axes[1,2].bar(d_rate.index, d_rate.values, color=["#e74c3c","#e67e22","#3498db"], edgecolor="white")
axes[1,2].set_title("Churn Rate by Device Type"); axes[1,2].set_ylabel("Churn Rate")
for bar, val in zip(bars, d_rate.values):
    axes[1,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                   f"{val:.2%}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(PLOT_EDA_DEMOGRAPHICS, dpi=130, bbox_inches="tight")
plt.close()
print(f"\n[Saved] {PLOT_EDA_DEMOGRAPHICS}")

# ── FIGURE 2: Feature Distributions ──
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Feature Distributions vs Churn", fontsize=15, fontweight="bold")

features = ["Satisfaction Score", "Login Frequency", "Customer Support Requests",
            "Tenure", "Total Spending", "Days Since Last Activity"]
for ax, col in zip(axes.flat, features):
    for label, grp in df.groupby("ChurnLabel"):
        ax.hist(grp[col], bins=25, alpha=0.6, density=True, label=label,
                color="#e74c3c" if label=="Churned" else "#2ecc71", edgecolor="white")
    ax.set_title(col, fontweight="bold"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOT_EDA_DISTRIBUTIONS, dpi=130, bbox_inches="tight")
plt.close()
print(f"[Saved] {PLOT_EDA_DISTRIBUTIONS}")

# ── FIGURE 3: Correlation Heatmap ──
num_cols = ["Age","Monthly Spending","Tenure","Number of Purchases",
            "Customer Support Requests","Login Frequency","Satisfaction Score",
            "Total Spending","Days Since Last Activity", TARGET_COLUMN]
fig, ax = plt.subplots(figsize=(12, 9))
corr = df[num_cols].corr()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
            annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax,
            linewidths=0.5, annot_kws={"size":9})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_EDA_CORRELATION, dpi=130, bbox_inches="tight")
plt.close()
print(f"[Saved] {PLOT_EDA_CORRELATION}")

# ── FIGURE 4: Behavior Trends ──
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Customer Behavior Trends", fontsize=14, fontweight="bold")

df["Tenure Bucket"] = pd.cut(df["Tenure"], bins=[0,6,12,24,48],
                              labels=["0-6m","6-12m","12-24m","24m+"])
tb = df.groupby("Tenure Bucket", observed=True)[TARGET_COLUMN].mean()
bars = axes[0].bar(tb.index, tb.values, color=["#e74c3c","#e67e22","#f39c12","#2ecc71"], edgecolor="white")
axes[0].set_title("Churn Rate by Tenure"); axes[0].set_ylabel("Churn Rate")
for bar, val in zip(bars, tb.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{val:.2%}", ha="center", fontsize=9)

churned  = df[df[TARGET_COLUMN]==1]
retained = df[df[TARGET_COLUMN]==0]
axes[1].scatter(retained["Satisfaction Score"], retained["Customer Support Requests"],
                alpha=0.3, color="#2ecc71", label="Retained", s=20)
axes[1].scatter(churned["Satisfaction Score"],  churned["Customer Support Requests"],
                alpha=0.4, color="#e74c3c", label="Churned",  s=20)
axes[1].set_xlabel("Satisfaction Score"); axes[1].set_ylabel("Support Requests")
axes[1].set_title("Satisfaction vs Support Requests"); axes[1].legend()

axes[2].scatter(retained["Login Frequency"], retained["Monthly Spending"],
                alpha=0.3, color="#2ecc71", label="Retained", s=20)
axes[2].scatter(churned["Login Frequency"],  churned["Monthly Spending"],
                alpha=0.4, color="#e74c3c", label="Churned",  s=20)
axes[2].set_xlabel("Login Frequency"); axes[2].set_ylabel("Monthly Spending ($)")
axes[2].set_title("Login Frequency vs Spending"); axes[2].legend()

plt.tight_layout()
plt.savefig(PLOT_EDA_BEHAVIOR, dpi=130, bbox_inches="tight")
plt.close()
print(f"[Saved] {PLOT_EDA_BEHAVIOR}")

print("\n── Mean Values: Retained vs Churned ──")
cols = ["Monthly Spending","Login Frequency","Satisfaction Score",
        "Customer Support Requests","Tenure","Total Spending"]
print(df.groupby(TARGET_COLUMN)[cols].mean().round(2).to_string())
print(f"\nEDA complete — 4 plots saved in {PLOTS_DIR}/")
