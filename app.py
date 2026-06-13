import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder

# ── Page Config ──
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .section-title {
        font-size: 18px; font-weight: 700;
        color: #e2e8f0; margin: 24px 0 12px;
        border-bottom: 1px solid #2d3748; padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──
def preprocess(df):
    """Same preprocessing as notebook — no training, just cleaning."""
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if col not in ['Churn', 'customerID']:
            df[col] = le.fit_transform(df[col])
    if 'Churn' in df.columns and df['Churn'].dtype == object:
        df['Churn'] = le.fit_transform(df['Churn'])

    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    if 'tenure' in df.columns:
        df['TenureGroup'] = pd.cut(df['tenure'],
                                   bins=[0,12,24,48,72,9999],
                                   labels=[1,2,3,4,5]).astype('Int64').astype(float)
    if 'SeniorCitizen' in df.columns and 'tenure' in df.columns:
        df['SeniorTenureInteraction'] = df['SeniorCitizen'] * df['tenure']
    return df


def risk_label(prob):
    if prob >= 0.7:   return 'High Risk'
    elif prob >= 0.4: return 'Medium Risk'
    else:             return 'Low Risk'


def segment(val, q33, q66):
    if val >= q66:   return 'High Value'
    elif val >= q33: return 'Medium Value'
    else:            return 'Low Value'


# ── Load saved models ──
MODEL_DIR = 'models'

@st.cache_resource
def load_models():
    rf     = joblib.load(os.path.join(MODEL_DIR, 'random_forest.pkl'))
    lr     = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    return rf, lr, scaler

if not os.path.exists(MODEL_DIR):
    st.error("❌ 'models/' folder not present, run notebook first.")
    st.stop()

rf, lr, scaler = load_models()

# Feature columns (same order as training) — saved from notebook
FEATURE_COLS_FILE = os.path.join(MODEL_DIR, 'feature_cols.pkl')
if os.path.exists(FEATURE_COLS_FILE):
    feature_cols = joblib.load(FEATURE_COLS_FILE)
else:
    feature_cols = None


# ── Sidebar ──
st.sidebar.title("📊 Churn Dashboard")
st.sidebar.markdown("**Teyzix Core — ML-INT-1**")
st.sidebar.markdown("---")
st.sidebar.markdown("**CSV Upload** *(optional)*")
predict_file = st.sidebar.file_uploader("Upload customers CSV", type=["csv"])
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Overview", "Visualizations", "Predict Customers"])

# ── CSV load karo agar upload hui ho ──
df = None
X = None
probs = None
id_col = None

if predict_file is not None:
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)

    raw_df = load_csv(predict_file)
    df = preprocess(raw_df.copy())

    id_col = df['customerID'] if 'customerID' in df.columns else pd.Series(range(len(df)), name='Index')
    drop_cols = [c for c in ['Churn', 'CustomerSegment', 'customerID'] if c in df.columns]
    X = df.drop(columns=drop_cols)

    if feature_cols is not None:
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols]

    X = X.fillna(0)
    probs = rf.predict_proba(X)[:, 1]
    df['ChurnProbability'] = probs
    df['RiskCategory']     = [risk_label(p) for p in probs]

    charge_col = "MonthlyCharges" if "MonthlyCharges" in X.columns else None
    if charge_col:
        q33 = X[charge_col].quantile(0.33)
        q66 = X[charge_col].quantile(0.66)
        df["CustomerSegment"] = X[charge_col].apply(lambda v: segment(v, q33, q66))
    else:
        df["CustomerSegment"] = "N/A"


# ══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════
if page == "Overview":
    st.title("📊 Customer Behavior Analytics")
    if df is None:
        st.info("👈 upload a CSV file to see the overview.")
    else:
        st.markdown("---")
        total     = len(df)
        high_risk = (df['RiskCategory'] == 'High Risk').sum()
        med_risk  = (df['RiskCategory'] == 'Medium Risk').sum()
        low_risk  = (df['RiskCategory'] == 'Low Risk').sum()
    
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{total:,}")
        c2.metric("High Risk",       f"{high_risk:,}")
        c3.metric("Medium Risk",     f"{med_risk:,}")
        c4.metric("Low Risk",        f"{low_risk:,}")
    
        if df is not None and 'Churn' in df.columns:
            churned    = int(df['Churn'].sum()) if 'Churn' in df.columns else 0
            churn_rate = churned / total * 100
            st.info(f"📌 Actual Churn in this file: **{churned}** customers ({churn_rate:.1f}%)")
    
        st.markdown("---")
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown('<div class="section-title">Customer Segmentation</div>', unsafe_allow_html=True)
            seg_counts = df['CustomerSegment'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            seg_counts.plot(kind='bar', color=['#f7b731','#a29bfe','#cd6133'], ax=ax)
            ax.set_title('Customer Segments', color='white')
            ax.tick_params(colors='white'); ax.set_ylabel('Count', color='white')
            plt.xticks(rotation=15)
            st.pyplot(fig)
    
        with col2:
            st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
            risk_counts = df['RiskCategory'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            fig2.patch.set_facecolor('#1e2130'); ax2.set_facecolor('#1e2130')
            color_map = {'High Risk': '#fc5c65', 'Medium Risk': '#f7b731', 'Low Risk': '#26de81'}
            bars  = [risk_counts.get(k, 0) for k in color_map]
            ax2.bar(list(color_map.keys()), bars, color=list(color_map.values()))
            ax2.set_title('Churn Risk Categories', color='white')
            ax2.tick_params(colors='white'); ax2.set_ylabel('Count', color='white')
            st.pyplot(fig2)
    
        if 'MonthlyCharges' in df.columns:
            st.markdown("---")
            rev_at_risk = df[df['RiskCategory'] == 'High Risk']['MonthlyCharges'].sum()
            total_rev   = df['MonthlyCharges'].sum()
            st.markdown(f"💰 **Revenue at Risk:** ${rev_at_risk:,.2f} / ${total_rev:,.2f} total "
                        f"({rev_at_risk/total_rev*100:.1f}%)")
    
    

# ══════════════════════════════════════════════
# PAGE 2 — VISUALIZATIONS
# ══════════════════════════════════════════════
elif page == "Visualizations":
    st.title("📈 Data Visualizations")
    if df is None:
        st.info("👈 For visualizations, please upload a CSV file.")
    else:
        tab1, tab2, tab3 = st.tabs(["Revenue & Churn", "Correlation Heatmap", "Feature Importance"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
                df['RiskCategory'].value_counts().plot(kind='bar',
                    color=['#fc5c65','#f7b731','#26de81'], ax=ax)
                ax.set_title('Risk Category Distribution', color='white')
                ax.tick_params(colors='white'); ax.set_ylabel('Count', color='white')
                plt.xticks(rotation=15)
                st.pyplot(fig)

            with col2:
                if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
                    trend = df.groupby('tenure')['MonthlyCharges'].mean()
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    fig2.patch.set_facecolor('#1e2130'); ax2.set_facecolor('#1e2130')
                    trend.plot(ax=ax2, color='#26de81')
                    ax2.set_title('Avg Monthly Charges by Tenure', color='white')
                    ax2.tick_params(colors='white')
                    ax2.set_xlabel('Tenure (months)', color='white')
                    ax2.set_ylabel('Avg Charges', color='white')
                    st.pyplot(fig2)

        with tab2:
            fig3, ax3 = plt.subplots(figsize=(10, 7))
            fig3.patch.set_facecolor('#1e2130'); ax3.set_facecolor('#1e2130')
            corr = df.select_dtypes(include='number').corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                        linewidths=0.5, ax=ax3, annot_kws={'color':'white','size':7})
            ax3.set_title('Correlation Heatmap', color='white')
            ax3.tick_params(colors='white')
            st.pyplot(fig3)

        with tab3:
            importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            fig4.patch.set_facecolor('#1e2130'); ax4.set_facecolor('#1e2130')
            importances.sort_values().plot(kind='barh', color='#4f8ef7', ax=ax4)
            ax4.set_title('Top 10 Feature Importances (Random Forest)', color='white')
            ax4.tick_params(colors='white')
            st.pyplot(fig4)


# ══════════════════════════════════════════════
# PAGE 3 — PREDICT CUSTOMERS
# ══════════════════════════════════════════════
elif page == "Predict Customers":
    st.title("🔍 Customer Churn Predictions")

    tab1, tab2 = st.tabs(["📋 All Customers (CSV)", "👤 Single Customer"])

    with tab1:
        if df is None:
            st.info("👈 Upload a CSV file to view customer predictions.")
        else:
            result_df = pd.DataFrame({
                'CustomerID':       id_col.values,
                'ChurnProbability': (probs * 100).round(2),
                'RiskCategory':     [risk_label(p) for p in probs]
            })

            c1, c2, c3 = st.columns(3)
            c1.metric("High Risk",   (result_df['RiskCategory'] == 'High Risk').sum())
            c2.metric("Medium Risk", (result_df['RiskCategory'] == 'Medium Risk').sum())
            c3.metric("Low Risk",    (result_df['RiskCategory'] == 'Low Risk').sum())

            st.markdown("---")

            filter_risk = st.selectbox("Filter by Risk", ["All", "High Risk", "Medium Risk", "Low Risk"])
            show_df = result_df if filter_risk == "All" else result_df[result_df['RiskCategory'] == filter_risk]

            st.dataframe(show_df, use_container_width=True)

            csv_out = show_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

    with tab2:
        st.markdown("Increase cunstomer details manually to predict churn.")
        st.markdown("---")

        # feature_cols use karo — CSV upload ki zaroorat nahi
        use_cols = feature_cols if feature_cols is not None else (list(X.columns) if X is not None else [])

        if not use_cols:
            st.warning("Models ka feature_cols.pkl not present, run notebook first.")
        else:
            input_vals = {}
            cols = st.columns(3)
            for i, col in enumerate(use_cols):
                input_vals[col] = cols[i % 3].number_input(
                    col, min_value=0.0, max_value=999999.0, value=0.0, key=f"single_{col}"
                )

            st.markdown("---")
            if st.button("Predict", type="primary"):
                input_df = pd.DataFrame([input_vals])
                prob = rf.predict_proba(input_df)[0][1]
                risk = risk_label(prob)
                color = {"High Risk": "#fc5c65", "Medium Risk": "#f7b731", "Low Risk": "#26de81"}[risk]
                c1, c2 = st.columns(2)
                c1.metric("Churn Probability", f"{prob*100:.1f}%")
                c2.markdown(
                    f"**Risk Category:** <span style=\'color:{color}; font-size:24px; font-weight:700\'>{risk}</span>",
                    unsafe_allow_html=True
                )
                st.progress(float(prob))