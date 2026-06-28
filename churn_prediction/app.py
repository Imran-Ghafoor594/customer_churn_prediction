import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import sys, os, time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from config import (
    BEST_MODEL_PATH, BEST_MODEL_NAME_PATH, FEATURE_NAMES_PATH,
    SCALER_PATH, LE_GENDER_PATH, LE_CITY_PATH, LE_SUB_PATH,
    LE_DEVICE_PATH, LE_TENURE_PATH, MODEL_RF_PATH, MODEL_XGB_PATH,
    MODEL_LR_PATH, MODEL_SVM_PATH, MODEL_COMPARISON_PATH, RAW_DATA_PATH,
    NUMERIC_COLS, CHURN_THRESHOLD, RISK_LOW_MAX, RISK_MEDIUM_MAX,
    GENDER_OPTIONS, CITY_OPTIONS, SUBSCRIPTION_OPTIONS, DEVICE_OPTIONS
)

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ────────────────────────────────────────────
st.markdown("""
<style>
/* ── MAIN BACKGROUND ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
}

/* ── HIDE SIDEBAR ── */
[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* ── TEXT COLORS ── */
.stApp p, .stApp span, .stApp div, .stApp label {
    color: #ffffff !important;
}
h1, h2, h3, h4, h5 { color: #ffffff !important; }

/* ── METRIC CARDS ── */
[data-testid="metric-container"] {
    background: #1e3a5f !important;
    border: 1px solid #2196F3 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="metric-container"] label {
    color: #90caf9 !important;
    font-size: 0.85rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #90caf9 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: #1565c0 !important;
    color: #ffffff !important;
}

/* ── BUTTON ── */
div.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 40px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    box-shadow: 0 4px 20px rgba(21,101,192,0.6) !important;
    transition: all 0.3s ease !important;
    width: auto !important;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(21,101,192,0.8) !important;
}

/* ── ALERT BOXES ── */
.stAlert { background: rgba(33,150,243,0.15) !important; color: #ffffff !important; }
.stAlert p { color: #ffffff !important; }

/* ── SELECTBOX / INPUTS ── */
[data-testid="stSelectbox"] > div > div {
    background: #1e3a5f !important;
    color: #ffffff !important;
    border: 1px solid #2196F3 !important;
}
.stNumberInput input {
    background: #1e3a5f !important;
    color: #ffffff !important;
    border: 1px solid #2196F3 !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] p { color: #ffffff !important; }

/* ── DIVIDER ── */
hr { border-color: #1565c0 !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { background: #1e3a5f !important; }
[data-testid="stDataFrame"] * { color: #ffffff !important; }

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── CUSTOM CARDS ── */
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #0f2440);
    border: 1px solid #2196F3;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(33,150,243,0.25);
    margin: 8px 0;
}
.metric-card h3 { color: #64b5f6 !important; margin: 8px 0 4px 0; }
.metric-card p  { color: #b0bec5 !important; margin: 0; font-size: 0.88rem; }

.risk-high   { background: linear-gradient(135deg,#c0392b,#e74c3c); border-radius:12px; padding:18px; text-align:center; font-size:1.5rem; font-weight:800; box-shadow:0 0 30px rgba(231,76,60,0.6); color:#ffffff !important; }
.risk-medium { background: linear-gradient(135deg,#d35400,#e67e22); border-radius:12px; padding:18px; text-align:center; font-size:1.5rem; font-weight:800; box-shadow:0 0 30px rgba(230,126,34,0.6); color:#ffffff !important; }
.risk-low    { background: linear-gradient(135deg,#1a6b3c,#27ae60); border-radius:12px; padding:18px; text-align:center; font-size:1.5rem; font-weight:800; box-shadow:0 0 30px rgba(39,174,96,0.6); color:#ffffff !important; }

.rec-box    { background:rgba(33,150,243,0.12); border-left:4px solid #2196F3; border-radius:8px; padding:12px 16px; margin:8px 0; }
.rec-urgent { background:rgba(231,76,60,0.15); border-left:4px solid #e74c3c; border-radius:8px; padding:12px 16px; margin:8px 0; }
.rec-box span, .rec-urgent span { color: #e0e0e0 !important; }

.summary-row {
    display:flex; justify-content:space-between;
    padding:6px 10px; border-bottom:1px solid rgba(255,255,255,0.1);
}
.summary-row .sk { color:#90caf9 !important; font-size:0.85rem; }
.summary-row .sv { color:#ffffff !important; font-weight:700; font-size:0.85rem; }

.input-section {
    background: rgba(30, 58, 95, 0.4);
    border: 1px solid rgba(33, 150, 243, 0.3);
    border-radius: 16px;
    padding: 25px;
    margin: 20px 0;
    backdrop-filter: blur(5px);
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ────────────────────────────────────
@st.cache_resource
def load_artifacts():
    return {
        "best_model":  joblib.load(BEST_MODEL_PATH),
        "best_name":   joblib.load(BEST_MODEL_NAME_PATH),
        "rf":          joblib.load(MODEL_RF_PATH),
        "xgb":         joblib.load(MODEL_XGB_PATH),
        "lr":          joblib.load(MODEL_LR_PATH),
        "svm":         joblib.load(MODEL_SVM_PATH),
        "scaler":      joblib.load(SCALER_PATH),
        "le_gender":   joblib.load(LE_GENDER_PATH),
        "le_city":     joblib.load(LE_CITY_PATH),
        "le_sub":      joblib.load(LE_SUB_PATH),
        "le_device":   joblib.load(LE_DEVICE_PATH),
        "le_tenure":   joblib.load(LE_TENURE_PATH),
        "feat_names":  joblib.load(FEATURE_NAMES_PATH),
    }

art = load_artifacts()

MODEL_COLORS      = ["#42a5f5","#ef5350","#66bb6a","#ab47bc"]
MODEL_FILL_COLORS = ["rgba(66,165,245,0.2)","rgba(239,83,80,0.2)",
                     "rgba(102,187,106,0.2)","rgba(171,71,188,0.2)"]
CHART_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,40,0.6)",
                font=dict(color="#ffffff", size=12))
GRID = "rgba(255,255,255,0.12)"

# ─── PREDICTION FUNCTIONS ──────────────────────────
def preprocess(c):
    tenure = c["Tenure"]
    spt = c["Monthly Spending"] / (tenure + 1)
    eng = c["Login Frequency"]*0.5 + c["Number of Purchases"]*0.3 - c["Days Since Last Activity"]*0.2
    sb  = c["Customer Support Requests"] / (c["Satisfaction Score"] + 0.1)
    hv  = int(c["Monthly Spending"] > 80 and c["Number of Purchases"] > 8)
    if   tenure <= 6:  tb = "New(0-6m)"
    elif tenure <= 12: tb = "Short(6-12m)"
    elif tenure <= 24: tb = "Mid(12-24m)"
    else:              tb = "Long(24m+)"
    row = {
        "Age":c["Age"],
        "Gender":                    art["le_gender"].transform([c["Gender"]])[0],
        "City":                      art["le_city"].transform([c["City"]])[0],
        "Subscription Type":         art["le_sub"].transform([c["Subscription Type"]])[0],
        "Monthly Spending":          c["Monthly Spending"],
        "Tenure":                    tenure,
        "Number of Purchases":       c["Number of Purchases"],
        "Customer Support Requests": c["Customer Support Requests"],
        "Login Frequency":           c["Login Frequency"],
        "Satisfaction Score":        c["Satisfaction Score"],
        "Device Type":               art["le_device"].transform([c["Device Type"]])[0],
        "Total Spending":            c["Total Spending"],
        "Days Since Last Activity":  c["Days Since Last Activity"],
        "Spending Per Tenure":       spt,
        "Engagement Score":          eng,
        "Support Burden":            sb,
        "Tenure Bucket":             art["le_tenure"].transform([tb])[0],
        "High Value":                hv,
    }
    X = pd.DataFrame([row])[art["feat_names"]]
    X[NUMERIC_COLS] = art["scaler"].transform(X[NUMERIC_COLS])
    return X

def get_recs(c):
    recs = []
    if c["Satisfaction Score"] <= 2:
        recs.append(("URGENT", "Satisfaction critically low! Assign account manager immediately.", "urgent"))
    elif c["Satisfaction Score"] <= 3:
        recs.append(("Action", "Low satisfaction. Schedule customer feedback call this week.", "warn"))
    if c["Customer Support Requests"] >= 5:
        recs.append(("Support", "High support volume. Escalate tickets and offer priority SLA.", "warn"))
    if c["Login Frequency"] <= 3:
        recs.append(("Re-engage", "Very low logins. Launch personalized email/SMS campaign.", "info"))
    if c["Days Since Last Activity"] >= 30:
        recs.append(("Offer", f"Inactive {c['Days Since Last Activity']} days. Send loyalty discount.", "info"))
    if c["Tenure"] <= 6:
        recs.append(("Onboard", "New customer ≤6 months. Assign onboarding specialist.", "info"))
    if not recs:
        recs.append(("Healthy", "Customer looks stable. Maintain regular check-ins.", "ok"))
    return recs

# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:20px 0 10px 0;'>
    <h1 style='font-size:2.5rem; font-weight:900; color:#64b5f6 !important; margin:0;'>
        Customer Churn Prediction System
    </h1>
    <p style='color:#90caf9 !important; font-size:0.95rem; margin-top:6px;'>
        Teyzix Core Internship &nbsp;·&nbsp; Task ML-2 &nbsp;·&nbsp; Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Prediction", "Analytics", "Model Performance", "About"])

# ══════════════════════════════════════════
# TAB 1 — PREDICTION (WITH INPUTS)
# ══════════════════════════════════════════
with tabs[0]:
    # Initialize session state for prediction
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    
    # ─── INPUT SECTION ──────────────────
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.slider("Age", 18, 70, 32, key="age")
        gender = st.selectbox("Gender", GENDER_OPTIONS, key="gender")
        city = st.selectbox("City", CITY_OPTIONS, key="city")
    
    with col2:
        st.markdown("#### 📋 Subscription")
        sub = st.selectbox("Subscription Type", SUBSCRIPTION_OPTIONS, key="sub")
        device = st.selectbox("Device Type", DEVICE_OPTIONS, key="device")
        
        st.markdown("#### 💰 Financial")
        spend = st.number_input("Monthly Spending ($)", 10.0, 200.0, 55.0, step=0.5, key="spend")
        tenure = st.slider("Tenure (months)", 1, 48, 12, key="tenure")
    
    with col3:
        st.markdown("#### 📊 Behaviour")
        purchases = st.slider("Number of Purchases", 1, 24, 8, key="purchases")
        support = st.slider("Support Requests", 0, 7, 2, key="support")
        login = st.slider("Login Frequency / month", 1, 30, 12, key="login")
        days = st.slider("Days Since Last Activity", 0, 180, 15, key="days")
        satis = st.select_slider(
            "Satisfaction Score",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: f"{'⭐'*x} ({x}/5)",
            key="satis"
        )
    
    # Calculate total spending
    total = round(spend * tenure, 2)
    st.markdown(f"""
    <div style='background:#1e3a5f; border:1px solid #2196F3; border-radius:10px;
                padding:12px 20px; margin:10px 0; text-align:center;'>
        <span style='color:#90caf9; font-size:0.9rem;'>💰 Total Spending</span><br>
        <span style='color:#ffffff; font-size:1.4rem; font-weight:800;'>${total:,.2f}</span>
    </div>""", unsafe_allow_html=True)
    
    # ─── PREDICT BUTTON ──────────────────
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_btn = st.button("🔮  PREDICT CHURN", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ─── PREDICTION RESULTS ──────────────
    if predict_btn:
        st.session_state.show_prediction = True
        
        customer = {
            "Age": age,
            "Gender": gender,
            "City": city,
            "Subscription Type": sub,
            "Device Type": device,
            "Monthly Spending": spend,
            "Tenure": tenure,
            "Number of Purchases": purchases,
            "Customer Support Requests": support,
            "Login Frequency": login,
            "Days Since Last Activity": days,
            "Satisfaction Score": satis,
            "Total Spending": total,
        }
        
        with st.spinner("Analyzing customer profile..."):
            time.sleep(0.4)
            X = preprocess(customer)
            prob = art["best_model"].predict_proba(X)[0][1]
            pred = int(prob >= CHURN_THRESHOLD)
            all_probs = {
                "Random Forest": art["rf"].predict_proba(X)[0][1],
                "XGBoost": art["xgb"].predict_proba(X)[0][1],
                "Logistic Regression": art["lr"].predict_proba(X)[0][1],
                "SVM": art["svm"].predict_proba(X)[0][1],
            }
            if prob < RISK_LOW_MAX:
                risk, rc, em = "LOW RISK", "risk-low", "🟢"
            elif prob < RISK_MEDIUM_MAX:
                risk, rc, em = "MEDIUM RISK", "risk-medium", "🟡"
            else:
                risk, rc, em = "HIGH RISK", "risk-high", "🔴"
        
        st.divider()
        st.markdown("### 📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='{rc}'>{em} {risk}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            vc = "#e74c3c" if pred else "#2ecc71"
            vt = "⚠️ WILL CHURN" if pred else "✅ WILL STAY"
            st.markdown(f"""
            <div style='background:#1e3a5f; border-radius:12px; padding:22px;
                        text-align:center; border:2px solid {vc};
                        box-shadow: 0 0 25px {vc}55;'>
                <div style='font-size:1.9rem; font-weight:900; color:{vc};'>{vt}</div>
                <div style='color:#90caf9; margin-top:10px; font-size:1rem;'>
                    Churn Probability:
                    <b style='font-size:1.5rem; color:{vc};'>{prob*100:.1f}%</b>
                </div>
                <div style='color:#78909c; font-size:0.82rem; margin-top:6px;'>
                    Model: {art["best_name"]}
                </div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            bc = "#e74c3c" if prob>=0.6 else ("#e67e22" if prob>=0.35 else "#2ecc71")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob*100,1),
                number={"suffix":"%","font":{"size":34,"color":"#ffffff"}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#ffffff","tickfont":{"color":"#ffffff"}},
                    "bar":{"color":bc,"thickness":0.35},
                    "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                    "steps":[{"range":[0,35],"color":"rgba(39,174,96,0.2)"},
                              {"range":[35,60],"color":"rgba(230,126,34,0.2)"},
                              {"range":[60,100],"color":"rgba(231,76,60,0.2)"}],
                    "threshold":{"line":{"color":"#ffffff","width":3},"value":50}
                },
                title={"text":"Churn Probability","font":{"color":"#90caf9","size":15}}
            ))
            fig.update_layout(**CHART_BG, height=230, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        col3, col4 = st.columns([1.3, 0.7])
        with col3:
            st.markdown("#### 🔍 Top Contributing Factors")
            rf_imps = art["rf"].feature_importances_
            feat_df = pd.DataFrame({"Feature": art["feat_names"], "Importance": rf_imps})
            feat_df = feat_df.sort_values("Importance", ascending=True).tail(10)
            fig2 = go.Figure(go.Bar(
                x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
                marker_color="#42a5f5",
                text=[f"{v:.4f}" for v in feat_df["Importance"]],
                textposition="outside", textfont=dict(color="#ffffff",size=10)
            ))
            fig2.update_layout(**CHART_BG, height=330,
                               margin=dict(t=10,b=10,l=10,r=70),
                               xaxis=dict(showgrid=True,gridcolor=GRID,color="#ffffff"),
                               yaxis=dict(color="#ffffff",tickfont=dict(size=11,color="#ffffff")))
            st.plotly_chart(fig2, use_container_width=True)
        
        with col4:
            st.markdown("#### 🤖 All Model Predictions")
            for mname, mprob in all_probs.items():
                mpct = mprob*100
                bc2 = "#e74c3c" if mprob>=0.6 else ("#e67e22" if mprob>=0.35 else "#2ecc71")
                star = "⭐ " if mname==art["best_name"] else ""
                st.markdown(f"""
                <div style='margin:10px 0;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
                        <span style='color:#e0e0e0; font-size:0.85rem; font-weight:600;'>{star}{mname}</span>
                        <span style='color:{bc2}; font-weight:800; font-size:0.95rem;'>{mpct:.1f}%</span>
                    </div>
                    <div style='background:rgba(255,255,255,0.12); border-radius:20px; height:12px; overflow:hidden;'>
                        <div style='background:{bc2}; width:{mpct}%; height:100%; border-radius:20px;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        
        st.divider()
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("#### 📋 Customer Summary")
            for k, v in [
                ("Age", f"{age} yrs"), ("Gender", gender), ("City", city),
                ("Device", device), ("Plan", sub), ("Monthly", f"${spend:.2f}"),
                ("Tenure", f"{tenure} months"), ("Purchases", purchases),
                ("Support Tickets", support), ("Logins/Month", login),
                ("Inactive Days", days), ("Satisfaction", f"{'⭐'*satis} ({satis}/5)"),
                ("Total Spend", f"${total:,.2f}")
            ]:
                st.markdown(f"""
                <div class='summary-row'>
                    <span class='sk'>{k}</span>
                    <span class='sv'>{v}</span>
                </div>""", unsafe_allow_html=True)
        
        with col6:
            st.markdown("#### 💡 Recommendations")
            for label, text, rtype in get_recs(customer):
                css = "rec-urgent" if rtype == "urgent" else "rec-box"
                lc = "#ef5350" if rtype == "urgent" else "#64b5f6"
                st.markdown(f"""
                <div class='{css}'>
                    <b style='color:{lc}; font-size:0.95rem;'>{label}</b><br>
                    <span style='color:#e0e0e0; font-size:0.88rem;'>{text}</span>
                </div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📡 Customer Radar")
            rvals = [satis/5, min(login/30,1), min(tenure/48,1), min(spend/120,1), 1-min(support/7,1)]
            rcats = ["Satisfaction", "Engagement", "Loyalty", "Spending", "Support OK"]
            fig3 = go.Figure(go.Scatterpolar(
                r=rvals+[rvals[0]], theta=rcats+[rcats[0]],
                fill="toself",
                fillcolor="rgba(66,165,245,0.2)",
                line=dict(color="#42a5f5", width=2),
                marker=dict(color="#42a5f5", size=7)
            ))
            fig3.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)",
                           radialaxis=dict(visible=True, range=[0, 1],
                                          gridcolor=GRID, color="#ffffff",
                                          tickfont=dict(color="#ffffff")),
                           angularaxis=dict(color="#ffffff", tickfont=dict(size=11, color="#ffffff"))),
                **CHART_BG, height=260,
                margin=dict(t=20, b=10, l=20, r=20), showlegend=False
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    else:
        # Show info cards when no prediction
        c1, c2, c3 = st.columns(3)
        for col, icon, title, desc in [
            (c1, "🎯", "91% Accuracy", "Random Forest best model"),
            (c2, "⚡", "0.98 ROC-AUC", "Excellent discrimination power"),
            (c3, "💡", "AI Insights", "Actionable retention recommendations"),
        ]:
            col.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:2.2rem;'>{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("📝 Fill in the customer details above and click **PREDICT CHURN** to get insights.")

# ══════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════
with tabs[1]:
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        df["Churn Label"] = df["Churn Status"].map({0: "Retained", 1: "Churned"})
        st.markdown("### 📊 Dataset Analytics")
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Customers", f"{len(df):,}")
        k2.metric("Churned", f"{(df['Churn Status']==1).sum():,}")
        k3.metric("Churn Rate", f"{df['Churn Status'].mean()*100:.1f}%")
        k4.metric("Avg Satisfaction", f"{df['Satisfaction Score'].mean():.1f}/5")
        k5.metric("Avg Monthly Spend", f"${df['Monthly Spending'].mean():.0f}")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            sd = df.groupby(["Subscription Type", "Churn Label"]).size().reset_index(name="Count")
            fig = px.bar(sd, x="Subscription Type", y="Count", color="Churn Label",
                         barmode="group",
                         color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
                         title="Churn by Subscription Type")
            fig.update_layout(**CHART_BG)
            fig.update_xaxes(color="#ffffff", gridcolor=GRID)
            fig.update_yaxes(color="#ffffff", gridcolor=GRID)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="Satisfaction Score", color="Churn Label",
                               nbins=5, barmode="overlay", opacity=0.75,
                               color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
                               title="Satisfaction Score Distribution")
            fig.update_layout(**CHART_BG)
            fig.update_xaxes(color="#ffffff", gridcolor=GRID)
            fig.update_yaxes(color="#ffffff", gridcolor=GRID)
            st.plotly_chart(fig, use_container_width=True)
        
        c3, c4 = st.columns(2)
        with c3:
            fig = px.scatter(df, x="Monthly Spending", y="Total Spending",
                             color="Churn Label", size="Tenure", opacity=0.7,
                             color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
                             title="Monthly vs Total Spending")
            fig.update_layout(**CHART_BG)
            fig.update_xaxes(color="#ffffff", gridcolor=GRID)
            fig.update_yaxes(color="#ffffff", gridcolor=GRID)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            cr = df.groupby("City")["Churn Status"].mean().reset_index()
            cr.columns = ["City", "Churn Rate"]
            cr["Churn Rate %"] = (cr["Churn Rate"]*100).round(1)
            fig = px.bar(cr, x="City", y="Churn Rate %",
                         color="Churn Rate %",
                         color_continuous_scale=["#2ecc71", "#e67e22", "#e74c3c"],
                         title="Churn Rate by City (%)")
            fig.update_layout(**CHART_BG, showlegend=False)
            fig.update_xaxes(color="#ffffff", gridcolor=GRID)
            fig.update_yaxes(color="#ffffff", gridcolor=GRID)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔗 Correlation Heatmap")
        nc = ["Age", "Monthly Spending", "Tenure", "Number of Purchases",
              "Customer Support Requests", "Login Frequency",
              "Satisfaction Score", "Total Spending", "Churn Status"]
        fig = px.imshow(df[nc].corr().round(2), text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu", color_continuous_midpoint=0,
                        title="Feature Correlation Matrix")
        fig.update_layout(**CHART_BG, height=440)
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error: {e}")

# ══════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════
with tabs[2]:
    try:
        metrics_df = pd.read_csv(MODEL_COMPARISON_PATH, index_col=0)
        best = art["best_name"]
        st.markdown("### 🏆 Model Performance")
        
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Best Model", best)
        b2.metric("Accuracy", f"{metrics_df.loc[best, 'Accuracy']*100:.2f}%")
        b3.metric("ROC-AUC", f"{metrics_df.loc[best, 'ROC-AUC']:.4f}")
        b4.metric("F1 Score", f"{metrics_df.loc[best, 'F1']:.4f}")
        st.divider()
        
        radar_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for i, (mname, row) in enumerate(metrics_df.iterrows()):
                star = " ⭐" if mname == best else ""
                fig.add_trace(go.Bar(
                    name=mname+star,
                    x=radar_metrics,
                    y=[row[m] for m in radar_metrics],
                    marker_color=MODEL_COLORS[i], opacity=0.85
                ))
            fig.update_layout(**CHART_BG, barmode="group", title="Metric Comparison",
                              yaxis=dict(range=[0, 1.1], color="#ffffff", gridcolor=GRID),
                              xaxis=dict(color="#ffffff"),
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")))
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            fig = go.Figure()
            for i, (mname, row) in enumerate(metrics_df.iterrows()):
                vals = [row[m] for m in radar_metrics]
                fig.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]],
                    theta=radar_metrics+[radar_metrics[0]],
                    name=mname,
                    line=dict(color=MODEL_COLORS[i], width=2),
                    fill="toself",
                    fillcolor=MODEL_FILL_COLORS[i]
                ))
            fig.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)",
                           radialaxis=dict(range=[0.5, 1.0], color="#ffffff", gridcolor=GRID,
                                          tickfont=dict(color="#ffffff")),
                           angularaxis=dict(color="#ffffff", tickfont=dict(color="#ffffff"))),
                **CHART_BG, title="Performance Radar",
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Full Metrics Table")
        disp = metrics_df[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV-F1(Train)"]].copy()
        disp = (disp*100).round(2).astype(str) + "%"
        st.dataframe(disp, use_container_width=True)
    
    except Exception as e:
        st.error(f"⚠️ Run `python src/model_training.py` first.\nError: {e}")

# ══════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    <div style='max-width:800px; margin:auto;'>
    <h2 style='color:#64b5f6;'>About This Project</h2>

    <p style='color:#b0bec5; line-height:1.8;'>
    <b style='color:#e0e0e0;'>ChurnGuard AI</b> is a Machine Learning system built for the
    <b style='color:#42a5f5;'>Teyzix Core Internship — Task ML-2</b>. It predicts whether a
    customer is likely to stop using a subscription service based on their behavioral,
    demographic, and transactional data.
    </p>

    <h3 style='color:#64b5f6; margin-top:24px;'>🧠 Models Used</h3>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, name, acc, color in zip(
        [m1,m2,m3,m4],
        ["Random Forest","XGBoost","Logistic Reg.","SVM"],
        ["91.0%","89.7%","89.7%","87.3%"],
        ["#42a5f5","#ef5350","#66bb6a","#ab47bc"]
    ):
        col.markdown(f"""
        <div style='background:rgba(255,255,255,0.05); border:1px solid {color};
                    border-radius:12px; padding:16px; text-align:center;'>
            <div style='color:{color}; font-size:1.5rem; font-weight:800;'>{acc}</div>
            <div style='color:#90caf9; font-size:0.85rem; margin-top:4px;'>{name}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='max-width:800px; margin:32px auto 0 auto;'>
    <h3 style='color:#64b5f6;'>🔧 Tech Stack</h3>
    <div style='display:flex; flex-wrap:wrap; gap:10px; margin-top:12px;'>
    """, unsafe_allow_html=True)

    techs = ["Python 3.12","Pandas","NumPy","Scikit-learn","XGBoost","Streamlit","Plotly","Joblib"]
    badges = "".join([
        f"<span style='background:rgba(33,150,243,0.2); border:1px solid #2196F3; "
        f"border-radius:20px; padding:6px 16px; color:#90caf9; font-size:0.85rem;'>{t}</span>"
        for t in techs
    ])
    st.markdown(f"<div style='display:flex; flex-wrap:wrap; gap:10px;'>{badges}</div>", unsafe_allow_html=True)

    st.markdown("""
    <br>
    <h3 style='color:#64b5f6;'>📁 Pipeline Modules</h3>
    <table style='width:100%; color:#b0bec5; border-collapse:collapse;'>
        <tr style='border-bottom:1px solid #1565c0;'>
            <th style='padding:10px; color:#64b5f6; text-align:left;'>File</th>
            <th style='padding:10px; color:#64b5f6; text-align:left;'>Description</th>
        </tr>
         <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
            <td style='padding:10px; color:#42a5f5;'>generate_data.py</td>
            <td style='padding:10px;'>Generates synthetic customer data for churn prediction</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
            <td style='padding:10px; color:#42a5f5;'>data_preprocessing.py</td>
            <td style='padding:10px;'>Cleaning, encoding, feature engineering, train-test split</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
            <td style='padding:10px; color:#42a5f5;'>eda.py</td>
            <td style='padding:10px;'>4 multi-panel EDA visualizations</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
            <td style='padding:10px; color:#42a5f5;'>model_training.py</td>
            <td style='padding:10px;'>Train 4 ML models + evaluation metrics + plots</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>
            <td style='padding:10px; color:#42a5f5;'>app.py</td>
            <td style='padding:10px;'>This Streamlit web application</td>
        </tr>
        <tr>
            <td style='padding:10px; color:#42a5f5;'>generate_report.py</td>
            <td style='padding:10px;'>Auto-generates full analysis report in Markdown</td>
        </tr>
    </table>
    </div>
    <br><br>
    <div style='text-align:center; color:#546e7a; font-size:0.85rem;'>
        Teyzix Core Internship · Task ML-2 · June 2026 · ChurnGuard AI
    </div>
    """, unsafe_allow_html=True)
