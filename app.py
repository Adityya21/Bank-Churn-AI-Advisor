"""
app.py — Main Streamlit application for Bank Customer Churn Prediction.

This is the entry point. It provides 5 tabs:
  1. 🔮 Predict Churn — Live predictions with SHAP explanations
  2. 🤖 AI Retention Advisor — LLM-powered retention strategy chat
  3. 📊 Model Performance — Training metrics, ROC curves, confusion matrix
  4. 📈 Data Insights — Interactive EDA dashboard
  5. ℹ️ About — Project info, tech stack, how to run

Run with: streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load .env from project root explicitly — must happen before src imports
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)
except ImportError:
    pass

from src.data_loader import load_data, get_feature_names
from src.preprocessor import Preprocessor
from src.trainer import train_all_models, evaluate_model, save_model, load_model
from src.explainer import (
    get_shap_explainer, get_shap_values,
    plot_summary, plot_waterfall_single,
    get_top_features, get_recommendation,
)
from src.ai_advisor import (
    get_initial_strategy, get_followup_response,
    build_customer_context, is_api_available, get_demo_strategy,
)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Bank Churn Predictor — AI-Powered Retention",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for a professional look
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ────────────────────────────────────────────── */
.stApp {
    background: #F4F6FB;
}
section[data-testid="stSidebar"] {
    background: #0A1628 !important;
    border-right: 1px solid rgba(255,255,255,.07);
}
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stNumberInput > label,
section[data-testid="stSidebar"] .stCheckbox > label {
    color: #94A3B8 !important;
    font-size: .8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,.1) !important;
}
/* Slider accent */
section[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background: #2563EB !important;
}

/* ── Sidebar header layout ──────── */
/* Target the sidebar header container to position the arrow at the top left */
[data-testid="stSidebarHeader"] {
    position: absolute !important;
    top: 1rem !important;
    left: 1.5rem !important;
    padding: 0 !important;
    z-index: 100 !important;
    width: auto !important;
    background: transparent !important;
}

/* Position the Customer Profile title below the arrow */
section[data-testid="stSidebar"] .stMarkdown h2 {
    margin-top: 2.5rem !important; /* Space to clear the arrow above it */
    margin-left: 0 !important; /* Reset margin-left */
}

/* Button styling - keep existing colors and size */
section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"] {
    width: 24px !important;
    height: 24px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #CBD5E1 !important;
    background: transparent !important;
    border: none !important;
}

section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"] svg {
    width: 16px !important;
    height: 16px !important;
    color: #CBD5E1 !important;
}

section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"]:hover {
    color: #E2E8F0 !important;
    background: rgba(255, 255, 255, .05) !important;
    border-radius: 4px;
}

/* ── Main header ───────────────────────────────────────────────── */
.main-header {
    background: #0A1628;
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
    border-bottom: 3px solid #2563EB;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 300px; height: 100%;
    background: linear-gradient(135deg, transparent 50%, rgba(37,99,235,.1));
    pointer-events: none;
}
.main-header h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: -.03em;
    color: #F8FAFC;
}
.main-header p {
    margin: .4rem 0 0;
    font-size: .9rem;
    color: #94A3B8;
    font-weight: 400;
}

/* ── Tabs ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 4px;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    padding: .65rem 1.25rem;
    border-radius: 8px 8px 0 0 !important;
    font-size: .85rem;
    font-weight: 500;
    color: #64748B !important;
    border-bottom: 2px solid transparent !important;
    transition: color .15s, border-color .15s;
}
.stTabs [aria-selected="true"] {
    color: #1D4ED8 !important;
    border-bottom: 2px solid #2563EB !important;
    background: #EFF6FF !important;
}

/* ── Metric cards ──────────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border: 0.5px solid #E2E8F0;
    border-bottom-width: 3px;
    margin-bottom: 1rem;
    transition: transform .15s ease, box-shadow .15s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,.06);
}
.metric-card.green  { border-bottom-color: #16A34A; }
.metric-card.yellow { border-bottom-color: #D97706; }
.metric-card.red    { border-bottom-color: #DC2626; }
.metric-card.blue   { border-bottom-color: #2563EB; }
.metric-card.purple { border-bottom-color: #7C3AED; }

.metric-card h3 {
    margin: 0 0 .3rem;
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #94A3B8;
    font-weight: 600;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 600;
    color: #0F172A;
    letter-spacing: -.03em;
    margin-bottom: .15rem;
}
.metric-card .subtext {
    font-size: .72rem;
    color: #94A3B8;
}

/* ── Risk badges ───────────────────────────────────────────────── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: .45rem 1.1rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: .78rem;
    letter-spacing: .06em;
    text-transform: uppercase;
}
.risk-badge::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: currentColor;
    opacity: .8;
}
.risk-low    { background: #DCFCE7; color: #166534; }
.risk-medium { background: #FEF3C7; color: #92400E; }
.risk-high   { background: #FEE2E2; color: #991B1B; }

/* ── Feature impact cards ──────────────────────────────────────── */
.feature-card {
    background: #F8FAFC;
    border-radius: 8px;
    padding: .9rem 1.1rem;
    margin-bottom: .6rem;
    border-left: 3px solid;
    transition: background .15s, transform .15s;
}
.feature-card:hover {
    background: #F1F5F9;
    transform: translateX(3px);
}
.feature-card.increase { border-left-color: #DC2626; }
.feature-card.decrease { border-left-color: #16A34A; }

/* ── AI Strategy box ───────────────────────────────────────────── */
.strategy-box {
    background: #F8FAFF;
    border: 0.5px solid #BFDBFE;
    border-left: 3px solid #2563EB;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    margin: 1rem 0;
    line-height: 1.75;
    font-size: .92rem;
    color: #1E293B;
}

/* ── Chat bubbles ──────────────────────────────────────────────── */
.chat-user {
    background: #1E3A5F;
    color: white;
    padding: .75rem 1.1rem;
    border-radius: 12px 12px 3px 12px;
    margin: .5rem 0;
    max-width: 80%;
    margin-left: auto;
    font-size: .88rem;
    line-height: 1.6;
}
.chat-ai {
    background: #F1F5F9;
    color: #1E293B;
    padding: .75rem 1.1rem;
    border-radius: 12px 12px 12px 3px;
    margin: .5rem 0;
    max-width: 80%;
    font-size: .88rem;
    line-height: 1.6;
}

/* ── Recommendation box ────────────────────────────────────────── */
.recommendation-box {
    background: #FFFBEB;
    border: 0.5px solid #FDE68A;
    border-left: 3px solid #D97706;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-size: .9rem;
    line-height: 1.7;
    color: #1C1917;
}

/* ── About cards ───────────────────────────────────────────────── */
.about-card {
    background: white;
    border-radius: 10px;
    padding: 1.4rem;
    border: 0.5px solid #E2E8F0;
    border-top: 3px solid #2563EB;
    height: 100%;
}

/* ── Streamlit native element polish ──────────────────────────── */
/* Buttons */
.stButton > button {
    background: #2563EB !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    letter-spacing: .01em !important;
    padding: .55rem 1.5rem !important;
    transition: background .15s, transform .1s !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    background: #1E40AF !important;
}
/* Secondary buttons (non-primary) */
.stButton > button[kind="secondary"] {
    background: white !important;
    color: #2563EB !important;
    border: 1px solid #BFDBFE !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #EFF6FF !important;
}

/* Inputs, selects */
.stTextInput input,
.stNumberInput input,
.stSelectbox select,
[data-baseweb="input"] input,
[data-baseweb="select"] div[role="combobox"] {
    border-radius: 7px !important;
    border: 0.5px solid #CBD5E1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .875rem !important;
    transition: border-color .15s, box-shadow .15s !important;
    background-color: #ffffff !important;
    color: #0F172A !important;
}
.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: #CBD5E1 !important;
}
/* Force combobox input text color */
input[role="combobox"] {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
input[role="combobox"]::placeholder {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
/* Autofill styling for combobox */
input[role="combobox"]:-webkit-autofill {
    -webkit-box-shadow: 0 0 0 1000px #ffffff inset !important;
    -webkit-text-fill-color: #0F172A !important;
}
input[role="combobox"]:-webkit-autofill::first-line {
    color: #0F172A !important;
}
/* Force all select component text to be dark */
[data-baseweb="select"] * {
    color: #0F172A !important;
}
/* Strong override for select values inside the sidebar */
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-baseweb="select"] div[role="combobox"] *,
section[data-testid="stSidebar"] [data-baseweb="select"] div.st-dv {
    color: #0F172A !important;
}
/* Target the selected value display div */
[data-baseweb="select"] div.st-dv {
    color: #0F172A !important;
}
/* Streamlit selectbox styling */
.stSelectbox input {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
[data-baseweb="select"] div[role="listbox"] {
    background-color: #ffffff !important;
}
[data-baseweb="select"] div[role="option"] {
    color: #0F172A !important;
}
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.1) !important;
    background-color: #ffffff !important;
    color: #0F172A !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    border-radius: 10px;
}
.stProgress > div {
    background: #E2E8F0;
    border-radius: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 0.5px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Alerts / info boxes */
.stAlert {
    border-radius: 9px !important;
    border-left-width: 3px !important;
    font-size: .875rem !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 0.5px solid #E2E8F0 !important;
    border-radius: 9px !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    border-radius: 10px !important;
    border: 0.5px solid #CBD5E1 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.08) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #2563EB !important;
}

/* Section headers */
h1, h2, h3 {
    letter-spacing: -.02em;
    color: #0F172A;
}
h3 { font-weight: 600; font-size: 1.15rem; }

/* ── Dividers ──────────────────────────────────────────────────── */
hr {
    border-color: #E2E8F0 !important;
}

/* ── Caption / subtext ─────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #94A3B8 !important;
    font-size: .78rem !important;
}

/* ── Hide Streamlit branding ───────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Monospace numbers ─────────────────────────────────────────── */
.value, .metric-card .value, .feat-card .fshap {
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state keys to avoid KeyError crashes."""
    defaults = {
        "prediction_made": False,
        "current_customer": {},
        "churn_probability": 0.0,
        "risk_level": "Low",
        "top_features": [],
        "initial_strategy": None,
        "advisor_chat_history": [],
        "customer_context": "",
        "model_trained": False,
        "training_results": {},
        "best_model_name": "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session_state()


# ═════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING (CACHED)
# ═════════════════════════════════════════════════════════════════════════════

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Churn_Modelling.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")


@st.cache_data(show_spinner=False)
def cached_load_data():
    """Load and cache the dataset."""
    return load_data(DATA_PATH)


@st.cache_resource(show_spinner=False)
def cached_load_model():
    """Load and cache the trained model."""
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def cached_load_preprocessor():
    """Load and cache the preprocessor (scaler + feature names)."""
    pp = Preprocessor()
    pp.load(SCALER_PATH)
    return pp


def train_and_save():
    """
    Train all models from scratch, save the best one.
    Called when no saved model exists (e.g., fresh deployment).
    """
    with st.spinner("🔄 Training models... This takes ~30 seconds on first run."):
        df = cached_load_data()
        pp = Preprocessor()
        X_train, X_test, y_train, y_test = pp.fit_transform(df)

        results, models = train_all_models(X_train, y_train, X_test, y_test)

        # Pick the best model by AUC
        best_name = max(results, key=lambda k: results[k]["roc_auc"])
        best_model = models[best_name]

        # Save model and scaler
        save_model(best_model, MODEL_PATH)
        pp.save(SCALER_PATH)

        # Store results in session state
        st.session_state["model_trained"] = True
        st.session_state["training_results"] = results
        st.session_state["best_model_name"] = best_name
        st.session_state["all_models"] = models
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["feature_names"] = pp.feature_names

        # Clear model/preprocessor cache so they reload from disk
        cached_load_model.clear()
        cached_load_preprocessor.clear()

    st.success(f"✅ Training complete! Best model: **{best_name}** (AUC: {results[best_name]['roc_auc']:.4f})")
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Customer Churn Predictor</h1>
    <p>AI-powered predictions & personalised retention strategies — powered by XGBoost + SHAP + Llama 3.1</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# CHECK DATA & MODEL AVAILABILITY
# ═════════════════════════════════════════════════════════════════════════════

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    st.warning("📁 Dataset not found. Please upload `Churn_Modelling.csv`.")
    uploaded = st.file_uploader("Upload Churn_Modelling.csv", type="csv")
    if uploaded:
        os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
        df_uploaded = pd.read_csv(uploaded)
        df_uploaded.to_csv(DATA_PATH, index=False)
        st.success("✅ Dataset saved! Reloading...")
        st.rerun()
    st.stop()

# Check if model exists — offer training if not
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.info("🧠 No trained model found. Train the model to get started.")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Train Model Now", type="primary", use_container_width=True):
            train_and_save()
    with col2:
        st.caption("This will train 4 ML models on the dataset and save the best one. Takes ~30 seconds.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES (data + model + preprocessor)
# ═════════════════════════════════════════════════════════════════════════════

try:
    df = cached_load_data()
    model = cached_load_model()
    pp = cached_load_preprocessor()
except Exception as e:
    st.error(f"❌ Error loading resources: {str(e)}")
    if st.button("🔄 Retrain Model"):
        # Delete corrupt files and retrain
        for f in [MODEL_PATH, SCALER_PATH]:
            if os.path.exists(f):
                os.remove(f)
        train_and_save()
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# 5 TABS
# ═════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Churn",
    "🤖 AI Retention Advisor",
    "📊 Model Performance",
    "📈 Data Insights",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT CHURN
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### 🔮 Customer Churn Prediction")
    st.caption("Enter customer details in the sidebar and click **Predict** to see churn probability with SHAP explanations.")

    # ── SIDEBAR: Customer Input Form ──────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📋 Customer Profile")
        st.markdown("---")

        credit_score = st.slider("Credit Score", 300, 850, 650, step=10,
                                 help="Customer's credit score (300-850)")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"],
                                 help="Customer's country of residence")
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        age = st.slider("Age", 18, 92, 35,
                         help="Customer's age in years")
        tenure = st.slider("Tenure (years)", 0, 10, 5,
                            help="How many years the customer has been with the bank")
        balance = st.number_input("Account Balance (₹)", 0, 250000, 50000, step=5000,
                                   help="Current account balance")
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4],
                                     help="Number of bank products used")
        has_cr_card = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Is Active Member", value=True)
        salary = st.number_input("Estimated Salary (₹)", 0, 200000, 75000, step=5000,
                                  help="Customer's estimated annual salary")

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

    # ── PREDICTION LOGIC ──────────────────────────────────────────────────
    if predict_btn:
        # Build customer dict
        customer_data = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": float(balance),
            "NumOfProducts": num_products,
            "HasCrCard": int(has_cr_card),
            "IsActiveMember": int(is_active),
            "EstimatedSalary": float(salary),
        }

        with st.spinner("🔄 Analyzing customer profile..."):
            try:
                # Transform and predict
                X_input = pp.transform_single(customer_data)
                churn_prob = float(model.predict_proba(X_input)[0][1])
                churn_pred = int(churn_prob > 0.5)

                # Determine risk level
                if churn_prob > 0.7:
                    risk_level = "High"
                elif churn_prob > 0.4:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"

                # SHAP explanation
                # get_shap_values() always returns 2D (n_samples, n_features)
                # for the positive (churn) class — take row 0 for this customer
                explainer = get_shap_explainer(model, X_input)
                shap_vals = get_shap_values(explainer, X_input)   # shape (1, n_features)
                shap_vals_row = np.asarray(shap_vals[0]).ravel()  # shape (n_features,)
                top_feats = get_top_features(shap_vals_row, pp.feature_names, n=3)

                # Save to session state for Tab 2
                st.session_state["prediction_made"] = True
                st.session_state["current_customer"] = customer_data
                st.session_state["churn_probability"] = churn_prob
                st.session_state["risk_level"] = risk_level
                st.session_state["top_features"] = top_feats
                st.session_state["initial_strategy"] = None  # reset on new prediction
                st.session_state["advisor_chat_history"] = []  # reset chat
                st.session_state["shap_vals_row"] = shap_vals_row
                st.session_state["X_input"] = X_input

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.stop()

    # ── DISPLAY PREDICTION RESULTS ────────────────────────────────────────
    if st.session_state["prediction_made"]:
        churn_prob = st.session_state["churn_probability"]
        risk_level = st.session_state["risk_level"]
        top_feats = st.session_state["top_features"]

        # ── Big metric + risk badge ──────────────────────────────────────
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            risk_colors = {"Low": "green", "Medium": "yellow", "High": "red"}
            risk_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            color = risk_colors[risk_level]

            st.markdown(f"""
            <div class="metric-card {color}">
                <h3>Churn Probability</h3>
                <div class="value">{churn_prob*100:.1f}%</div>
                <div class="subtext">Based on XGBoost + 17 features</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            risk_css = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}
            st.markdown(f"""
            <div style="padding-top: 1rem;">
                <span class="risk-badge {risk_css[risk_level]}">
                    {risk_emojis[risk_level]} {risk_level} Risk
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(churn_prob, 1.0))

        with col3:
            prediction_label = "Will Churn" if churn_prob > 0.5 else "Will Stay"
            pred_icon = "🚪" if churn_prob > 0.5 else "✅"
            st.markdown(f"""
            <div class="metric-card blue">
                <h3>Prediction</h3>
                <div class="value" style="font-size:1.3rem;">{pred_icon} {prediction_label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── SHAP waterfall chart ─────────────────────────────────────────
        st.markdown("#### 📊 Why This Prediction? (SHAP Explanation)")

        try:
            X_input = st.session_state["X_input"]
            fig = plot_waterfall_single(
                get_shap_explainer(model, X_input),
                X_input[0],
                pp.feature_names
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate SHAP waterfall plot: {str(e)}")

        # ── Top 3 features as styled cards ───────────────────────────────
        st.markdown("#### 🔍 Key Risk Factors")
        cols = st.columns(3)
        for i, (feat_name, direction, shap_val) in enumerate(top_feats):
            card_class = "increase" if "increases" in direction else "decrease"
            arrow = "↑" if "increases" in direction else "↓"
            with cols[i]:
                st.markdown(f"""
                <div class="feature-card {card_class}">
                    <strong>{feat_name}</strong><br/>
                    <span style="font-size:0.9rem; color: {'#EF4444' if 'increases' in direction else '#22C55E'};">
                        {arrow} {direction}
                    </span><br/>
                    <span style="font-size:0.8rem; color:#6B7280;">SHAP: {shap_val:+.4f}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Rule-based recommendation ────────────────────────────────────
        st.markdown("#### 💡 Retention Recommendation")
        recommendation = get_recommendation(churn_prob, top_feats)
        st.markdown(f'<div class="recommendation-box">{recommendation}</div>',
                     unsafe_allow_html=True)

        # ── Link to AI Advisor ───────────────────────────────────────────
        st.markdown("---")
        st.info("🤖 **Want a personalised AI-generated retention strategy?** Switch to the **AI Retention Advisor** tab for detailed, LLM-powered advice!")

    else:
        # No prediction yet — show instructions
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#6B7280;">
            <h3>👈 Enter customer details in the sidebar</h3>
            <p>Fill in the customer profile and click <strong>Predict Churn</strong> to get started.</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: AI RETENTION ADVISOR
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🤖 AI Retention Advisor")
    st.caption("Powered by Llama 3.1 via Groq API — Get personalised, LLM-generated retention strategies")

    # ── API key status indicator ─────────────────────────────────────────
    api_available = is_api_available()

    with st.sidebar:
        with st.expander("🔑 Groq API Key Setup", expanded=not api_available):
            if api_available:
                st.success("✅ Groq API key detected!")
            else:
                st.warning("⚠️ No API key found")

            st.markdown("""
            **Get a FREE API key in 2 minutes:**

            1. Go to [console.groq.com](https://console.groq.com)
            2. Sign up (free — no credit card needed)
            3. Click **API Keys** → **Create API Key**
            4. Copy the key

            **For local development:**
            - Create a `.env` file in the project root
            - Add: `GROQ_API_KEY=your_key_here`

            **For Streamlit Cloud:**
            - Go to your app dashboard
            - Click **Settings** → **Secrets**
            - Add: `GROQ_API_KEY = "your_key_here"`
            """)

    # ── Check if prediction has been made ────────────────────────────────
    if not st.session_state["prediction_made"]:
        st.markdown("""
        <div style="text-align:center; padding:3rem; background:#F9FAFB; border-radius:14px; margin:2rem 0;">
            <h3 style="color:#6B7280;">📋 No Customer Data Yet</h3>
            <p style="color:#9CA3AF;">
                Make a prediction in the <strong>🔮 Predict Churn</strong> tab first,<br/>
                then come back here for an AI-powered retention strategy.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Customer summary card ────────────────────────────────────────
        customer = st.session_state["current_customer"]
        churn_prob = st.session_state["churn_probability"]
        risk_level = st.session_state["risk_level"]
        top_feats = st.session_state["top_features"]

        risk_colors_bg = {"Low": "#DCFCE7", "Medium": "#FEF3C7", "High": "#FEE2E2"}
        risk_colors_text = {"Low": "#166534", "Medium": "#92400E", "High": "#991B1B"}

        st.markdown(f"""
        <div style="background:white; border-radius:14px; padding:1.5rem; box-shadow:0 2px 12px rgba(0,0,0,0.06); margin-bottom:1.5rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <strong style="font-size:1.1rem;">Customer Profile</strong><br/>
                    <span style="color:#6B7280;">
                        {customer['Age']}yo {customer['Gender']} • {customer['Geography']} •
                        Balance: ₹{customer['Balance']:,.0f} • {customer['NumOfProducts']} products •
                        {'Active' if customer['IsActiveMember'] else 'Inactive'}
                    </span>
                </div>
                <div style="text-align:right;">
                    <span class="risk-badge" style="background:{risk_colors_bg[risk_level]};color:{risk_colors_text[risk_level]};">
                        {churn_prob*100:.0f}% Churn Risk — {risk_level}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show top SHAP factors as quick summary
        shap_cols = st.columns(len(top_feats))
        for i, (feat, direction, val) in enumerate(top_feats):
            with shap_cols[i]:
                emoji = "🔴" if "increases" in direction else "🟢"
                st.caption(f"{emoji} **{feat}**: {direction} (SHAP: {val:+.3f})")

        st.markdown("---")

        # ── AI Strategy Panel ────────────────────────────────────────────
        st.markdown("#### 🧠 AI-Generated Retention Strategy")
        st.caption("Powered by Llama 3.1 via Groq API")

        # Build model metrics dict for context
        model_metrics = {"roc_auc": 0.91, "accuracy": 0.86}

        # Build customer context (reusable for chat)
        if not st.session_state["customer_context"]:
            st.session_state["customer_context"] = build_customer_context(
                customer, churn_prob, risk_level, top_feats, model_metrics
            )

        # Generate Strategy button
        if st.button("✨ Generate Retention Strategy", type="primary", use_container_width=True):
            if api_available:
                with st.spinner("🧠 Consulting AI advisor... (2-3 seconds)"):
                    strategy = get_initial_strategy(
                        customer, churn_prob, risk_level, top_feats, model_metrics
                    )
                    st.session_state["initial_strategy"] = strategy
            else:
                # Demo mode
                strategy = get_demo_strategy(churn_prob, top_feats)
                st.session_state["initial_strategy"] = strategy
                st.info("💡 **Demo Mode** — Connect your FREE Groq API key for personalised AI strategies. See sidebar for setup instructions.")

        # Display strategy
        if st.session_state["initial_strategy"]:
            st.markdown(f"""
            <div class="strategy-box">
                {st.session_state["initial_strategy"]}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Follow-up Chat Interface ─────────────────────────────────────
        st.markdown("#### 💬 Ask the AI Advisor")
        st.caption("Ask follow-up questions about the retention strategy")

        # Display conversation history
        for msg in st.session_state["advisor_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input(
            "Ask about this customer's retention strategy...",
            disabled=not api_available and len(st.session_state["advisor_chat_history"]) > 0
        )

        if user_input:
            # Add user message to history
            st.session_state["advisor_chat_history"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Get AI response
            if api_available:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_followup_response(
                            st.session_state["advisor_chat_history"][:-1],  # exclude the just-added user msg
                            user_input,
                            st.session_state["customer_context"]
                        )
                        st.markdown(response)
                st.session_state["advisor_chat_history"].append(
                    {"role": "assistant", "content": response}
                )
            else:
                with st.chat_message("assistant"):
                    st.markdown("⚠️ Connect your Groq API key to enable the chat feature. See sidebar for setup instructions.")
                st.session_state["advisor_chat_history"].append(
                    {"role": "assistant", "content": "⚠️ Connect your Groq API key to enable the chat feature."}
                )

        # Clear chat button
        if st.session_state["advisor_chat_history"]:
            if st.button("🗑️ Clear Conversation"):
                st.session_state["advisor_chat_history"] = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 📊 Model Performance Dashboard")

    # If we have training results in session state, use them
    # Otherwise, evaluate the current model on the dataset
    if st.session_state.get("training_results"):
        results = st.session_state["training_results"]
        best_name = st.session_state["best_model_name"]
    else:
        # Run evaluation on saved model
        with st.spinner("📊 Evaluating model performance..."):
            pp_eval = Preprocessor()
            pp_eval.load(SCALER_PATH)
            X_train_eval, X_test_eval, y_train_eval, y_test_eval = pp_eval.fit_transform(df)
            eval_metrics = evaluate_model(model, X_test_eval, y_test_eval)
            results = {"Best Model": eval_metrics}
            best_name = "Best Model"
            # Cache for future tab switches
            st.session_state["training_results"] = results
            st.session_state["best_model_name"] = best_name
            st.session_state["X_test"] = X_test_eval
            st.session_state["y_test"] = y_test_eval
            st.session_state["feature_names"] = pp_eval.feature_names

    best_metrics = results[best_name]

    # ── Top metrics cards ────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>Accuracy</h3>
            <div class="value">{best_metrics['accuracy']*100:.1f}%</div>
            <div class="subtext">{best_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card green">
            <h3>F1 Score</h3>
            <div class="value">{best_metrics['f1']*100:.1f}%</div>
            <div class="subtext">Harmonic mean</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card purple">
            <h3>ROC AUC</h3>
            <div class="value">{best_metrics['roc_auc']*100:.1f}%</div>
            <div class="subtext">Discrimination ability</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card yellow">
            <h3>Recall</h3>
            <div class="value">{best_metrics['recall']*100:.1f}%</div>
            <div class="subtext">Churn detection rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model comparison table ───────────────────────────────────────────
    if len(results) > 1:
        st.markdown("#### 📋 Model Comparison")
        comparison_data = []
        for name, metrics in results.items():
            comparison_data.append({
                "Model": name,
                "Accuracy": f"{metrics['accuracy']*100:.2f}%",
                "Precision": f"{metrics['precision']*100:.2f}%",
                "Recall": f"{metrics['recall']*100:.2f}%",
                "F1 Score": f"{metrics['f1']*100:.2f}%",
                "ROC AUC": f"{metrics['roc_auc']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # ── Charts row ───────────────────────────────────────────────────────
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("#### 📈 ROC Curve")
        if "fpr" in best_metrics and "tpr" in best_metrics:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=best_metrics["fpr"], y=best_metrics["tpr"],
                mode="lines",
                name=f"{best_name} (AUC={best_metrics['roc_auc']:.3f})",
                line=dict(color="#1E3A5F", width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random (AUC=0.5)",
                line=dict(color="#D1D5DB", width=2, dash="dash")
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                height=400,
                legend=dict(x=0.4, y=0.1),
                margin=dict(l=40, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    with chart2:
        st.markdown("#### 🎯 Confusion Matrix")
        if "confusion_matrix" in best_metrics:
            cm = np.array(best_metrics["confusion_matrix"])
            labels = ["Stayed (0)", "Churned (1)"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels, y=labels,
                text=[[str(v) for v in row] for row in cm],
                texttemplate="%{text}",
                textfont=dict(size=18, color="white"),
                colorscale=[[0, "#E8F0FE"], [1, "#1E3A5F"]],
                showscale=False
            ))
            fig_cm.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                template="plotly_white",
                height=400,
                margin=dict(l=40, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature importance ───────────────────────────────────────────────
    st.markdown("#### 📊 Top 15 Feature Importance")
    try:
        if hasattr(model, "feature_importances_"):
            feat_imp = model.feature_importances_
            feat_names = st.session_state.get("feature_names", pp.feature_names)
            if len(feat_imp) == len(feat_names):
                imp_df = pd.DataFrame({
                    "Feature": feat_names,
                    "Importance": feat_imp
                }).sort_values("Importance", ascending=True).tail(15)

                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#E8F0FE", "#1E3A5F"],
                )
                fig_imp.update_layout(
                    template="plotly_white",
                    height=500,
                    showlegend=False,
                    margin=dict(l=40, r=20, t=20, b=40),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.info(f"Feature importance not available for this model type: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: DATA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### 📈 Data Insights & EDA Dashboard")

    # ── Dataset overview ─────────────────────────────────────────────────
    total = len(df)
    churned = df["Exited"].sum()
    churn_rate = churned / total * 100

    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>Total Customers</h3>
            <div class="value">{total:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with ov2:
        st.markdown(f"""
        <div class="metric-card red">
            <h3>Churned</h3>
            <div class="value">{churned:,}</div>
            <div class="subtext">{churn_rate:.1f}% churn rate</div>
        </div>
        """, unsafe_allow_html=True)
    with ov3:
        st.markdown(f"""
        <div class="metric-card green">
            <h3>Retained</h3>
            <div class="value">{total - churned:,}</div>
            <div class="subtext">{100-churn_rate:.1f}% retention</div>
        </div>
        """, unsafe_allow_html=True)
    with ov4:
        st.markdown(f"""
        <div class="metric-card purple">
            <h3>Features</h3>
            <div class="value">{df.shape[1]}</div>
            <div class="subtext">Original columns</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── EDA Charts (6 key visualizations) ────────────────────────────────
    df_eda = df.copy()
    df_eda["Churn Status"] = df_eda["Exited"].map({0: "Stayed", 1: "Churned"})

    # Row 1: Geography + Age
    eda1, eda2 = st.columns(2)

    with eda1:
        st.markdown("#### 🌍 Churn by Geography")
        geo_churn = df_eda.groupby(["Geography", "Churn Status"]).size().reset_index(name="Count")
        fig_geo = px.bar(
            geo_churn, x="Geography", y="Count", color="Churn Status",
            barmode="group",
            color_discrete_map={"Stayed": "#3B82F6", "Churned": "#EF4444"},
            template="plotly_white"
        )
        fig_geo.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig_geo, use_container_width=True)

    with eda2:
        st.markdown("#### 👤 Churn by Age Group")
        fig_age = px.histogram(
            df_eda, x="Age", color="Churn Status",
            nbins=30, barmode="overlay",
            color_discrete_map={"Stayed": "#3B82F6", "Churned": "#EF4444"},
            template="plotly_white", opacity=0.7
        )
        fig_age.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig_age, use_container_width=True)

    # Row 2: Balance + Correlation
    eda3, eda4 = st.columns(2)

    with eda3:
        st.markdown("#### 💰 Balance Distribution by Churn")
        fig_bal = px.violin(
            df_eda, y="Balance", x="Churn Status", color="Churn Status",
            box=True,
            color_discrete_map={"Stayed": "#3B82F6", "Churned": "#EF4444"},
            template="plotly_white"
        )
        fig_bal.update_layout(height=380, showlegend=False, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig_bal, use_container_width=True)

    with eda4:
        st.markdown("#### 🔗 Feature Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            template="plotly_white",
            aspect="auto"
        )
        fig_corr.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Row 3: Products + Active Members
    eda5, eda6 = st.columns(2)

    with eda5:
        st.markdown("#### 📦 Churn by Number of Products")
        prod_churn = df_eda.groupby(["NumOfProducts", "Churn Status"]).size().reset_index(name="Count")
        fig_prod = px.bar(
            prod_churn, x="NumOfProducts", y="Count", color="Churn Status",
            barmode="group",
            color_discrete_map={"Stayed": "#3B82F6", "Churned": "#EF4444"},
            template="plotly_white"
        )
        fig_prod.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig_prod, use_container_width=True)

    with eda6:
        st.markdown("#### 👥 Active vs Inactive Member Churn")
        active_churn = df_eda.groupby(["IsActiveMember", "Churn Status"]).size().reset_index(name="Count")
        active_churned = active_churn[active_churn["Churn Status"] == "Churned"]
        active_churned = active_churned.copy()
        active_churned["Label"] = active_churned["IsActiveMember"].map({0: "Inactive", 1: "Active"})
        fig_active = px.pie(
            active_churned, values="Count", names="Label",
            color_discrete_sequence=["#EF4444", "#3B82F6"],
            template="plotly_white"
        )
        fig_active.update_traces(textposition="inside", textinfo="percent+label")
        fig_active.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig_active, use_container_width=True)

    st.markdown("---")

    # ── Business Impact Calculator ───────────────────────────────────────
    st.markdown("#### 💰 Business Impact Calculator")
    st.caption("Estimate the potential savings from retaining at-risk customers.")

    biz1, biz2 = st.columns(2)
    with biz1:
        at_risk = st.slider("Number of customers identified at risk", 50, 5000, 500, step=50)
    with biz2:
        clv = st.slider("Average Customer Lifetime Value (₹)", 10000, 500000, 100000, step=10000)

    retention_rate = 0.30  # assume we can retain 30% of at-risk customers
    savings = at_risk * retention_rate * clv

    sav1, sav2, sav3 = st.columns(3)
    with sav1:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>At-Risk Customers</h3>
            <div class="value">{at_risk:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with sav2:
        st.markdown(f"""
        <div class="metric-card green">
            <h3>Est. Retained (30%)</h3>
            <div class="value">{int(at_risk * retention_rate):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with sav3:
        st.markdown(f"""
        <div class="metric-card purple">
            <h3>Estimated Annual Savings</h3>
            <div class="value">₹{savings:,.0f}</div>
            <div class="subtext">{at_risk} × 30% × ₹{clv:,}</div>
        </div>
        """, unsafe_allow_html=True)

