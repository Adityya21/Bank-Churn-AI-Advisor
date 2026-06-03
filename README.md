<div align="center">

<h1>🏦 Bank Customer Churn Prediction + AI Retention Advisor</h1>
<h3>Predict who's leaving. Understand why. Retain them with AI.</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-ROC_AUC_0.91-FF0000?style=flat-square"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-8E75B2?style=flat-square"/>
  <img src="https://img.shields.io/badge/Groq-Llama_3.1-F97316?style=flat-square"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Optuna-Bayesian_Tuning-0ea5e9?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square"/>
</p>

<p><i>10,000 customers &nbsp;·&nbsp; XGBoost + LightGBM + Logistic Regression &nbsp;·&nbsp; SHAP explainability &nbsp;·&nbsp; LLM retention advisor</i></p>

</div>

---

## The Problem

Banks lose **15–25% of customers annually** to churn. Acquiring a replacement costs **5–7× more** than keeping an existing one. Most banks only find out a customer has left *after* it's already happened.

This project solves that in three steps:

1. **Predict** — XGBoost (AUC 0.91) flags customers likely to leave before they do
2. **Explain** — SHAP pinpoints *which features* drive each individual's risk score
3. **Act** — Llama 3.1 (via Groq) generates a personalised retention strategy on the spot

Not a demo. An end-to-end production-ready ML + GenAI pipeline.

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~79% | ~55% | ~71% | ~62% | ~0.77 |
| Random Forest | ~86% | ~72% | ~51% | ~60% | ~0.86 |
| XGBoost | ~86% | ~74% | ~55% | ~63% | ~0.88 |
| **XGBoost + Optuna ✅** | **~87%** | **~76%** | **~57%** | **~65%** | **~0.91** |
| LightGBM | ~86% | ~73% | ~53% | ~62% | ~0.87 |

> **Why AUC, not accuracy?** The dataset is 80/20 imbalanced — a model predicting "never churns" gets 80% accuracy for free. AUC measures true discrimination ability across all thresholds.

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 🔮 | **ML Churn Prediction** | XGBoost tuned with Optuna Bayesian search · AUC ~0.91 |
| 🔍 | **SHAP Explainability** | Waterfall + beeswarm plots per customer — *why* they're at risk |
| 🤖 | **AI Retention Advisor** | Llama 3.1 (Groq API) generates structured retention strategy + follow-up chat |
| 📊 | **Interactive EDA** | 6 Plotly charts — geography, age, balance, products, activity |
| 💰 | **Business Impact Calc** | Estimate annual revenue saved from targeted retention campaigns |
| ☁️ | **One-click Deploy** | Runs on Streamlit Community Cloud — zero server management |

---

## 🤖 AI Retention Advisor — How It Works

This is the standout feature. Most churn projects stop at a probability score. This one goes further.

```
Customer Profile
      │
      ▼
XGBoost (churn probability)
      │
      ▼
SHAP TreeExplainer (top 3 risk factors)
      │
      ▼
Context prompt → Groq API (Llama 3.1 8B Instant)
      │
      ▼
Structured Retention Strategy
  ├── Risk Assessment
  ├── Root Causes
  ├── Action Plan (48h · 1 week · 1 month)
  ├── Talking Points
  └── Conversational follow-up chat
```

**Works without an API key** — demo mode serves realistic pre-built strategies.

**Example output (abbreviated):**
```
🎯 Risk Assessment
84% churn probability (HIGH RISK). Primary drivers: account inactivity
and zero balance over the last 90 days...

📋 Action Plan
• 48 hours: Personal call from relationship manager
• This week: Re-activation bonus — 2% cashback on next 10 transactions
• This month: Schedule financial planning consultation

💬 Talking Point
"I noticed your account has been quieter recently — is there anything
we can do to better meet your needs?"
```

---

## 🔍 Key SHAP Insights

- **IsActiveMember** is the single strongest predictor — inactive customers are 2× more likely to churn
- **Age 40–60** is the highest-risk band — mid-career customers have the most product alternatives
- **NumOfProducts = 2** is the loyalty sweet spot; 3–4 products paradoxically *increases* churn (over-selling effect)

---

## 💰 Business Impact

On a 10,000-customer portfolio with ₹1,00,000 average CLV:

```
Without intervention:   ~2,014 churners/year  →  ₹20.1 Cr revenue lost
With model (30% lift):     604 churners saved  →  ₹6.0 Cr retained
Campaign cost:                                 →  ₹3.5 L
──────────────────────────────────────────────────────────────────────
Net ROI: ₹6.0 Cr  |  1,700%+ return  |  Breaks even retaining just 4 customers
```

---

## 🛠️ Tech Stack

```
ML Models       →  XGBoost · LightGBM · Random Forest · Logistic Regression
Tuning          →  Optuna (Bayesian hyperparameter search)
Explainability  →  SHAP (TreeExplainer — waterfall + beeswarm)
GenAI           →  Groq API · Llama 3.1 8B Instant · Custom prompt engineering
Frontend        →  Streamlit · Plotly · Custom CSS (deep blue theme)
Data            →  Pandas · NumPy · Scikit-learn · imbalanced-learn (SMOTE)
Deploy          →  Streamlit Community Cloud
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Churn_Modelling.csv from Kaggle](https://www.kaggle.com/datasets/shubh0799/churn-modelling)
- (Optional) Free [Groq API key](https://console.groq.com) — takes 2 minutes, no credit card

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/bank-churn-prediction.git
cd bank-churn-prediction
pip install -r requirements.txt
```

### 2. Add the dataset

```bash
mkdir data
mv ~/Downloads/Churn_Modelling.csv data/
```

### 3. Configure Groq (optional)

```bash
cp .env.example .env
# Add your key: GROQ_API_KEY=your_key_here
```

### 4. Launch

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). On first run, click **"Train Model Now"** — takes ~30 seconds.

---

## ☁️ Deploy to Streamlit Cloud

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "init"
git remote add origin https://github.com/<your-username>/bank-churn-prediction.git
git push -u origin main
```

Then:
1. Go to [share.streamlit.io](https://share.streamlit.io) → **New App** → select your repo → main file: `app.py`
2. Under **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
3. Deploy — the app auto-trains on first launch if no `.pkl` files are found.

> ⚠️ Never commit `.env` or API keys. Verify `.gitignore` before pushing.

---

## 🗂️ Project Structure

```
bank-churn-prediction/
├── app.py                      # Main Streamlit app (5 tabs)
├── requirements.txt
├── .env.example                # GROQ_API_KEY=your_key_here
├── .streamlit/
│   └── config.toml             # Deep blue theme
├── data/
│   └── Churn_Modelling.csv     # 10,000 customers · 14 features
├── models/
│   ├── best_model.pkl          # Trained XGBoost
│   └── scaler.pkl              # Fitted StandardScaler
├── src/
│   ├── data_loader.py          # Load & validate CSV
│   ├── preprocessor.py         # Feature engineering + scaling
│   ├── trainer.py              # Train 4 models + Optuna tuning
│   ├── explainer.py            # SHAP plots + rule-based recommendations
│   └── ai_advisor.py           # Groq API + LLM prompt engineering
└── notebooks/
    ├── 01_eda.py               # 15-cell EDA · 12+ visualisations
    ├── 02_preprocessing.py     # Feature engineering walkthrough
    ├── 03_modeling.py          # Model training & comparison
    └── 04_explainability.py    # SHAP deep dive
```

---

## 📦 Dataset

**Churn_Modelling.csv** — [Kaggle](https://www.kaggle.com/datasets/shubh0799/churn-modelling)

| Property | Value |
|---|---|
| Rows | 10,000 customers |
| Features | 14 (geography, age, balance, products, activity, salary…) |
| Target | `Exited` (1 = churned, 0 = stayed) |
| Class split | ~80% stayed · ~20% churned |

---

## 🔮 Future Extensions

- Real-time pipeline via Kafka + live transaction feeds
- Automated WhatsApp/SMS alerts for high-risk customers
- A/B testing framework to measure which retention strategies actually work
- Model drift detection with automated retraining triggers
- Per-segment models (cluster customers before predicting)
- GPT-4 for high-value customers · Llama for standard tier

---

## 👤 Author

**Aditya Yashovardhan**

---

## 📄 Licence

MIT — open-source and free to use with attribution.

---

<div align="center">
<sub>Dataset: Kaggle Churn Modelling &nbsp;·&nbsp; AI: Groq · Llama 3.1 &nbsp;·&nbsp; ML: XGBoost · SHAP · Optuna &nbsp;·&nbsp; App: Streamlit</sub>
</div>
