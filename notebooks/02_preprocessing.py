# ============================================================
# notebooks/02_preprocessing.ipynb — Feature Engineering & Preprocessing
# ============================================================

# In[1]: Load raw data
# ─────────────────────────────────────────────────────────────
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Add project root to path
sys.path.insert(0, os.path.abspath(".."))
from src.preprocessor import Preprocessor

DATA_PATH = "../data/Churn_Modelling.csv"
df = pd.read_csv(DATA_PATH)

print(f"Raw dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nSample row:\n{df.iloc[0].to_dict()}")


# In[2]: Drop irrelevant columns (explain why each is dropped)
# ─────────────────────────────────────────────────────────────
print("Columns to drop and WHY:")
print("  RowNumber  → Sequential index — pure identifier, no predictive signal")
print("  CustomerId → Unique customer ID — carries no meaning for ML models")
print("  Surname    → Customer name — no generalizable pattern, causes data leakage risk")

df_clean = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

print(f"\nAfter dropping: {df_clean.shape[1]} columns remain")
print(f"Remaining: {list(df_clean.columns)}")

# Verify the target column
print(f"\nTarget column 'Exited' distribution:")
print(df_clean["Exited"].value_counts())


# In[3]: Encode categorical variables (before and after)
# ─────────────────────────────────────────────────────────────
print("BEFORE encoding:")
print(f"  Gender unique values: {df_clean['Gender'].unique()}")
print(f"  Geography unique values: {df_clean['Geography'].unique()}")

df_encoded = df_clean.copy()

# Gender: binary label encoding (Female=0, Male=1)
# We use label encoding here because Gender is binary — one-hot would be redundant
df_encoded["Gender"] = df_encoded["Gender"].map({"Female": 0, "Male": 1})

# Geography: one-hot encoding with drop_first=True to avoid multicollinearity
# France becomes the baseline (all zeros) — Germany_Germany=1 and Geography_Spain=1 are the new columns
df_encoded = pd.get_dummies(df_encoded, columns=["Geography"], drop_first=True, dtype=int)

print("\nAFTER encoding:")
print(f"  Gender sample values: {df_encoded['Gender'].unique()}")
print(f"  New Geography columns: {[c for c in df_encoded.columns if 'Geography' in c]}")
print(f"\nEncoded shape: {df_encoded.shape}")
print(f"\nFirst 3 rows after encoding:\n{df_encoded.head(3)}")


# In[4]: Feature engineering — create 7 new features
# ─────────────────────────────────────────────────────────────
print("Engineering 7 new features...")

df_feat = df_encoded.copy()

# 1. Balance-to-salary ratio — how much of their income is in this bank?
df_feat["balance_salary_ratio"] = df_feat["Balance"] / (df_feat["EstimatedSalary"] + 1)
print("  ✅ balance_salary_ratio: proportion of salary held as bank balance")

# 2. Age group — customers in different life stages have different churn patterns
df_feat["age_group"] = pd.cut(
    df_feat["Age"],
    bins=[0, 30, 45, 60, 100],
    labels=[0, 1, 2, 3]  # young=0, mid=1, senior=2, elderly=3
).astype(int)
print("  ✅ age_group: 0=young(<30), 1=mid(30-45), 2=senior(45-60), 3=elderly(60+)")

# 3. Age-tenure ratio — loyalty signal: old tenure at young age = very loyal
df_feat["age_tenure_ratio"] = df_feat["Age"] / (df_feat["Tenure"] + 1)
print("  ✅ age_tenure_ratio: customer age divided by tenure (higher = joined later in life)")

# 4. Zero balance flag — binary signal for empty accounts
df_feat["zero_balance"] = (df_feat["Balance"] == 0).astype(int)
zero_count = df_feat["zero_balance"].sum()
print(f"  ✅ zero_balance: {zero_count:,} customers have zero balance ({zero_count/len(df_feat)*100:.1f}%)")

# 5. High value flag — premium customer signal
df_feat["high_value"] = (df_feat["Balance"] > 100000).astype(int)
hv_count = df_feat["high_value"].sum()
print(f"  ✅ high_value: {hv_count:,} customers have balance > ₹100,000")

# 6. Engagement score — composite of activity, products, and credit card
df_feat["engagement_score"] = (
    df_feat["IsActiveMember"] * df_feat["NumOfProducts"] * (df_feat["HasCrCard"] + 1)
)
print("  ✅ engagement_score: IsActiveMember × NumOfProducts × (HasCrCard+1)")

# 7. Products per year — adoption rate signal
df_feat["products_per_year"] = df_feat["NumOfProducts"] / (df_feat["Tenure"] + 1)
print("  ✅ products_per_year: product adoption rate per year")

print(f"\nShape after feature engineering: {df_feat.shape}")
print(f"\nSample of engineered features:\n{df_feat[['balance_salary_ratio','age_group','zero_balance','high_value','engagement_score']].head()}")


# In[5]: Class imbalance check
# ─────────────────────────────────────────────────────────────
y = df_feat["Exited"].values
stayed = (y == 0).sum()
churned = (y == 1).sum()
ratio = stayed / churned

print(f"Class distribution:")
print(f"  Stayed (0): {stayed:,} ({stayed/len(y)*100:.1f}%)")
print(f"  Churned (1): {churned:,} ({churned/len(y)*100:.1f}%)")
print(f"  Imbalance ratio: {ratio:.1f}:1")
print(f"\n⚠️ The dataset is imbalanced — {ratio:.0f}x more 'Stayed' than 'Churned'.")
print("   Without handling this, the model will bias toward predicting 'Stayed'.")
print("   We handle this using class_weight='balanced' in models (preferred in production)")
print("   and demonstrate SMOTE below for reference.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(["Stayed", "Churned"], [stayed, churned], color=["#3B82F6", "#EF4444"], edgecolor="white")
axes[0].set_title("Original Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate([stayed, churned]):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")


# In[6]: SMOTE demonstration
# ─────────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE

    X_temp = df_feat.drop(columns=["Exited"]).values
    y_temp = df_feat["Exited"].values

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_temp, y_temp)

    stayed_r = (y_resampled == 0).sum()
    churned_r = (y_resampled == 1).sum()

    print(f"\nAfter SMOTE:")
    print(f"  Stayed (0):  {stayed_r:,}")
    print(f"  Churned (1): {churned_r:,}")
    print(f"  Total: {len(y_resampled):,} (added {len(y_resampled)-len(y):,} synthetic samples)")
    print("\n  NOTE: We use class_weight='balanced' in production (simpler, no synthetic data risk)")
    print("  SMOTE is shown here as a concept demonstration only.")

    axes[1].bar(["Stayed", "Churned"], [stayed_r, churned_r], color=["#3B82F6", "#EF4444"], edgecolor="white")
    axes[1].set_title("After SMOTE (Balanced)")
    axes[1].set_ylabel("Count")

except ImportError:
    print("⚠️ imbalanced-learn not installed. Run: pip install imbalanced-learn")
    axes[1].set_title("SMOTE (not available)")

plt.tight_layout()
plt.show()


# In[7]: Feature scaling — StandardScaler
# ─────────────────────────────────────────────────────────────
X = df_feat.drop(columns=["Exited"]).values
y = df_feat["Exited"].values
feature_names = list(df_feat.drop(columns=["Exited"]).columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling statistics (BEFORE → AFTER StandardScaler):")
print(f"{'Feature':<25} {'Mean (before)':>15} {'Std (before)':>13} {'Mean (after)':>13} {'Std (after)':>12}")
print("-" * 80)
for i, feat in enumerate(feature_names[:10]):  # show first 10
    raw_mean = X[:, i].mean()
    raw_std  = X[:, i].std()
    sc_mean  = X_scaled[:, i].mean()
    sc_std   = X_scaled[:, i].std()
    print(f"{feat:<25} {raw_mean:>15.2f} {raw_std:>13.2f} {sc_mean:>13.4f} {sc_std:>12.4f}")

print("\n✅ All features now have mean ≈ 0 and std ≈ 1")
print("   This prevents large-magnitude features (like Balance) from dominating distance-based models")


# In[8]: Train-test split (stratified 80/20)
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

train_churn = y_train.mean() * 100
test_churn  = y_test.mean()  * 100

print(f"Train-Test Split (80/20, stratified):")
print(f"  Training set:  {X_train.shape[0]:,} rows | Churn rate: {train_churn:.1f}%")
print(f"  Test set:      {X_test.shape[0]:,}  rows | Churn rate: {test_churn:.1f}%")
print(f"\n✅ Stratified split preserves the churn ratio in both sets.")
print(f"   This ensures evaluation is done on a representative sample.")


# In[9]: Save processed data and scaler
# ─────────────────────────────────────────────────────────────
# Save using our Preprocessor class (which bundles scaler + feature names)
pp = Preprocessor()
X_train_pp, X_test_pp, y_train_pp, y_test_pp = pp.fit_transform(
    pd.read_csv(DATA_PATH)
)

os.makedirs("../models", exist_ok=True)
pp.save("../models/scaler.pkl")

# Also save the processed arrays for use in notebook 3
np.save("../models/X_train.npy", X_train_pp)
np.save("../models/X_test.npy", X_test_pp)
np.save("../models/y_train.npy", y_train_pp)
np.save("../models/y_test.npy", y_test_pp)

print("✅ Preprocessor saved to ../models/scaler.pkl")
print("✅ Processed arrays saved as .npy files")


# In[10]: Final feature list printout
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"FINAL FEATURE LIST ({len(pp.feature_names)} features total)")
print(f"{'='*60}")
for i, name in enumerate(pp.feature_names, 1):
    category = ""
    if name in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                 "HasCrCard", "IsActiveMember", "EstimatedSalary"]:
        category = "  ← original"
    elif name.startswith("Geography"):
        category = "  ← one-hot encoded"
    elif name == "Gender":
        category = "  ← label encoded"
    else:
        category = "  ← engineered"
    print(f"  {i:2d}. {name:<30}{category}")

print(f"\n  Original features: 8 numeric + 2 categorical = 10 inputs")
print(f"  After encoding:    +1 Geography_Germany +1 Geography_Spain")
print(f"  After engineering: +7 new features")
print(f"  TOTAL: {len(pp.feature_names)} features")
