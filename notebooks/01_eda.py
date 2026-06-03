# ============================================================
# notebooks/01_eda.ipynb — Exploratory Data Analysis
# Run as a Jupyter notebook OR as a plain Python script.
# Each "# In[N]:" marker represents a notebook cell.
# ============================================================

# In[1]: Imports and setup
# ─────────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent plot styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "font.family": "sans-serif",
})

print("✅ Imports complete.")


# In[2]: Load data, inspect shape, dtypes, and basic stats
# ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(".")), "data", "Churn_Modelling.csv")
# Fallback path if running directly from notebooks/ directory
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/Churn_Modelling.csv"

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic statistics:\n{df.describe().T.round(2)}")
print(f"\nChurn rate: {df['Exited'].mean()*100:.2f}%")


# In[3]: Missing values check + heatmap
# ─────────────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
}).sort_values("Missing %", ascending=False)

print("Missing Values Report:")
print(missing_df[missing_df["Missing Count"] > 0] if missing_df["Missing Count"].sum() > 0 else "✅ No missing values!")

# Visualise missingness as a heatmap (useful for larger datasets)
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", ax=ax)
ax.set_title("Missing Values Heatmap (yellow = missing)", pad=15)
plt.tight_layout()
plt.show()


# In[4]: Churn distribution — pie chart + countplot
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Countplot
churn_counts = df["Exited"].value_counts().rename({0: "Stayed", 1: "Churned"})
colors = ["#3B82F6", "#EF4444"]
churn_counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white", width=0.5)
axes[0].set_title("Churn vs Retained Customers")
axes[0].set_xlabel("Customer Status")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(["Stayed", "Churned"], rotation=0)
for bar, count in zip(axes[0].patches, churn_counts):
    axes[0].annotate(f"{count:,}\n({count/len(df)*100:.1f}%)",
                     (bar.get_x() + bar.get_width()/2, bar.get_height()/2),
                     ha="center", va="center", color="white", fontsize=13, fontweight="bold")

# Pie chart
axes[1].pie(churn_counts, labels=["Stayed", "Churned"],
            autopct="%1.1f%%", colors=colors, startangle=90,
            textprops={"fontsize": 12}, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Class Distribution")

plt.suptitle("Churn Distribution Analysis", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print(f"\nClass imbalance ratio: {churn_counts[0]}/{churn_counts[1]} = {churn_counts[0]/churn_counts[1]:.1f}:1 (Stayed:Churned)")


# In[5]: Churn rate by Geography (grouped bar)
# ─────────────────────────────────────────────────────────────
geo_churn = df.groupby("Geography")["Exited"].agg(["sum", "count"]).reset_index()
geo_churn.columns = ["Geography", "Churned", "Total"]
geo_churn["Churn Rate (%)"] = (geo_churn["Churned"] / geo_churn["Total"] * 100).round(1)
geo_churn["Retained"] = geo_churn["Total"] - geo_churn["Churned"]

print("Churn by Geography:")
print(geo_churn.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grouped bar: count
geo_churn.set_index("Geography")[["Churned", "Retained"]].plot(
    kind="bar", ax=axes[0], color=["#EF4444", "#3B82F6"], edgecolor="white"
)
axes[0].set_title("Churned vs Retained by Geography")
axes[0].set_xlabel("Geography")
axes[0].set_ylabel("Number of Customers")
axes[0].legend(loc="upper right")
axes[0].tick_params(axis="x", rotation=0)

# Churn rate bar
bars = axes[1].bar(geo_churn["Geography"], geo_churn["Churn Rate (%)"],
                    color=["#F87171", "#FB923C", "#60A5FA"], edgecolor="white")
axes[1].set_title("Churn Rate (%) by Geography")
axes[1].set_xlabel("Geography")
axes[1].set_ylabel("Churn Rate (%)")
for bar, rate in zip(bars, geo_churn["Churn Rate (%)"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{rate}%", ha="center", fontweight="bold")

plt.suptitle("Geographic Churn Analysis", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()

# 💡 Insight
print("\n💡 KEY INSIGHT: Germany has the highest churn rate (~32%), nearly 2x that of France (~16%) and Spain (~17%).")


# In[6]: Churn rate by Gender
# ─────────────────────────────────────────────────────────────
gender_churn = df.groupby("Gender")["Exited"].agg(["sum", "count"]).reset_index()
gender_churn.columns = ["Gender", "Churned", "Total"]
gender_churn["Churn Rate (%)"] = (gender_churn["Churned"] / gender_churn["Total"] * 100).round(1)

print("Churn by Gender:")
print(gender_churn.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(data=df, x="Gender", hue=df["Exited"].map({0: "Stayed", 1: "Churned"}),
              palette={"Stayed": "#3B82F6", "Churned": "#EF4444"}, ax=axes[0])
axes[0].set_title("Customer Count by Gender and Churn")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Count")
axes[0].legend(title="Status")

bars = axes[1].bar(gender_churn["Gender"], gender_churn["Churn Rate (%)"],
                    color=["#F9A8D4", "#93C5FD"], edgecolor="white", width=0.4)
for bar, rate in zip(bars, gender_churn["Churn Rate (%)"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{rate}%", ha="center", fontweight="bold")
axes[1].set_title("Churn Rate by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()

print("\n💡 KEY INSIGHT: Female customers have a significantly higher churn rate (~25%) vs males (~16%).")


# In[7]: Age distribution by churn (overlapping histograms)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overlapping histograms
for status, color, label in [(0, "#3B82F6", "Stayed"), (1, "#EF4444", "Churned")]:
    axes[0].hist(df[df["Exited"] == status]["Age"], bins=30,
                 alpha=0.6, color=color, label=label, edgecolor="white")
axes[0].set_title("Age Distribution by Churn Status")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].axvline(df[df["Exited"]==1]["Age"].mean(), color="#EF4444", linestyle="--", label="Churn mean")
axes[0].axvline(df[df["Exited"]==0]["Age"].mean(), color="#3B82F6", linestyle="--", label="Stay mean")

# Boxplot
df_age = df.copy()
df_age["Status"] = df_age["Exited"].map({0: "Stayed", 1: "Churned"})
df_age.boxplot(column="Age", by="Status", ax=axes[1],
               boxprops=dict(color="#1E3A5F"),
               medianprops=dict(color="#EF4444", linewidth=2))
axes[1].set_title("Age Boxplot by Churn Status")
axes[1].set_xlabel("Customer Status")
axes[1].set_ylabel("Age")
plt.suptitle("")

plt.tight_layout()
plt.show()

churned_age = df[df["Exited"]==1]["Age"].mean()
stayed_age = df[df["Exited"]==0]["Age"].mean()
print(f"\n💡 KEY INSIGHT: Churned customers have a higher average age ({churned_age:.1f}) vs retained ({stayed_age:.1f}).")
print("   Customers aged 40-60 are disproportionately at risk.")


# In[8]: Balance distribution by churn (KDE plot)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# KDE plot
for status, color, label in [(0, "#3B82F6", "Stayed"), (1, "#EF4444", "Churned")]:
    subset = df[df["Exited"] == status]["Balance"]
    subset.plot.kde(ax=axes[0], color=color, label=label, linewidth=2.5)
axes[0].set_title("Balance Distribution — KDE")
axes[0].set_xlabel("Account Balance (₹)")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))

# Zero-balance analysis
zero_bal_churn = df[df["Balance"] == 0]["Exited"].value_counts(normalize=True) * 100
nonzero_bal_churn = df[df["Balance"] > 0]["Exited"].value_counts(normalize=True) * 100

categories = ["Zero Balance", "Non-Zero Balance"]
churn_rates = [zero_bal_churn.get(1, 0), nonzero_bal_churn.get(1, 0)]
bars = axes[1].bar(categories, churn_rates, color=["#EF4444", "#3B82F6"], edgecolor="white", width=0.4)
for bar, rate in zip(bars, churn_rates):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{rate:.1f}%", ha="center", fontweight="bold")
axes[1].set_title("Churn Rate: Zero vs Non-Zero Balance")
axes[1].set_ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()

print("\n💡 KEY INSIGHT: Counterintuitively, customers WITH higher balances churn more.")
print("   Zero-balance accounts have unique churn patterns — these customers may be 'dormant'.")


# In[9]: Correlation heatmap (all numeric features)
# ─────────────────────────────────────────────────────────────
numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # upper triangle mask
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Feature Correlation Matrix", pad=15)
plt.tight_layout()
plt.show()

print("\nTop correlations with Churn (Exited):")
target_corr = corr_matrix["Exited"].drop("Exited").abs().sort_values(ascending=False)
for feat, corr_val in target_corr.items():
    direction = "+" if corr_matrix.loc[feat, "Exited"] > 0 else "-"
    print(f"  {feat:20s}: {direction}{corr_val:.3f}")


# In[10]: NumOfProducts vs Churn
# ─────────────────────────────────────────────────────────────
prod_analysis = df.groupby("NumOfProducts")["Exited"].agg(["mean", "count"]).reset_index()
prod_analysis.columns = ["NumOfProducts", "Churn Rate", "Count"]
prod_analysis["Churn Rate %"] = prod_analysis["Churn Rate"] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Churn rate by product count
colors = ["#3B82F6" if r < 25 else "#F59E0B" if r < 50 else "#EF4444"
          for r in prod_analysis["Churn Rate %"]]
bars = axes[0].bar(prod_analysis["NumOfProducts"], prod_analysis["Churn Rate %"],
                    color=colors, edgecolor="white", width=0.5)
for bar, rate in zip(bars, prod_analysis["Churn Rate %"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{rate:.1f}%", ha="center", fontweight="bold")
axes[0].set_title("Churn Rate by Number of Products")
axes[0].set_xlabel("Number of Products")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].set_xticks([1, 2, 3, 4])

# Customer count by product
axes[1].bar(prod_analysis["NumOfProducts"], prod_analysis["Count"],
             color="#93C5FD", edgecolor="white", width=0.5)
axes[1].set_title("Customer Count by Number of Products")
axes[1].set_xlabel("Number of Products")
axes[1].set_ylabel("Number of Customers")
axes[1].set_xticks([1, 2, 3, 4])

plt.tight_layout()
plt.show()

print(prod_analysis.to_string(index=False))
print("\n💡 KEY INSIGHT: Customers with 3-4 products have extremely high churn (>80%)!")
print("   This is counter-intuitive — perhaps forced cross-selling creates dissatisfaction.")


# In[11]: Tenure vs Churn (boxplot)
# ─────────────────────────────────────────────────────────────
tenure_churn = df.groupby("Tenure")["Exited"].agg(["mean", "count"]).reset_index()
tenure_churn.columns = ["Tenure", "Churn Rate", "Count"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Line plot of churn rate by tenure
axes[0].plot(tenure_churn["Tenure"], tenure_churn["Churn Rate"] * 100,
              marker="o", linewidth=2.5, color="#1E3A5F", markersize=8)
axes[0].fill_between(tenure_churn["Tenure"], tenure_churn["Churn Rate"] * 100,
                      alpha=0.15, color="#1E3A5F")
axes[0].set_title("Churn Rate by Tenure (Years)")
axes[0].set_xlabel("Tenure (Years)")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].set_xticks(range(0, 11))

# Boxplot
df_ten = df.copy()
df_ten["Status"] = df_ten["Exited"].map({0: "Stayed", 1: "Churned"})
df_ten.boxplot(column="Tenure", by="Status", ax=axes[1])
plt.suptitle("")
axes[1].set_title("Tenure Distribution by Churn Status")
axes[1].set_xlabel("Customer Status")
axes[1].set_ylabel("Tenure (Years)")

plt.tight_layout()
plt.show()

print("\n💡 KEY INSIGHT: Churn rate is relatively uniform across tenure groups.")
print("   This means even long-standing customers can leave — relationship quality matters throughout.")


# In[12]: CreditScore distribution by Geography
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

geo_colors = {"France": "#3B82F6", "Germany": "#EF4444", "Spain": "#F59E0B"}
for geo, color in geo_colors.items():
    subset = df[df["Geography"] == geo]["CreditScore"]
    subset.plot.kde(ax=ax, color=color, label=geo, linewidth=2.5)

ax.axvline(df["CreditScore"].mean(), color="#1E3A5F", linestyle="--",
           linewidth=1.5, label=f"Overall mean ({df['CreditScore'].mean():.0f})")
ax.set_title("Credit Score Distribution by Geography")
ax.set_xlabel("Credit Score")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.show()

print("\nCredit Score statistics by Geography:")
print(df.groupby("Geography")["CreditScore"].agg(["mean", "std", "min", "max"]).round(1))


# In[13]: IsActiveMember vs Churn (countplot)
# ─────────────────────────────────────────────────────────────
df_active = df.copy()
df_active["Active Status"] = df_active["IsActiveMember"].map({0: "Inactive", 1: "Active"})
df_active["Churn Status"] = df_active["Exited"].map({0: "Stayed", 1: "Churned"})

active_churn = df_active.groupby(["Active Status", "Churn Status"]).size().unstack()
active_churn_pct = active_churn.div(active_churn.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

active_churn.plot(kind="bar", ax=axes[0], color=["#EF4444", "#3B82F6"], edgecolor="white")
axes[0].set_title("Churn Count: Active vs Inactive Members")
axes[0].set_xlabel("Member Status")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(title="Churn Status")

active_churn_pct.plot(kind="bar", ax=axes[1], color=["#EF4444", "#3B82F6"], edgecolor="white")
axes[1].set_title("Churn Rate (%): Active vs Inactive Members")
axes[1].set_xlabel("Member Status")
axes[1].set_ylabel("Percentage (%)")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Churn Status")

plt.tight_layout()
plt.show()

inactive_churn = df_active[df_active["Active Status"]=="Inactive"]["Exited"].mean()*100
active_churn_rate = df_active[df_active["Active Status"]=="Active"]["Exited"].mean()*100
print(f"\n💡 KEY INSIGHT: Inactive members churn at {inactive_churn:.1f}% vs {active_churn_rate:.1f}% for active members.")
print("   IsActiveMember is consistently one of the top SHAP features in the model.")


# In[14]: Multi-variable scatter: Age + Balance coloured by Churn
# ─────────────────────────────────────────────────────────────
fig = px.scatter(
    df, x="Age", y="Balance",
    color=df["Exited"].map({0: "Stayed", 1: "Churned"}),
    color_discrete_map={"Stayed": "#3B82F6", "Churned": "#EF4444"},
    opacity=0.5,
    hover_data=["Geography", "NumOfProducts", "IsActiveMember"],
    title="Age vs Balance — Coloured by Churn Status",
    labels={"color": "Status"},
    template="plotly_white",
    height=550
)
fig.update_traces(marker=dict(size=5))
fig.show()

print("\n💡 KEY INSIGHT: Churned customers span all balance levels but cluster in the 40-60 age range.")
print("   Both zero-balance AND high-balance customers churn — different reasons require different strategies.")


# In[15]: Key findings markdown summary
# ─────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    KEY BUSINESS INSIGHTS FROM EDA                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  1. 🌍 GEOGRAPHY: German customers churn at ~32% — nearly twice      ║
║     the rate of French (~16%) or Spanish (~17%) customers.           ║
║     Country-specific retention programs are essential.               ║
║                                                                       ║
║  2. 👩 GENDER: Female customers churn at ~25% vs ~16% for males.    ║
║     Women may have unmet needs — financial planning products         ║
║     designed for their life stages could reduce this gap.            ║
║                                                                       ║
║  3. 👤 AGE: Customers aged 40-60 disproportionately churn.           ║
║     Average churner age is ~45 vs ~37 for retained customers.        ║
║     Mid-life customers need wealth management and retirement tools.  ║
║                                                                       ║
║  4. 📦 PRODUCTS: Customers with 3-4 products churn >80%!             ║
║     This suggests over-selling is counterproductive.                 ║
║     Focus on quality of product fit, not quantity.                   ║
║                                                                       ║
║  5. 🏃 ACTIVITY: Inactive members churn at 26% vs 14% for active.   ║
║     Re-engagement campaigns should target inactive members first.    ║
║     IsActiveMember is the single strongest predictor of churn.       ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")
