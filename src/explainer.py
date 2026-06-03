"""
explainer.py — SHAP-based model explainability for churn predictions.

Uses TreeExplainer for tree-based models (XGBoost, LightGBM, Random Forest)
and LinearExplainer for Logistic Regression. Provides:
  - Global feature importance (bar + beeswarm plots)
  - Local explanations (waterfall for individual predictions)
  - Top feature extraction for AI Advisor context
  - Rule-based retention recommendations

Compatible with both:
  - SHAP < 0.43  : shap_values() returns list [class_0, class_1]
  - SHAP >= 0.43 : shap_values() returns 3-D ndarray (samples, features, classes)
                   or 2-D ndarray (samples, features) for some models
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import shap


def get_shap_explainer(model, X_train):
    """
    Create the appropriate SHAP explainer based on model type.

    TreeExplainer is ~100x faster than KernelExplainer for tree models,
    so we prefer it whenever the model supports it.
    """
    model_type = type(model).__name__

    if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
        explainer = shap.TreeExplainer(model)
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # Fallback to KernelExplainer (slow but universal)
        explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])

    return explainer


def _extract_class1_shap(shap_values):
    """
    Normalise SHAP output to always return a 2-D array of shape
    (n_samples, n_features) containing the class-1 (churn) SHAP values.

    Handles three output formats produced by different SHAP versions:
      1. list  [arr_class0, arr_class1]       → old API, tree models
      2. ndarray (n_samples, n_features, 2)   → new API (SHAP >= 0.43)
      3. ndarray (n_samples, n_features)      → already the positive class
    """
    if isinstance(shap_values, list):
        # Old SHAP API — binary classification returns a 2-element list
        return np.array(shap_values[1])

    arr = np.array(shap_values)

    if arr.ndim == 3:
        # New SHAP API — shape is (n_samples, n_features, n_classes)
        return arr[:, :, 1]

    # Already 2-D — (n_samples, n_features) for the positive class
    return arr


def _extract_expected_value(explainer):
    """
    Return the scalar base (expected) value for the positive class (churn).

    Handles:
      - scalar float
      - list or ndarray of length 2  → take index 1
      - ndarray of length 1          → take index 0
    """
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.asarray(ev).ravel()
        return float(ev[1]) if len(ev) > 1 else float(ev[0])
    return float(ev)


def get_shap_values(explainer, X):
    """
    Compute SHAP values for the given data.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        SHAP values for the positive class (churn = 1).
    """
    raw = explainer.shap_values(X)
    return _extract_class1_shap(raw)


def plot_summary(shap_values, X, feature_names, plot_type="bar"):
    """
    Create a SHAP summary plot (global feature importance).

    Parameters
    ----------
    plot_type : str
        "bar" for mean |SHAP| bar chart, "dot" for beeswarm plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        plot_type=plot_type,
        show=False,
        max_display=15
    )

    plt.title("Feature Impact on Churn Prediction", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    return plt.gcf()


def plot_waterfall_single(explainer, X_row, feature_names):
    """
    Create a SHAP waterfall plot for a single prediction.

    This shows how each feature pushes the prediction from the base value
    (average churn probability) toward the final prediction.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Compute SHAP values for the single row — ensure 2-D output
    raw = explainer.shap_values(X_row.reshape(1, -1))
    shap_vals_2d = _extract_class1_shap(raw)   # shape (1, n_features)
    shap_vals_1d = shap_vals_2d[0]              # shape (n_features,)

    expected_value = _extract_expected_value(explainer)

    # Create a SHAP Explanation object for the waterfall plot
    explanation = shap.Explanation(
        values=shap_vals_1d,
        base_values=expected_value,
        data=X_row,
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(explanation, show=False, max_display=10)
    plt.title("How Each Feature Affects This Prediction", fontsize=12, fontweight="bold")
    plt.tight_layout()

    return plt.gcf()


def get_top_features(shap_values_row, feature_names, n=3):
    """
    Extract the top-N most impactful features for a single prediction.

    Parameters
    ----------
    shap_values_row : 1-D array-like of shape (n_features,)
        SHAP values for a single sample (positive class).
    feature_names : list of str
    n : int

    Returns
    -------
    list of tuples: (feature_name, impact_direction, shap_value)
        impact_direction is either "increases churn risk" or "decreases churn risk"
    """
    # Ensure it's a flat 1-D numpy array
    shap_values_row = np.asarray(shap_values_row).ravel()

    # Pair each feature with its SHAP value
    feature_impacts = list(zip(feature_names, shap_values_row))

    # Sort by absolute SHAP value (most impactful first)
    feature_impacts.sort(key=lambda x: abs(float(x[1])), reverse=True)

    top_features = []
    for feat_name, shap_val in feature_impacts[:n]:
        shap_val = float(shap_val)
        direction = "increases churn risk" if shap_val > 0 else "decreases churn risk"
        top_features.append((feat_name, direction, round(shap_val, 4)))

    return top_features


def get_recommendation(churn_prob: float, top_features: list) -> str:
    """
    Generate a rule-based retention recommendation based on churn probability
    and the top SHAP features.

    This is the non-AI fallback — gives immediate, structured advice.
    """
    # Build feature-specific advice
    feature_advice = []
    for feat_name, direction, _ in top_features:
        if "increases" in direction:
            advice = _get_feature_advice(feat_name)
            if advice:
                feature_advice.append(advice)

    # Risk-level-based recommendation
    if churn_prob > 0.7:
        risk_header = "🔴 **HIGH RISK — Immediate retention action needed!**"
        risk_body = (
            "This customer has a very high probability of leaving. "
            "Prioritize personal outreach within the next 48 hours."
        )
    elif churn_prob > 0.4:
        risk_header = "🟡 **MEDIUM RISK — Proactive engagement recommended**"
        risk_body = (
            "This customer shows warning signs. "
            "Schedule a relationship review within the next 2 weeks."
        )
    else:
        risk_header = "🟢 **LOW RISK — Standard engagement**"
        risk_body = (
            "This customer appears satisfied. "
            "Continue regular check-ins and loyalty program engagement."
        )

    # Combine all parts
    recommendation = f"{risk_header}\n\n{risk_body}"
    if feature_advice:
        recommendation += "\n\n**Recommended Actions:**\n"
        for i, advice in enumerate(feature_advice, 1):
            recommendation += f"\n{i}. {advice}"

    return recommendation


def _get_feature_advice(feature_name: str) -> str:
    """Return specific advice based on which feature is driving churn risk."""
    advice_map = {
        "IsActiveMember": "Re-engage this inactive customer — offer exclusive benefits or a personal call from their relationship manager.",
        "Balance": "Address the account balance concern — offer competitive interest rates or a savings goal program.",
        "zero_balance": "This customer has a zero balance — reach out to understand why and offer incentives to re-deposit.",
        "NumOfProducts": "Cross-sell additional products — suggest a savings account, investment plan, or insurance to deepen the relationship.",
        "Age": "Age-specific retention — for older customers, emphasize stability and personalized service; for younger ones, push digital features.",
        "age_group": "Tailor engagement to this age segment — different life stages need different banking solutions.",
        "Geography_Germany": "German market customers have higher churn — consider region-specific loyalty programs.",
        "engagement_score": "Low engagement detected — launch a targeted re-activation campaign with personalized offers.",
        "CreditScore": "Credit score is a factor — consider offering credit-building products or financial advisory sessions.",
        "products_per_year": "Product adoption rate is concerning — focus on value demonstration for existing products.",
        "balance_salary_ratio": "Review the balance-to-salary ratio — ensure products match the customer's financial capacity.",
        "age_tenure_ratio": "Customer tenure relative to age suggests risk — strengthen the relationship with loyalty rewards.",
        "high_value": "This is a high-value customer — assign a dedicated relationship manager immediately.",
        "HasCrCard": "Credit card status is a factor — review card benefits and ensure they match customer spending patterns.",
        "Tenure": "Tenure-related risk — for short-tenure customers, focus on onboarding experience; for long-tenure, show appreciation.",
        "Gender": "Consider gender-specific financial product recommendations based on behavioral data.",
    }
    return advice_map.get(feature_name, "")
