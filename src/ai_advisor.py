"""
ai_advisor.py — LLM-powered Retention Strategy Advisor using Groq API.

This is the STANDOUT FEATURE of the project. It combines:
  - ML model predictions (churn probability)
  - SHAP explanations (which features drive the prediction)
  - LLM reasoning (Llama 3.1 via Groq) to generate personalised retention strategies

The module supports:
  1. Initial strategy generation from customer profile + ML output
  2. Follow-up conversational Q&A for deeper analysis
  3. Graceful fallback to demo mode when no API key is available

API: Groq (https://console.groq.com) — free tier, very fast inference.
Model: llama-3.1-8b-instant (free, fast, good enough for structured advice)
"""

import os

# Try to import Groq — graceful fallback if not installed
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Load .env explicitly from project root
try:
    from dotenv import load_dotenv
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_HERE)
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _secrets_file_exists() -> bool:
    """
    Return True only when a Streamlit secrets.toml actually exists on disk.
    Calling st.secrets when no file exists triggers a visible UI warning,
    so we guard against it with this pre-check.
    """
    candidates = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(_PROJECT_ROOT, ".streamlit", "secrets.toml"),
    ]
    return any(os.path.isfile(p) for p in candidates)


# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def get_groq_client():
    """
    Initialize and return a Groq API client.

    Reads the API key from (in order of priority):
      1. Streamlit secrets (for Streamlit Cloud deployment)
      2. Environment variable GROQ_API_KEY (for local .env file)

    Raises a clear error if no key is found.
    """
    api_key = None

    # Priority 1: Streamlit secrets (used on Streamlit Community Cloud)
    if _secrets_file_exists():
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", None)
        except (FileNotFoundError, KeyError, Exception):
            api_key = None

    # Priority 2: Environment variable (used locally with .env file)
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", None)

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in one of these locations:\n"
            "  • Local: Create a .env file with GROQ_API_KEY=your_key_here\n"
            "  • Streamlit Cloud: Add to app Settings → Secrets\n"
            "  • Get a FREE key at: https://console.groq.com"
        )

    if not GROQ_AVAILABLE:
        raise ImportError(
            "The 'groq' package is not installed. Run: pip install groq"
        )

    return Groq(api_key=api_key)


def is_api_available() -> bool:
    """
    Check if a valid GROQ_API_KEY exists (without making an API call).
    Used to show/hide API setup instructions in the UI.
    """
    # Check Streamlit secrets first (only if the file actually exists)
    if _secrets_file_exists():
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY", None)
            if key and key.strip() and key != "your_key_here":
                return True
        except (FileNotFoundError, KeyError, Exception):
            pass

    # Check environment variable
    key = os.environ.get("GROQ_API_KEY", None)
    if key and key.strip() and key != "your_key_here":
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    """
    Build the system prompt that shapes the LLM's behaviour as a bank advisor.

    The prompt instructs the LLM to:
    - Act as a senior bank relationship manager with data science expertise
    - Use the ML model's predictions and SHAP explanations as evidence
    - Provide practical, specific, empathetic retention strategies
    - Structure responses clearly with action items
    """
    return """You are a Senior Bank Relationship Manager and Customer Retention Expert with 15 years of experience in retail banking. You also have a strong background in data science and machine learning.

You have access to an ML-powered churn prediction system that uses XGBoost and SHAP explainability. Your role is to:

1. ANALYZE the customer profile and ML model's prediction
2. INTERPRET the SHAP feature importance to understand WHY the model flagged this customer
3. RECOMMEND specific, practical, empathetic retention strategies

Your responses should be structured as follows:

**🎯 Risk Assessment**
Brief summary of the customer's risk level and key stats.

**🔍 Root Causes**
Why is this customer likely to churn? Reference the SHAP factors directly.

**📋 Action Plan**
- **Immediate (48 hours):** Urgent actions
- **This Week:** Short-term engagement steps
- **This Month:** Longer-term relationship building

**💬 Talking Points**
2-3 specific conversation starters a relationship manager could use when contacting this customer.

IMPORTANT GUIDELINES:
- Be specific and practical — avoid generic advice like "improve customer satisfaction"
- Reference the actual data (e.g., "With a balance of ₹0 and only 1 product...")
- Be empathetic — remember this is about helping a real person
- Keep responses concise but complete (200-300 words)
- Use professional but warm language — like a trusted advisor, not a robot
- When discussing numbers, use the customer's actual values from the profile"""


def build_customer_context(
    customer_data: dict,
    churn_probability: float,
    risk_level: str,
    top_features: list,
    model_metrics: dict
) -> str:
    """
    Build a rich context string from the customer's data and ML predictions.

    This gives the LLM everything it needs to generate a specific,
    evidence-based retention strategy.

    Parameters
    ----------
    customer_data : dict — all customer input fields
    churn_probability : float — 0 to 1
    risk_level : str — "Low" / "Medium" / "High"
    top_features : list of (feature_name, direction, shap_value) tuples
    model_metrics : dict with model accuracy, AUC, etc.
    """
    # Format customer profile
    geography = customer_data.get("Geography", "N/A")
    gender = customer_data.get("Gender", "N/A")
    age = customer_data.get("Age", "N/A")
    credit_score = customer_data.get("CreditScore", "N/A")
    tenure = customer_data.get("Tenure", "N/A")
    balance = customer_data.get("Balance", 0)
    salary = customer_data.get("EstimatedSalary", 0)
    num_products = customer_data.get("NumOfProducts", "N/A")
    has_card = "Yes" if customer_data.get("HasCrCard", 0) else "No"
    is_active = "Yes" if customer_data.get("IsActiveMember", 0) else "No"

    # Format SHAP features
    shap_lines = []
    for i, (feat_name, direction, shap_val) in enumerate(top_features, 1):
        sign = "+" if shap_val > 0 else ""
        shap_lines.append(
            f"  {i}. {feat_name} → {direction.upper()} (SHAP: {sign}{shap_val:.3f})"
        )
    shap_section = "\n".join(shap_lines)

    # Format model metrics
    model_auc = model_metrics.get("roc_auc", "N/A")
    model_accuracy = model_metrics.get("accuracy", "N/A")
    if isinstance(model_auc, float):
        model_auc = f"{model_auc:.3f}"
    if isinstance(model_accuracy, float):
        model_accuracy = f"{model_accuracy:.3f}"

    context = f"""CUSTOMER PROFILE:
- Age: {age}, Geography: {geography}, Gender: {gender}
- Credit Score: {credit_score}
- Account Balance: ₹{balance:,.0f}, Estimated Salary: ₹{salary:,.0f}
- Number of Products: {num_products}
- Active Member: {is_active}, Has Credit Card: {has_card}
- Tenure: {tenure} years

ML MODEL PREDICTION:
- Churn Probability: {churn_probability*100:.1f}% ({risk_level.upper()} RISK)
- Model: XGBoost (AUC: {model_auc}, Accuracy: {model_accuracy})

TOP CHURN RISK FACTORS (from SHAP analysis):
{shap_section}

Based on this data, provide a complete retention strategy for this customer."""

    return context


# ─────────────────────────────────────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────────────────────────────────────

def get_initial_strategy(
    customer_data: dict,
    churn_probability: float,
    risk_level: str,
    top_features: list,
    model_metrics: dict
) -> str:
    """
    Call the Groq API to generate the first retention strategy.

    Uses Llama 3.1 8B Instant — free tier, fast inference (~2s response time).
    """
    try:
        client = get_groq_client()
    except (ValueError, ImportError) as e:
        return f"⚠️ API Error: {str(e)}\n\nShowing demo strategy instead.\n\n{get_demo_strategy(churn_probability, top_features)}"

    system_prompt = build_system_prompt()
    customer_context = build_customer_context(
        customer_data, churn_probability, risk_level, top_features, model_metrics
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this customer and give me a complete retention strategy.\n\n{customer_context}"}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ API call failed: {str(e)}\n\nShowing demo strategy instead.\n\n{get_demo_strategy(churn_probability, top_features)}"


def get_followup_response(
    conversation_history: list,
    user_question: str,
    customer_context: str
) -> str:
    """
    Continue the retention strategy conversation with follow-up questions.

    Maintains full conversation history so the LLM can reference prior context.
    """
    try:
        client = get_groq_client()
    except (ValueError, ImportError) as e:
        return f"⚠️ API Error: {str(e)}\n\nPlease set up your Groq API key to use the chat feature."

    # Build message list: system + context reminder + full conversation history + new question
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Here is the customer context for reference:\n\n{customer_context}"},
        {"role": "assistant", "content": "Understood. I have the customer's profile, ML prediction, and SHAP analysis. How can I help?"},
    ]

    # Add all previous conversation turns
    messages.extend(conversation_history)

    # Add the new user question
    messages.append({"role": "user", "content": user_question})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ API call failed: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_strategy_as_sections(strategy_text: str) -> dict:
    """
    Parse the LLM response and extract structured sections.

    Returns a dict with keys like 'risk_assessment', 'root_causes', etc.
    Falls back to putting everything in 'full_text' if parsing fails.
    """
    sections = {
        "risk_assessment": "",
        "root_causes": "",
        "immediate_actions": "",
        "weekly_plan": "",
        "talking_points": "",
        "full_text": strategy_text,
    }

    try:
        text = strategy_text

        # Try to extract each section by looking for header patterns
        section_markers = {
            "risk_assessment": ["Risk Assessment", "🎯 Risk Assessment"],
            "root_causes": ["Root Causes", "🔍 Root Causes"],
            "immediate_actions": ["Action Plan", "📋 Action Plan", "Immediate"],
            "talking_points": ["Talking Points", "💬 Talking Points"],
        }

        for key, markers in section_markers.items():
            for marker in markers:
                if marker.lower() in text.lower():
                    # Find the section content (between this marker and the next)
                    start_idx = text.lower().find(marker.lower())
                    # Find the next section marker
                    remaining = text[start_idx + len(marker):]
                    # Look for the next ** or ## marker
                    end_markers = ["**🎯", "**🔍", "**📋", "**💬", "## ", "---"]
                    end_idx = len(remaining)
                    for em in end_markers:
                        idx = remaining.find(em)
                        if idx > 0 and idx < end_idx:
                            end_idx = idx
                    sections[key] = remaining[:end_idx].strip().lstrip("*:").strip()
                    break

    except Exception:
        # If parsing fails, the full_text fallback is already set
        pass

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / FALLBACK MODE
# ─────────────────────────────────────────────────────────────────────────────

def get_demo_strategy(churn_probability: float, top_features: list) -> str:
    """
    Return a realistic demo strategy when no API key is available.

    This ensures the app works and looks professional even without Groq setup.
    The content varies based on the top feature driving churn.
    """
    risk_level = "HIGH" if churn_probability > 0.7 else ("MEDIUM" if churn_probability > 0.4 else "LOW")

    # Get the primary risk driver to customize the demo response
    primary_feature = top_features[0][0] if top_features else "general"

    # Build feature-specific advice
    feature_strategies = {
        "IsActiveMember": (
            "The customer's inactive status is the strongest churn signal. "
            "Inactive members are 2.5x more likely to leave within 6 months. "
            "**Immediate action:** Schedule a personal call from their relationship manager. "
            "Offer a re-activation bonus (e.g., cashback on next 5 transactions). "
            "Enrol them in the monthly financial wellness newsletter."
        ),
        "Balance": (
            "The customer's account balance is a significant concern. "
            "**Immediate action:** Review their account for any recent large withdrawals. "
            "Offer competitive fixed deposit rates to encourage balance retention. "
            "Consider a tiered loyalty program that rewards maintaining higher balances."
        ),
        "zero_balance": (
            "A zero-balance account is one of the strongest churn indicators — these "
            "customers have essentially stopped using their account. "
            "**Immediate action:** Call within 48 hours to understand the situation. "
            "Offer a salary account migration with zero-fee benefits. "
            "Provide a welcome-back bonus for depositing within 30 days."
        ),
        "NumOfProducts": (
            "Single-product customers have significantly higher churn rates. "
            "They lack the 'stickiness' that comes with multiple banking relationships. "
            "**Immediate action:** Identify cross-sell opportunities based on their profile. "
            "Offer bundled product packages with fee waivers. "
            "Schedule a financial planning consultation to match products to needs."
        ),
        "Age": (
            "Age is a key demographic factor in this prediction. "
            "**For older customers (45+):** Emphasize stability, personal service, and wealth "
            "management options. **For younger customers (<30):** Highlight digital banking "
            "features, investment apps, and rewards programs. "
            "**Immediate action:** Align communication channel and product offerings with age preferences."
        ),
    }

    specific_advice = feature_strategies.get(
        primary_feature,
        "Based on the model's analysis, this customer shows signs of disengagement. "
        "**Immediate action:** Schedule a personal touchpoint within 48 hours. "
        "Review their product usage patterns and identify unmet needs. "
        "Consider offering loyalty rewards proportional to their relationship value."
    )

    return f"""**🎯 Risk Assessment**
This customer has a **{churn_probability*100:.0f}% churn probability** ({risk_level} RISK). The ML model (AUC: 0.91) has identified several concerning patterns in their banking behaviour.

**🔍 Root Causes**
{specific_advice}

**📋 Action Plan**
- **Immediate (48 hours):** Personal outreach from relationship manager
- **This Week:** Product review and tailored offer preparation
- **This Month:** Follow-up meeting to assess engagement improvement

**💬 Talking Points**
1. "I noticed your account hasn't been as active recently — is there anything we can do to better serve your needs?"
2. "We have some exclusive offers for valued customers like you — can I walk you through them?"
3. "I'd love to schedule a quick financial health check to make sure you're getting the most from your banking relationship."

---
*⚡ This is a demo strategy. Connect your FREE Groq API key for personalized AI-generated strategies.*"""
