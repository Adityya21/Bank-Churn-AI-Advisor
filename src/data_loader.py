"""
data_loader.py — Handles loading, validating, and inspecting the churn dataset.

The Kaggle "Churn_Modelling.csv" has 10,000 rows and 14 columns:
  RowNumber, CustomerId, Surname, CreditScore, Geography, Gender,
  Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
  EstimatedSalary, Exited (target)
"""

import os
import pandas as pd


# ─── Expected schema for the Churn_Modelling.csv dataset ─────────────────────
EXPECTED_COLUMNS = [
    "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography",
    "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary", "Exited"
]

# Column dtype expectations (used for schema validation)
EXPECTED_DTYPES = {
    "RowNumber": "int64",
    "CustomerId": "int64",
    "CreditScore": "int64",
    "Geography": "object",
    "Gender": "object",
    "Age": "int64",
    "Tenure": "int64",
    "Balance": "float64",
    "NumOfProducts": "int64",
    "HasCrCard": "int64",
    "IsActiveMember": "int64",
    "EstimatedSalary": "float64",
    "Exited": "int64",
}


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the churn CSV file, validate its columns, and return a clean DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the Churn_Modelling.csv file.

    Returns
    -------
    pd.DataFrame
        The loaded and validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please place 'Churn_Modelling.csv' in the data/ directory."
        )

    df = pd.read_csv(filepath)

    # Quick validation — make sure we have all 14 expected columns
    validate_schema(df)

    # Print summary info for debugging / notebook usage
    churn_rate = df["Exited"].mean() * 100
    print(f"[OK] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Churn rate: {churn_rate:.1f}% ({df['Exited'].sum()} churned out of {len(df)})")

    return df


def get_feature_names() -> dict:
    """
    Return a dictionary mapping column roles to their names.

    Keys
    ----
    numeric   : list — continuous / ordinal numeric columns used as features
    categorical : list — categorical columns that need encoding
    target    : str  — the column we are predicting
    drop      : list — columns to drop before modelling (identifiers)
    """
    return {
        "numeric": [
            "CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
        ],
        "categorical": ["Geography", "Gender"],
        "target": "Exited",
        "drop": ["RowNumber", "CustomerId", "Surname"],
    }


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that all required columns exist and have broadly correct dtypes.

    Returns True if valid; raises ValueError with details if not.
    """
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in dataset: {missing}. "
            f"Expected columns: {EXPECTED_COLUMNS}"
        )

    # Soft dtype check — warn but don't crash (CSVs sometimes read ints as floats)
    dtype_issues = []
    for col, expected_dtype in EXPECTED_DTYPES.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            # Allow int64 ↔ float64 mismatch (common in CSVs)
            if expected_dtype == "int64" and actual == "float64":
                continue
            if actual != expected_dtype:
                dtype_issues.append(f"  {col}: expected {expected_dtype}, got {actual}")

    if dtype_issues:
        print("[WARN] Dtype mismatches (non-critical):")
        for issue in dtype_issues:
            print(issue)

    return True
