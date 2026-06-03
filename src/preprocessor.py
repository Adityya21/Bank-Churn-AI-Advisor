"""
preprocessor.py — Feature engineering, encoding, scaling, and train-test splitting.

Pipeline:
  1. Drop identifier columns (RowNumber, CustomerId, Surname)
  2. Label-encode Gender (Female→0, Male→1)
  3. One-hot-encode Geography (drop_first=True → France is baseline)
  4. Engineer 7 new features that capture behavioural signals
  5. Scale all numeric features with StandardScaler
  6. Split into train/test (80/20 stratified)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class Preprocessor:
    """
    End-to-end preprocessor for the bank churn dataset.

    Usage
    -----
    >>> pp = Preprocessor()
    >>> X_train, X_test, y_train, y_test = pp.fit_transform(df)
    >>> pp.save("models/scaler.pkl")

    For single-row prediction:
    >>> pp.load("models/scaler.pkl")
    >>> X_single = pp.transform_single(customer_dict)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []  # populated during fit_transform

    # ─── Full dataset pipeline ────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Run the entire preprocessing pipeline and return train/test splits.

        Returns (X_train, X_test, y_train, y_test) — all as numpy arrays.
        """
        df = df.copy()

        # 1. Separate target before any transformations
        y = df["Exited"].values

        # 2. Drop identifier columns — they carry no predictive signal
        df = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"], errors="ignore")

        # 3. Encode Gender: Female → 0, Male → 1 (binary label encoding)
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(int)

        # 4. One-hot encode Geography (drop France as baseline to avoid multicollinearity)
        df = pd.get_dummies(df, columns=["Geography"], drop_first=True, dtype=int)

        # 5. Feature engineering — 7 new columns that capture real banking behaviour
        df = self._engineer_features(df)

        # 6. Store final feature list (needed for transform_single later)
        self.feature_names = list(df.columns)

        # 7. Scale all features to zero-mean, unit-variance
        X = self.scaler.fit_transform(df.values)

        # 8. Stratified train-test split (preserves churn ratio in both sets)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"[OK] Preprocessing complete:")
        print(f"   Features: {len(self.feature_names)} columns")
        print(f"   Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
        print(f"   Churn rate — Train: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")

        return X_train, X_test, y_train, y_test

    # ─── Single-row transformation (for live predictions) ─────────────────
    def transform_single(self, input_dict: dict) -> np.ndarray:
        """
        Transform a single customer input dictionary into a scaled feature vector.

        Parameters
        ----------
        input_dict : dict
            Must contain keys: CreditScore, Geography, Gender, Age, Tenure,
            Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

        Returns
        -------
        np.ndarray — shape (1, n_features), ready for model.predict()
        """
        # Build a one-row DataFrame to reuse the same column logic
        row = pd.DataFrame([input_dict])

        # Encode Gender
        row["Gender"] = row["Gender"].map({"Female": 0, "Male": 1}).astype(int)

        # One-hot encode Geography (must match training columns exactly)
        row = pd.get_dummies(row, columns=["Geography"], drop_first=False, dtype=int)

        # Ensure all Geography columns exist (some might be missing for a single row)
        for geo_col in ["Geography_Germany", "Geography_Spain"]:
            if geo_col not in row.columns:
                row[geo_col] = 0
        # Drop France column if it was created (drop_first logic)
        if "Geography_France" in row.columns:
            row = row.drop(columns=["Geography_France"])

        # Engineer features — same 7 as training
        row = self._engineer_features(row)

        # Reorder columns to match training feature order exactly
        # Add any missing columns as 0, drop any extra columns
        for col in self.feature_names:
            if col not in row.columns:
                row[col] = 0
        row = row[self.feature_names]

        # Scale using the fitted scaler
        X = self.scaler.transform(row.values)

        return X

    # ─── Feature engineering ──────────────────────────────────────────────
    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 7 engineered features capturing banking customer behaviour.
        """
        # Ratio of balance to salary — high ratio = more invested in the bank
        df["balance_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

        # Age group as ordinal — different age groups have different churn patterns
        df["age_group"] = pd.cut(
            df["Age"],
            bins=[0, 30, 45, 60, 100],
            labels=[0, 1, 2, 3]  # young=0, mid=1, senior=2, elderly=3
        ).astype(int)

        # Age relative to tenure — long tenure at young age = loyal customer
        df["age_tenure_ratio"] = df["Age"] / (df["Tenure"] + 1)

        # Zero balance flag — empty accounts are a strong churn signal
        df["zero_balance"] = (df["Balance"] == 0).astype(int)

        # High-value flag — customers with >100k balance are worth retaining
        df["high_value"] = (df["Balance"] > 100000).astype(int)

        # Engagement score — composite of activity, products, and credit card
        df["engagement_score"] = (
            df["IsActiveMember"] * df["NumOfProducts"] * (df["HasCrCard"] + 1)
        )

        # Products per year of tenure — rapid product adoption = engaged customer
        df["products_per_year"] = df["NumOfProducts"] / (df["Tenure"] + 1)

        return df

    # ─── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str):
        """Save the fitted scaler and feature names to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }, path)
        print(f"[OK] Preprocessor saved to {path}")

    def load(self, path: str):
        """Load a previously fitted scaler and feature names from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at '{path}'")
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        print(f"[OK] Preprocessor loaded from {path} ({len(self.feature_names)} features)")
