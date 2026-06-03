"""
trainer.py — Train, tune, evaluate, and persist ML models for churn prediction.

Models trained:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost (with Optuna tuning)
  4. LightGBM

All models use class-weight balancing or scale_pos_weight to handle the
~20% churn vs 80% non-churn imbalance without needing SMOTE in production.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

# Optuna is only needed for hyperparameter tuning (not at inference time)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all 4 models, evaluate each, and return results + model objects.

    Returns
    -------
    results : dict
        Keys are model names, values are dicts of metrics.
    models : dict
        Keys are model names, values are fitted model objects.
    """
    # Compute positive-class weight for XGBoost (handles imbalance)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count

    # Define all 4 models with their configurations
    model_configs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # auto-adjusts weights inversely proportional to class frequency
            random_state=42,
            solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,  # use all CPU cores
            max_depth=10
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,  # handles class imbalance
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced",
            random_state=42,
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            verbose=-1  # suppress LightGBM training output
        ),
    }

    results = {}
    models = {}

    for name, model in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        models[name] = model

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    # Find and report the best model by AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n[BEST] Best model: {best_name} (AUC: {results[best_name]['roc_auc']:.4f})")

    return results, models


def tune_xgboost(X_train, y_train, n_trials: int = 50):
    """
    Use Optuna to find optimal XGBoost hyperparameters that maximize AUC.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials (more = better but slower).

    Returns
    -------
    dict — best hyperparameters found.
    """
    if not OPTUNA_AVAILABLE:
        print("[WARN] Optuna not installed. Returning default hyperparameters.")
        return {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "use_label_encoder": False,
            "eval_metric": "auc",
            "random_state": 42,
        }

        # Use 3-fold stratified CV for robust evaluation
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[OPTUNA] Best AUC: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")

    return study.best_params


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a fitted model and return a dict of metrics + confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve data for plotting
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def save_model(model, path: str):
    """Save a trained model to disk with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[OK] Model saved to {path}")


def load_model(path: str):
    """Load a trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'")
    model = joblib.load(path)
    print(f"[OK] Model loaded from {path}")
    return model
