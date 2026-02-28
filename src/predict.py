"""
Predict module — load a trained model and score a single application
or a batch DataFrame.

Usage:
    python src/predict.py --input data/processed/test_features.csv --output data/processed/predictions.csv
"""

import argparse
import pathlib

import joblib
import numpy as np
import pandas as pd
import shap


MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


def load_model(model_name: str = "xgboost_challenger"):
    """Load a pickled model from models/."""
    path = MODELS_DIR / f"{model_name}.pkl"
    return joblib.load(path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature columns and encode categoricals."""
    drop_cols = ["SK_ID_CURR", "SK_ID_PREV", "TARGET"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes
    return X


def score_batch(df: pd.DataFrame, model_name: str = "xgboost_challenger") -> pd.DataFrame:
    """Score a DataFrame; returns original df with prediction columns added."""
    model = load_model(model_name)
    X = prepare_features(df)

    probs = model.predict_proba(X)[:, 1]

    # Convert to credit score (log-odds → points scale)
    from woe_encoder import WoEEncoder
    log_odds = np.log(probs / (1 - probs + 1e-10))
    scores = [WoEEncoder.log_odds_to_score(lo) for lo in log_odds]

    df = df.copy()
    df["default_probability"] = probs
    df["credit_score"] = scores
    df["risk_tier"] = pd.cut(
        probs,
        bins=[0, 0.05, 0.15, 1.0],
        labels=["Low", "Medium", "High"],
    )
    df["recommendation"] = df["risk_tier"].map({
        "Low": "Approve",
        "Medium": "Review",
        "High": "Decline",
    })
    return df


def explain_prediction(
    model, X_single: pd.DataFrame, n_top: int = 5
) -> list[dict]:
    """Return top SHAP factors for a single observation."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_single)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # class=1 for binary

    abs_shap = np.abs(shap_vals[0])
    top_idx = abs_shap.argsort()[-n_top:][::-1]
    return [
        {
            "feature": X_single.columns[i],
            "shap_value": round(float(shap_vals[0][i]), 4),
            "direction": "increases risk" if shap_vals[0][i] > 0 else "decreases risk",
        }
        for i in top_idx
    ]


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/predictions.csv")
    parser.add_argument("--model", type=str, default="xgboost_challenger")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = score_batch(df, args.model)
    result.to_csv(args.output, index=False)
    print(f"✓ Predictions written to {args.output}  ({len(result):,} rows)")
