"""
Training script — Champion (Logistic Scorecard) & Challenger (XGBoost).

Usage:
    python src/train.py --model scorecard --data data/processed
    python src/train.py --model xgboost  --data data/processed
"""

import argparse
import pathlib
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from woe_encoder import WoEEncoder
from evaluate import compute_ks_statistic, compute_gini


RANDOM_STATE = 42
N_FOLDS = 5
MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


# ── Data loading ──────────────────────────────────────────────
def load_data(data_dir: pathlib.Path):
    df = pd.read_csv(data_dir / "train_features.csv")
    target = "TARGET"
    feature_cols = [
        c for c in df.columns
        if c not in (target, "SK_ID_CURR", "SK_ID_PREV")
    ]
    X = df[feature_cols]
    y = df[target]
    return X, y


# ── Champion: Logistic Scorecard ─────────────────────────────
def train_scorecard(X: pd.DataFrame, y: pd.Series):
    mlflow.set_experiment("credit_risk_champion")

    with mlflow.start_run(run_name="scorecard_v1"):
        # WoE transform (numeric cols only for scorecard)
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        X_num = X[numeric_cols].copy()

        encoder = WoEEncoder(bins=10)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_num, y)):
            X_tr, X_val = X_num.iloc[train_idx], X_num.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            enc = WoEEncoder(bins=10).fit(X_tr, y_tr)
            X_tr_woe = enc.transform(X_tr).fillna(0)
            X_val_woe = enc.transform(X_val).fillna(0)

            model = LogisticRegression(
                class_weight="balanced", max_iter=1000,
                solver="lbfgs", random_state=RANDOM_STATE,
            )
            model.fit(X_tr_woe, y_tr)

            y_prob = model.predict_proba(X_val_woe)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            gini = compute_gini(y_val, y_prob)
            ks = compute_ks_statistic(y_val, y_prob)

            cv_results.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
            print(f"  Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

        # Final fit on full data
        encoder_full = WoEEncoder(bins=10).fit(X_num, y)
        X_woe_full = encoder_full.transform(X_num).fillna(0)
        final_model = LogisticRegression(
            class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=RANDOM_STATE,
        )
        final_model.fit(X_woe_full, y)

        # Log to MLflow
        mlflow.log_params({"model_type": "scorecard", "bins": 10, "folds": N_FOLDS})
        mlflow.log_metric("mean_auc", np.mean([r["auc"] for r in cv_results]))
        mlflow.log_metric("mean_gini", np.mean([r["gini"] for r in cv_results]))
        mlflow.log_metric("mean_ks", np.mean([r["ks"] for r in cv_results]))

        # Save artefacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, MODELS_DIR / "scorecard_champion.pkl")
        joblib.dump(encoder_full, MODELS_DIR / "woe_encoder.pkl")

        iv_summary = encoder_full.get_iv_summary()
        iv_summary.to_csv(MODELS_DIR / "iv_summary.csv")
        mlflow.log_artifact(str(MODELS_DIR / "iv_summary.csv"))

        print(f"\nChampion scorecard saved  |  Mean AUC={np.mean([r['auc'] for r in cv_results]):.4f}")
        return final_model, encoder_full


# ── Challenger: XGBoost ──────────────────────────────────────
def train_xgboost(X: pd.DataFrame, y: pd.Series):
    mlflow.set_experiment("credit_risk_challenger")

    # Encode categoricals
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X_encoded[col] = X_encoded[col].astype("category").cat.codes

    with mlflow.start_run(run_name="xgboost_v1"):
        pos_weight = (y == 0).sum() / (y == 1).sum()
        params = {
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "scale_pos_weight": float(pos_weight),
            "eval_metric": "auc",
            "early_stopping_rounds": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        }
        mlflow.log_params(params)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
            X_tr = X_encoded.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_val = X_encoded.iloc[val_idx]
            y_val = y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            gini = compute_gini(y_val, y_prob)
            ks = compute_ks_statistic(y_val, y_prob)

            cv_results.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
            print(f"  Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

        # Final fit
        final_model = xgb.XGBClassifier(**{k: v for k, v in params.items() if k != "early_stopping_rounds"})
        final_model.fit(X_encoded, y, verbose=False)

        mlflow.log_metric("mean_auc", np.mean([r["auc"] for r in cv_results]))
        mlflow.log_metric("mean_gini", np.mean([r["gini"] for r in cv_results]))
        mlflow.log_metric("mean_ks", np.mean([r["ks"] for r in cv_results]))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, MODELS_DIR / "xgboost_challenger.pkl")
        mlflow.log_artifact(str(MODELS_DIR / "xgboost_challenger.pkl"))

        print(f"\nChallenger XGBoost saved  |  Mean AUC={np.mean([r['auc'] for r in cv_results]):.4f}")
        return final_model


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scorecard", "xgboost"], required=True)
    parser.add_argument("--data", type=str, default="data/processed")
    args = parser.parse_args()

    X, y = load_data(pathlib.Path(args.data))
    print(f"Data: {X.shape[0]:,} rows × {X.shape[1]} features  |  Default rate: {y.mean():.2%}\n")

    if args.model == "scorecard":
        train_scorecard(X, y)
    else:
        train_xgboost(X, y)
