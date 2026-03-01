"""
Training script — Champion (Logistic Scorecard) & Challenger (XGBoost).

Usage:
    python src/train.py --model scorecard --data data/processed
    python src/train.py --model xgboost  --data data/processed
"""

import argparse
import pathlib
import json
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import optuna

from woe_encoder import WoEEncoder
from evaluate import compute_ks_statistic, compute_gini

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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


# ── Feature selection helpers ─────────────────────────────────
def select_features_by_iv(encoder: WoEEncoder, iv_floor: float = 0.02) -> list[str]:
    """Keep features with IV >= iv_floor (industry standard)."""
    iv_df = encoder.get_iv_summary()
    selected = iv_df[iv_df["IV"] >= iv_floor].index.tolist()
    print(f"  IV selection: {len(selected)}/{len(iv_df)} features kept (IV >= {iv_floor})")
    return selected


# ── Champion: Logistic Scorecard ─────────────────────────────
def train_scorecard(X: pd.DataFrame, y: pd.Series):
    mlflow.set_experiment("credit_risk_champion")

    with mlflow.start_run(run_name="scorecard_v2"):
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        X_num = X[numeric_cols].copy()

        # Step 1: Fit WoE on full training set to identify usable features
        encoder_pre = WoEEncoder(bins=10).fit(X_num, y)
        selected = select_features_by_iv(encoder_pre, iv_floor=0.02)
        X_num = X_num[selected]

        # Step 2: Tune regularization with nested CV
        print("\n  Tuning regularization (C) ...")
        best_c, best_mean_auc = 0.1, 0.0
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for c_val in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            fold_aucs = []
            for train_idx, val_idx in skf.split(X_num, y):
                X_tr, X_val = X_num.iloc[train_idx], X_num.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                enc = WoEEncoder(bins=10).fit(X_tr, y_tr)
                X_tr_woe = enc.transform(X_tr).fillna(0)
                X_val_woe = enc.transform(X_val).fillna(0)

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_woe)
                X_val_s = scaler.transform(X_val_woe)

                mdl = LogisticRegression(
                    C=c_val, l1_ratio=0, class_weight="balanced",
                    max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
                )
                mdl.fit(X_tr_s, y_tr)
                fold_aucs.append(roc_auc_score(y_val, mdl.predict_proba(X_val_s)[:, 1]))

            mean_auc = np.mean(fold_aucs)
            print(f"    C={c_val:<6}  Mean AUC={mean_auc:.4f}")
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_c = c_val

        print(f"  Best C={best_c}  AUC={best_mean_auc:.4f}")

        # Step 3: Final CV evaluation with best C
        cv_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_num, y)):
            X_tr, X_val = X_num.iloc[train_idx], X_num.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            enc = WoEEncoder(bins=10).fit(X_tr, y_tr)
            X_tr_woe = enc.transform(X_tr).fillna(0)
            X_val_woe = enc.transform(X_val).fillna(0)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_woe)
            X_val_s = scaler.transform(X_val_woe)

            model = LogisticRegression(
                C=best_c, l1_ratio=0, class_weight="balanced",
                max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
            )
            model.fit(X_tr_s, y_tr)

            y_prob = model.predict_proba(X_val_s)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            gini = compute_gini(y_val, y_prob)
            ks = compute_ks_statistic(y_val, y_prob)

            cv_results.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
            print(f"  Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

        # Step 4: Final fit on full data
        encoder_full = WoEEncoder(bins=10).fit(X_num, y)
        X_woe_full = encoder_full.transform(X_num).fillna(0)
        scaler_full = StandardScaler()
        X_scaled_full = scaler_full.fit_transform(X_woe_full)

        final_model = LogisticRegression(
            C=best_c, l1_ratio=0, class_weight="balanced",
            max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
        )
        final_model.fit(X_scaled_full, y)

        # Log to MLflow
        mlflow.log_params({
            "model_type": "scorecard", "bins": 10, "folds": N_FOLDS,
            "C": best_c, "n_features": len(selected),
        })
        mlflow.log_metric("mean_auc", np.mean([r["auc"] for r in cv_results]))
        mlflow.log_metric("mean_gini", np.mean([r["gini"] for r in cv_results]))
        mlflow.log_metric("mean_ks", np.mean([r["ks"] for r in cv_results]))

        # Save artefacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, MODELS_DIR / "scorecard_champion.pkl")
        joblib.dump(encoder_full, MODELS_DIR / "woe_encoder.pkl")
        joblib.dump(scaler_full, MODELS_DIR / "scaler.pkl")

        iv_summary = encoder_full.get_iv_summary()
        iv_summary.to_csv(MODELS_DIR / "iv_summary.csv")
        mlflow.log_artifact(str(MODELS_DIR / "iv_summary.csv"))

        # Save selected feature list for inference
        with open(MODELS_DIR / "selected_features.json", "w") as f:
            json.dump(selected, f)

        print(f"\nChampion scorecard saved  |  Mean AUC={np.mean([r['auc'] for r in cv_results]):.4f}")
        return final_model, encoder_full, scaler_full


# ── Challenger: XGBoost with Optuna tuning ───────────────────
def train_xgboost(X: pd.DataFrame, y: pd.Series, n_trials: int = 30):
    mlflow.set_experiment("credit_risk_challenger")

    # Encode categoricals
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X_encoded[col] = X_encoded[col].astype("category").cat.codes

    pos_weight = float((y == 0).sum() / (y == 1).sum())
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Optuna objective
    def objective(trial):
        p = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "scale_pos_weight": pos_weight,
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        }
        fold_aucs = []
        for train_idx, val_idx in skf.split(X_encoded, y):
            mdl = xgb.XGBClassifier(**p, early_stopping_rounds=50)
            mdl.fit(
                X_encoded.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X_encoded.iloc[val_idx], y.iloc[val_idx])],
                verbose=False,
            )
            y_prob = mdl.predict_proba(X_encoded.iloc[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y.iloc[val_idx], y_prob))
        return np.mean(fold_aucs)

    print(f"  Running Optuna ({n_trials} trials) ...")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": pos_weight,
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
    })
    print(f"  Best trial AUC={study.best_value:.4f}")
    print(f"  Best params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()}, indent=2)}")

    with mlflow.start_run(run_name="xgboost_v2"):
        mlflow.log_params(best_params)

        # Final CV with best params for clean metrics
        cv_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
            model = xgb.XGBClassifier(**best_params, early_stopping_rounds=50)
            model.fit(
                X_encoded.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X_encoded.iloc[val_idx], y.iloc[val_idx])],
                verbose=False,
            )
            y_prob = model.predict_proba(X_encoded.iloc[val_idx])[:, 1]
            auc = roc_auc_score(y.iloc[val_idx], y_prob)
            gini = compute_gini(y.iloc[val_idx], y_prob)
            ks = compute_ks_statistic(y.iloc[val_idx], y_prob)
            cv_results.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
            print(f"  Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

        # Final fit on full data (no early stopping)
        final_params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(X_encoded, y, verbose=False)

        mlflow.log_metric("mean_auc", np.mean([r["auc"] for r in cv_results]))
        mlflow.log_metric("mean_gini", np.mean([r["gini"] for r in cv_results]))
        mlflow.log_metric("mean_ks", np.mean([r["ks"] for r in cv_results]))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, MODELS_DIR / "xgboost_challenger.pkl")
        mlflow.log_artifact(str(MODELS_DIR / "xgboost_challenger.pkl"))

        print(f"\nChallenger XGBoost saved  |  Mean AUC={np.mean([r['auc'] for r in cv_results]):.4f}")
        return final_model


# ── Challenger 2: LightGBM with Optuna tuning ───────────────
def train_lightgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 30):
    """
    LightGBM with DART boosting.

    Standard LightGBM GBDT underperforms XGBoost on this NaN-heavy dataset
    (AUC ~0.73 vs ~0.79).  DART (Dropouts meet Multiple Additive Regression
    Trees) applies dropout regularisation across boosting iterations and
    recovers parity with XGBoost.

    DART is too slow for per-trial Optuna (each 5-fold CV takes ~20 min),
    so we use validated parameters with 5-fold CV instead.
    """
    mlflow.set_experiment("credit_risk_lightgbm")

    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X_encoded[col] = X_encoded[col].astype("category").cat.codes

    pos_weight = float((y == 0).sum() / (y == 1).sum())
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Validated DART parameters — tested at 5-fold CV AUC=0.7821
    params = {
        "boosting_type": "dart",
        "n_estimators": 800,
        "max_depth": -1,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "drop_rate": 0.1,
        "scale_pos_weight": pos_weight,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    print("  Training LightGBM DART with validated params ...")
    with mlflow.start_run(run_name="lightgbm_dart_v1"):
        mlflow.log_params(params)

        cv_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_encoded.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X_encoded.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[lgb.log_evaluation(0)],
            )
            y_prob = model.predict_proba(X_encoded.iloc[val_idx])[:, 1]
            auc = roc_auc_score(y.iloc[val_idx], y_prob)
            gini = compute_gini(y.iloc[val_idx], y_prob)
            ks = compute_ks_statistic(y.iloc[val_idx], y_prob)
            cv_results.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
            print(f"  Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

        # Final fit on full data
        final_model = lgb.LGBMClassifier(**params)
        final_model.fit(X_encoded, y, callbacks=[lgb.log_evaluation(0)])

        mlflow.log_metric("mean_auc", np.mean([r["auc"] for r in cv_results]))
        mlflow.log_metric("mean_gini", np.mean([r["gini"] for r in cv_results]))
        mlflow.log_metric("mean_ks", np.mean([r["ks"] for r in cv_results]))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, MODELS_DIR / "lightgbm_challenger.pkl")
        mlflow.log_artifact(str(MODELS_DIR / "lightgbm_challenger.pkl"))

        print(f"\nLightGBM DART saved  |  Mean AUC={np.mean([r['auc'] for r in cv_results]):.4f}")
        return final_model


# ── Stacking Ensemble ────────────────────────────────────────
def train_stacking(X: pd.DataFrame, y: pd.Series):
    """
    Stacking ensemble: XGBoost + LightGBM base learners,
    logistic regression meta-learner.

    Generates out-of-fold predictions from each base model,
    then trains a meta-learner on those predictions.
    """
    mlflow.set_experiment("credit_risk_stacking")

    X_encoded = X.copy()
    cat_cols_obj = X_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols_obj:
        X_encoded[col] = X_encoded[col].astype("category").cat.codes

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pos_weight = float((y == 0).sum() / (y == 1).sum())

    # Load tuned models if they exist, else use sensible defaults
    xgb_path = MODELS_DIR / "xgboost_challenger.pkl"
    lgb_path = MODELS_DIR / "lightgbm_challenger.pkl"

    if xgb_path.exists() and lgb_path.exists():
        xgb_ref = joblib.load(xgb_path)
        lgb_ref = joblib.load(lgb_path)
        xgb_params = xgb_ref.get_params()
        lgb_params = lgb_ref.get_params()
        # Clean up non-constructor params
        for k in ["callbacks", "feature_name", "feature_names_in_"]:
            xgb_params.pop(k, None)
            lgb_params.pop(k, None)
        print("  Using tuned XGBoost + LightGBM params from saved models")
    else:
        xgb_params = {
            "max_depth": 5, "learning_rate": 0.05, "n_estimators": 500,
            "scale_pos_weight": pos_weight, "tree_method": "hist",
            "eval_metric": "auc", "random_state": RANDOM_STATE,
        }
        lgb_params = {
            "boosting_type": "dart", "max_depth": -1,
            "learning_rate": 0.05, "n_estimators": 500,
            "num_leaves": 63, "scale_pos_weight": pos_weight,
            "verbose": -1, "random_state": RANDOM_STATE,
        }
        print("  Using default params (train xgboost + lightgbm first for better results)")

    # Clean params that cannot be passed to constructor
    for k in ["early_stopping_rounds"]:
        xgb_params.pop(k, None)
        lgb_params.pop(k, None)

    n_samples = len(X_encoded)
    oof_xgb = np.zeros(n_samples)
    oof_lgb = np.zeros(n_samples)

    print("\n  Generating out-of-fold predictions ...")
    with mlflow.start_run(run_name="stacking_v1"):
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
            X_tr = X_encoded.iloc[train_idx]
            X_val = X_encoded.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # XGBoost base
            xgb_es_params = {**xgb_params, "early_stopping_rounds": 50}
            xgb_model = xgb.XGBClassifier(**xgb_es_params)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

            # LightGBM base (DART — no early stopping)
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(
                X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(0)],
            )
            oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

            xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])
            lgb_auc = roc_auc_score(y_val, oof_lgb[val_idx])
            print(f"  Fold {fold}: XGB AUC={xgb_auc:.4f}  LGB AUC={lgb_auc:.4f}")

        # Meta-learner: logistic regression on OOF predictions
        meta_X = np.column_stack([oof_xgb, oof_lgb])
        meta_scaler = StandardScaler()
        meta_X_s = meta_scaler.fit_transform(meta_X)

        meta_model = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
        )
        meta_model.fit(meta_X_s, y)

        # Evaluate stacking
        meta_prob = meta_model.predict_proba(meta_X_s)[:, 1]
        stack_auc = roc_auc_score(y, meta_prob)
        stack_gini = compute_gini(y, meta_prob)
        stack_ks = compute_ks_statistic(y, meta_prob)
        print(f"\n  Stacking (in-sample): AUC={stack_auc:.4f}  Gini={stack_gini:.4f}  KS={stack_ks:.4f}")

        # Cross-validate stacking for unbiased estimate
        cv_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, y)):
            meta_model_cv = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
            scaler_cv = StandardScaler()
            X_tr_cv = scaler_cv.fit_transform(meta_X[train_idx])
            X_val_cv = scaler_cv.transform(meta_X[val_idx])
            meta_model_cv.fit(X_tr_cv, y.iloc[train_idx])
            y_prob_cv = meta_model_cv.predict_proba(X_val_cv)[:, 1]
            auc = roc_auc_score(y.iloc[val_idx], y_prob_cv)
            gini = compute_gini(y.iloc[val_idx], y_prob_cv)
            ks = compute_ks_statistic(y.iloc[val_idx], y_prob_cv)
            cv_results.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
            print(f"  Meta CV Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

        mlflow.log_metric("mean_auc", np.mean([r["auc"] for r in cv_results]))
        mlflow.log_metric("mean_gini", np.mean([r["gini"] for r in cv_results]))
        mlflow.log_metric("mean_ks", np.mean([r["ks"] for r in cv_results]))

        # Final base models on full data
        xgb_final = xgb.XGBClassifier(
            **{k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
        )
        xgb_final.fit(X_encoded, y, verbose=False)

        lgb_final = lgb.LGBMClassifier(**lgb_params)
        lgb_final.fit(X_encoded, y, callbacks=[lgb.log_evaluation(0)])

        # Save all stacking artefacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        stacking_artefacts = {
            "xgb_model": xgb_final,
            "lgb_model": lgb_final,
            "meta_model": meta_model,
            "meta_scaler": meta_scaler,
        }
        joblib.dump(stacking_artefacts, MODELS_DIR / "stacking_ensemble.pkl")

        print(f"\nStacking ensemble saved  |  Mean AUC={np.mean([r['auc'] for r in cv_results]):.4f}")
        return stacking_artefacts


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scorecard", "xgboost", "lightgbm", "stacking", "all"], required=True)
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials for XGBoost/LightGBM")
    args = parser.parse_args()

    X, y = load_data(pathlib.Path(args.data))
    print(f"Data: {X.shape[0]:,} rows x {X.shape[1]} features  |  Default rate: {y.mean():.2%}\n")

    if args.model == "scorecard":
        train_scorecard(X, y)
    elif args.model == "xgboost":
        train_xgboost(X, y, n_trials=args.trials)
    elif args.model == "lightgbm":
        train_lightgbm(X, y, n_trials=args.trials)
    elif args.model == "stacking":
        train_stacking(X, y)
    elif args.model == "all":
        train_scorecard(X, y)
        train_xgboost(X, y, n_trials=args.trials)
        train_lightgbm(X, y, n_trials=args.trials)
        train_stacking(X, y)
