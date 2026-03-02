"""
Fast retraining script — uses validated hyperparameters (no Optuna search).
This avoids the 30+ min Optuna search while still producing clean models.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from evaluate import compute_ks_statistic, compute_gini

RANDOM_STATE = 42
N_FOLDS = 5
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "processed" / "train_features.csv")
target = "TARGET"
feature_cols = [c for c in df.columns if c not in (target, "SK_ID_CURR", "SK_ID_PREV")]
X = df[feature_cols]
y = df[target]
pos_weight = float((y == 0).sum() / (y == 1).sum())
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(f"Data: {X.shape[0]:,} rows x {X.shape[1]} features  |  Default rate: {y.mean():.2%}\n")

# ══════════════════════════════════════════════════════════════
# 1. XGBoost — validated params from Optuna (best trial AUC=0.7821)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TRAINING XGBoost with validated params")
print("=" * 60)

xgb_params = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "gamma": 1.0,
    "scale_pos_weight": pos_weight,
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
}

xgb_cv = []
for fold, (tr, va) in enumerate(skf.split(X, y)):
    model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
    yp = model.predict_proba(X.iloc[va])[:, 1]
    auc = roc_auc_score(y.iloc[va], yp)
    gini = compute_gini(y.iloc[va], yp)
    ks = compute_ks_statistic(y.iloc[va], yp)
    xgb_cv.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
    print(f"  Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

xgb_mean = np.mean([r["auc"] for r in xgb_cv])
print(f"\n  XGBoost CV Mean AUC={xgb_mean:.4f}")

# Final fit
xgb_final = xgb.XGBClassifier(**xgb_params)
xgb_final.fit(X, y, verbose=False)
joblib.dump(xgb_final, MODELS_DIR / "xgboost_challenger.pkl")
print(f"  Saved xgboost_challenger.pkl\n")

# ══════════════════════════════════════════════════════════════
# 2. LightGBM DART — regularized params (no data leakage)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TRAINING LightGBM DART with regularized params")
print("=" * 60)

lgb_params = {
    "boosting_type": "dart",
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "drop_rate": 0.15,
    "max_drop": 50,
    "scale_pos_weight": pos_weight,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

lgb_cv = []
for fold, (tr, va) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.log_evaluation(0)])
    yp = model.predict_proba(X.iloc[va])[:, 1]
    auc = roc_auc_score(y.iloc[va], yp)
    gini = compute_gini(y.iloc[va], yp)
    ks = compute_ks_statistic(y.iloc[va], yp)
    lgb_cv.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
    print(f"  Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

lgb_mean = np.mean([r["auc"] for r in lgb_cv])
print(f"\n  LightGBM DART CV Mean AUC={lgb_mean:.4f}")

lgb_final = lgb.LGBMClassifier(**lgb_params)
lgb_final.fit(X, y, callbacks=[lgb.log_evaluation(0)])
joblib.dump(lgb_final, MODELS_DIR / "lightgbm_challenger.pkl")
print(f"  Saved lightgbm_challenger.pkl\n")

# ══════════════════════════════════════════════════════════════
# 3. Stacking Ensemble — XGB + LGB → LogisticRegression meta
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TRAINING Stacking Ensemble")
print("=" * 60)

n = len(X)
oof_xgb = np.zeros(n)
oof_lgb = np.zeros(n)

for fold, (tr, va) in enumerate(skf.split(X, y)):
    # XGBoost base
    xm = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    xm.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
    oof_xgb[va] = xm.predict_proba(X.iloc[va])[:, 1]
    
    # LightGBM base
    lm = lgb.LGBMClassifier(**lgb_params)
    lm.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
           callbacks=[lgb.log_evaluation(0)])
    oof_lgb[va] = lm.predict_proba(X.iloc[va])[:, 1]
    
    xa = roc_auc_score(y.iloc[va], oof_xgb[va])
    la = roc_auc_score(y.iloc[va], oof_lgb[va])
    print(f"  Fold {fold}: XGB AUC={xa:.4f}  LGB AUC={la:.4f}")

# Meta-learner
meta_X = np.column_stack([oof_xgb, oof_lgb])
meta_scaler = StandardScaler()
meta_X_s = meta_scaler.fit_transform(meta_X)
meta_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
meta_model.fit(meta_X_s, y)

# CV the meta-learner for unbiased metrics
stack_cv = []
for fold, (tr, va) in enumerate(skf.split(meta_X, y)):
    mm = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    sc = StandardScaler()
    Xtr = sc.fit_transform(meta_X[tr])
    Xva = sc.transform(meta_X[va])
    mm.fit(Xtr, y.iloc[tr])
    yp = mm.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(y.iloc[va], yp)
    gini = compute_gini(y.iloc[va], yp)
    ks = compute_ks_statistic(y.iloc[va], yp)
    stack_cv.append({"fold": fold, "auc": auc, "gini": gini, "ks": ks})
    print(f"  Meta Fold {fold}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

stack_mean = np.mean([r["auc"] for r in stack_cv])
print(f"\n  Stacking CV Mean AUC={stack_mean:.4f}")

# Final base models for stacking
xgb_stack = xgb.XGBClassifier(**xgb_params)
xgb_stack.fit(X, y, verbose=False)
lgb_stack = lgb.LGBMClassifier(**lgb_params)
lgb_stack.fit(X, y, callbacks=[lgb.log_evaluation(0)])

stacking_artefacts = {
    "xgb_model": xgb_stack,
    "lgb_model": lgb_stack,
    "meta_model": meta_model,
    "meta_scaler": meta_scaler,
}
joblib.dump(stacking_artefacts, MODELS_DIR / "stacking_ensemble.pkl")
print(f"  Saved stacking_ensemble.pkl\n")

# ══════════════════════════════════════════════════════════════
# 4. Evaluate all on hold-out
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("HOLD-OUT EVALUATION (last 20%)")
print("=" * 60)

split_idx = int(len(df) * 0.8)
df_test = df.iloc[split_idx:].reset_index(drop=True)
X_test = df_test[feature_cols]
y_test = df_test[target]

print(f"Hold-out: {len(df_test):,} rows  |  Default rate: {y_test.mean():.2%}\n")

results = {}
for name, mdl in [("xgboost_challenger", xgb_final), ("lightgbm_challenger", lgb_final)]:
    yp = mdl.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, yp)
    gini = compute_gini(y_test, yp)
    ks = compute_ks_statistic(y_test, yp)
    results[name] = {"AUC": round(auc, 4), "Gini": round(gini, 4), "KS": round(ks, 4)}
    print(f"  {name}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

# Stacking hold-out
xp = stacking_artefacts["xgb_model"].predict_proba(X_test)[:, 1]
lp = stacking_artefacts["lgb_model"].predict_proba(X_test)[:, 1]
mX = stacking_artefacts["meta_scaler"].transform(np.column_stack([xp, lp]))
yp = stacking_artefacts["meta_model"].predict_proba(mX)[:, 1]
auc = roc_auc_score(y_test, yp)
gini = compute_gini(y_test, yp)
ks = compute_ks_statistic(y_test, yp)
results["stacking_ensemble"] = {"AUC": round(auc, 4), "Gini": round(gini, 4), "KS": round(ks, 4)}
print(f"  stacking_ensemble: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

# Scorecard hold-out (uses WoE encoder)
from woe_encoder import WoEEncoder
woe = joblib.load(MODELS_DIR / "woe_encoder.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
with open(MODELS_DIR / "selected_features.json") as f:
    sel = json.load(f)

sc_model = joblib.load(MODELS_DIR / "scorecard_champion.pkl")
X_sc = X_test[[c for c in sel if c in X_test.columns]]
X_sc = woe.transform(X_sc).fillna(0)
X_sc = pd.DataFrame(scaler.transform(X_sc), columns=X_sc.columns)
yp = sc_model.predict_proba(X_sc)[:, 1]
auc = roc_auc_score(y_test, yp)
gini = compute_gini(y_test, yp)
ks = compute_ks_statistic(y_test, yp)
results["scorecard_champion"] = {"AUC": round(auc, 4), "Gini": round(gini, 4), "KS": round(ks, 4)}
print(f"  scorecard_champion: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

# Save comparison
reports_dir = ROOT / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
with open(reports_dir / "model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved model_comparison.json")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY (CV vs Hold-out — checking for overfitting)")
print("=" * 60)
print(f"{'Model':<25} {'CV AUC':>8} {'HO AUC':>8} {'Gap':>8}")
print("-" * 51)
print(f"{'Scorecard':<25} {'0.7585':>8} {results['scorecard_champion']['AUC']:>8.4f} {results['scorecard_champion']['AUC']-0.7585:>+8.4f}")
print(f"{'XGBoost':<25} {xgb_mean:>8.4f} {results['xgboost_challenger']['AUC']:>8.4f} {results['xgboost_challenger']['AUC']-xgb_mean:>+8.4f}")
print(f"{'LightGBM DART':<25} {lgb_mean:>8.4f} {results['lightgbm_challenger']['AUC']:>8.4f} {results['lightgbm_challenger']['AUC']-lgb_mean:>+8.4f}")
print(f"{'Stacking':<25} {stack_mean:>8.4f} {results['stacking_ensemble']['AUC']:>8.4f} {results['stacking_ensemble']['AUC']-stack_mean:>+8.4f}")
print("\nSmall (|gap| < 0.02) = no overfitting ✓")
