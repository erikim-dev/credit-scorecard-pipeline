"""
Credit Risk Scoring API — FastAPI application.

Endpoints:
    GET  /health       → readiness check
    GET  /model/info   → model metadata and metrics
    POST /score        → score a single application
    POST /score/batch  → score multiple applications
"""

from __future__ import annotations

import os
import sys
import pathlib
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# ── Paths ─────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from schemas import (
    ApplicationInput, ScoringResult, HealthResponse,
    BatchInput, BatchResult, ModelInfoResponse,
)
from woe_encoder import WoEEncoder

MODEL_VERSION = "2.1.0"

# ── Load model at startup ────────────────────────────────────
MODEL_PATH = pathlib.Path(
    os.getenv("MODEL_PATH", str(ROOT / "models" / "xgboost_challenger.pkl"))
)
model = None
explainer = None


def _load_artefacts() -> None:
    global model, explainer
    if MODEL_PATH.exists():
        import shap  # lazy — heavy dependency

        model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
    else:
        print(f"Model not found at {MODEL_PATH} — /score will return 503")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artefacts()
    yield


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Production-grade credit scoring endpoint. "
        "Returns default probability, mapped credit score, "
        "risk tier, and SHAP-based explainability."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────

# Cache model feature names at startup
_model_feature_names: list[str] = []


def _get_model_features() -> list[str]:
    """Get the feature names the model was trained on."""
    global _model_feature_names
    if not _model_feature_names and model is not None:
        if hasattr(model, "get_booster"):
            _model_feature_names = model.get_booster().feature_names
        elif hasattr(model, "feature_names_in_"):
            _model_feature_names = list(model.feature_names_in_)
    return _model_feature_names


def application_to_features(app_in: ApplicationInput) -> pd.DataFrame:
    """Map API input fields to the model's expected feature vector.

    Builds a partial feature vector from API inputs, then pads
    all missing columns with 0 so the model gets exactly the
    features it was trained on.
    """
    # Coalesce nullable external sources to median (0.5) for safety
    ext1 = app_in.ext_source_1 if app_in.ext_source_1 is not None else 0.5
    ext2 = app_in.ext_source_2 if app_in.ext_source_2 is not None else 0.5
    ext3 = app_in.ext_source_3 if app_in.ext_source_3 is not None else 0.5

    data = {
        "AGE_YEARS": app_in.age,
        "AMT_INCOME_TOTAL": app_in.income,
        "AMT_CREDIT": app_in.loan_amount,
        "AMT_ANNUITY": app_in.annuity,
        "AMT_GOODS_PRICE": app_in.goods_price,
        "EMPLOYMENT_YEARS": app_in.employment_years,
        "bureau_loan_count": app_in.bureau_loan_count,
        "active_credits": app_in.active_credits,
        "total_debt": app_in.total_debt,
        "overdue_count": app_in.overdue_count,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
    }

    # Derived ratios
    income = app_in.income + 1
    goods = app_in.goods_price + 1 if app_in.goods_price > 0 else 1.0
    age = app_in.age

    data.update({
        "LOAN_INCOME_RATIO": app_in.loan_amount / income,
        "ANNUITY_INCOME_RATIO": app_in.annuity / income,
        "CREDIT_GOODS_RATIO": app_in.loan_amount / goods,
        "CREDIT_TERM": app_in.annuity / (app_in.loan_amount + 1),
        "GOODS_INCOME_RATIO": app_in.goods_price / income,
        "INCOME_PER_PERSON": app_in.income,
        "DAYS_BIRTH": -abs(age) * 365.25,
        "DAYS_EMPLOYED": -abs(app_in.employment_years) * 365.25,
        "REGISTRATION_YEARS": 5.0,
        "ID_PUBLISH_YEARS": 5.0,
        "EMPLOY_TO_AGE_RATIO": app_in.employment_years / (age + 0.01),
        "EXT_SOURCE_MEAN": np.mean([ext1, ext2, ext3]),
        "EXT_SOURCE_STD": np.std([ext1, ext2, ext3]),
        "EXT_SOURCE_MIN": min(ext1, ext2, ext3),
        "EXT_SOURCE_MAX": max(ext1, ext2, ext3),
        "EXT_SOURCE_RANGE": max(ext1, ext2, ext3) - min(ext1, ext2, ext3),
        "EXT_SRC_1x2": ext1 * ext2,
        "EXT_SRC_2x3": ext2 * ext3,
        "EXT_SRC_1x3": ext1 * ext3,
        "EXT_SRC2_x_AGE": ext2 * age,
        "EXT_SRC3_x_AGE": ext3 * age,
        "PAYMENT_RATE": app_in.annuity / (app_in.loan_amount + 1),
        "CREDIT_OVERCHARGE": (app_in.loan_amount - app_in.goods_price) / goods,
    })

    # Build full-width DataFrame matching training features
    feature_names = _get_model_features()
    if feature_names:
        row = {col: data.get(col, 0) for col in feature_names}
        return pd.DataFrame([row])[feature_names]
    return pd.DataFrame([data])


def get_top_shap_factors(
    shap_vals: np.ndarray, feature_names: list[str], n: int = 3
) -> list[dict]:
    abs_vals = np.abs(shap_vals)
    top_idx = abs_vals.argsort()[-n:][::-1]
    return [
        {
            "feature": feature_names[i],
            "shap_value": round(float(shap_vals[i]), 4),
            "direction": "increases risk" if shap_vals[i] > 0 else "decreases risk",
        }
        for i in top_idx
    ]


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        version=MODEL_VERSION,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name="xgboost_challenger",
        model_version=MODEL_VERSION,
        model_type="XGBClassifier",
        metrics={
            "note": "Run evaluate.py for actual metrics",
        },
        features_count=model.n_features_in_ if model and hasattr(model, "n_features_in_") else 0,
        training_date="2026-02",
    )


def _score_single(application: ApplicationInput) -> ScoringResult:
    """Core scoring logic shared by single and batch endpoints."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = application_to_features(application)

    # Prediction
    default_prob = float(model.predict_proba(features)[0][1])
    risk_tier = (
        "Low" if default_prob < 0.05
        else "Medium" if default_prob < 0.15
        else "High"
    )

    # Score mapping
    log_odds = np.log(default_prob / (1 - default_prob + 1e-10))
    credit_score = WoEEncoder.log_odds_to_score(log_odds)

    # SHAP explanation
    top_factors = []
    if explainer is not None:
        shap_vals = explainer.shap_values(features)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        top_factors = get_top_shap_factors(
            shap_vals[0], features.columns.tolist(), n=3
        )

    return ScoringResult(
        default_probability=round(default_prob, 4),
        credit_score=credit_score,
        risk_tier=risk_tier,
        recommendation=(
            "Approve" if risk_tier == "Low"
            else "Review" if risk_tier == "Medium"
            else "Decline"
        ),
        top_risk_factors=top_factors,
        model_version=MODEL_VERSION,
    )


@app.post("/score", response_model=ScoringResult)
def score_application(application: ApplicationInput):
    return _score_single(application)


@app.post("/score/batch", response_model=BatchResult)
def score_batch(batch: BatchInput):
    results = [_score_single(app_input) for app_input in batch.applications]
    return BatchResult(results=results, total=len(results))
