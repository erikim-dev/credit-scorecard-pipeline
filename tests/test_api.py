"""
Tests for the FastAPI Credit Risk Scoring API.

Run:  pytest tests/test_api.py -v
"""

import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# FastAPI test client (httpx-based)
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ── Valid application payload ─────────────────────────────────
VALID_PAYLOAD = {
    "age": 35,
    "income": 250_000,
    "loan_amount": 500_000,
    "annuity": 25_000,
    "goods_price": 450_000,
    "employment_years": 8,
    "education_type": "Higher education",
    "family_status": "Married",
    "housing_type": "House / apartment",
    "bureau_loan_count": 3,
    "active_credits": 1,
    "total_debt": 150_000,
    "overdue_count": 0,
    "ext_source_1": 0.5,
    "ext_source_2": 0.6,
    "ext_source_3": 0.4,
}


# ── Health Endpoint ──────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_is_ok(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


# ── Score Endpoint ───────────────────────────────────────────

class TestScoreEndpoint:
    def test_score_returns_200_or_503(self):
        """Returns 200 if model loaded, 503 if not — both are valid."""
        resp = client.post("/score", json=VALID_PAYLOAD)
        assert resp.status_code in (200, 503)

    def test_score_has_all_required_fields(self):
        resp = client.post("/score", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert "default_probability" in data
            assert "credit_score" in data
            assert "risk_tier" in data
            assert "recommendation" in data
            assert "top_risk_factors" in data
            assert "model_version" in data

    def test_score_risk_tier_values(self):
        resp = client.post("/score", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert data["risk_tier"] in ("Low", "Medium", "High")

    def test_score_recommendation_values(self):
        resp = client.post("/score", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert data["recommendation"] in ("Approve", "Review", "Decline")

    def test_score_probability_range(self):
        resp = client.post("/score", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            prob = resp.json()["default_probability"]
            assert 0 <= prob <= 1

    def test_score_credit_score_range(self):
        resp = client.post("/score", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            score = resp.json()["credit_score"]
            assert 200 <= score <= 900  # generous bounds


# ── Input Validation (422 on malformed input) ────────────────

class TestInputValidation:
    def test_missing_required_field(self):
        """Omitting a required field should return 422."""
        bad = {"income": 100_000}  # missing age, loan_amount, annuity
        resp = client.post("/score", json=bad)
        assert resp.status_code == 422

    def test_age_below_minimum(self):
        payload = {**VALID_PAYLOAD, "age": 10}
        resp = client.post("/score", json=payload)
        assert resp.status_code == 422

    def test_age_above_maximum(self):
        payload = {**VALID_PAYLOAD, "age": 150}
        resp = client.post("/score", json=payload)
        assert resp.status_code == 422

    def test_negative_income(self):
        payload = {**VALID_PAYLOAD, "income": -5000}
        resp = client.post("/score", json=payload)
        assert resp.status_code == 422

    def test_empty_body(self):
        resp = client.post("/score", json={})
        assert resp.status_code == 422

    def test_wrong_type(self):
        payload = {**VALID_PAYLOAD, "age": "thirty-five"}
        resp = client.post("/score", json=payload)
        assert resp.status_code == 422


# ── Model Info Endpoint ──────────────────────────────────────

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self):
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_fields(self):
        resp = client.get("/model/info")
        data = resp.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "model_type" in data


# ── Batch Endpoint ───────────────────────────────────────────

class TestBatchEndpoint:
    def test_batch_returns_200_or_503(self):
        payload = {"applications": [VALID_PAYLOAD, VALID_PAYLOAD]}
        resp = client.post("/score/batch", json=payload)
        assert resp.status_code in (200, 503)

    def test_batch_returns_correct_count(self):
        payload = {"applications": [VALID_PAYLOAD] * 3}
        resp = client.post("/score/batch", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert data["total"] == 3
            assert len(data["results"]) == 3

    def test_batch_rejects_empty_list(self):
        payload = {"applications": []}
        resp = client.post("/score/batch", json=payload)
        assert resp.status_code == 422
