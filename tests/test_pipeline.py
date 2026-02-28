"""
Tests for the credit risk pipeline.

Run:  pytest tests/ -v --cov=src
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from woe_encoder import WoEEncoder
from evaluate import (
    compute_gini, compute_ks_statistic, compute_psi,
    decile_analysis, compute_expected_loss, compute_vif,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Synthetic binary-classification data."""
    np.random.seed(42)
    n = 1_000
    X = pd.DataFrame({
        "income": np.random.lognormal(10, 1, n),
        "age": np.random.normal(40, 10, n).clip(18, 80),
        "loan_amount": np.random.lognormal(11, 0.5, n),
        "bureau_count": np.random.poisson(3, n),
    })
    logits = (
        -2
        + 0.5 * (X["loan_amount"] / X["income"])
        - 0.02 * X["age"]
        + 0.3 * X["bureau_count"]
    )
    prob = 1 / (1 + np.exp(-logits))
    y = pd.Series((np.random.rand(n) < prob).astype(int), name="TARGET")
    return X, y


# ── WoE Encoder Tests ────────────────────────────────────────

class TestWoEEncoder:
    def test_fit_transform_shape(self, sample_data):
        X, y = sample_data
        enc = WoEEncoder(bins=5)
        X_woe = enc.fit_transform(X, y)
        assert X_woe.shape == X.shape

    def test_iv_summary_has_all_features(self, sample_data):
        X, y = sample_data
        enc = WoEEncoder(bins=5).fit(X, y)
        iv = enc.get_iv_summary()
        assert set(iv.index) == set(X.columns)

    def test_iv_values_non_negative(self, sample_data):
        X, y = sample_data
        enc = WoEEncoder(bins=5).fit(X, y)
        iv = enc.get_iv_summary()
        assert (iv["IV"] >= 0).all()

    def test_score_conversion_monotonic(self):
        """Higher log-odds (better) → higher score."""
        scores = [WoEEncoder.log_odds_to_score(lo) for lo in [-2, -1, 0, 1, 2]]
        assert scores == sorted(scores)


# ── Metric Tests ─────────────────────────────────────────────

class TestMetrics:
    def test_gini_range(self, sample_data):
        X, y = sample_data
        probs = np.random.rand(len(y))
        gini = compute_gini(y, probs)
        assert -1 <= gini <= 1

    def test_ks_range(self, sample_data):
        X, y = sample_data
        probs = np.random.rand(len(y))
        ks = compute_ks_statistic(y, probs)
        assert 0 <= ks <= 1

    def test_perfect_model_gini(self):
        y = pd.Series([0, 0, 0, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        gini = compute_gini(y, probs)
        assert gini > 0.9

    def test_psi_identical_distributions(self):
        data = np.random.normal(0, 1, 5000)
        result = compute_psi(data, data)
        assert result["PSI"] < 0.01
        assert result["Status"] == "Stable"

    def test_psi_shifted_distribution(self):
        base = np.random.normal(0, 1, 5000)
        shifted = np.random.normal(2, 1, 5000)  # big shift
        result = compute_psi(base, shifted)
        assert result["PSI"] > 0.25
        assert "ALERT" in result["Status"]


# -- Decile / Expected Loss / VIF Tests ----------------------------

class TestDecileAnalysis:
    def test_decile_table_shape(self, sample_data):
        X, y = sample_data
        probs = np.random.rand(len(y))
        dec = decile_analysis(y.values, probs)
        assert len(dec) <= 10
        assert "Bad_Rate" in dec.columns
        assert "Lift" in dec.columns
        assert "Cumul_Bad_Pct" in dec.columns

    def test_cumulative_capture_reaches_one(self, sample_data):
        X, y = sample_data
        probs = np.random.rand(len(y))
        dec = decile_analysis(y.values, probs)
        assert abs(dec["Cumul_Bad_Pct"].iloc[-1] - 1.0) < 0.01


class TestExpectedLoss:
    def test_zero_pd_zero_loss(self):
        probs = np.zeros(100)
        result = compute_expected_loss(probs, ead=10000, lgd=0.45)
        assert result["total_expected_loss"] == 0

    def test_el_scales_with_lgd(self):
        probs = np.array([0.1, 0.2, 0.3])
        el_low = compute_expected_loss(probs, ead=1.0, lgd=0.20)
        el_high = compute_expected_loss(probs, ead=1.0, lgd=0.60)
        assert el_high["total_expected_loss"] > el_low["total_expected_loss"]

    def test_el_rate_between_0_and_1(self):
        probs = np.random.rand(500)
        result = compute_expected_loss(probs, ead=1.0, lgd=0.45)
        assert 0 <= result["el_rate"] <= 1


class TestVIF:
    def test_vif_returns_dataframe(self, sample_data):
        X, y = sample_data
        vif_df = compute_vif(X)
        assert isinstance(vif_df, pd.DataFrame)
        assert "Feature" in vif_df.columns
        assert "VIF" in vif_df.columns
        assert len(vif_df) == X.shape[1]

    def test_vif_values_positive(self, sample_data):
        X, y = sample_data
        vif_df = compute_vif(X)
        assert (vif_df["VIF"] >= 1.0).all()


class TestScorecardPoints:
    def test_points_table_generation(self, sample_data):
        X, y = sample_data
        enc = WoEEncoder(bins=5).fit(X, y)
        # Fake model coefficients
        coef = np.array([0.5, -0.3, 0.2, 0.1])
        intercept = -1.5
        features = list(X.columns)
        pts = enc.scorecard_points(coef, intercept, features)
        assert "Points" in pts.columns
        assert "Feature" in pts.columns
        assert len(pts) > 0
