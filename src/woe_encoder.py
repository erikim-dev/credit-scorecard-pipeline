"""
WoE (Weight of Evidence) Encoder for Credit Risk Scorecard Development.

Transforms continuous and categorical features into WoE values
and computes Information Value (IV) for variable selection.

Industry-standard approach used in credit risk modelling under
Basel II/III regulatory frameworks.
"""

import pandas as pd
import numpy as np
from typing import Optional


class WoEEncoder:
    """
    Weight of Evidence encoder with Information Value computation.

    Parameters
    ----------
    bins : int
        Number of quantile bins for continuous variables (default: 10).
    min_bin_size : float
        Minimum fraction of total population per bin (default: 0.05).
    regularization : float
        Laplace smoothing to avoid log(0) (default: 0.5).
    """

    def __init__(
        self,
        bins: int = 10,
        min_bin_size: float = 0.05,
        regularization: float = 0.5,
    ):
        self.bins = bins
        self.min_bin_size = min_bin_size
        self.regularization = regularization
        self.woe_dict: dict = {}
        self.iv_dict: dict = {}
        self.bin_edges: dict = {}

    # ── Fit ───────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        """Compute WoE mapping and IV for every column in X."""
        for col in X.columns:
            if X[col].dtype in ("object", "category"):
                self._fit_categorical(X[col], y, col)
            else:
                self._fit_numeric(X[col], y, col)
        return self

    def _fit_numeric(self, x: pd.Series, y: pd.Series, col: str):
        df = pd.DataFrame({"x": x, "y": y}).dropna(subset=["x"])

        # Quantile binning (merge small buckets automatically)
        df["bin"], edges = pd.qcut(
            df["x"], q=self.bins, retbins=True, duplicates="drop"
        )
        self.bin_edges[col] = edges
        self._compute_woe(df, col)

    def _fit_categorical(self, x: pd.Series, y: pd.Series, col: str):
        df = pd.DataFrame({"x": x, "y": y}).dropna(subset=["x"])
        df["bin"] = df["x"]
        self._compute_woe(df, col)

    def _compute_woe(self, df: pd.DataFrame, col: str):
        stats = df.groupby("bin")["y"].agg(events="sum", total="count")
        stats["non_events"] = stats["total"] - stats["events"]

        total_events = stats["events"].sum()
        total_non_events = stats["non_events"].sum()

        # Laplace smoothing
        r = self.regularization
        stats["event_rate"] = (stats["events"] + r) / (total_events + 2 * r)
        stats["non_event_rate"] = (stats["non_events"] + r) / (
            total_non_events + 2 * r
        )

        stats["woe"] = np.log(stats["non_event_rate"] / stats["event_rate"])
        stats["iv_component"] = (
            stats["non_event_rate"] - stats["event_rate"]
        ) * stats["woe"]

        self.woe_dict[col] = stats["woe"].to_dict()
        self.iv_dict[col] = stats["iv_component"].sum()

    # ── Transform ─────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace feature values with their WoE equivalents."""
        X_woe = X.copy()
        for col in self.woe_dict:
            if col not in X_woe.columns:
                continue
            if col in self.bin_edges:
                bins = pd.cut(X_woe[col], bins=self.bin_edges[col], include_lowest=True)
                X_woe[col] = bins.map(self.woe_dict[col]).astype(float)
            else:
                X_woe[col] = X_woe[col].map(self.woe_dict[col]).astype(float)
        return X_woe

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ── IV Summary ────────────────────────────────────────────
    def get_iv_summary(self) -> pd.DataFrame:
        """
        Return a ranked table of Information Value per feature.

        IV interpretation (industry standard):
            < 0.02  → Useless
            0.02–0.1 → Weak
            0.1–0.3  → Medium
            0.3–0.5  → Strong
            > 0.5    → Suspicious (possible overfit / info leakage)
        """
        iv_df = (
            pd.DataFrame.from_dict(self.iv_dict, orient="index", columns=["IV"])
            .sort_values("IV", ascending=False)
        )
        iv_df["Predictive_Power"] = iv_df["IV"].apply(
            lambda x: (
                "Useless" if x < 0.02
                else "Weak" if x < 0.1
                else "Medium" if x < 0.3
                else "Strong" if x < 0.5
                else "Suspicious"
            )
        )
        return iv_df

    # -- Scorecard Points Table ------------------------------------
    def scorecard_points(
        self,
        model_coef: np.ndarray,
        model_intercept: float,
        feature_names: list[str],
        pdo: float = 20,
        base_score: float = 600,
        base_odds: float = 1 / 19,
    ) -> pd.DataFrame:
        """
        Build a per-bin scorecard points table.

        Each row shows: Feature, Bin, WoE, Coefficient, Points.
        The sum of selected bin-points + base points = final credit score.

        Parameters
        ----------
        model_coef : array
            Logistic regression coefficients (shape = n_features).
        model_intercept : float
            Logistic regression intercept.
        feature_names : list
            Feature names corresponding to model_coef.
        pdo, base_score, base_odds :
            Scorecard calibration parameters.

        Returns
        -------
        pd.DataFrame with columns [Feature, Bin, WoE, Coefficient, Points]
        """
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        n = len(feature_names)

        # Base points from intercept, distributed evenly
        base_points_per_feat = (offset + factor * model_intercept) / n

        records = []
        for i, feat in enumerate(feature_names):
            coef = model_coef[i]
            if feat not in self.woe_dict:
                continue
            for bin_label, woe_val in self.woe_dict[feat].items():
                points = round(base_points_per_feat + factor * coef * woe_val, 1)
                records.append({
                    "Feature": feat,
                    "Bin": str(bin_label),
                    "WoE": round(woe_val, 4),
                    "Coefficient": round(coef, 4),
                    "Points": points,
                })

        return pd.DataFrame(records)

    # -- Scorecard conversion helpers ------------------------------
    @staticmethod
    def log_odds_to_score(
        log_odds: float,
        pdo: float = 20,
        base_score: float = 600,
        base_odds: float = 1 / 19,
    ) -> int:
        """
        Convert log-odds to a points-based credit score.

        Parameters
        ----------
        log_odds : float
            Model output in log-odds (logit) space.
        pdo : float
            Points to Double the Odds (industry default: 20).
        base_score : float
            Score at the base odds (industry default: 600).
        base_odds : float
            Assumed base odds of default (e.g. 1:19 → 5%).

        Returns
        -------
        int
            Integer credit score.
        """
        factor = pdo / np.log(2)
        offset = base_score - (factor * np.log(base_odds))
        return int(round(offset + factor * log_odds))
