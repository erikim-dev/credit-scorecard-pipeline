"""
Evaluation utilities -- credit-risk-specific metrics.

Includes:
    - AUC, Gini, KS
    - PSI (Population Stability Index)
    - Decile analysis with gains and lift tables
    - Expected Loss (EL = PD x LGD x EAD)
    - VIF (Variance Inflation Factor)
    - Segmented performance
    - ROC comparison plots

Usage:
    python src/evaluate.py --models models/ --data data/processed
"""

import argparse
import pathlib
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)


# ── Core Metrics ──────────────────────────────────────────────

def compute_gini(y_true, y_prob) -> float:
    """Gini = 2 × AUC − 1.  Industry-standard risk separation metric."""
    return 2 * roc_auc_score(y_true, y_prob) - 1


def compute_ks_statistic(y_true, y_prob) -> float:
    """
    Kolmogorov–Smirnov statistic: max separation between
    cumulative distributions of goods and bads.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_psi(expected, actual, bins: int = 10) -> dict:
    """
    Population Stability Index.

    PSI < 0.10 → Stable
    PSI 0.10–0.25 → Moderate shift, investigate
    PSI > 0.25 → Major shift, model review required
    """
    breakpoints = np.linspace(0, 1, bins + 1)
    exp_perc = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_perc = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    exp_perc = np.clip(exp_perc, 1e-4, None)
    act_perc = np.clip(act_perc, 1e-4, None)

    psi = float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))
    status = (
        "Stable" if psi < 0.10
        else "Monitor" if psi < 0.25
        else "ALERT: Model Review Required"
    )
    return {"PSI": round(psi, 4), "Status": status}


# -- Decile Analysis / Gains / Lift ----------------------------

def decile_analysis(y_true, y_prob) -> pd.DataFrame:
    """
    Build a 10-decile risk ranking table.

    Returns a DataFrame with columns:
        Decile, Count, Bads, Goods, Bad_Rate, Cumul_Bad_Rate,
        Cumul_Bad_Pct (capture), Lift
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], 10, labels=False, duplicates="drop")
    df["decile"] = 10 - df["decile"]  # 1 = highest risk

    grouped = df.groupby("decile").agg(
        count=("y_true", "size"),
        bads=("y_true", "sum"),
    ).sort_index()

    grouped["goods"] = grouped["count"] - grouped["bads"]
    grouped["bad_rate"] = grouped["bads"] / grouped["count"]
    grouped["cumul_bads"] = grouped["bads"].cumsum()
    grouped["cumul_bad_pct"] = grouped["cumul_bads"] / grouped["bads"].sum()
    overall_bad_rate = df["y_true"].mean()
    grouped["lift"] = grouped["bad_rate"] / overall_bad_rate

    grouped = grouped.reset_index()
    grouped.columns = [
        "Decile", "Count", "Bads", "Goods", "Bad_Rate",
        "Cumul_Bads", "Cumul_Bad_Pct", "Lift",
    ]
    return grouped


def plot_gains_chart(decile_df: pd.DataFrame, save_path: str | None = None):
    """Cumulative gains curve from decile analysis."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(decile_df) + 1) * 10
    ax.plot(x, decile_df["Cumul_Bad_Pct"].values * 100,
            marker="o", linewidth=2, label="Model")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("Percentage of Population (ranked by risk)")
    ax.set_ylabel("Cumulative % of Defaults Captured")
    ax.set_title("Cumulative Gains Chart")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_lift_chart(decile_df: pd.DataFrame, save_path: str | None = None):
    """Lift chart from decile analysis."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(decile_df["Decile"], decile_df["Lift"], color="steelblue", edgecolor="white")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.6, label="Baseline (lift=1)")
    ax.set_xlabel("Decile (1 = highest risk)")
    ax.set_ylabel("Lift over Random")
    ax.set_title("Lift Chart by Decile")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


# -- Expected Loss (Basel) -------------------------------------

def compute_expected_loss(
    y_prob: np.ndarray,
    ead: np.ndarray | float = 1.0,
    lgd: float = 0.45,
) -> dict:
    """
    Expected Loss = PD x LGD x EAD.

    Parameters
    ----------
    y_prob : array
        Predicted probability of default for each observation.
    ead : array or float
        Exposure at default (use loan amounts if available).
    lgd : float
        Loss given default (Basel IRB default = 0.45 for unsecured).

    Returns
    -------
    dict with total_el, mean_pd, mean_el_per_loan, el_rate
    """
    pd_arr = np.asarray(y_prob)
    ead_arr = np.broadcast_to(np.asarray(ead, dtype=float), pd_arr.shape)

    el_per_loan = pd_arr * lgd * ead_arr
    total_el = float(np.sum(el_per_loan))
    total_ead = float(np.sum(ead_arr))

    return {
        "total_expected_loss": round(total_el, 2),
        "total_exposure": round(total_ead, 2),
        "mean_pd": round(float(np.mean(pd_arr)), 4),
        "lgd_assumption": lgd,
        "el_rate": round(total_el / total_ead if total_ead > 0 else 0, 6),
        "mean_el_per_loan": round(float(np.mean(el_per_loan)), 2),
    }


# -- VIF (Variance Inflation Factor) --------------------------

def compute_vif(X: pd.DataFrame, max_features: int = 50) -> pd.DataFrame:
    """
    Compute VIF for numeric features.  VIF > 5 suggests multicollinearity,
    VIF > 10 is severe.  Uses only a sample of features if the matrix is wide.

    Returns a DataFrame with columns: Feature, VIF  (sorted descending).
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    numeric_df = X.select_dtypes(include="number").dropna(axis=1)
    if numeric_df.shape[1] > max_features:
        # Pick top-variance features to keep runtime reasonable
        variances = numeric_df.var().nlargest(max_features)
        numeric_df = numeric_df[variances.index]

    # Add constant column for intercept
    numeric_df = numeric_df.copy()
    numeric_df.insert(0, "_const", 1.0)

    vif_data = []
    for i in range(1, numeric_df.shape[1]):  # skip constant
        vif_val = variance_inflation_factor(numeric_df.values, i)
        vif_data.append({
            "Feature": numeric_df.columns[i],
            "VIF": round(vif_val, 2),
        })

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df


# -- Segmented Performance ────────────────────────────────────

def segment_performance(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    pred_col: str = "PRED_PROB",
    segment_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute AUC and Gini for each segment."""
    if segment_cols is None:
        segment_cols = ["AGE_BAND", "INCOME_BAND"]

    records = []
    for seg in segment_cols:
        if seg not in df.columns:
            continue
        for val, grp in df.groupby(seg):
            if grp[target_col].nunique() < 2:
                continue
            auc = roc_auc_score(grp[target_col], grp[pred_col])
            records.append({
                "Segment": seg,
                "Value": val,
                "N": len(grp),
                "Default_Rate": grp[target_col].mean(),
                "AUC": round(auc, 4),
                "Gini": round(2 * auc - 1, 4),
            })
    return pd.DataFrame(records)


# ── Plotting ─────────────────────────────────────────────────

def plot_roc_comparison(models: dict[str, tuple], save_path: str | None = None):
    """
    Plot overlaid ROC curves for champion / challenger.

    Parameters
    ----------
    models : dict
        {"model_name": (y_true, y_prob), ...}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_prob) in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Champion vs Challenger — ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved ROC plot -> {save_path}")
    plt.close(fig)


def plot_score_distribution(
    scores: pd.Series, default: pd.Series, save_path: str | None = None
):
    """Plot score distributions for goods vs bads."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[default == 0], bins=50, alpha=0.6, label="Good (non-default)", density=True)
    ax.hist(scores[default == 1], bins=50, alpha=0.6, label="Bad (default)", density=True)
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution — Goods vs Bads")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────

def main(models_dir: pathlib.Path, data_dir: pathlib.Path):
    reports_dir = pathlib.Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "train_features.csv")

    # Use a proper hold-out split (last 20% by row order, reproducible)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Hold-out test set: {len(df_test):,} rows  |  Default rate: {df_test['TARGET'].mean():.2%}\n")

    # Load WoE encoder and scaler if available (needed for scorecard)
    woe_encoder = None
    scaler = None
    woe_path = models_dir / "woe_encoder.pkl"
    scaler_path = models_dir / "scaler.pkl"
    selected_path = models_dir / "selected_features.json"

    if woe_path.exists():
        woe_encoder = joblib.load(woe_path)
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    selected_features = None
    if selected_path.exists():
        import json as _json
        with open(selected_path) as f:
            selected_features = _json.load(f)

    results = {}
    for pkl in sorted(models_dir.glob("*.pkl")):
        if "encoder" in pkl.stem or "scaler" in pkl.stem:
            continue
        model = joblib.load(pkl)
        name = pkl.stem

        feature_cols = [c for c in df_test.columns if c not in ("TARGET", "SK_ID_CURR", "SK_ID_PREV")]
        X = df_test[feature_cols].copy()

        if "scorecard" in name and woe_encoder is not None:
            numeric_cols = X.select_dtypes(include="number").columns.tolist()
            if selected_features:
                numeric_cols = [c for c in selected_features if c in numeric_cols]
            X = woe_encoder.transform(X[numeric_cols]).fillna(0)
            if scaler is not None:
                X = pd.DataFrame(scaler.transform(X), columns=X.columns)
        else:
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            for c in cat_cols:
                X[c] = X[c].astype("category").cat.codes

        y_prob = model.predict_proba(X)[:, 1]
        y_true = df_test["TARGET"]

        auc = roc_auc_score(y_true, y_prob)
        gini = compute_gini(y_true, y_prob)
        ks = compute_ks_statistic(y_true, y_prob)

        results[name] = {
            "AUC": round(auc, 4),
            "Gini": round(gini, 4),
            "KS": round(ks, 4),
        }
        print(f"{name}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

    # Save comparison
    with open(reports_dir / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved -> {reports_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="models")
    parser.add_argument("--data", type=str, default="data/processed")
    args = parser.parse_args()
    main(pathlib.Path(args.models), pathlib.Path(args.data))
