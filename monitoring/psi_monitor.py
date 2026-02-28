"""
PSI Monitoring — Population Stability Index tracker.

Detects score / feature distribution drift between a baseline
(training) population and the current (production) population.

Usage:
    python monitoring/psi_monitor.py \
        --baseline data/processed/train_features.csv \
        --current  data/processed/latest.csv
"""

import argparse
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> dict:
    """
    Population Stability Index.

    PSI < 0.10  → No significant change
    PSI 0.10–0.25 → Moderate change — investigate
    PSI > 0.25  → Major shift — model review required
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_perc = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_perc = np.clip(expected_perc, 1e-4, None)
    actual_perc = np.clip(actual_perc, 1e-4, None)

    psi = float(np.sum(
        (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
    ))
    status = (
        "Stable" if psi < 0.10
        else "Monitor" if psi < 0.25
        else "ALERT: Model Review Required"
    )
    return {"PSI": round(psi, 4), "Status": status}


def monitor_features(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Compute PSI for every numeric feature."""
    if feature_cols is None:
        feature_cols = baseline_df.select_dtypes(include="number").columns.tolist()

    records = []
    for col in feature_cols:
        if col not in current_df.columns:
            continue
        base = baseline_df[col].dropna().values
        curr = current_df[col].dropna().values
        if len(base) < 50 or len(curr) < 50:
            continue
        result = calculate_psi(base, curr, bins=bins)
        result["Feature"] = col
        records.append(result)

    return (
        pd.DataFrame(records)
        .sort_values("PSI", ascending=False)
        .reset_index(drop=True)
    )


def generate_report(psi_df: pd.DataFrame) -> str:
    """Human-readable monitoring report."""
    lines = [
        "=" * 60,
        f"  PSI MONITORING REPORT — {datetime.now():%Y-%m-%d %H:%M}",
        "=" * 60,
        "",
    ]

    alerts = psi_df[psi_df["Status"].str.contains("ALERT")]
    monitors = psi_df[psi_df["Status"] == "Monitor"]
    stable = psi_df[psi_df["Status"] == "Stable"]

    lines.append(f"  Total features checked: {len(psi_df)}")
    lines.append(f"  🟢 Stable:     {len(stable)}")
    lines.append(f"  🟡 Monitor:    {len(monitors)}")
    lines.append(f"  🔴 Alert:      {len(alerts)}")
    lines.append("")

    if len(alerts) > 0:
        lines.append("  ─── ALERTS (PSI > 0.25) ───")
        for _, row in alerts.iterrows():
            lines.append(f"    {row['Feature']:40s}  PSI = {row['PSI']:.4f}")
        lines.append("")

    if len(monitors) > 0:
        lines.append("  ─── MONITOR (0.10 < PSI ≤ 0.25) ───")
        for _, row in monitors.iterrows():
            lines.append(f"    {row['Feature']:40s}  PSI = {row['PSI']:.4f}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSI drift monitor")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--current", type=str, required=True)
    parser.add_argument("--output", type=str, default="reports/psi_report.csv")
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline)
    current = pd.read_csv(args.current)

    psi_df = monitor_features(baseline, current)
    report = generate_report(psi_df)
    print(report)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psi_df.to_csv(out_path, index=False)
    print(f"\n✓ PSI results saved → {out_path}")
