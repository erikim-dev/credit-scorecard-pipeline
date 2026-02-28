"""
Data Drift Report -- Evidently-powered monitoring.

Generates an HTML report comparing a reference dataset (training)
against a current dataset to detect feature-level and target drift.

Usage:
    python monitoring/evidently_drift.py \
        --reference data/processed/train_features.csv \
        --current   data/processed/latest_features.csv \
        --output    reports/drift_report.html
"""

import argparse
import pathlib

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: pathlib.Path,
    sample_size: int = 10_000,
) -> pathlib.Path:
    """
    Build and save an Evidently drift + data quality report.

    Parameters
    ----------
    reference : DataFrame
        Baseline dataset (usually the training set).
    current : DataFrame
        New production data or a recent scoring batch.
    output_path : Path
        Where to save the HTML report.
    sample_size : int
        Cap each dataset to this many rows for speed.

    Returns
    -------
    Path to the saved HTML file.
    """
    # Sample for speed
    if len(reference) > sample_size:
        reference = reference.sample(sample_size, random_state=42)
    if len(current) > sample_size:
        current = current.sample(sample_size, random_state=42)

    # Keep only shared numeric columns
    shared = sorted(set(reference.columns) & set(current.columns))
    num_cols = reference[shared].select_dtypes(include="number").columns.tolist()

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(
        reference_data=reference[num_cols],
        current_data=current[num_cols],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    print(f"Drift report saved -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Evidently drift report")
    parser.add_argument("--reference", type=str, default="data/processed/train_features.csv")
    parser.add_argument("--current", type=str, default="data/processed/train_features.csv",
                        help="Path to current/production data (defaults to train for demo)")
    parser.add_argument("--output", type=str, default="reports/drift_report.html")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference)
    cur_df = pd.read_csv(args.current)

    generate_drift_report(ref_df, cur_df, pathlib.Path(args.output))


if __name__ == "__main__":
    main()
