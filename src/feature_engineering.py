"""
Feature Engineering Pipeline — DuckDB-powered SQL aggregations
joined to the main application table.

Usage:
    python src/feature_engineering.py --input data/raw --output data/processed
"""

import argparse
import pathlib

import duckdb
import pandas as pd


SQL_DIR = pathlib.Path(__file__).resolve().parent.parent / "sql"


def run_sql_file(con: duckdb.DuckDBPyConnection, path: pathlib.Path) -> pd.DataFrame:
    """Execute a .sql file and return the result as a DataFrame."""
    sql = path.read_text(encoding="utf-8")
    return con.execute(sql).fetchdf()


def load_tables(con: duckdb.DuckDBPyConnection, raw_dir: pathlib.Path):
    """Register every CSV in the raw directory as a DuckDB table."""
    skip = {"HomeCredit_columns_description", "sample_submission"}
    for csv_path in raw_dir.glob("*.csv"):
        table_name = csv_path.stem  # e.g. "bureau" from "bureau.csv"
        if table_name in skip:
            continue
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT * FROM read_csv_auto('{csv_path}', header=true)"
        )
        print(f"  Loaded {table_name} ({con.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]:,} rows)")


def build_features(raw_dir: pathlib.Path, out_dir: pathlib.Path):
    """Main pipeline: load CSVs → run SQL aggregations → merge → save."""
    con = duckdb.connect()

    print("Loading raw CSVs into DuckDB …")
    load_tables(con, raw_dir)

    # ── Run aggregation queries ──────────────────────────────
    print("\nRunning SQL feature aggregations …")
    agg_frames: dict[str, pd.DataFrame] = {}
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        print(f"  {sql_file.name}")
        agg_frames[sql_file.stem] = run_sql_file(con, sql_file)

    # ── Load application_train as base ───────────────────────
    app_train = con.execute("SELECT * FROM application_train").fetchdf()
    print(f"\nBase table: application_train — {len(app_train):,} rows")

    # ── Left-join every aggregation ──────────────────────────
    merged = app_train
    for name, agg_df in agg_frames.items():
        if "SK_ID_CURR" in agg_df.columns:
            merged = merged.merge(agg_df, on="SK_ID_CURR", how="left")
            print(f"  Joined {name}: +{len(agg_df.columns) - 1} features")

    # ── Simple derived features ──────────────────────────────
    merged = add_derived_features(merged)

    # ── Save ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows x {len(merged.columns)} cols -> {out_path}")

    con.close()
    return merged


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven ratios and flags."""
    # Loan-to-income ratio
    df["LOAN_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Annuity-to-income ratio (monthly burden)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Credit-to-goods ratio (financing markup)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    # Age in years
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25

    # Employment years
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"]) / 365.25
    df["EMPLOYMENT_YEARS"] = df["EMPLOYMENT_YEARS"].clip(lower=0)

    # External source mean
    ext_cols = [c for c in df.columns if c.startswith("EXT_SOURCE")]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)

    return df


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature table")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()

    build_features(pathlib.Path(args.input), pathlib.Path(args.output))
