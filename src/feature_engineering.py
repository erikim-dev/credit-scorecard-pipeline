"""
Feature Engineering Pipeline — DuckDB-powered SQL aggregations
joined to the main application table.

Usage:
    python src/feature_engineering.py --input data/raw --output data/processed
"""

import argparse
import pathlib

import duckdb
import numpy as np
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

    # ── Missing-value indicators ─────────────────────────────
    merged = add_missing_indicators(merged)

    # ── Frequency encoding for categoricals ──────────────────
    merged = add_frequency_encoding(merged)

    # ── Cross-table risk aggregates ──────────────────────────
    merged = add_cross_table_features(merged)

    # NOTE: Target encoding removed from batch pipeline to prevent
    # data leakage. Tree models (XGB/LGB) handle categoricals via
    # category codes; the scorecard uses per-fold WoE encoding.
    # If target encoding is needed, use fold-aware encoding inside
    # the training loop (see train.py).

    # Drop raw object columns — tree models use numeric features only
    cat_cols = merged.select_dtypes(include=["object", "category"]).columns.tolist()
    # Also catch extension dtypes that cannot be used directly by tree models
    for col in merged.columns:
        dtype_str = str(merged[col].dtype).lower()
        if dtype_str in ("object", "category", "boolean", "string"):
            if col not in cat_cols:
                cat_cols.append(col)
        elif hasattr(merged[col].dtype, "kind") and merged[col].dtype.kind in ("O", "U", "S"):
            if col not in cat_cols:
                cat_cols.append(col)
    # Explicit safety: check every column for non-numeric content
    for col in list(merged.columns):
        if col in ("SK_ID_CURR", "TARGET"):
            continue
        if col not in cat_cols:
            try:
                pd.to_numeric(merged[col].dropna(), errors="raise")
            except (ValueError, TypeError):
                cat_cols.append(col)
    print(f"  Dropping {len(cat_cols)} categorical/boolean columns")
    # Convert boolean extension type columns to int before drop (keep as 0/1 features)
    bool_cols = [c for c in cat_cols if str(merged[c].dtype).lower() == "boolean"]
    for col in bool_cols:
        merged[col] = merged[col].astype("Int64").astype("float64")
        cat_cols.remove(col)
        print(f"    Converted {col} (boolean -> float)")
    merged = merged.drop(columns=cat_cols)

    # Final safety: force all columns to numeric, drop any stragglers
    for col in merged.columns:
        if col in ("SK_ID_CURR", "TARGET"):
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # ── Save ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows x {len(merged.columns)} cols -> {out_path}")

    con.close()
    return merged


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary flags for key missing columns.

    Missingness is often informative in credit data -- for example,
    no bureau history means the applicant has no credit track record.
    Also impute EXT_SOURCE columns with median (they are the strongest predictors).
    """
    # Key columns where missingness is a risk signal
    indicator_cols = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "AMT_GOODS_PRICE", "AMT_ANNUITY",
        "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
    ]
    # Add aggregation-level missingness indicators
    agg_prefixes = ["bureau_", "bb_", "cc_", "inst_", "pos_", "prev_"]
    for prefix in agg_prefixes:
        sample_col = next((c for c in df.columns if c.startswith(prefix)), None)
        if sample_col:
            indicator_cols.append(sample_col)

    for col in indicator_cols:
        if col in df.columns:
            flag_name = f"MISS_{col}"
            df[flag_name] = df[col].isnull().astype(int)

    # Impute EXT_SOURCE with median (they drive ~22% of predictive power)
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    return df


def target_encode_categoricals(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    min_samples: int = 100,
    smoothing: float = 10.0,
) -> pd.DataFrame:
    """
    Smoothed target encoding for categorical columns.

    Uses Bayesian smoothing: encoded = (count * mean + smoothing * global_mean) / (count + smoothing)
    This avoids overfitting on rare categories while preserving the risk signal.
    """
    global_mean = df[target_col].mean()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        stats = df.groupby(col, observed=True)[target_col].agg(["mean", "count"])
        # Bayesian smoothing
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        encoded_name = f"TE_{col}"
        df[encoded_name] = df[col].map(smooth).astype(float)
        # Fill missing with global mean
        df[encoded_name] = df[encoded_name].fillna(global_mean)

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven ratios, interactions, and flags."""

    # ── Basic ratios ─────────────────────────────────────────
    df["LOAN_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    # Credit term in months (higher = longer exposure)
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    # Goods-to-income ratio
    df["GOODS_INCOME_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_INCOME_TOTAL"] + 1)
    # Income per family member
    cnt_fam = df.get("CNT_FAM_MEMBERS", df.get("CNT_CHILDREN", pd.Series(0, index=df.index)) + 1)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (cnt_fam.clip(lower=1))

    # ── Time-based features ──────────────────────────────────
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"]).clip(lower=0) / 365.25
    df["REGISTRATION_YEARS"] = (-df["DAYS_REGISTRATION"]) / 365.25
    df["ID_PUBLISH_YEARS"] = (-df["DAYS_ID_PUBLISH"]) / 365.25

    # Employment-to-age ratio (work stability)
    df["EMPLOY_TO_AGE_RATIO"] = df["EMPLOYMENT_YEARS"] / (df["AGE_YEARS"] + 0.01)
    # Years since last phone change
    if "DAYS_LAST_PHONE_CHANGE" in df.columns:
        df["PHONE_CHANGE_YEARS"] = (-df["DAYS_LAST_PHONE_CHANGE"]) / 365.25

    # ── External source features (strongest predictors) ──────
    ext_cols = [c for c in df.columns if c.startswith("EXT_SOURCE_") and c[-1].isdigit()]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
        df["EXT_SOURCE_RANGE"] = df["EXT_SOURCE_MAX"] - df["EXT_SOURCE_MIN"]
        # Product interactions (capture non-linear risk signal)
        if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_2" in df.columns:
            df["EXT_SRC_1x2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
        if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
            df["EXT_SRC_2x3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_3" in df.columns:
            df["EXT_SRC_1x3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
        # EXT_SOURCE weighted by age (risk evolves over lifetime)
        if "EXT_SOURCE_2" in df.columns:
            df["EXT_SRC2_x_AGE"] = df["EXT_SOURCE_2"] * df["AGE_YEARS"]
        if "EXT_SOURCE_3" in df.columns:
            df["EXT_SRC3_x_AGE"] = df["EXT_SOURCE_3"] * df["AGE_YEARS"]

        # Polynomial features (capture non-linear risk surface)
        if "EXT_SOURCE_2" in df.columns:
            df["EXT_SRC2_SQ"] = df["EXT_SOURCE_2"] ** 2
            df["EXT_SRC2_CB"] = df["EXT_SOURCE_2"] ** 3
        if "EXT_SOURCE_3" in df.columns:
            df["EXT_SRC3_SQ"] = df["EXT_SOURCE_3"] ** 2
        if "EXT_SOURCE_1" in df.columns:
            df["EXT_SRC1_SQ"] = df["EXT_SOURCE_1"] ** 2
        # Triple interaction
        if all(f"EXT_SOURCE_{i}" in df.columns for i in [1, 2, 3]):
            df["EXT_SRC_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        # Weighted EXT score (EXT2 is strongest predictor)
        if all(f"EXT_SOURCE_{i}" in df.columns for i in [1, 2, 3]):
            df["EXT_SOURCE_WEIGHTED"] = (
                0.2 * df["EXT_SOURCE_1"]
                + 0.5 * df["EXT_SOURCE_2"]
                + 0.3 * df["EXT_SOURCE_3"]
            )

    # ── Document count (proxy for documentation completeness) ─
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)

    # ── Address mismatch flags ───────────────────────────────
    reg_cols = [c for c in df.columns if c.startswith("REG_") and "NOT" in c]
    if reg_cols:
        df["ADDR_MISMATCH_COUNT"] = df[reg_cols].sum(axis=1)

    # ── Bureau enquiry intensity ─────────────────────────────
    bureau_req_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    if bureau_req_cols:
        df["BUREAU_ENQUIRY_TOTAL"] = df[bureau_req_cols].sum(axis=1)

    # ── Annuity / credit capacity ────────────────────────────
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    # How much of credit goes beyond goods price (fees/interest proxy)
    df["CREDIT_OVERCHARGE"] = (df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]) / (df["AMT_GOODS_PRICE"] + 1)

    # ── Income-loan cross features ───────────────────────────
    df["ANNUITY_x_EMPLOYMENT"] = df["AMT_ANNUITY"] * df.get("EMPLOYMENT_YEARS", 0)
    df["INCOME_x_EXT2"] = df["AMT_INCOME_TOTAL"] * df.get("EXT_SOURCE_2", 0.5)
    df["CREDIT_x_AGE"] = df["AMT_CREDIT"] * df.get("AGE_YEARS", 40)

    return df


def add_frequency_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency-encode categorical columns before dropping them."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        df[f"FREQ_{col}"] = df[col].map(freq).astype(float).fillna(0)
    return df


def add_cross_table_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-table risk aggregates combining bureau, CC, installment signals."""
    # Worst DPD across all sub-tables
    dpd_cols = [c for c in df.columns if "dpd" in c.lower() and "max" in c.lower()]
    if dpd_cols:
        df["WORST_DPD_ALL"] = df[dpd_cols].max(axis=1)

    # Total DPD months across all sub-tables
    dpd_month_cols = [c for c in df.columns if "months_with_dpd" in c.lower() or "dpd_count" in c.lower()]
    if dpd_month_cols:
        df["TOTAL_DPD_MONTHS"] = df[dpd_month_cols].sum(axis=1)

    # Bureau risk composite
    if all(c in df.columns for c in ["debt_to_credit_ratio", "overdue_count", "bureau_loan_count"]):
        df["BUREAU_RISK_SCORE"] = (
            df["debt_to_credit_ratio"].fillna(0) * 0.5
            + (df["overdue_count"] / df["bureau_loan_count"].clip(lower=1)) * 0.3
            + (1 - df.get("bb_current_ratio", pd.Series(0.5, index=df.index)).fillna(0.5)) * 0.2
        )

    # Payment discipline composite
    if "late_payment_ratio" in df.columns and "payment_to_due_ratio" in df.columns:
        df["PAYMENT_DISCIPLINE"] = (
            (1 - df["late_payment_ratio"].fillna(0)) * 0.5
            + df["payment_to_due_ratio"].fillna(1).clip(upper=1.5) / 1.5 * 0.5
        )

    # Total credit exposure ratio
    if "total_credit_amount" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["TOTAL_EXPOSURE_RATIO"] = (
            (df["total_credit_amount"].fillna(0) + df["AMT_CREDIT"])
            / (df["AMT_INCOME_TOTAL"] + 1)
        )

    return df


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature table")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()

    build_features(pathlib.Path(args.input), pathlib.Path(args.output))
