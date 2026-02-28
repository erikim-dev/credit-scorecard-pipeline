"""
Credit Risk Scoring -- Streamlit Application

Operates in two modes:
  - Standalone (default): loads models directly via joblib.
    This is the mode used on Streamlit Cloud.
  - API mode: forwards requests to the FastAPI backend.
    Activated when API_URL env var is set and non-empty.

Pages:
  1. Single Application Scoring
  2. Batch Scoring (CSV upload)
  3. Model Performance Dashboard
"""

import sys
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# -- Paths ----------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# -- Mode selection -------------------------------------------------
API_URL = os.getenv("API_URL", "").strip()
USE_API = bool(API_URL)

st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="CRS",
    layout="wide",
)


# -- Model Loading (standalone mode) --------------------------------
@st.cache_resource
def load_models():
    """Load serialized models + encoder + scaler once."""
    import joblib
    models_dir = ROOT / "models"
    artefacts = {}
    xgb_path = models_dir / "xgboost_challenger.pkl"
    sc_path = models_dir / "scorecard_champion.pkl"
    woe_path = models_dir / "woe_encoder.pkl"
    scaler_path = models_dir / "scaler.pkl"
    feats_path = models_dir / "selected_features.json"
    if xgb_path.exists():
        artefacts["xgb"] = joblib.load(xgb_path)
    if sc_path.exists():
        artefacts["scorecard"] = joblib.load(sc_path)
    if woe_path.exists():
        artefacts["woe_encoder"] = joblib.load(woe_path)
    if scaler_path.exists():
        artefacts["scaler"] = joblib.load(scaler_path)
    if feats_path.exists():
        with open(feats_path) as f:
            artefacts["selected_features"] = json.load(f)
    return artefacts


@st.cache_data
def load_test_data():
    """Load processed features for the dashboard."""
    csv_path = ROOT / "data" / "processed" / "train_features.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def build_feature_vector(data: dict, feature_cols: list) -> pd.DataFrame:
    """Convert application dict into a 1-row DataFrame aligned to model features."""
    row = {}

    age = abs(data.get("age", 35))
    income = data.get("income", 100000) or 1
    credit = data.get("loan_amount", 200000)
    annuity = data.get("annuity", 15000)
    goods = data.get("goods_price", 1) or 1
    emp_yrs = max(data.get("employment_years", 0), 0)
    ext1 = data.get("ext_source_1", 0.5)
    ext2 = data.get("ext_source_2", 0.5)
    ext3 = data.get("ext_source_3", 0.5)

    mapping = {
        "DAYS_BIRTH": -age * 365.25,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "DAYS_EMPLOYED": -emp_yrs * 365.25,
        "bureau_loan_count": data.get("bureau_loan_count", 0),
        "active_loans": data.get("active_credits", 0),
        "total_debt": data.get("total_debt", 0),
        "overdue_loan_count": data.get("overdue_count", 0),
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        # Basic ratios
        "LOAN_INCOME_RATIO": credit / income,
        "ANNUITY_INCOME_RATIO": annuity / income,
        "CREDIT_GOODS_RATIO": credit / goods,
        "CREDIT_TERM": annuity / (credit + 1),
        "GOODS_INCOME_RATIO": goods / income,
        "INCOME_PER_PERSON": income,
        "AGE_YEARS": age,
        "EMPLOYMENT_YEARS": emp_yrs,
        "REGISTRATION_YEARS": 5.0,
        "ID_PUBLISH_YEARS": 5.0,
        "EMPLOY_TO_AGE_RATIO": emp_yrs / (age + 0.01),
        # EXT_SOURCE interactions
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
        "PAYMENT_RATE": annuity / (credit + 1),
        "CREDIT_OVERCHARGE": (credit - goods) / goods,
    }

    for col in feature_cols:
        row[col] = mapping.get(col, 0)

    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


def score_standalone(data: dict) -> dict | None:
    """Score an application using locally loaded models."""
    try:
        artefacts = load_models()
        model = artefacts.get("xgb")
        if model is None:
            st.error("Model file not found. Run training first.")
            return None

        feature_cols = list(model.feature_names_in_)
        X = build_feature_vector(data, feature_cols)

        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes

        prob = float(model.predict_proba(X)[:, 1][0])

        base_score, base_odds, pdo = 600, 1 / 19, 20
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        odds = (1 - prob) / max(prob, 1e-10)
        credit_score = int(np.clip(offset + factor * np.log(max(odds, 1e-10)), 300, 850))

        if prob < 0.05:
            tier, rec = "Low", "Approve"
        elif prob < 0.15:
            tier, rec = "Medium", "Review"
        else:
            tier, rec = "High", "Decline"

        top_factors = []
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            abs_shap = np.abs(shap_vals[0])
            top_idx = np.argsort(abs_shap)[::-1][:5]
            for idx in top_idx:
                feat = feature_cols[idx]
                sv = float(shap_vals[0][idx])
                top_factors.append({
                    "feature": feat,
                    "shap_value": round(sv, 4),
                    "direction": "increases risk" if sv > 0 else "decreases risk",
                })
        except Exception:
            pass

        return {
            "default_probability": round(prob, 4),
            "credit_score": credit_score,
            "risk_tier": tier,
            "recommendation": rec,
            "top_risk_factors": top_factors,
            "model_version": "1.0.0",
        }
    except Exception as e:
        st.error(f"Scoring error: {e}")
        return None


def score_via_api(data: dict) -> dict | None:
    """Score by posting to the FastAPI backend."""
    import requests
    try:
        resp = requests.post(f"{API_URL}/score", json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is the FastAPI server running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} -- {e.response.text}")
        return None


def score_application(data: dict) -> dict | None:
    if USE_API:
        return score_via_api(data)
    return score_standalone(data)


def get_model_info() -> dict | None:
    if USE_API:
        import requests
        try:
            resp = requests.get(f"{API_URL}/model/info", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None
    return {
        "model": "xgboost_challenger",
        "version": "2.0.0",
        "features": "230",
        "mode": "standalone",
    }


# -- Sidebar Navigation --------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Single Scoring", "Batch Scoring", "Model Dashboard"],
)

st.sidebar.markdown("---")
mode_label = "API mode" if USE_API else "Standalone mode"
st.sidebar.markdown(
    f"**Credit Risk Scorecard**\n\n"
    f"Champion / Challenger framework.\n\n"
    f"Mode: {mode_label}"
)

# ==================================================================
# PAGE 1 -- Single Application Scoring
# ==================================================================
if page == "Single Scoring":
    st.title("Single Application Scoring")
    st.markdown("Enter applicant details and get an instant credit decision.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Info")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income", min_value=1_000, value=250_000, step=10_000)
        employment_years = st.number_input("Years Employed", min_value=0.0, value=8.0, step=0.5)
        education = st.selectbox("Education", [
            "Higher education", "Secondary", "Incomplete higher",
            "Lower secondary", "Academic degree",
        ])
        family_status = st.selectbox("Family Status", [
            "Married", "Single", "Civil marriage",
            "Separated", "Widow",
        ])
        housing = st.selectbox("Housing Type", [
            "House / apartment", "Rented apartment",
            "With parents", "Municipal apartment",
            "Office apartment", "Co-op apartment",
        ])

    with col2:
        st.subheader("Loan Details")
        loan_amount = st.number_input("Loan Amount", min_value=1_000, value=500_000, step=10_000)
        annuity = st.number_input("Monthly Annuity", min_value=100, value=25_000, step=1_000)
        goods_price = st.number_input("Goods Price", min_value=0, value=450_000, step=10_000)

    with col3:
        st.subheader("Bureau Data")
        bureau_loan_count = st.number_input("Bureau Loan Count", min_value=0, value=3)
        active_credits = st.number_input("Active Credits", min_value=0, value=1)
        total_debt = st.number_input("Total Debt", min_value=0.0, value=150_000.0, step=10_000.0)
        overdue_count = st.number_input("Overdue Count", min_value=0, value=0)
        st.markdown("**External Scores** (optional)")
        ext1 = st.number_input("EXT_SOURCE_1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        ext2 = st.number_input("EXT_SOURCE_2", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        ext3 = st.number_input("EXT_SOURCE_3", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.markdown("---")

    if st.button("Score Application", type="primary", use_container_width=True):
        payload = {
            "age": age,
            "income": float(income),
            "loan_amount": float(loan_amount),
            "annuity": float(annuity),
            "goods_price": float(goods_price),
            "employment_years": float(employment_years),
            "education_type": education,
            "family_status": family_status,
            "housing_type": housing,
            "bureau_loan_count": bureau_loan_count,
            "active_credits": active_credits,
            "total_debt": float(total_debt),
            "overdue_count": overdue_count,
            "ext_source_1": ext1,
            "ext_source_2": ext2,
            "ext_source_3": ext3,
        }

        with st.spinner("Scoring..."):
            result = score_application(payload)

        if result:
            tier_colors = {"Low": "green", "Medium": "orange", "High": "red"}

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Credit Score", result["credit_score"])
            r2.metric("Default Probability", f"{result['default_probability']:.2%}")
            r3.metric("Risk Tier", result["risk_tier"])
            r4.metric("Recommendation", result["recommendation"])

            if result.get("top_risk_factors"):
                st.subheader("Top Risk Factors (SHAP)")
                for factor in result["top_risk_factors"]:
                    st.markdown(
                        f"- **{factor['feature']}** -- "
                        f"SHAP = {factor['shap_value']:+.4f} ({factor['direction']})"
                    )

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["credit_score"],
                title={"text": "Credit Score"},
                gauge={
                    "axis": {"range": [300, 850]},
                    "bar": {"color": tier_colors.get(result["risk_tier"], "gray")},
                    "steps": [
                        {"range": [300, 500], "color": "#ffcccc"},
                        {"range": [500, 650], "color": "#fff3cd"},
                        {"range": [650, 850], "color": "#d4edda"},
                    ],
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# PAGE 2 -- Batch Scoring
# ==================================================================
elif page == "Batch Scoring":
    st.title("Batch Scoring")
    st.markdown("Upload a CSV of applicants to score them all at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"**Uploaded:** {len(df):,} rows x {len(df.columns)} columns")
        st.dataframe(df.head(10), use_container_width=True)

        required = ["age", "income", "loan_amount", "annuity"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.warning(
                f"Missing required columns: {missing}. "
                "Expected: age, income, loan_amount, annuity, "
                "goods_price, employment_years, bureau_loan_count, "
                "active_credits, total_debt, overdue_count"
            )
        else:
            if st.button("Score All Applications", type="primary"):
                results = []
                progress = st.progress(0)

                for i, row in df.iterrows():
                    payload = {
                        "age": int(row.get("age", 30)),
                        "income": float(row.get("income", 100000)),
                        "loan_amount": float(row.get("loan_amount", 200000)),
                        "annuity": float(row.get("annuity", 15000)),
                        "goods_price": float(row.get("goods_price", 0)),
                        "employment_years": float(row.get("employment_years", 0)),
                        "bureau_loan_count": int(row.get("bureau_loan_count", 0)),
                        "active_credits": int(row.get("active_credits", 0)),
                        "total_debt": float(row.get("total_debt", 0)),
                        "overdue_count": int(row.get("overdue_count", 0)),
                        "ext_source_1": float(row.get("ext_source_1", 0.5)) if pd.notna(row.get("ext_source_1")) else None,
                        "ext_source_2": float(row.get("ext_source_2", 0.5)) if pd.notna(row.get("ext_source_2")) else None,
                        "ext_source_3": float(row.get("ext_source_3", 0.5)) if pd.notna(row.get("ext_source_3")) else None,
                    }
                    result = score_application(payload)
                    if result:
                        results.append(result)
                    progress.progress((i + 1) / len(df))

                if results:
                    results_df = pd.DataFrame(results)
                    scored = pd.concat([df.reset_index(drop=True), results_df], axis=1)

                    st.success(f"Scored {len(results):,} applications")
                    st.dataframe(scored, use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Score", f"{results_df['credit_score'].mean():.0f}")
                    c2.metric("Avg PD", f"{results_df['default_probability'].mean():.2%}")
                    c3.metric("Decline Rate", f"{(results_df['recommendation'] == 'Decline').mean():.1%}")

                    fig = px.histogram(
                        results_df, x="credit_score", color="risk_tier",
                        title="Score Distribution by Risk Tier",
                        color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    csv = scored.to_csv(index=False)
                    st.download_button(
                        "Download Scored CSV",
                        data=csv,
                        file_name="scored_applications.csv",
                        mime="text/csv",
                    )


# ==================================================================
# PAGE 3 -- Model Performance Dashboard
# ==================================================================
elif page == "Model Dashboard":
    st.title("Model Performance Dashboard")

    # -- Champion vs Challenger metrics ----------------------------
    st.subheader("Champion vs Challenger Comparison")

    metrics_data = {
        "Metric": ["AUC-ROC", "Gini", "KS Statistic", "PSI"],
        "Champion (Scorecard)": [0.7566, 0.5131, 0.3842, 0.0002],
        "Challenger (XGBoost)": [0.7823, 0.5646, 0.4273, 0.0002],
    }

    report_path = ROOT / "reports" / "model_comparison.json"
    if report_path.exists():
        with open(report_path) as f:
            actual = json.load(f)
        if "scorecard_champion" in actual and "xgboost_challenger" in actual:
            sc = actual["scorecard_champion"]
            xg = actual["xgboost_challenger"]
            metrics_data = {
                "Metric": list(sc.keys()),
                "Champion (Scorecard)": list(sc.values()),
                "Challenger (XGBoost)": list(xg.values()),
            }

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Champion (Scorecard)",
        x=metrics_df["Metric"][:3],
        y=metrics_df["Champion (Scorecard)"][:3],
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        name="Challenger (XGBoost)",
        x=metrics_df["Metric"][:3],
        y=metrics_df["Challenger (XGBoost)"][:3],
        marker_color="coral",
    ))
    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        yaxis_title="Score",
    )
    st.plotly_chart(fig, use_container_width=True)

    # -- Score Distribution (real data) ----------------------------
    st.subheader("Score Distribution")

    test_df = load_test_data()
    artefacts = load_models()

    if test_df is not None and "xgb" in artefacts:
        from sklearn.model_selection import train_test_split

        target = "TARGET"
        exclude = {"SK_ID_CURR", "SK_ID_PREV", target}
        feat_cols = [c for c in test_df.columns if c not in exclude]
        X_all = test_df[feat_cols].copy()
        y_all = test_df[target]
        _, X_hold, _, y_hold = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )

        for col in X_hold.select_dtypes(include=["object", "category"]).columns:
            X_hold[col] = X_hold[col].astype("category").cat.codes

        xgb_model = artefacts["xgb"]
        y_prob = xgb_model.predict_proba(X_hold)[:, 1]

        base_score, base_odds, pdo = 600, 1 / 19, 20
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        odds = (1 - y_prob) / np.clip(y_prob, 1e-10, None)
        credit_scores = np.clip(offset + factor * np.log(np.clip(odds, 1e-10, None)), 300, 850)

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=credit_scores[y_hold == 0], name="Non-Default", opacity=0.6,
            marker_color="steelblue", nbinsx=50, histnorm="probability density",
        ))
        fig2.add_trace(go.Histogram(
            x=credit_scores[y_hold == 1], name="Default", opacity=0.6,
            marker_color="coral", nbinsx=50, histnorm="probability density",
        ))
        fig2.update_layout(
            barmode="overlay",
            title="Score Distribution -- Goods vs Bads (Hold-out Set)",
            xaxis_title="Credit Score", yaxis_title="Density",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Default rate by score band
        st.subheader("Default Rate by Score Band")
        score_band_df = pd.DataFrame({
            "score": credit_scores,
            "default": y_hold.values,
        })
        score_band_df["band"] = pd.cut(
            score_band_df["score"],
            bins=[300, 450, 550, 620, 680, 750, 850],
            labels=["300-450", "450-550", "550-620", "620-680", "680-750", "750-850"],
        )
        band_rates = score_band_df.groupby("band", observed=False)["default"].agg(
            ["mean", "count"]
        ).reset_index()
        band_rates.columns = ["Score Band", "Default Rate", "Count"]

        fig3 = px.bar(
            band_rates, x="Score Band", y="Default Rate",
            title="Default Rate by Score Band",
            color="Default Rate",
            color_continuous_scale=["green", "orange", "red"],
            text=band_rates["Default Rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

        # Decile table
        st.subheader("Decile Risk Ranking")
        from evaluate import decile_analysis
        dec_df = decile_analysis(y_hold.values, y_prob)
        st.dataframe(dec_df.style.format({
            "Bad_Rate": "{:.2%}",
            "Cumul_Bad_Pct": "{:.2%}",
            "Lift": "{:.2f}",
        }), use_container_width=True, hide_index=True)

    else:
        st.info("Processed data or model not found. Showing placeholder metrics.")

    # -- PSI Status ------------------------------------------------
    st.subheader("PSI Monitoring Status")

    psi_data = pd.DataFrame({
        "Feature": ["Overall Score", "EXT_SOURCE_2", "AMT_CREDIT",
                     "AGE_YEARS", "LOAN_INCOME_RATIO", "bureau_loan_count"],
        "PSI": [0.0002, 0.003, 0.002, 0.001, 0.005, 0.004],
        "Status": ["Stable"] * 6,
    })

    psi_path = ROOT / "reports" / "psi_report.csv"
    if psi_path.exists():
        psi_data = pd.read_csv(psi_path).head(10)

    st.dataframe(psi_data, use_container_width=True, hide_index=True)

    # -- Model Info ------------------------------------------------
    st.subheader("Model Info")
    info = get_model_info()
    if info:
        st.json(info)
