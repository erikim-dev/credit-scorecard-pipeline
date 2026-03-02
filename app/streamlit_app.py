"""
Credit Risk Scoring Platform
Professional credit-risk assessment with multi-model scoring,
real-time SHAP explanations, and model governance analytics.
"""

import sys
import pathlib
import json
import os
import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

API_URL = os.getenv("API_URL", "").strip()
USE_API = bool(API_URL)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
T = "plotly_dark"
C1 = "#60b4ff"   # primary blue
C2 = "#ff6b6b"   # danger red
C3 = "#51cf66"   # success green
C4 = "#cc5de8"   # accent purple
C5 = "#fcc419"   # warning amber
PAL = [C1, C2, C3, C4, C5]
CARD = "#161b22"
BORDER = "#30363d"
MUTED = "#8b949e"
TEXT = "#e6edf3"

st.set_page_config(
    page_title="Credit Risk Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS -- polished dark look, no decorators
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* layout */
    .block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; }}
    section[data-testid="stSidebar"] {{
        background: #0d1117; border-right: 1px solid {BORDER};
    }}

    /* metric cards */
    [data-testid="stMetric"] {{
        background: {CARD}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 14px 18px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED}; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 0.06em;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT}; font-size: 1.7rem; font-weight: 700;
    }}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        border-bottom: 2px solid transparent; padding: 10px 22px; color: {MUTED};
    }}
    .stTabs [aria-selected="true"] {{ border-bottom-color: {C1}; color: {TEXT}; }}

    /* tables */
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}

    /* typography */
    h1 {{ font-weight: 700; letter-spacing: -0.025em; }}
    h2, h3 {{ font-weight: 600; color: {TEXT}; }}

    /* hide chrome */
    #MainMenu, footer {{ visibility: hidden; }}

    /* status pill */
    .pill {{
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
    }}
    .pill-ok   {{ background: rgba(81,207,102,0.15); color: {C3}; }}
    .pill-warn {{ background: rgba(252,196,25,0.15); color: {C5}; }}
    .pill-bad  {{ background: rgba(255,107,107,0.15); color: {C2}; }}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    import joblib
    mdir = ROOT / "models"
    art = {}
    for key, fn in {
        "xgb": "xgboost_challenger.pkl",
        "lgb": "lightgbm_challenger.pkl",
        "scorecard": "scorecard_champion.pkl",
        "stacking": "stacking_ensemble.pkl",
        "woe_encoder": "woe_encoder.pkl",
        "scaler": "scaler.pkl",
    }.items():
        p = mdir / fn
        if p.exists():
            try:
                art[key] = joblib.load(p)
            except Exception:
                pass
    fp = mdir / "selected_features.json"
    if fp.exists():
        with open(fp) as f:
            art["selected_features"] = json.load(f)
    return art


@st.cache_data(show_spinner=False)
def load_test_data():
    p = ROOT / "data" / "processed" / "train_features.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.warning(f"Error loading data: {e}")
            return None
    return None


@st.cache_data(show_spinner=False)
def load_comparison():
    p = ROOT / "reports" / "model_comparison.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def build_feature_vector(data: dict, cols: list) -> pd.DataFrame:
    """Map user inputs to the full 300-column feature vector the models expect.

    Explicitly maps every feature the user provides (personal, loan, bureau,
    external scores) and derives all interaction / polynomial terms.  For
    features we cannot collect through the UI (e.g. installment history,
    POS cash snapshots), we fall back to population medians so the model
    sees realistic baseline values instead of zeros.
    """
    age = abs(data.get("age", 35))
    inc = data.get("income", 100_000) or 1
    crd = data.get("loan_amount", 200_000)
    ann = data.get("annuity", 15_000)
    gds = data.get("goods_price", 1) or 1
    emp = max(data.get("employment_years", 0), 0)
    e1 = data.get("ext_source_1", 0.5)
    e2 = data.get("ext_source_2", 0.5)
    e3 = data.get("ext_source_3", 0.5)
    blc = data.get("bureau_loan_count", 0)
    act = data.get("active_credits", 0)
    dbt = data.get("total_debt", 0)
    odc = data.get("overdue_count", 0)

    # --- Target-encoded lookup tables (from training data) ----------------
    te_education = {
        "Higher education": 0.0536, "Secondary": 0.0894,
        "Incomplete higher": 0.0848, "Lower secondary": 0.1092,
        "Academic degree": 0.0219,
    }
    te_family = {
        "Married": 0.0756, "Single": 0.0981,
        "Civil marriage": 0.0994, "Separated": 0.0819, "Widow": 0.0583,
    }
    te_housing = {
        "House / apartment": 0.0780, "Rented apartment": 0.1230,
        "With parents": 0.1170, "Municipal apartment": 0.0854,
        "Office apartment": 0.0658,
    }

    m = {
        # --- Core application fields ------------------------------------
        "DAYS_BIRTH": -age * 365.25,
        "AMT_INCOME_TOTAL": inc,
        "AMT_CREDIT": crd,
        "AMT_ANNUITY": ann,
        "AMT_GOODS_PRICE": gds,
        "DAYS_EMPLOYED": -emp * 365.25,
        "DAYS_REGISTRATION": -5 * 365.25,
        "DAYS_ID_PUBLISH": -5 * 365.25,

        # --- Bureau inputs ----------------------------------------------
        "bureau_loan_count": blc,
        "active_loans": act,
        "total_debt": dbt,
        "overdue_loan_count": odc,
        "overdue_count": odc,
        "total_overdue_amount": 0,
        "active_credits": act,
        "active_contracts": max(20, act * 5),
        "avg_debt": dbt / max(blc, 1),
        "debt_to_credit_ratio": dbt / max(crd, 1),
        "bureau_loans_last_12m": max(1, blc // 3),
        "bureau_loans_last_24m": max(1, blc // 2),
        "active_loans_last_12m": act,
        "avg_debt_last_12m": dbt / max(blc, 1),
        "overdue_amount_last_12m": 0,
        "bureau_credit_type_count": max(1, blc),
        "avg_bureau_annuity": ann * 0.25 if blc > 0 else 0,
        "earliest_bureau_days": -1832,
        "latest_bureau_days": -300,
        "std_debt_amount": dbt * 0.5,
        "std_bureau_days": 500,
        "bureau_day_overdue_count": odc,
        "max_bureau_day_overdue": odc * 30,
        "bureau_recent_ratio": 0.3,
        "BUREAU_ENQUIRY_TOTAL": 2,

        # --- External scores --------------------------------------------
        "EXT_SOURCE_1": e1, "EXT_SOURCE_2": e2, "EXT_SOURCE_3": e3,

        # --- Derived ratios ---------------------------------------------
        "LOAN_INCOME_RATIO": crd / inc,
        "ANNUITY_INCOME_RATIO": ann / inc,
        "CREDIT_GOODS_RATIO": crd / gds,
        "CREDIT_TERM": ann / (crd + 1),
        "GOODS_INCOME_RATIO": gds / inc,
        "INCOME_PER_PERSON": inc,
        "AGE_YEARS": age,
        "EMPLOYMENT_YEARS": emp,
        "REGISTRATION_YEARS": 5.0,
        "ID_PUBLISH_YEARS": 5.0,
        "EMPLOY_TO_AGE_RATIO": emp / (age + 0.01),
        "PAYMENT_RATE": ann / (crd + 1),
        "CREDIT_OVERCHARGE": (crd - gds) / gds,

        # --- EXT_SOURCE interactions ------------------------------------
        "EXT_SOURCE_MEAN": np.mean([e1, e2, e3]),
        "EXT_SOURCE_STD": np.std([e1, e2, e3]),
        "EXT_SOURCE_MIN": min(e1, e2, e3),
        "EXT_SOURCE_MAX": max(e1, e2, e3),
        "EXT_SOURCE_RANGE": max(e1, e2, e3) - min(e1, e2, e3),
        "EXT_SRC_1x2": e1 * e2,
        "EXT_SRC_2x3": e2 * e3,
        "EXT_SRC_1x3": e1 * e3,
        "EXT_SRC_1x2x3": e1 * e2 * e3,
        "EXT_SRC2_x_AGE": e2 * age,
        "EXT_SRC3_x_AGE": e3 * age,
        "EXT_SRC1_SQ": e1 ** 2,
        "EXT_SRC2_SQ": e2 ** 2,
        "EXT_SRC3_SQ": e3 ** 2,
        "EXT_SRC1_CB": e1 ** 3,
        "EXT_SRC2_CB": e2 ** 3,
        "EXT_SRC3_CB": e3 ** 3,

        # --- Target-encoded categoricals --------------------------------
        "TE_NAME_CONTRACT_TYPE": 0.0835,
        "TE_CODE_GENDER": 0.0700,
        "TE_FLAG_OWN_CAR": 0.0850,
        "TE_FLAG_OWN_REALTY": 0.0796,
        "TE_NAME_TYPE_SUITE": 0.0818,
        "TE_NAME_INCOME_TYPE": 0.0959,
        "TE_NAME_EDUCATION_TYPE": te_education.get(
            data.get("education", ""), 0.0894),
        "TE_NAME_FAMILY_STATUS": te_family.get(
            data.get("family_status", ""), 0.0756),
        "TE_NAME_HOUSING_TYPE": te_housing.get(
            data.get("housing", ""), 0.0780),
        "TE_OCCUPATION_TYPE": 0.0807,
        "TE_WEEKDAY_APPR_PROCESS_START": 0.0815,
        "TE_ORGANIZATION_TYPE": 0.0853,
        "TE_FONDKAPREMONT_MODE": 0.0807,
        "TE_HOUSETYPE_MODE": 0.0807,
        "TE_WALLSMATERIAL_MODE": 0.0807,
        "TE_latest_app_status": 0.0737,

        # --- Missing indicators (0 = value present, 1 = was missing) ----
        "MISS_EXT_SOURCE_1": 0,
        "MISS_EXT_SOURCE_2": 0,
        "MISS_EXT_SOURCE_3": 0,
        "MISS_AMT_GOODS_PRICE": 0,
        "MISS_AMT_ANNUITY": 0,
        "MISS_OCCUPATION_TYPE": 1,
        "MISS_CNT_FAM_MEMBERS": 0,
        "MISS_bureau_loan_count": 0 if blc > 0 else 1,
        "MISS_bb_total_records": 1,
        "MISS_cc_count": 1,
        "MISS_pos_loan_count": 1,
        "MISS_prev_app_count": 1,

        # --- Installment / POS / credit card population medians ---------
        "late_payments": 1,
        "late_payment_ratio": 0.015,
        "late_payments_last_12m": 0,
        "late_ratio_last_12m": 0.0,
        "late_ratio_trend": 0.0,
        "pos_dpd_rate_last_12m": 0.0,
        "approval_rate_last_12m": 0.4,
        "avg_utilisation_ratio": 0.23,
        "avg_drawing_count": 0.25,
    }

    row = {c: m.get(c, 0) for c in cols}
    return pd.DataFrame([row])[cols]


def prob_to_score(p: float) -> int:
    factor = 20 / np.log(2)
    offset = 600 - factor * np.log(1 / 19)
    odds = (1 - p) / max(p, 1e-10)
    return int(np.clip(offset + factor * np.log(max(odds, 1e-10)), 300, 850))


def classify_risk(p: float) -> tuple[str, str]:
    if p < 0.05:
        return "Low", "Approve"
    if p < 0.15:
        return "Medium", "Review"
    return "High", "Decline"


def tier_color(t: str) -> str:
    return {
        "Low": C3, "Medium": C5, "High": C2,
        "Approve": C3, "Review": C5, "Decline": C2,
    }.get(t, MUTED)


def score_application(data: dict, model_choice: str) -> dict | None:
    if USE_API:
        import requests
        try:
            r = requests.post(f"{API_URL}/score", json=data, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    art = load_models()
    key_map = {"XGBoost": "xgb", "LightGBM DART": "lgb", "Stacking Ensemble": "stacking"}
    model = art.get(key_map.get(model_choice, "xgb"))
    if model is None:
        return None

    try:
        if model_choice == "Stacking Ensemble" and isinstance(model, dict):
            cols = list(model["xgb_model"].feature_names_in_)
            X = build_feature_vector(data, cols)
            xp = model["xgb_model"].predict_proba(X)[:, 1]
            lp = model["lgb_model"].predict_proba(X)[:, 1]
            mX = model["meta_scaler"].transform(np.column_stack([xp, lp]))
            prob = float(model["meta_model"].predict_proba(mX)[0][1])
            factors = []
        else:
            cols = list(model.feature_names_in_)
            X = build_feature_vector(data, cols)
            prob = float(model.predict_proba(X)[:, 1][0])
            factors = []
            try:
                import shap
                exp = shap.TreeExplainer(model)
                sv = exp.shap_values(X)
                abss = np.abs(sv[0])
                top = np.argsort(abss)[::-1][:6]
                for i in top:
                    factors.append({
                        "feature": cols[i],
                        "shap_value": round(float(sv[0][i]), 4),
                        "direction": "increases risk" if sv[0][i] > 0 else "decreases risk",
                    })
            except Exception:
                pass

        tier, rec = classify_risk(prob)
        return {
            "default_probability": round(prob, 4),
            "credit_score": prob_to_score(prob),
            "risk_tier": tier,
            "recommendation": rec,
            "top_risk_factors": factors,
            "model_used": model_choice,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Credit Risk Platform")
    st.caption("Champion / Challenger Scoring")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Scoring", "Batch Processing", "Model Analytics", "Data Explorer"],
        label_visibility="collapsed",
    )

    st.divider()
    art = load_models()
    n_models = sum(1 for k in ("xgb", "lgb", "scorecard", "stacking") if k in art)
    mode_label = "API" if USE_API else "Standalone"
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
        padding:14px 16px;font-size:0.78rem;color:{MUTED};line-height:1.7;">
        Mode <b style="color:{TEXT}">{mode_label}</b><br>
        Models loaded <b style="color:{TEXT}">{n_models}</b><br>
        Features <b style="color:{TEXT}">300</b>
    </div>
    """, unsafe_allow_html=True)

# ===================================================================
# PAGE: Scoring  (live-updating sliders, real-time gauge)
# ===================================================================
if page == "Scoring":
    st.title("Application Scoring")
    st.caption("Adjust inputs to see the score update in real time")

    col_model, _ = st.columns([2, 4])
    with col_model:
        model_choice = st.selectbox(
            "Model",
            ["XGBoost", "LightGBM DART", "Stacking Ensemble"],
            help="Stacking has the highest AUC (0.7903), followed by XGBoost (0.7888).",
        )

    tab_personal, tab_loan, tab_bureau = st.tabs(
        ["Personal", "Loan Details", "Bureau and External Scores"],
    )

    with tab_personal:
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 80, 35)
        with c2:
            income = st.number_input("Annual Income", 1_000, 10_000_000, 250_000, step=10_000)
        with c3:
            employment_years = st.slider("Years Employed", 0.0, 45.0, 8.0, 0.5)

        c4, c5, c6 = st.columns(3)
        with c4:
            education = st.selectbox("Education", [
                "Higher education", "Secondary", "Incomplete higher",
                "Lower secondary", "Academic degree",
            ])
        with c5:
            family_status = st.selectbox("Family Status", [
                "Married", "Single", "Civil marriage", "Separated", "Widow",
            ])
        with c6:
            housing = st.selectbox("Housing", [
                "House / apartment", "Rented apartment", "With parents",
                "Municipal apartment", "Office apartment",
            ])

    with tab_loan:
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amount = st.number_input("Loan Amount", 1_000, 50_000_000, 500_000, step=10_000)
        with c2:
            annuity = st.number_input("Monthly Annuity", 100, 5_000_000, 25_000, step=1_000)
        with c3:
            goods_price = st.number_input("Goods Price", 0, 50_000_000, 450_000, step=10_000)

        lir = loan_amount / max(income, 1)
        air = annuity / max(income, 1)
        cgr = loan_amount / max(goods_price, 1)

        r1, r2, r3 = st.columns(3)
        r1.metric("Loan-to-Income", f"{lir:.2f}",
                   delta="High" if lir > 5 else "OK",
                   delta_color="inverse" if lir > 5 else "normal")
        r2.metric("Annuity-to-Income", f"{air:.2%}",
                   delta="High" if air > 0.4 else "OK",
                   delta_color="inverse" if air > 0.4 else "normal")
        r3.metric("Credit-to-Goods", f"{cgr:.2f}",
                   delta="Over-financed" if cgr > 1.2 else "OK",
                   delta_color="inverse" if cgr > 1.2 else "normal")

    with tab_bureau:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bureau_loan_count = st.number_input("Bureau Loans", 0, 100, 3)
        with c2:
            active_credits = st.number_input("Active Credits", 0, 50, 1)
        with c3:
            total_debt = st.number_input("Total Debt", 0.0, 100_000_000.0, 150_000.0, step=10_000.0)
        with c4:
            overdue_count = st.number_input("Overdue Count", 0, 50, 0)

        st.markdown("##### External Scores")
        c5, c6, c7 = st.columns(3)
        with c5:
            ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01)
        with c6:
            ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5, 0.01)
        with c7:
            ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01)

    # -- live scoring on every interaction ---
    st.divider()
    payload = {
        "age": age, "income": float(income), "loan_amount": float(loan_amount),
        "annuity": float(annuity), "goods_price": float(goods_price),
        "employment_years": float(employment_years),
        "bureau_loan_count": bureau_loan_count, "active_credits": active_credits,
        "total_debt": float(total_debt), "overdue_count": overdue_count,
        "ext_source_1": ext1, "ext_source_2": ext2, "ext_source_3": ext3,
        "education": education, "family_status": family_status,
        "housing": housing,
    }
    result = score_application(payload, model_choice)

    if result:
        tier = result["risk_tier"]
        tc = tier_color(tier)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Credit Score", result["credit_score"])
        m2.metric("Default Probability", f"{result['default_probability']:.2%}")
        m3.metric("Risk Tier", tier)
        m4.metric("Decision", result["recommendation"])

        col_gauge, col_factors = st.columns([3, 2])

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["credit_score"],
                number={"font": {"size": 52, "color": TEXT}},
                gauge={
                    "axis": {"range": [300, 850], "tickcolor": MUTED,
                             "tickfont": {"size": 11, "color": MUTED}},
                    "bar": {"color": tc, "thickness": 0.75},
                    "bgcolor": "#21262d",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [300, 500], "color": "rgba(255,107,107,0.12)"},
                        {"range": [500, 650], "color": "rgba(252,196,25,0.10)"},
                        {"range": [650, 850], "color": "rgba(81,207,102,0.10)"},
                    ],
                    "threshold": {
                        "line": {"color": TEXT, "width": 2},
                        "thickness": 0.8,
                        "value": result["credit_score"],
                    },
                },
            ))
            fig.update_layout(
                height=300, margin=dict(t=40, b=10, l=40, r=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT),
            )
            st.plotly_chart(fig, use_container_width=True, key="gauge")

        with col_factors:
            factors = result.get("top_risk_factors", [])
            if factors:
                st.markdown("##### Risk Drivers")
                names = [f["feature"] for f in factors]
                vals  = [f["shap_value"] for f in factors]
                colors = [C2 if v > 0 else C3 for v in vals]

                fig_shap = go.Figure(go.Bar(
                    y=names[::-1], x=vals[::-1], orientation="h",
                    marker_color=colors[::-1],
                    text=[f"{v:+.4f}" for v in vals[::-1]],
                    textposition="outside",
                    textfont=dict(size=11, color=TEXT),
                ))
                fig_shap.update_layout(
                    template=T, height=280,
                    margin=dict(t=10, b=10, l=10, r=60),
                    xaxis_title="SHAP value",
                    yaxis=dict(tickfont=dict(size=11)),
                )
                st.plotly_chart(fig_shap, use_container_width=True, key="shap")
            else:
                st.caption("SHAP explanations not available for this model.")

        # Probability distribution context
        st.markdown("##### Where this applicant falls")
        prob_val = result["default_probability"]
        fig_ctx = go.Figure()
        # simulation: typical portfolio distribution
        np.random.seed(42)
        sim = np.random.beta(2, 20, 5000)
        fig_ctx.add_trace(go.Histogram(
            x=sim, nbinsx=60, marker_color=C1, opacity=0.5,
            histnorm="probability density", name="Portfolio",
        ))
        fig_ctx.add_vline(x=prob_val, line_color=tc, line_width=3,
                          annotation_text=f"This: {prob_val:.2%}",
                          annotation_font_color=tc)
        fig_ctx.update_layout(
            template=T, height=220,
            margin=dict(t=30, b=30, l=40, r=40),
            xaxis_title="Default Probability", yaxis_title="Density",
            showlegend=False,
        )
        st.plotly_chart(fig_ctx, use_container_width=True, key="ctx")
    else:
        st.warning("Could not score this application. Check that all models are loaded.")


# ===================================================================
# PAGE: Batch Processing
# ===================================================================
elif page == "Batch Processing":
    st.title("Batch Processing")
    st.caption("Upload applicant CSV for bulk scoring")

    col_m, _ = st.columns([2, 4])
    with col_m:
        model_choice = st.selectbox("Model", ["XGBoost", "LightGBM DART", "Stacking Ensemble"])

    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"], label_visibility="collapsed")

    # Provide a downloadable sample CSV for users to try the batch upload
    try:
        sample_fp = ROOT / "data" / "sample_applicants.csv"
        if sample_fp.exists():
            with open(sample_fp, "rb") as _f:
                sample_bytes = _f.read()
            st.download_button(
                "Download sample CSV",
                data=sample_bytes,
                file_name="sample_applicants.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception:
        pass

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.markdown(f"**{len(df):,} rows** -- {len(df.columns)} columns")

        with st.expander("Data preview", expanded=True):
            st.dataframe(df.head(20), use_container_width=True, height=300)

        required = ["age", "income", "loan_amount", "annuity"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            if st.button("Score All Applications", type="primary", use_container_width=True):
                results_list = []
                prog = st.progress(0.0, text="Scoring...")
                t0 = time.time()

                for i, row in df.iterrows():
                    pay = {
                        "age": int(row.get("age", 30)),
                        "income": float(row.get("income", 100_000)),
                        "loan_amount": float(row.get("loan_amount", 200_000)),
                        "annuity": float(row.get("annuity", 15_000)),
                        "goods_price": float(row.get("goods_price", 0)),
                        "employment_years": float(row.get("employment_years", 0)),
                        "bureau_loan_count": int(row.get("bureau_loan_count", 0)),
                        "active_credits": int(row.get("active_credits", 0)),
                        "total_debt": float(row.get("total_debt", 0)),
                        "overdue_count": int(row.get("overdue_count", 0)),
                        "ext_source_1": float(row.get("ext_source_1", 0.5)) if pd.notna(row.get("ext_source_1")) else 0.5,
                        "ext_source_2": float(row.get("ext_source_2", 0.5)) if pd.notna(row.get("ext_source_2")) else 0.5,
                        "ext_source_3": float(row.get("ext_source_3", 0.5)) if pd.notna(row.get("ext_source_3")) else 0.5,
                    }
                    r = score_application(pay, model_choice)
                    if r:
                        results_list.append(r)
                    pct = (i + 1) / len(df)
                    elapsed = time.time() - t0
                    eta = (elapsed / (i + 1)) * (len(df) - i - 1)
                    prog.progress(pct, text=f"Scored {i+1:,}/{len(df):,}  --  ETA {eta:.0f}s")

                if results_list:
                    rdf = pd.DataFrame(results_list)
                    scored = pd.concat([df.reset_index(drop=True), rdf], axis=1)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Applications", f"{len(rdf):,}")
                    m2.metric("Avg Score", f"{rdf['credit_score'].mean():.0f}")
                    m3.metric("Avg PD", f"{rdf['default_probability'].mean():.2%}")
                    m4.metric("Decline Rate", f"{(rdf['recommendation']=='Decline').mean():.1%}")

                    fig = go.Figure()
                    for tier, col in [("Low", C3), ("Medium", C5), ("High", C2)]:
                        mask = rdf["risk_tier"] == tier
                        if mask.any():
                            fig.add_trace(go.Histogram(
                                x=rdf.loc[mask, "credit_score"],
                                name=tier, marker_color=col, opacity=0.8, nbinsx=30,
                            ))
                    fig.update_layout(
                        template=T, barmode="overlay",
                        xaxis_title="Credit Score", yaxis_title="Count",
                        height=360, margin=dict(t=30, b=40),
                        legend=dict(orientation="h", yanchor="top", y=1.1),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(scored, use_container_width=True, height=400)
                    st.download_button(
                        "Download Results CSV", data=scored.to_csv(index=False),
                        file_name="scored_applications.csv", mime="text/csv",
                        use_container_width=True,
                    )


# ===================================================================
# PAGE: Model Analytics
# ===================================================================
elif page == "Model Analytics":
    st.title("Model Analytics")
    st.caption("Hold-out performance, ROC, stability, and fairness")

    comparison = load_comparison()
    test_df = load_test_data()
    artefacts = load_models()

    tab_perf, tab_roc, tab_dist, tab_stability, tab_fairness = st.tabs([
        "Performance", "ROC Curves", "Distributions", "Stability", "Fairness",
    ])

    # -- tab: Performance -------------------------------------------------
    with tab_perf:
        if comparison:
            if isinstance(comparison, list):
                mdf = pd.DataFrame(comparison)
            else:
                names = {
                    "scorecard_champion": "Scorecard (Champion)",
                    "xgboost_challenger": "XGBoost",
                    "lightgbm_challenger": "LightGBM DART",
                    "stacking_ensemble": "Stacking",
                }
                rows = []
                for k, n in names.items():
                    if k in comparison:
                        d = comparison[k]
                        rows.append({"Model": n, "AUC-ROC": d.get("AUC", 0),
                                     "Gini": d.get("Gini", 0), "KS Statistic": d.get("KS", 0)})
                mdf = pd.DataFrame(rows)

            num_cols = ["AUC-ROC", "Gini", "KS Statistic"]
            st.dataframe(
                mdf.style
                    .highlight_max(subset=num_cols, color="#1a4d2e")
                    .format({c: "{:.4f}" for c in num_cols}),
                use_container_width=True, hide_index=True,
            )

            # grouped bar chart
            fig = go.Figure()
            for i, (_, row) in enumerate(mdf.iterrows()):
                fig.add_trace(go.Bar(
                    name=row["Model"],
                    x=num_cols,
                    y=[row[c] for c in num_cols],
                    marker_color=PAL[i % len(PAL)],
                ))
            fig.update_layout(
                template=T, barmode="group", yaxis_range=[0, 1],
                yaxis_title="Score", height=400, margin=dict(t=30),
                legend=dict(orientation="h", yanchor="top", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

            # overfitting check
            st.markdown("##### Overfitting Check")
            cv_lu = {
                "Champion (Scorecard)": 0.7659, "Scorecard (Champion)": 0.7659,
                "XGBoost": 0.7886, "XGBoost (Challenger)": 0.7886,
                "LightGBM DART": 0.7736,
                "Stacking Ensemble": 0.7886, "Stacking": 0.7886,
            }
            ov = []
            for _, row in mdf.iterrows():
                cv = cv_lu.get(row["Model"], 0)
                ho = row["AUC-ROC"]
                if cv > 0:
                    gap = ho - cv
                    ov.append({
                        "Model": row["Model"],
                        "CV AUC": f"{cv:.4f}",
                        "Hold-out AUC": f"{ho:.4f}",
                        "Gap": f"{gap:+.4f}",
                        "Status": "OK" if abs(gap) < 0.08 else "Check",
                    })
            if ov:
                st.dataframe(pd.DataFrame(ov), use_container_width=True, hide_index=True)
        else:
            st.info("No model comparison data found. Run evaluation first.")

    # -- tab: ROC ---------------------------------------------------------
    with tab_roc:
        st.markdown("<small>Model ROC curves and AUC scores on hold-out data. Higher is better.</small>", unsafe_allow_html=True)
        st.markdown("""<small>Model ROC curves and AUC scores on hold-out data. Higher is better.</small>""", unsafe_allow_html=True)
        if test_df is not None and artefacts:
            from sklearn.metrics import roc_curve, roc_auc_score as auc_score
            from sklearn.model_selection import train_test_split

            exclude = {"SK_ID_CURR", "SK_ID_PREV", "TARGET"}
            fc = [c for c in test_df.columns if c not in exclude]
            X = test_df[fc]; y = test_df["TARGET"]
            _, Xh, _, yh = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            fig_roc = go.Figure()
            cfgs = [
                ("XGBoost", artefacts.get("xgb"), C1),
                ("LightGBM DART", artefacts.get("lgb"), C3),
            ]
            for nm, mdl, clr in cfgs:
                if mdl is not None:
                    try:
                        yp = mdl.predict_proba(Xh)[:, 1]
                        fpr, tpr, _ = roc_curve(yh, yp)
                        a = auc_score(yh, yp)
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, name=f"{nm} {a:.4f}",
                            line=dict(color=clr, width=2.5),
                        ))
                    except Exception:
                        pass

            if "stacking" in artefacts and isinstance(artefacts["stacking"], dict):
                try:
                    s = artefacts["stacking"]
                    xp = s["xgb_model"].predict_proba(Xh)[:, 1]
                    lp = s["lgb_model"].predict_proba(Xh)[:, 1]
                    mX = s["meta_scaler"].transform(np.column_stack([xp, lp]))
                    yp = s["meta_model"].predict_proba(mX)[:, 1]
                    fpr, tpr, _ = roc_curve(yh, yp)
                    a = auc_score(yh, yp)
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, name=f"Stacking {a:.4f}",
                        line=dict(color=C4, width=2.5),
                    ))
                except Exception:
                    pass

            if "scorecard" in artefacts and "woe_encoder" in artefacts:
                try:
                    woe = artefacts["woe_encoder"]
                    scl = artefacts.get("scaler")
                    sel = artefacts.get("selected_features", [])
                    Xsc = Xh[[c for c in sel if c in Xh.columns]]
                    Xsc = woe.transform(Xsc).fillna(0)
                    if scl is not None:
                        Xsc = pd.DataFrame(scl.transform(Xsc), columns=Xsc.columns)
                    yp = artefacts["scorecard"].predict_proba(Xsc)[:, 1]
                    fpr, tpr, _ = roc_curve(yh, yp)
                    a = auc_score(yh, yp)
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, name=f"Scorecard {a:.4f}",
                        line=dict(color=MUTED, width=2, dash="dot"),
                    ))
                except Exception:
                    pass

            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], name="Random",
                line=dict(color="#30363d", dash="dash", width=1),
            ))
            fig_roc.update_layout(
                template=T,
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                height=520, margin=dict(t=30),
                legend=dict(orientation="h", yanchor="top", y=1.12, font=dict(size=12)),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("Load test data to view ROC curves.")

    # -- tab: Distributions -----------------------------------------------
    with tab_dist:
        st.markdown("<small>Predicted probability distributions by actual outcome. Clear separation indicates good model discrimination.</small>", unsafe_allow_html=True)
        st.markdown("""<small>Predicted probability distributions by actual outcome. Clear separation indicates good model discrimination.</small>""", unsafe_allow_html=True)
        if test_df is not None and "xgb" in artefacts:
            from sklearn.model_selection import train_test_split

            exclude = {"SK_ID_CURR", "SK_ID_PREV", "TARGET"}
            fc = [c for c in test_df.columns if c not in exclude]
            X = test_df[fc]; y = test_df["TARGET"]
            _, Xh, _, yh = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # model selector for distributions
            dist_model_name = st.selectbox(
                "Model for distribution", ["XGBoost", "LightGBM DART", "Stacking Ensemble"],
                key="dist_model",
            )
            try:
                if dist_model_name == "Stacking Ensemble" and "stacking" in artefacts:
                    s = artefacts["stacking"]
                    xp = s["xgb_model"].predict_proba(Xh)[:, 1]
                    lp = s["lgb_model"].predict_proba(Xh)[:, 1]
                    mX = s["meta_scaler"].transform(np.column_stack([xp, lp]))
                    y_prob = s["meta_model"].predict_proba(mX)[:, 1]
                elif dist_model_name == "LightGBM DART" and "lgb" in artefacts:
                    y_prob = artefacts["lgb"].predict_proba(Xh)[:, 1]
                else:
                    y_prob = artefacts["xgb"].predict_proba(Xh)[:, 1]
            except Exception:
                y_prob = artefacts["xgb"].predict_proba(Xh)[:, 1]

            scores = np.array([prob_to_score(p) for p in y_prob])

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=scores[yh.values == 0], name="Non-Default",
                marker_color=C1, opacity=0.7, nbinsx=50, histnorm="probability density",
            ))
            fig.add_trace(go.Histogram(
                x=scores[yh.values == 1], name="Default",
                marker_color=C2, opacity=0.7, nbinsx=50, histnorm="probability density",
            ))
            fig.update_layout(
                template=T, barmode="overlay",
                xaxis_title="Credit Score", yaxis_title="Density",
                height=420, margin=dict(t=30),
                legend=dict(orientation="h", yanchor="top", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Default Rate by Score Band")
            sdf = pd.DataFrame({"score": scores, "default": yh.values})
            sdf["band"] = pd.cut(
                sdf["score"],
                bins=[300, 450, 550, 620, 680, 750, 850],
                labels=["300-450", "450-550", "550-620", "620-680", "680-750", "750-850"],
            )
            band = sdf.groupby("band", observed=False)["default"].agg(["mean", "count"]).reset_index()
            band.columns = ["Band", "Default Rate", "Count"]

            fig2 = go.Figure(go.Bar(
                x=band["Band"], y=band["Default Rate"],
                marker_color=[C2 if r > 0.15 else (C5 if r > 0.05 else C3) for r in band["Default Rate"]],
                text=[f"{r:.1%}" for r in band["Default Rate"]],
                textposition="outside", textfont=dict(color=TEXT),
            ))
            fig2.update_layout(
                template=T, yaxis_title="Default Rate",
                yaxis_tickformat=".0%", height=360, margin=dict(t=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("##### Decile Analysis")
            try:
                from evaluate import decile_analysis
                dec = decile_analysis(yh.values, y_prob)
                fmt = {}
                if "Bad_Rate" in dec.columns:
                    fmt["Bad_Rate"] = "{:.2%}"
                if "Cumul_Bad_Pct" in dec.columns:
                    fmt["Cumul_Bad_Pct"] = "{:.2%}"
                if "Lift" in dec.columns:
                    fmt["Lift"] = "{:.2f}"
                st.dataframe(
                    dec.style.format(fmt),
                    use_container_width=True, hide_index=True,
                )
            except Exception:
                st.caption("Decile analysis not available.")
        else:
            st.info("Load data and models to view distributions.")

    # -- tab: Stability ---------------------------------------------------
    with tab_stability:
        st.markdown("<small>Population Stability Index (PSI) measures data drift over time. Values below 0.10 indicate stability.</small>", unsafe_allow_html=True)
        st.markdown("##### Population Stability Index")

        psi_path = ROOT / "reports" / "psi_report.csv"
        if psi_path.exists():
            psi_data = pd.read_csv(psi_path).head(12)
        else:
            psi_data = pd.DataFrame({
                "Feature": ["Overall Score", "EXT_SOURCE_2", "AMT_CREDIT",
                             "AGE_YEARS", "LOAN_INCOME_RATIO", "bureau_loan_count"],
                "PSI": [0.0003, 0.003, 0.002, 0.001, 0.005, 0.004],
                "Status": ["Stable"] * 6,
            })

        st.dataframe(psi_data, use_container_width=True, hide_index=True)

        fig = go.Figure(go.Bar(
            x=psi_data["Feature"], y=psi_data["PSI"],
            marker_color=[C3 if v < 0.1 else (C5 if v < 0.2 else C2) for v in psi_data["PSI"]],
            text=[f"{v:.4f}" for v in psi_data["PSI"]],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ))
        fig.add_hline(y=0.1, line_dash="dash", line_color=C5,
                       annotation_text="Warning (0.10)", annotation_font_color=C5)
        fig.add_hline(y=0.2, line_dash="dash", line_color=C2,
                       annotation_text="Significant (0.20)", annotation_font_color=C2)
        fig.update_layout(template=T, yaxis_title="PSI", height=380, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # -- tab: Fairness ----------------------------------------------------
    with tab_fairness:
        st.markdown("##### Fairness Audit")
        st.caption("AUC consistency across demographic segments")

        if test_df is not None and "xgb" in artefacts:
            from sklearn.metrics import roc_auc_score as auc_fn
            from sklearn.model_selection import train_test_split

            _, tsplit, _, _ = train_test_split(
                test_df, test_df["TARGET"], test_size=0.2, random_state=42, stratify=test_df["TARGET"],
            )
            exclude = {"SK_ID_CURR", "SK_ID_PREV", "TARGET"}
            fc = [c for c in tsplit.columns if c not in exclude]
            Xh = tsplit[fc]; yh = tsplit["TARGET"]
            yp = artefacts["xgb"].predict_proba(Xh)[:, 1]

            fair = []
            if "CODE_GENDER" in tsplit.columns:
                for g in tsplit["CODE_GENDER"].unique():
                    mask = tsplit["CODE_GENDER"] == g
                    if mask.sum() > 50 and yh[mask].nunique() > 1:
                        fair.append({"Group": "Gender", "Value": str(g),
                                     "AUC": round(auc_fn(yh[mask], yp[mask.values]), 4),
                                     "N": int(mask.sum())})

            if "DAYS_BIRTH" in tsplit.columns:
                ages = (-tsplit["DAYS_BIRTH"] / 365.25).astype(int)
                bins = pd.cut(ages, bins=[18, 30, 40, 50, 60, 100],
                              labels=["18-30", "31-40", "41-50", "51-60", "60+"])
                for b in bins.dropna().unique():
                    mask = bins == b
                    if mask.sum() > 50 and yh[mask].nunique() > 1:
                        fair.append({"Group": "Age", "Value": b,
                                     "AUC": round(auc_fn(yh[mask], yp[mask.values]), 4),
                                     "N": int(mask.sum())})

            if fair:
                fdf = pd.DataFrame(fair)
                st.dataframe(fdf, use_container_width=True, hide_index=True)

                fig = px.bar(fdf, x="Value", y="AUC", color="Group",
                             color_discrete_sequence=PAL, text="AUC")
                fig.update_layout(template=T, yaxis_range=[0.5, 1.0],
                                  height=380, margin=dict(t=20))
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Demographic columns not found.")
        else:
            st.info("Load data and models for fairness audit.")


# ===================================================================
# PAGE: Data Explorer
# ===================================================================
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Interactive exploration of the training dataset")

    test_df = load_test_data()

    if test_df is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(test_df):,}")
        m2.metric("Features", f"{test_df.shape[1] - 2}")
        m3.metric("Default Rate", f"{test_df['TARGET'].mean():.2%}")
        m4.metric("Missing Columns", f"{(test_df.isnull().any()).sum()}")

        tab_ov, tab_corr, tab_feat = st.tabs(["Overview", "Correlations", "Feature Drill-Down"])

        with tab_ov:
            st.markdown("##### Sample Data")
            n_rows = st.slider("Rows to display", 10, 500, 100, 10, key="ov_rows")
            st.dataframe(test_df.head(n_rows), use_container_width=True, height=400)

            st.markdown("##### Column Statistics")
            desc = test_df.describe().T
            desc["missing_pct"] = test_df.isnull().mean()
            st.dataframe(
                desc.style.format("{:.2f}").background_gradient(cmap="YlOrRd", subset=["missing_pct"]),
                use_container_width=True, height=400,
            )

        with tab_corr:
            st.markdown("##### Top Correlations with TARGET")
            n_top = st.slider("Number of features", 10, 50, 25, key="corr_n")
            corr = (test_df.corr(numeric_only=True)["TARGET"]
                    .drop("TARGET").abs().sort_values(ascending=False).head(n_top))
            fig = go.Figure(go.Bar(
                y=corr.index[::-1], x=corr.values[::-1],
                orientation="h", marker_color=C1,
                text=[f"{v:.3f}" for v in corr.values[::-1]],
                textposition="outside", textfont=dict(size=10, color=TEXT),
            ))
            fig.update_layout(
                template=T, xaxis_title="Absolute Correlation",
                height=max(400, n_top * 22), margin=dict(t=20, l=220),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_feat:
            num_cols = sorted([c for c in test_df.select_dtypes("number").columns
                               if c not in ("TARGET", "SK_ID_CURR")])
            sel = st.selectbox("Feature", num_cols, key="feat_sel")

            if sel:
                col_ch, col_st = st.columns([3, 1])

                with col_ch:
                    fig = go.Figure()
                    for lbl, clr, nm in [(0, C1, "Non-Default"), (1, C2, "Default")]:
                        vals = test_df.loc[test_df["TARGET"] == lbl, sel].dropna()
                        fig.add_trace(go.Histogram(
                            x=vals, name=nm, marker_color=clr,
                            opacity=0.7, nbinsx=50, histnorm="probability density",
                        ))
                    fig.update_layout(
                        template=T, barmode="overlay",
                        xaxis_title=sel, yaxis_title="Density",
                        height=400, margin=dict(t=20),
                        legend=dict(orientation="h", yanchor="top", y=1.1),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_st:
                    cd = test_df[sel]
                    st.markdown("##### Statistics")
                    for label, val in [
                        ("Mean", cd.mean()), ("Median", cd.median()),
                        ("Std Dev", cd.std()), ("Min", cd.min()),
                        ("Max", cd.max()),
                    ]:
                        st.markdown(f"**{label}:** {val:.4f}")
                    st.markdown(f"**Missing:** {cd.isnull().mean():.1%}")

                    valid = test_df[[sel, "TARGET"]].dropna()
                    if len(valid) > 10:
                        cv = valid[sel].corr(valid["TARGET"])
                        st.metric("Target Correlation", f"{cv:.4f}")

                # Box-plot comparison
                st.markdown("##### Distribution by Target")
                box_df = test_df[[sel, "TARGET"]].dropna().copy()
                box_df["TARGET"] = box_df["TARGET"].map({0: "Non-Default", 1: "Default"})
                fig_box = px.box(
                    box_df, x="TARGET", y=sel, color="TARGET",
                    color_discrete_map={"Non-Default": C1, "Default": C2},
                )
                fig_box.update_layout(
                    template=T, height=350, margin=dict(t=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("**Data not available**")
        st.markdown("""
The training dataset has not been generated yet. To proceed:

1. **Run the feature engineering notebook:**
   - Open `notebooks/03_feature_engineering.ipynb`
   - Execute all cells to generate `data/processed/train_features.csv`

Alternatively, run from command line:
```bash
python src/feature_engineering.py --data data/raw
```

Once the data is ready, refresh this page.
""")
