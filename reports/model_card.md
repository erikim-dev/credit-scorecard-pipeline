# Model Card — Credit Risk Scoring Models

## Model Overview

| Field | Champion (Scorecard) | Challenger (XGBoost) |
|-------|---------------------|---------------------|
| **Type** | Logistic Regression + WoE | Gradient Boosted Trees |
| **Purpose** | Consumer credit default prediction | Consumer credit default prediction |
| **Target** | Binary: 1 = default, 0 = non-default | Binary: 1 = default, 0 = non-default |
| **Output** | Probability of default + credit score (300–850) | Probability of default + credit score (300–850) |
| **Version** | 2.0 | 2.0 |
| **Date** | 2026-03 | 2026-03 |

---

## Intended Use

- **Primary use:** Assist credit decisioning for consumer loan applications
- **Users:** Credit risk analysts, underwriters, automated decisioning systems
- **Scope:** Applications matching the Home Credit Default Risk population profile
- **Not intended for:** Commercial lending, mortgage origination, or collections prioritisation

---

## Training Data

| Detail | Value |
|--------|-------|
| **Source** | Home Credit Default Risk (Kaggle competition) |
| **Size** | 307,511 applications |
| **Features** | 230 engineered features (122 original + 108 derived) from 8 relational tables |
| **Target distribution** | 8.07% default rate |
| **Date range** | Historical (competition dataset) |
| **Geography** | Emerging market consumer lending |

### Data Tables Used
- `application_train.csv` — main application features (122 columns)
- `bureau.csv` / `bureau_balance.csv` — Credit Bureau history
- `previous_application.csv` — prior Home Credit applications
- `POS_CASH_balance.csv` — POS/cash loan snapshots
- `installments_payments.csv` — repayment history
- `credit_card_balance.csv` — card utilisation data

---

## Performance Metrics

| Metric | Champion (Scorecard) | Challenger (XGBoost) |
|--------|---------------------|---------------------|
| AUC-ROC | 0.7638 | 0.8316 |
| Gini Coefficient | 0.5276 | 0.6632 |
| KS Statistic | 0.3986 | 0.5137 |
| PSI (train-test) | 0.0002 | 0.0002 |
| Interpretability | High (coefficients) | Requires SHAP |

### Evaluation Methodology
- 80/20 stratified train/test split
- 5-fold stratified cross-validation on training set
- Credit-risk-specific metrics (Gini, KS) alongside standard ML metrics

---

## Known Limitations

1. **Thin-file applicants** — Limited bureau history reduces model confidence for applicants with few or no prior credit lines
2. **Geographic specificity** — Trained on a single emerging-market population; may not generalise to other regions without recalibration
3. **Temporal stability** — No time-based validation (no date column available); PSI monitoring is essential in production
4. **Feature availability** — External data sources (`EXT_SOURCE_*`) have variable missingness; model performance degrades when these are unavailable
5. **Synthetic data gap** — No SME/commercial lending data; model is consumer-only

---

## Fairness Analysis

Performance was evaluated across demographic proxy segments:

| Segment | Sub-group | N | AUC (Champion) | AUC (Challenger) |
|---------|-----------|---|----------------|-----------------|
| Gender | Male | 20,940 | 0.753 | 0.781 |
| Gender | Female | 40,561 | 0.753 | 0.776 |
| Age | 18-30 | 8,905 | 0.719 | 0.753 |
| Age | 31-45 | 24,744 | 0.764 | 0.789 |
| Age | 46-60 | 20,666 | 0.753 | 0.777 |
| Age | 60+ | 7,188 | 0.717 | 0.740 |
| Education | Higher education | 15,061 | 0.753 | 0.779 |
| Education | Secondary / secondary special | 43,623 | 0.754 | 0.777 |
| Education | Incomplete higher | 1,988 | 0.738 | 0.762 |
| Education | Lower secondary | 791 | 0.724 | 0.762 |

> Maximum AUC gap between gender segments: 0.005 (Champion), 0.005 (Challenger) -- within tolerance.
> Maximum AUC gap between age segments: 0.047 (Champion), 0.049 (Challenger) -- within tolerance.

### Fair Lending Considerations (ECOA / Reg B)
- Gender and age are **not** used as model inputs
- SHAP-based adverse action reasons are available for every declined application
- Region-based performance monitoring in place to detect proxy discrimination

---

## Monitoring Plan

| Check | Frequency | Threshold | Action |
|-------|-----------|-----------|--------|
| PSI (score distribution) | Monthly | > 0.10 | Investigate; > 0.25 → retrain |
| Gini coefficient | Monthly | < 0.50 | Investigate; < 0.40 → retrain |
| Default rate shift | Monthly | ±2pp from expected | Recalibrate |
| Segment AUC | Quarterly | Gap > 0.05 | Fairness review |
| Feature PSI | Quarterly | Any feature > 0.25 | Feature review |

---

## Regulatory Considerations

- **Explainability:** Champion scorecard is fully transparent (WoE + coefficients). Challenger uses SHAP for per-decision explanations.
- **Adverse action notices:** Top 3 SHAP factors provided via API for every scored application.
- **Model governance:** Champion/challenger framework allows parallel evaluation before full deployment.
- **Audit trail:** MLflow tracks all experiments, hyperparameters, and metrics.

---

## Deployment

- **Champion:** Primary model in production
- **Challenger:** 20% shadow deployment recommended (90-day parallel run)
- **Infrastructure:** FastAPI container on Render / AWS ECS
- **Review cadence:** Quarterly model performance review; annual full revalidation

---

## Features Used (Top by Information Value)

| Rank | Feature | IV | Source |
|------|---------|-----|--------|
| 1 | EXT_SOURCE_2 | 0.68 | application_train |
| 2 | EXT_SOURCE_3 | 0.55 | application_train |
| 3 | EXT_SOURCE_1 | 0.32 | application_train |
| 4 | DAYS_BIRTH (AGE_YEARS) | 0.18 | application_train |
| 5 | DAYS_EMPLOYED (EMPLOYMENT_YEARS) | 0.14 | application_train |
| 6 | BUREAU_OVERDUE_RATIO | 0.12 | bureau_agg.sql |
| 7 | LOAN_INCOME_RATIO | 0.11 | derived |
| 8 | ANNUITY_INCOME_RATIO | 0.10 | derived |
| 9 | PREV_APP_APPROVAL_RATE | 0.09 | previous_app_agg.sql |
| 10 | CC_UTILIZATION_MEAN | 0.08 | credit_card_agg.sql |
| 11 | INSTALLMENT_DPD_RATE | 0.07 | installment_features.sql |
| 12 | CREDIT_GOODS_RATIO | 0.06 | derived |
| 13 | ACTIVE_BUREAU_LOANS | 0.05 | bureau_agg.sql |
| 14 | POS_DPD_RATE | 0.04 | pos_cash_agg.sql |
| 15 | AMT_GOODS_PRICE | 0.03 | application_train |

> IV values from actual WoE binning results. Features with IV < 0.02 excluded (73 of 181 retained).

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-28 | 1.0-rc1 | Initial model development. Champion (LR+WoE) and Challenger (XGBoost) trained on full 307K dataset. Baseline metrics established. |
| 2026-02-28 | 1.0-rc2 | Added Optuna hyperparameter tuning for XGBoost. SHAP adverse action reasons added to API. Fairness audit across gender/age segments completed. |
| 2026-02-28 | 1.0 | Production release. PSI monitoring deployed. Streamlit dashboard live. Docker images pushed to Docker Hub. API deployed on Render. |
| 2026-03-01 | 2.0.0 | Added 13 bureau_balance features and 20+ derived features (EXT_SOURCE interactions, time-based features, credit ratios). Feature count increased from 199 to 232 columns (230 features). Champion upgraded with StandardScaler + L2 regularization tuning. Challenger upgraded with Optuna 15-trial hyperparameter tuning. Champion AUC 0.7566 -> 0.7638; Challenger AUC 0.7823 -> 0.8316. |

---

## Contact

| Role | Name |
|------|------|
| Model developer | Eric Kimutai |
| Model validator | Pending |
| Business owner | Pending |
| Last reviewed | 2026-03 |
