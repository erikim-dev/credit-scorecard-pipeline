-- ============================================================
-- Bureau Aggregation: Credit Bureau history per applicant
-- Source: bureau.csv
-- Grain: one row per SK_ID_CURR
-- ============================================================

SELECT
    SK_ID_CURR,

    -- Volume
    COUNT(*)                                             AS bureau_loan_count,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active'  THEN 1 ELSE 0 END) AS active_credits,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Closed'  THEN 1 ELSE 0 END) AS closed_credits,

    -- Amounts
    AVG(AMT_CREDIT_SUM)                                  AS avg_credit_amount,
    SUM(AMT_CREDIT_SUM)                                  AS total_credit_amount,
    SUM(AMT_CREDIT_SUM_DEBT)                             AS total_debt,
    AVG(AMT_CREDIT_SUM_DEBT)                             AS avg_debt,
    SUM(AMT_CREDIT_SUM_OVERDUE)                          AS total_overdue_amount,

    -- Overdue flags
    SUM(CASE WHEN AMT_CREDIT_SUM_OVERDUE > 0 THEN 1 ELSE 0 END) AS overdue_count,

    -- Timing
    MIN(DAYS_CREDIT)                                     AS earliest_bureau_days,
    MAX(DAYS_CREDIT)                                     AS latest_bureau_days,
    MAX(DAYS_CREDIT_ENDDATE)                             AS latest_credit_enddate,
    AVG(DAYS_CREDIT_UPDATE)                              AS avg_days_since_update,

    -- Ratios
    CASE
        WHEN SUM(AMT_CREDIT_SUM) > 0
        THEN SUM(AMT_CREDIT_SUM_DEBT) / SUM(AMT_CREDIT_SUM)
        ELSE NULL
    END                                                  AS debt_to_credit_ratio,

    -- ── Variability ─────────────────────────────────────────
    STDDEV(AMT_CREDIT_SUM)                               AS std_credit_amount,
    STDDEV(AMT_CREDIT_SUM_DEBT)                          AS std_debt_amount,
    STDDEV(DAYS_CREDIT)                                  AS std_bureau_days,

    -- ── Recency windows ─────────────────────────────────────
    SUM(CASE WHEN DAYS_CREDIT >= -365  THEN 1 ELSE 0 END)  AS bureau_loans_last_12m,
    SUM(CASE WHEN DAYS_CREDIT >= -730  THEN 1 ELSE 0 END)  AS bureau_loans_last_24m,
    SUM(CASE WHEN DAYS_CREDIT >= -365 AND CREDIT_ACTIVE = 'Active'
             THEN 1 ELSE 0 END)                             AS active_loans_last_12m,
    AVG(CASE WHEN DAYS_CREDIT >= -365
             THEN AMT_CREDIT_SUM_DEBT END)                  AS avg_debt_last_12m,
    SUM(CASE WHEN DAYS_CREDIT >= -365
             THEN AMT_CREDIT_SUM_OVERDUE ELSE 0 END)       AS overdue_amount_last_12m,

    -- ── Credit type diversity ───────────────────────────────
    COUNT(DISTINCT CREDIT_TYPE)                           AS bureau_credit_type_count,

    -- ── Prolongation / annuity behaviour ────────────────────
    AVG(AMT_ANNUITY)                                     AS avg_bureau_annuity,
    SUM(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END) AS bureau_day_overdue_count,
    MAX(CREDIT_DAY_OVERDUE)                              AS max_bureau_day_overdue,

    -- ── Credit freshness ratio (recent / total) ─────────────
    CASE
        WHEN COUNT(*) > 0
        THEN SUM(CASE WHEN DAYS_CREDIT >= -365 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
        ELSE 0
    END                                                  AS bureau_recent_ratio

FROM bureau
GROUP BY SK_ID_CURR
