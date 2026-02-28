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
    END                                                  AS debt_to_credit_ratio

FROM bureau
GROUP BY SK_ID_CURR
