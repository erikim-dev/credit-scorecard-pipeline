-- ============================================================
-- Previous Application Aggregation: Loan application history
-- Source: previous_application.csv
-- Grain: one row per SK_ID_CURR
-- ============================================================

SELECT
    SK_ID_CURR,

    -- Application counts
    COUNT(*)                                             AS prev_app_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved'  THEN 1 ELSE 0 END) AS approved_apps,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'   THEN 1 ELSE 0 END) AS refused_apps,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled'  THEN 1 ELSE 0 END) AS canceled_apps,

    -- Approval rate
    CASE
        WHEN COUNT(*) > 0
        THEN SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved'
                      THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
        ELSE NULL
    END                                                  AS approval_rate,

    -- Amount analysis
    AVG(AMT_APPLICATION)                                 AS avg_application_amount,
    AVG(AMT_CREDIT)                                      AS avg_credit_amount,
    AVG(AMT_ANNUITY)                                     AS avg_annuity,

    -- Credit-to-application ratio (how much was actually granted vs requested)
    CASE
        WHEN SUM(AMT_APPLICATION) > 0
        THEN SUM(AMT_CREDIT) / SUM(AMT_APPLICATION)
        ELSE NULL
    END                                                  AS credit_to_application_ratio,

    -- Annuity-to-credit ratio (burden indicator)
    CASE
        WHEN SUM(AMT_CREDIT) > 0
        THEN SUM(AMT_ANNUITY) / SUM(AMT_CREDIT)
        ELSE NULL
    END                                                  AS annuity_to_credit_ratio,

    -- Contract types
    SUM(CASE WHEN NAME_CONTRACT_TYPE = 'Cash loans'      THEN 1 ELSE 0 END) AS cash_loans,
    SUM(CASE WHEN NAME_CONTRACT_TYPE = 'Consumer loans'  THEN 1 ELSE 0 END) AS consumer_loans,
    SUM(CASE WHEN NAME_CONTRACT_TYPE = 'Revolving loans' THEN 1 ELSE 0 END) AS revolving_loans,

    -- Timing
    MAX(DAYS_DECISION)                                   AS latest_decision_days,
    MIN(DAYS_DECISION)                                   AS earliest_decision_days

FROM previous_application
GROUP BY SK_ID_CURR
