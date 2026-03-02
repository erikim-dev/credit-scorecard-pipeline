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
    MIN(DAYS_DECISION)                                   AS earliest_decision_days,

    -- ── Variability ─────────────────────────────────────────
    STDDEV(AMT_APPLICATION)                              AS std_application_amount,
    STDDEV(AMT_CREDIT)                                   AS std_prev_credit_amount,

    -- ── Recency windows ─────────────────────────────────────
    SUM(CASE WHEN DAYS_DECISION >= -365 THEN 1 ELSE 0 END) AS prev_apps_last_12m,
    SUM(CASE WHEN DAYS_DECISION >= -365 AND NAME_CONTRACT_STATUS = 'Refused'
             THEN 1 ELSE 0 END)                          AS refusals_last_12m,
    SUM(CASE WHEN DAYS_DECISION >= -365 AND NAME_CONTRACT_STATUS = 'Approved'
             THEN 1 ELSE 0 END)                          AS approvals_last_12m,

    -- ── Recent approval trend ───────────────────────────────
    CASE
        WHEN SUM(CASE WHEN DAYS_DECISION >= -365 THEN 1 ELSE 0 END) > 0
        THEN SUM(CASE WHEN DAYS_DECISION >= -365 AND NAME_CONTRACT_STATUS = 'Approved'
                      THEN 1 ELSE 0 END) * 1.0
             / SUM(CASE WHEN DAYS_DECISION >= -365 THEN 1 ELSE 0 END)
        ELSE NULL
    END                                                  AS approval_rate_last_12m,

    -- ── Most recent application outcome ─────────────────────
    CASE
        WHEN MAX(DAYS_DECISION) IS NOT NULL
        THEN (SELECT pa2.NAME_CONTRACT_STATUS
              FROM previous_application pa2
              WHERE pa2.SK_ID_CURR = previous_application.SK_ID_CURR
              ORDER BY pa2.DAYS_DECISION DESC
              LIMIT 1)
        ELSE NULL
    END                                                  AS latest_app_status

FROM previous_application
GROUP BY SK_ID_CURR
