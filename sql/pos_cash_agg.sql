-- ============================================================
-- POS / Cash Balance Aggregation: Loan lifecycle behaviour
-- Source: POS_CASH_balance.csv  →  previous_application.csv
-- Grain: one row per SK_ID_CURR
-- ============================================================

SELECT
    prev.SK_ID_CURR,

    -- Volume
    COUNT(DISTINCT pc.SK_ID_PREV)                        AS pos_loan_count,
    COUNT(*)                                             AS pos_balance_records,

    -- Contract status distribution
    SUM(CASE WHEN pc.NAME_CONTRACT_STATUS = 'Completed'
             THEN 1 ELSE 0 END)                          AS completed_contracts,
    SUM(CASE WHEN pc.NAME_CONTRACT_STATUS = 'Active'
             THEN 1 ELSE 0 END)                          AS active_contracts,

    -- DPD (Days Past Due) analysis
    AVG(pc.SK_DPD)                                       AS avg_dpd,
    MAX(pc.SK_DPD)                                       AS max_dpd,
    SUM(CASE WHEN pc.SK_DPD > 0 THEN 1 ELSE 0 END)      AS months_with_dpd,

    -- DPD tolerance (internal threshold)
    AVG(pc.SK_DPD_DEF)                                   AS avg_dpd_def,
    MAX(pc.SK_DPD_DEF)                                   AS max_dpd_def,

    -- Remaining instalments
    AVG(pc.CNT_INSTALMENT_FUTURE)                        AS avg_remaining_instalments,
    MIN(pc.CNT_INSTALMENT_FUTURE)                        AS min_remaining_instalments,

    -- Contract completion rate
    CASE
        WHEN COUNT(DISTINCT pc.SK_ID_PREV) > 0
        THEN SUM(CASE WHEN pc.NAME_CONTRACT_STATUS = 'Completed'
                      THEN 1 ELSE 0 END) * 1.0
             / COUNT(DISTINCT pc.SK_ID_PREV)
        ELSE NULL
    END                                                  AS contract_completion_rate,

    -- DPD rate (fraction of months with any DPD)
    CASE
        WHEN COUNT(*) > 0
        THEN SUM(CASE WHEN pc.SK_DPD > 0 THEN 1 ELSE 0 END) * 1.0
             / COUNT(*)
        ELSE 0
    END                                                  AS dpd_rate

FROM POS_CASH_balance pc
JOIN previous_application prev
    ON pc.SK_ID_PREV = prev.SK_ID_PREV
GROUP BY prev.SK_ID_CURR
