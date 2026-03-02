-- ============================================================
-- Credit Card Balance Aggregation: Card usage behaviour
-- Source: credit_card_balance.csv  →  previous_application.csv
-- Grain: one row per SK_ID_CURR
-- ============================================================

SELECT
    prev.SK_ID_CURR,

    -- Volume
    COUNT(DISTINCT cc.SK_ID_PREV)                        AS cc_count,
    COUNT(*)                                             AS cc_balance_records,

    -- Balance & Limit
    AVG(cc.AMT_BALANCE)                                  AS avg_cc_balance,
    MAX(cc.AMT_BALANCE)                                  AS max_cc_balance,
    AVG(cc.AMT_CREDIT_LIMIT_ACTUAL)                      AS avg_credit_limit,

    -- Utilisation
    CASE
        WHEN AVG(cc.AMT_CREDIT_LIMIT_ACTUAL) > 0
        THEN AVG(cc.AMT_BALANCE) / AVG(cc.AMT_CREDIT_LIMIT_ACTUAL)
        ELSE NULL
    END                                                  AS avg_utilisation_ratio,

    -- Drawing behaviour
    AVG(cc.AMT_DRAWINGS_CURRENT)                         AS avg_drawings,
    SUM(cc.AMT_DRAWINGS_CURRENT)                         AS total_drawings,
    AVG(cc.CNT_DRAWINGS_CURRENT)                         AS avg_drawing_count,

    -- Payment behaviour
    AVG(cc.AMT_PAYMENT_CURRENT)                          AS avg_payment,
    SUM(cc.AMT_PAYMENT_CURRENT)                          AS total_payment,

    -- Minimum payment misses
    SUM(CASE WHEN cc.AMT_PAYMENT_CURRENT < cc.AMT_INST_MIN_REGULARITY
             THEN 1 ELSE 0 END)                          AS min_payment_misses,

    -- DPD
    AVG(cc.SK_DPD)                                       AS avg_cc_dpd,
    MAX(cc.SK_DPD)                                       AS max_cc_dpd,
    SUM(CASE WHEN cc.SK_DPD > 0 THEN 1 ELSE 0 END)      AS cc_months_with_dpd,

    -- ── Variability ─────────────────────────────────────────
    STDDEV(cc.AMT_BALANCE)                               AS std_cc_balance,
    STDDEV(cc.AMT_DRAWINGS_CURRENT)                      AS std_cc_drawings,

    -- ── Peak utilisation ────────────────────────────────────
    MAX(CASE WHEN cc.AMT_CREDIT_LIMIT_ACTUAL > 0
             THEN cc.AMT_BALANCE * 1.0 / cc.AMT_CREDIT_LIMIT_ACTUAL
             ELSE NULL END)                              AS max_utilisation_ratio,

    -- ── Recency windows ─────────────────────────────────────
    AVG(CASE WHEN cc.MONTHS_BALANCE >= -6
             THEN cc.AMT_BALANCE END)                    AS avg_balance_last_6m,
    AVG(CASE WHEN cc.MONTHS_BALANCE >= -12
             THEN cc.AMT_BALANCE END)                    AS avg_balance_last_12m,
    SUM(CASE WHEN cc.MONTHS_BALANCE >= -12 AND cc.SK_DPD > 0
             THEN 1 ELSE 0 END)                          AS cc_dpd_last_12m,

    -- ── Balance trend (recent minus older) ──────────────────
    AVG(CASE WHEN cc.MONTHS_BALANCE >= -6
             THEN cc.AMT_BALANCE END) -
    AVG(CASE WHEN cc.MONTHS_BALANCE < -6 AND cc.MONTHS_BALANCE >= -18
             THEN cc.AMT_BALANCE END)                    AS cc_balance_trend,

    -- ── Utilisation trend ───────────────────────────────────
    AVG(CASE WHEN cc.MONTHS_BALANCE >= -6 AND cc.AMT_CREDIT_LIMIT_ACTUAL > 0
             THEN cc.AMT_BALANCE * 1.0 / cc.AMT_CREDIT_LIMIT_ACTUAL END) -
    AVG(CASE WHEN cc.MONTHS_BALANCE < -6 AND cc.MONTHS_BALANCE >= -18
              AND cc.AMT_CREDIT_LIMIT_ACTUAL > 0
             THEN cc.AMT_BALANCE * 1.0 / cc.AMT_CREDIT_LIMIT_ACTUAL END)
                                                         AS cc_utilisation_trend

FROM credit_card_balance cc
JOIN previous_application prev
    ON cc.SK_ID_PREV = prev.SK_ID_PREV
GROUP BY prev.SK_ID_CURR
