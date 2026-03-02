-- ============================================================
-- Installment Features: Payment behaviour per applicant
-- Source: installments_payments.csv  →  previous_application.csv  →  application
-- Grain: one row per SK_ID_CURR
-- ============================================================

SELECT
    prev.SK_ID_CURR,

    -- Payment counts
    COUNT(*)                                             AS total_installments,
    SUM(CASE WHEN ip.DAYS_ENTRY_PAYMENT > ip.DAYS_INSTALMENT
             THEN 1 ELSE 0 END)                          AS late_payments,
    SUM(CASE WHEN ip.DAYS_ENTRY_PAYMENT <= ip.DAYS_INSTALMENT
             THEN 1 ELSE 0 END)                          AS on_time_payments,

    -- Late-payment ratio
    CASE
        WHEN COUNT(*) > 0
        THEN SUM(CASE WHEN ip.DAYS_ENTRY_PAYMENT > ip.DAYS_INSTALMENT
                      THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
        ELSE 0
    END                                                  AS late_payment_ratio,

    -- Days late statistics
    AVG(ip.DAYS_ENTRY_PAYMENT - ip.DAYS_INSTALMENT)     AS avg_days_late,
    MAX(ip.DAYS_ENTRY_PAYMENT - ip.DAYS_INSTALMENT)     AS max_days_late,

    -- Payment amount analysis
    AVG(ip.AMT_PAYMENT)                                  AS avg_payment_amount,
    SUM(ip.AMT_PAYMENT)                                  AS total_paid,
    AVG(ip.AMT_INSTALMENT)                               AS avg_instalment_amount,
    SUM(ip.AMT_INSTALMENT)                               AS total_due,

    -- Payment shortfall
    SUM(ip.AMT_INSTALMENT - ip.AMT_PAYMENT)              AS total_underpayment,
    AVG(ip.AMT_INSTALMENT - ip.AMT_PAYMENT)              AS avg_underpayment,

    -- Payment-to-instalment ratio (>1 = overpaying, <1 = underpaying)
    CASE
        WHEN SUM(ip.AMT_INSTALMENT) > 0
        THEN SUM(ip.AMT_PAYMENT) / SUM(ip.AMT_INSTALMENT)
        ELSE NULL
    END                                                  AS payment_to_due_ratio,

    -- ── Variability ─────────────────────────────────────────
    STDDEV(ip.AMT_PAYMENT)                               AS std_payment_amount,
    STDDEV(ip.DAYS_ENTRY_PAYMENT - ip.DAYS_INSTALMENT)   AS std_days_late,

    -- ── Recency windows ─────────────────────────────────────
    SUM(CASE WHEN ip.DAYS_INSTALMENT >= -365
              AND ip.DAYS_ENTRY_PAYMENT > ip.DAYS_INSTALMENT
             THEN 1 ELSE 0 END)                          AS late_payments_last_12m,
    SUM(CASE WHEN ip.DAYS_INSTALMENT >= -365
             THEN 1 ELSE 0 END)                          AS total_installments_last_12m,
    CASE
        WHEN SUM(CASE WHEN ip.DAYS_INSTALMENT >= -365 THEN 1 ELSE 0 END) > 0
        THEN SUM(CASE WHEN ip.DAYS_INSTALMENT >= -365
                       AND ip.DAYS_ENTRY_PAYMENT > ip.DAYS_INSTALMENT
                      THEN 1 ELSE 0 END) * 1.0
             / SUM(CASE WHEN ip.DAYS_INSTALMENT >= -365 THEN 1 ELSE 0 END)
        ELSE 0
    END                                                  AS late_ratio_last_12m,

    -- ── Trend: recent vs older late ratio ────────────────────
    CASE
        WHEN SUM(CASE WHEN ip.DAYS_INSTALMENT < -365 THEN 1 ELSE 0 END) > 0
        THEN (
            SUM(CASE WHEN ip.DAYS_INSTALMENT >= -365
                      AND ip.DAYS_ENTRY_PAYMENT > ip.DAYS_INSTALMENT
                     THEN 1 ELSE 0 END) * 1.0
            / NULLIF(SUM(CASE WHEN ip.DAYS_INSTALMENT >= -365 THEN 1 ELSE 0 END), 0)
        ) - (
            SUM(CASE WHEN ip.DAYS_INSTALMENT < -365
                      AND ip.DAYS_ENTRY_PAYMENT > ip.DAYS_INSTALMENT
                     THEN 1 ELSE 0 END) * 1.0
            / SUM(CASE WHEN ip.DAYS_INSTALMENT < -365 THEN 1 ELSE 0 END)
        )
        ELSE NULL
    END                                                  AS late_ratio_trend,

    -- ── Early payment signal ─────────────────────────────────
    SUM(CASE WHEN ip.DAYS_ENTRY_PAYMENT < ip.DAYS_INSTALMENT - 5
             THEN 1 ELSE 0 END)                          AS early_payment_count,
    AVG(CASE WHEN ip.DAYS_ENTRY_PAYMENT <= ip.DAYS_INSTALMENT
             THEN ip.DAYS_INSTALMENT - ip.DAYS_ENTRY_PAYMENT
             ELSE NULL END)                              AS avg_days_early

FROM installments_payments ip
JOIN previous_application prev
    ON ip.SK_ID_PREV = prev.SK_ID_PREV
GROUP BY prev.SK_ID_CURR
