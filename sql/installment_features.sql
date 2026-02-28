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
    END                                                  AS payment_to_due_ratio

FROM installments_payments ip
JOIN previous_application prev
    ON ip.SK_ID_PREV = prev.SK_ID_PREV
GROUP BY prev.SK_ID_CURR
