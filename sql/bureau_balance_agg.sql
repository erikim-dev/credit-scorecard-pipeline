-- bureau_balance_agg.sql
-- Aggregate monthly bureau balance snapshots into per-applicant features.
-- Captures: payment discipline, history depth, and delinquency trajectory.

WITH balance_stats AS (
    SELECT
        b.SK_ID_CURR,
        COUNT(*)                                    AS bb_total_records,
        COUNT(DISTINCT bb.SK_ID_BUREAU)             AS bb_distinct_accounts,
        COUNT(DISTINCT bb.MONTHS_BALANCE)           AS bb_months_covered,
        MIN(bb.MONTHS_BALANCE)                      AS bb_earliest_month,
        MAX(bb.MONTHS_BALANCE)                       AS bb_latest_month,

        -- DPD-related statuses (1-5 = days past due buckets, C = closed, X = unknown)
        SUM(CASE WHEN bb.STATUS IN ('1','2','3','4','5') THEN 1 ELSE 0 END)
                                                    AS bb_dpd_count,
        SUM(CASE WHEN bb.STATUS = 'C' THEN 1 ELSE 0 END)
                                                    AS bb_closed_count,
        SUM(CASE WHEN bb.STATUS = '0' THEN 1 ELSE 0 END)
                                                    AS bb_current_count,
        SUM(CASE WHEN bb.STATUS = 'X' THEN 1 ELSE 0 END)
                                                    AS bb_unknown_count,

        -- Worst delinquency bucket ever observed
        MAX(CASE
            WHEN bb.STATUS = '5' THEN 5
            WHEN bb.STATUS = '4' THEN 4
            WHEN bb.STATUS = '3' THEN 3
            WHEN bb.STATUS = '2' THEN 2
            WHEN bb.STATUS = '1' THEN 1
            ELSE 0
        END)                                        AS bb_worst_dpd_bucket

    FROM bureau_balance bb
    INNER JOIN bureau b ON bb.SK_ID_BUREAU = b.SK_ID_BUREAU
    GROUP BY b.SK_ID_CURR
)
SELECT
    SK_ID_CURR,
    bb_total_records,
    bb_distinct_accounts,
    bb_months_covered,
    bb_earliest_month,
    bb_latest_month,
    bb_dpd_count,
    bb_closed_count,
    bb_current_count,
    bb_unknown_count,
    bb_worst_dpd_bucket,

    -- Ratios
    ROUND(bb_dpd_count * 1.0 / NULLIF(bb_total_records, 0), 4)
                                                    AS bb_dpd_ratio,
    ROUND(bb_closed_count * 1.0 / NULLIF(bb_total_records, 0), 4)
                                                    AS bb_closed_ratio,
    ROUND(bb_current_count * 1.0 / NULLIF(bb_total_records, 0), 4)
                                                    AS bb_current_ratio
FROM balance_stats
