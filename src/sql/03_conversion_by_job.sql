-- ============================================================
-- BankInsight AI — Q02: Conversion by Job Type
-- Business question: Which occupations show the highest and
--   lowest term deposit subscription rates?
-- Technique: GROUP BY + window function for rank
-- ============================================================

WITH job_stats AS (
    SELECT
        job,
        COUNT(*)                                              AS total_clients,
        SUM(y_encoded)                                        AS subscribed,
        ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)          AS conv_rate_pct
    FROM campaign_contacts
    GROUP BY job
),
ranked AS (
    SELECT
        job,
        total_clients,
        subscribed,
        conv_rate_pct,
        RANK() OVER (ORDER BY conv_rate_pct DESC)            AS rank_by_conv,
        ROUND(100.0 * total_clients /
              SUM(total_clients) OVER (), 2)                 AS pct_of_dataset
    FROM job_stats
)
SELECT
    rank_by_conv,
    job,
    total_clients,
    subscribed,
    conv_rate_pct,
    pct_of_dataset,
    CASE
        WHEN conv_rate_pct >= 20 THEN 'HIGH'
        WHEN conv_rate_pct >= 10 THEN 'AVERAGE'
        ELSE 'LOW'
    END AS conversion_tier
FROM ranked
ORDER BY rank_by_conv;
