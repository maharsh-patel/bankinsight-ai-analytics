-- ============================================================
-- BankInsight AI — Q03: Conversion by Education Level
-- Business question: Does education level affect subscription?
-- Technique: GROUP BY + CASE for ordinal ordering + window rank
-- ============================================================

WITH edu_stats AS (
    SELECT
        education,
        COUNT(*)                                            AS total_clients,
        SUM(y_encoded)                                      AS subscribed,
        ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct,
        CASE education
            WHEN 'illiterate'          THEN 1
            WHEN 'basic.4y'            THEN 2
            WHEN 'basic.6y'            THEN 3
            WHEN 'basic.9y'            THEN 4
            WHEN 'high.school'         THEN 5
            WHEN 'professional.course' THEN 6
            WHEN 'university.degree'   THEN 7
            WHEN 'unknown'             THEN 8
            ELSE 9
        END AS edu_order
    FROM campaign_contacts
    GROUP BY education
)
SELECT
    edu_order,
    education,
    total_clients,
    subscribed,
    conv_rate_pct,
    ROUND(100.0 * total_clients / SUM(total_clients) OVER (), 2) AS pct_of_dataset,
    ROUND(conv_rate_pct - AVG(conv_rate_pct) OVER (), 2)         AS diff_from_mean_pp
FROM edu_stats
ORDER BY edu_order;
