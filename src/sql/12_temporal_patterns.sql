-- ============================================================
-- BankInsight AI — Q11: Temporal Conversion Patterns
-- Business question: Which months and days-of-week have the
--   highest subscription conversion rates and call volumes?
-- Technique: Multi-level GROUP BY + window percent rank
-- ============================================================

-- Monthly analysis
SELECT
    'month'                                             AS time_dimension,
    month                                               AS period,
    COUNT(*)                                            AS total_calls,
    SUM(y_encoded)                                      AS subscribed,
    ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_calls,
    ROUND(100.0 * PERCENT_RANK() OVER (
          ORDER BY ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)), 1) AS conv_percent_rank
FROM campaign_contacts
GROUP BY month

UNION ALL

-- Day-of-week analysis
SELECT
    'day_of_week'                                       AS time_dimension,
    day_of_week                                         AS period,
    COUNT(*)                                            AS total_calls,
    SUM(y_encoded)                                      AS subscribed,
    ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_calls,
    ROUND(100.0 * PERCENT_RANK() OVER (
          ORDER BY ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)), 1) AS conv_percent_rank
FROM campaign_contacts
GROUP BY day_of_week

ORDER BY time_dimension, conv_rate_pct DESC;
