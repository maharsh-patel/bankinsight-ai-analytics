-- ============================================================
-- BankInsight AI — Q09: Campaign Duration & Contact Statistics
-- Business question: How do call duration and contact patterns
--   differ between subscribers and non-subscribers?
-- Note: 'duration' is included here for descriptive/analytical
--   purposes only. It is excluded from predictive models (leakage).
-- Technique: CASE-based segmentation + aggregate stats
-- ============================================================

-- Part A: Duration statistics by outcome
SELECT
    y                                                   AS outcome,
    COUNT(*)                                            AS clients,
    ROUND(AVG(duration), 0)                             AS avg_duration_sec,
    ROUND(AVG(duration) / 60.0, 1)                      AS avg_duration_min,
    MIN(duration)                                       AS min_sec,
    MAX(duration)                                       AS max_sec,
    -- Approximate median using percentile-style NTILE
    ROUND(AVG(CASE WHEN duration_rank BETWEEN 0.49 AND 0.51
                   THEN duration END), 0)               AS approx_median_sec,
    ROUND(AVG(campaign), 2)                             AS avg_contacts_this_campaign,
    ROUND(AVG(previous), 2)                             AS avg_prior_contacts
FROM (
    SELECT
        y, duration, campaign, previous,
        1.0 * ROW_NUMBER() OVER (PARTITION BY y ORDER BY duration)
            / COUNT(*) OVER (PARTITION BY y)             AS duration_rank
    FROM campaign_contacts
) sub
GROUP BY y;

-- Part B: Duration bucket analysis
SELECT
    CASE
        WHEN duration <  60  THEN '< 1 min'
        WHEN duration < 180  THEN '1-3 min'
        WHEN duration < 300  THEN '3-5 min'
        WHEN duration < 600  THEN '5-10 min'
        ELSE '> 10 min'
    END AS duration_bucket,
    COUNT(*)                                            AS total_clients,
    SUM(y_encoded)                                      AS subscribed,
    ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_calls
FROM campaign_contacts
GROUP BY duration_bucket
ORDER BY MIN(duration);
