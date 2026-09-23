-- ============================================================
-- BankInsight AI — Q10: Customer Segment Analysis
-- Business question: Can we identify distinct customer
--   segments based on age, job, education, and loan status
--   and rank them by conversion rate?
-- Technique: Multi-dimension CTE + window ranking
-- ============================================================

-- Segment: Age band × Job tier × Previously Contacted
WITH age_banded AS (
    SELECT
        *,
        CASE
            WHEN age BETWEEN 17 AND 25 THEN '18-25'
            WHEN age BETWEEN 26 AND 35 THEN '26-35'
            WHEN age BETWEEN 36 AND 45 THEN '36-45'
            WHEN age BETWEEN 46 AND 55 THEN '46-55'
            WHEN age BETWEEN 56 AND 65 THEN '56-65'
            ELSE '66+'
        END AS age_band,
        CASE
            WHEN job IN ('student','retired')             THEN 'HIGH_VALUE'
            WHEN job IN ('management','admin.','technician') THEN 'MID_VALUE'
            ELSE 'LOWER_VALUE'
        END AS job_tier
    FROM campaign_contacts
),
segment_stats AS (
    SELECT
        age_band,
        job_tier,
        previously_contacted,
        COUNT(*)                                             AS clients,
        SUM(y_encoded)                                       AS subscribed,
        ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)         AS conv_rate_pct,
        ROUND(AVG(age), 1)                                   AS avg_age,
        ROUND(AVG(campaign), 2)                              AS avg_contacts
    FROM age_banded
    GROUP BY age_band, job_tier, previously_contacted
    HAVING COUNT(*) >= 30    -- filter out thin segments
)
SELECT
    age_band,
    job_tier,
    previously_contacted,
    clients,
    subscribed,
    conv_rate_pct,
    avg_age,
    avg_contacts,
    RANK() OVER (ORDER BY conv_rate_pct DESC) AS segment_rank,
    ROUND(100.0 * clients / SUM(clients) OVER (), 2) AS pct_of_dataset
FROM segment_stats
ORDER BY segment_rank
LIMIT 20;
