-- ============================================================
-- BankInsight AI — Q07: Conversion by Campaign Contact Count
-- Business question: Does contacting a client more times in
--   one campaign improve or diminish subscription rates?
-- Technique: GROUP BY campaign count + running totals via CTE
-- ============================================================

WITH campaign_stats AS (
    SELECT
        campaign                                            AS contact_count,
        COUNT(*)                                            AS total_clients,
        SUM(y_encoded)                                      AS subscribed,
        ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct
    FROM campaign_contacts
    GROUP BY campaign
),
cumulative AS (
    SELECT
        contact_count,
        total_clients,
        subscribed,
        conv_rate_pct,
        SUM(total_clients) OVER (ORDER BY contact_count
                                  ROWS BETWEEN UNBOUNDED PRECEDING
                                  AND CURRENT ROW)          AS cumulative_clients,
        ROUND(100.0 *
              SUM(total_clients) OVER (ORDER BY contact_count
                                        ROWS BETWEEN UNBOUNDED PRECEDING
                                        AND CURRENT ROW)
              / SUM(total_clients) OVER (), 1)              AS pct_clients_reached_by_this_count
    FROM campaign_stats
)
SELECT
    contact_count,
    total_clients,
    subscribed,
    conv_rate_pct,
    pct_clients_reached_by_this_count,
    CASE
        WHEN contact_count = 1           THEN 'FIRST CONTACT'
        WHEN contact_count BETWEEN 2 AND 3 THEN 'OPTIMAL ZONE'
        WHEN contact_count BETWEEN 4 AND 6 THEN 'DIMINISHING RETURNS'
        ELSE 'OVER-CONTACTED'
    END AS contact_zone
FROM cumulative
ORDER BY contact_count;
