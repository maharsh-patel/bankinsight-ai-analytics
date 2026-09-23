-- ============================================================
-- BankInsight AI — Q08: Conversion by Previous Campaign Outcome
-- Business question: Is a client's prior campaign outcome
--   predictive of current subscription?
-- Technique: GROUP BY poutcome + previously_contacted flag +
--            percentage-of-total window function
-- ============================================================

WITH poutcome_stats AS (
    SELECT
        poutcome,
        previously_contacted,
        COUNT(*)                                            AS total_clients,
        SUM(y_encoded)                                      AS subscribed,
        ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct,
        ROUND(AVG(previous), 2)                             AS avg_prior_contacts
    FROM campaign_contacts
    GROUP BY poutcome, previously_contacted
)
SELECT
    poutcome,
    previously_contacted,
    total_clients,
    subscribed,
    conv_rate_pct,
    avg_prior_contacts,
    ROUND(100.0 * total_clients / SUM(total_clients) OVER (), 2) AS pct_of_dataset,
    -- Index: how much better/worse than overall average?
    ROUND(conv_rate_pct / AVG(conv_rate_pct) OVER (), 2)         AS conv_index
FROM poutcome_stats
ORDER BY conv_rate_pct DESC;
