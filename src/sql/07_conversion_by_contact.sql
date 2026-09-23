-- ============================================================
-- BankInsight AI — Q06: Conversion by Contact Method
-- Business question: Does cellular vs telephone contact affect
--   subscription rates?
-- ============================================================

SELECT
    contact,
    COUNT(*)                                            AS total_clients,
    SUM(y_encoded)                                      AS subscribed,
    ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_dataset,
    ROUND(AVG(duration), 0)                             AS avg_call_duration_sec,
    ROUND(AVG(campaign), 2)                             AS avg_contacts
FROM campaign_contacts
GROUP BY contact
ORDER BY conv_rate_pct DESC;
