-- ============================================================
-- BankInsight AI — Q04: Conversion by Marital Status
-- Business question: Does marital status influence subscription?
-- ============================================================

SELECT
    marital,
    COUNT(*)                                            AS total_clients,
    SUM(y_encoded)                                      AS subscribed,
    ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_dataset,
    ROUND(AVG(age), 1)                                  AS avg_age,
    ROUND(AVG(campaign), 2)                             AS avg_contacts
FROM campaign_contacts
GROUP BY marital
ORDER BY conv_rate_pct DESC;
