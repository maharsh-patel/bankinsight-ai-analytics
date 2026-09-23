-- ============================================================
-- BankInsight AI — Q01: Dataset Overview
-- Business question: What is the total size of the dataset
--   and the overall campaign subscription rate?
-- ============================================================

SELECT
    COUNT(*)                                          AS total_clients,
    SUM(y_encoded)                                    AS total_subscribed,
    COUNT(*) - SUM(y_encoded)                         AS total_not_subscribed,
    ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)       AS subscription_rate_pct,
    ROUND(100.0 * (COUNT(*) - SUM(y_encoded))
          / COUNT(*), 2)                              AS non_subscription_rate_pct,
    ROUND(1.0 * (COUNT(*) - SUM(y_encoded))
          / NULLIF(SUM(y_encoded), 0), 1)             AS imbalance_ratio_no_per_yes
FROM campaign_contacts;
