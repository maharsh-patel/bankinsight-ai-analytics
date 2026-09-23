-- ============================================================
-- BankInsight AI — Q05: Conversion by Housing & Loan Status
-- Business question: Do existing debt obligations reduce the
--   likelihood of subscribing to a term deposit?
-- Technique: CROSS-dimension analysis via CTE + UNION
-- ============================================================

-- Housing loan breakdown
WITH housing_stats AS (
    SELECT
        'housing'                                           AS dimension,
        housing                                             AS category,
        COUNT(*)                                            AS total_clients,
        SUM(y_encoded)                                      AS subscribed,
        ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct
    FROM campaign_contacts
    GROUP BY housing
),
-- Personal loan breakdown
loan_stats AS (
    SELECT
        'loan'                                              AS dimension,
        loan                                                AS category,
        COUNT(*)                                            AS total_clients,
        SUM(y_encoded)                                      AS subscribed,
        ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct
    FROM campaign_contacts
    GROUP BY loan
),
-- Credit default breakdown
default_stats AS (
    SELECT
        'credit_default'                                    AS dimension,
        credit_default                                      AS category,
        COUNT(*)                                            AS total_clients,
        SUM(y_encoded)                                      AS subscribed,
        ROUND(100.0 * SUM(y_encoded) / COUNT(*), 2)        AS conv_rate_pct
    FROM campaign_contacts
    GROUP BY credit_default
)
SELECT * FROM housing_stats
UNION ALL
SELECT * FROM loan_stats
UNION ALL
SELECT * FROM default_stats
ORDER BY dimension, conv_rate_pct DESC;
