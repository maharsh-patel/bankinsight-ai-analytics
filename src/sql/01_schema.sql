-- ============================================================
-- BankInsight AI — SQL Analytics Layer
-- Script : 01_schema.sql
-- Purpose: Define the campaign_contacts table schema.
-- DB     : data/db/bankinsight.db  (SQLite 3)
-- Note   : 'balance' is NOT present in the UCI Bank Marketing
--          Full dataset (v2). That column exists only in the
--          older reduced dataset (v1). This schema reflects the
--          actual cleaned dataset from Phase 2.
-- ============================================================

DROP TABLE IF EXISTS campaign_contacts;

CREATE TABLE campaign_contacts (
    -- ── Row identifier ──────────────────────────────────────
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,

    -- ── Client demographics ──────────────────────────────────
    age                  INTEGER     NOT NULL,   -- client age in years
    job                  TEXT        NOT NULL,   -- occupation type
    marital              TEXT        NOT NULL,   -- marital status
    education            TEXT        NOT NULL,   -- education level
    credit_default       TEXT        NOT NULL,   -- has credit in default? yes/no/unknown
    housing              TEXT        NOT NULL,   -- has housing loan? yes/no/unknown
    loan                 TEXT        NOT NULL,   -- has personal loan? yes/no/unknown

    -- ── Contact information ───────────────────────────────────
    contact              TEXT        NOT NULL,   -- communication type: cellular/telephone
    month                TEXT        NOT NULL,   -- last contact month (3-letter lowercase)
    day_of_week          TEXT        NOT NULL,   -- last contact day (mon/tue/wed/thu/fri)
    duration             INTEGER     NOT NULL,   -- last contact duration in seconds (>0)

    -- ── Campaign data ─────────────────────────────────────────
    campaign             INTEGER     NOT NULL,   -- contacts this campaign (capped at 99th pct)
    pdays                INTEGER     NOT NULL,   -- days since last prev contact (999=none)
    previous             INTEGER     NOT NULL,   -- contacts before this campaign
    poutcome             TEXT        NOT NULL,   -- previous campaign outcome

    -- ── Socio-economic context ────────────────────────────────
    emp_var_rate         REAL        NOT NULL,   -- employment variation rate (quarterly)
    cons_price_idx       REAL        NOT NULL,   -- consumer price index (monthly)
    cons_conf_idx        REAL        NOT NULL,   -- consumer confidence index (monthly)
    euribor3m            REAL        NOT NULL,   -- euribor 3-month rate (daily)
    nr_employed          REAL        NOT NULL,   -- number of employees (quarterly)

    -- ── Target & engineered flags ─────────────────────────────
    y                    TEXT        NOT NULL,   -- subscribed term deposit? yes/no
    y_encoded            INTEGER     NOT NULL,   -- binary: 1=yes, 0=no
    previously_contacted INTEGER     NOT NULL    -- binary flag: pdays != 999
);

-- Indexes for common GROUP BY / WHERE columns
CREATE INDEX IF NOT EXISTS idx_job        ON campaign_contacts(job);
CREATE INDEX IF NOT EXISTS idx_education  ON campaign_contacts(education);
CREATE INDEX IF NOT EXISTS idx_marital    ON campaign_contacts(marital);
CREATE INDEX IF NOT EXISTS idx_contact    ON campaign_contacts(contact);
CREATE INDEX IF NOT EXISTS idx_poutcome   ON campaign_contacts(poutcome);
CREATE INDEX IF NOT EXISTS idx_y_encoded  ON campaign_contacts(y_encoded);
CREATE INDEX IF NOT EXISTS idx_month      ON campaign_contacts(month);
