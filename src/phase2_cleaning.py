"""
BankInsight AI - Phase 2: Data Cleaning Pipeline
=================================================
Reads  : bank_marketing.csv  (raw - never modified)
Writes : data/processed/bank_marketing_clean.csv
         reports/phase2/cleaning_report.txt
         reports/phase2/cleaning_log.csv

Cleaning decisions are fully documented in-code and in the report.
Every major step records rows_before / rows_after.
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV   = os.path.join(BASE_DIR, "bank_marketing.csv")
OUT_CSV   = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_clean.csv")
RPT_DIR   = os.path.join(BASE_DIR, "reports", "phase2")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)

# ── Cleaning log accumulator ──────────────────────────────────────────────────
_log: list[dict] = []

def _record(step: int, rule: str, decision: str, rows_before: int,
            rows_after: int, cols_affected: str, notes: str = "") -> None:
    removed = rows_before - rows_after
    _log.append({
        "step":          step,
        "rule":          rule,
        "decision":      decision,
        "rows_before":   rows_before,
        "rows_after":    rows_after,
        "rows_removed":  removed,
        "cols_affected": cols_affected,
        "notes":         notes,
    })
    flag = "(-{})".format(removed) if removed > 0 else "(no rows removed)"
    print(f"  Step {step:02d} | {rule:<45} | {rows_before:>6} -> {rows_after:>6} {flag}")


# ── Step helpers ──────────────────────────────────────────────────────────────

def step01_load(path: str) -> pd.DataFrame:
    """Load raw CSV; confirm integrity."""
    df = pd.read_csv(path)
    assert len(df) > 0, "Dataset is empty"
    assert len(df.columns) == 21, f"Unexpected col count: {len(df.columns)}"
    if len(df) != 41188:
        print(f"  Note: Custom dataset size ({len(df)} rows) loaded.")
    return df


def step02_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Remove exact duplicate rows (all 21 columns identical).
    Rationale: 12 duplicate row-pairs were confirmed in Phase 1; each pair is
    an identical record appearing twice consecutively, indicating a data-entry
    or extraction error — not a different customer.  We keep the first
    occurrence.
    """
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    _record(2, "Remove exact duplicates", "Drop 2nd copy, keep first",
            before, len(df), "all", "12 pairs -> 12 rows removed")
    return df


def step03_drop_duration_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Drop the 4 rows where duration == 0.
    Rationale: duration == 0 means the call was never made.  All 4 rows have
    y='no', but they do not represent a real campaign contact and would
    introduce noise.  4 rows is < 0.01% of the dataset — negligible loss.
    """
    before = len(df)
    df = df[df["duration"] > 0].reset_index(drop=True)
    _record(3, "Drop duration == 0", "Remove non-contact records",
            before, len(df), "duration", "Call never happened; y is always 'no'")
    return df


def step04_fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Standardise data types.
    - Categorical string columns -> Python str (already str in pandas 3, but
      explicitly cast to ensure downstream consistency).
    - Numeric columns -> correct numeric dtypes (already numeric from read_csv).
    - 'y' binary target -> 0/1 integer (y_encoded) and retain original string 'y'.
    No rows removed.
    """
    before = len(df)

    # Ensure all string-category columns are lowercase and stripped
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Confirm numeric columns
    num_cols = ["age", "duration", "campaign", "pdays", "previous",
                "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                "euribor3m", "nr.employed"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Add binary-encoded target (0 = no, 1 = yes) as a separate column
    df["y_encoded"] = (df["y"] == "yes").astype(int)

    _record(4, "Fix dtypes + add y_encoded", "Cast categories to lowercase str; add y_encoded",
            before, len(df), "all cat cols + y_encoded", "No rows removed")
    return df


def step05_handle_unknown_default(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Retain 'unknown' in 'default' as a valid category.
    Rationale: 8,597 rows (20.87%) have default='unknown'. Imputing 'no' or
    'yes' for 20% of rows would inject substantial noise.  'unknown' itself
    carries information — a bank cannot confirm whether the client has credit
    in default.  Keeping it as a three-level categorical is the safest approach
    for modelling.  No rows removed.
    """
    before = len(df)
    # Already lower-cased in step04; document the decision only.
    _record(5, "Retain 'unknown' in 'default'",
            "Keep as 3-level category (no / yes / unknown)",
            before, len(df), "default",
            "20.87% unknown — imputing would introduce > 20% noise; treat as category")
    return df


def step06_handle_unknown_education(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Retain 'unknown' in 'education' as a valid category.
    Rationale: 1,731 rows (4.2%). Education level is not easily imputed from
    other columns without external reference.  'unknown' is a legitimate
    response category in survey/CRM data.  Keeping it avoids spurious
    imputation.  No rows removed.
    """
    before = len(df)
    _record(6, "Retain 'unknown' in 'education'",
            "Keep as 8-level category including 'unknown'",
            before, len(df), "education",
            "4.2% unknown — retain as category to preserve information")
    return df


def step07_handle_unknown_housing_loan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Retain 'unknown' in 'housing' and 'loan' as valid categories.
    Rationale: 990 rows each (2.4%). These are tristate yes/no/unknown fields
    in the original UCI dataset definition. Collapsing to 'no' would misrepresent
    reality.  No rows removed.
    """
    before = len(df)
    _record(7, "Retain 'unknown' in housing + loan",
            "Keep as 3-level category (yes / no / unknown)",
            before, len(df), "housing, loan",
            "2.4% each — tristate field by design; retain unknown")
    return df


def step08_handle_unknown_job_marital(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Retain 'unknown' in 'job' (0.8%) and 'marital' (0.19%).
    Rationale: Both percentages are small, but we have no basis to impute job
    or marital status without additional data.  'unknown' is retained as a
    valid label.  No rows removed.
    """
    before = len(df)
    _record(8, "Retain 'unknown' in job + marital",
            "Keep as valid category label",
            before, len(df), "job, marital",
            "0.8% and 0.19% — too small to reliably impute; retain category")
    return df


def step09_engineer_pdays_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Add binary flag 'previously_contacted' (1 = pdays != 999).
    Rationale: 96.3% of pdays is 999 (sentinel = not previously contacted).
    The raw pdays value is near-useless as a numeric predictor because of this
    extreme skew.  The binary flag captures the meaningful information.
    pdays is RETAINED (not dropped) for downstream feature engineering in Phase 3.
    No rows removed.
    """
    before = len(df)
    df["previously_contacted"] = (df["pdays"] != 999).astype(int)
    contacted_count = df["previously_contacted"].sum()
    _record(9, "Engineer previously_contacted flag",
            "Add binary flag: pdays != 999",
            before, len(df), "pdays (new col: previously_contacted)",
            f"{contacted_count} rows with actual pdays value; 999 -> flag=0")
    return df


def step10_cap_campaign_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Cap 'campaign' at its 99th percentile (7 contacts).
    Rationale: The distribution is heavily right-skewed (max=56, skewness~4.76).
    406 rows exceed the 99th percentile.  Capping (Winsorisation) prevents
    extreme values from distorting models without removing valid records.
    No rows removed.
    """
    before = len(df)
    cap_val = int(df["campaign"].quantile(0.99))
    n_capped = int((df["campaign"] > cap_val).sum())
    df["campaign"] = df["campaign"].clip(upper=cap_val)
    _record(10, f"Cap campaign at 99th pct ({cap_val})",
            f"Winsorise values > {cap_val} to {cap_val}",
            before, len(df), "campaign",
            f"{n_capped} values capped; no rows removed")
    return df


def step11_rename_dot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Rename columns with dots to use underscores.
    Rationale: Column names like 'emp.var.rate' cause issues in most ML
    frameworks and SQL-style queries.  Pure cosmetic/compatibility fix.
    No rows removed, no data changed.
    """
    before = len(df)
    rename_map = {
        "emp.var.rate":   "emp_var_rate",
        "cons.price.idx": "cons_price_idx",
        "cons.conf.idx":  "cons_conf_idx",
        "nr.employed":    "nr_employed",
    }
    df = df.rename(columns=rename_map)
    _record(11, "Rename dot columns to underscores",
            "emp.var.rate->emp_var_rate etc.",
            before, len(df), str(list(rename_map.keys())),
            "Compatibility fix; no data changed")
    return df


def step12_validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Assert valid ranges for numeric columns; do not drop on range.
    Rationale: age > 80 has 119 rows but these are plausible (elderly retirees).
    We document the range validation and flag but do not remove.
    No rows removed.
    """
    before = len(df)
    issues = []

    if (df["age"] < 17).any() or (df["age"] > 100).any():
        issues.append(f"age out of [17,100]: {int(((df['age']<17)|(df['age']>100)).sum())} rows")

    if (df["campaign"] < 1).any():
        issues.append(f"campaign < 1: {int((df['campaign']<1).sum())} rows")

    if (df["previous"] < 0).any():
        issues.append(f"previous < 0: {int((df['previous']<0).sum())} rows")

    notes = "; ".join(issues) if issues else "All ranges valid after capping"
    _record(12, "Validate numeric ranges",
            "Assert ranges; flag only (no rows dropped)",
            before, len(df), "age, campaign, previous",
            notes)
    return df


def step13_verify_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision: Verify target column 'y' contains only 'yes'/'no'.
    No rows removed if clean.
    """
    before = len(df)
    bad = df[~df["y"].isin(["yes", "no"])]
    if len(bad) > 0:
        raise ValueError(f"Unexpected target values: {bad['y'].unique()}")
    _record(13, "Verify target labels",
            "Assert y in {yes, no}",
            before, len(df), "y",
            "Target labels are consistent: only 'yes' and 'no' present")
    return df


def step14_final_check(df: pd.DataFrame) -> pd.DataFrame:
    """Final integrity assertions before saving."""
    before = len(df)
    assert df.isnull().sum().sum() == 0, "Unexpected nulls after cleaning"
    assert df["y"].isin(["yes", "no"]).all(), "Unexpected target labels"
    assert df["y_encoded"].isin([0, 1]).all(), "y_encoded must be 0/1"
    assert df["previously_contacted"].isin([0, 1]).all(), "Flag must be 0/1"
    assert (df["duration"] > 0).all(), "duration must be > 0"
    assert (df["campaign"] >= 1).all(), "campaign must be >= 1"
    _record(14, "Final integrity assertions",
            "All assertions passed",
            before, len(df), "all",
            "Dataset is clean and consistent")
    return df


# ── Report builder ────────────────────────────────────────────────────────────

def build_cleaning_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                          log: list[dict]) -> str:
    sep = "=" * 70
    lines = [sep,
             "BankInsight AI - Phase 2: Data Cleaning Report",
             sep, ""]

    orig_rows  = len(df_raw)
    final_rows = len(df_clean)
    removed    = orig_rows - final_rows

    lines += [
        "SUMMARY",
        f"  Original rows        : {orig_rows:>7,}",
        f"  Final rows           : {final_rows:>7,}",
        f"  Records removed      : {removed:>7,}  "
        f"({removed/orig_rows*100:.3f}% of original)",
        f"  Original columns     : {len(df_raw.columns):>7}",
        f"  Final columns        : {len(df_clean.columns):>7}  "
        "(added: y_encoded, previously_contacted)",
        "",
    ]

    lines += [
        "MISSING / NULL VALUES",
        f"  Before cleaning : {int(df_raw.isnull().sum().sum())} null cells",
        f"  After cleaning  : {int(df_clean.isnull().sum().sum())} null cells",
        "  Note: 'unknown' is a valid categorical label, not a null.",
        "",
    ]

    # Duplicates
    lines += [
        "DUPLICATE RECORDS",
        f"  Before cleaning : {int(df_raw.duplicated().sum())} exact duplicates",
        f"  After cleaning  : {int(df_clean.duplicated().sum())} exact duplicates",
        "",
    ]

    # 'unknown' counts
    unknown_cols = ["job", "marital", "education", "default", "housing", "loan"]
    lines.append("'unknown' STRING LABEL COUNTS (retained as category)")
    for col in unknown_cols:
        if col in df_clean.columns:
            n   = int((df_clean[col] == "unknown").sum())
            pct = round(n / len(df_clean) * 100, 2)
            lines.append(f"  {col:<20}: {n:>5,}  ({pct:.2f}%)")
    lines.append("")

    # Target distribution after cleaning
    vc    = df_clean["y"].value_counts()
    vc_pct = df_clean["y"].value_counts(normalize=True).mul(100).round(2)
    lines += [
        "TARGET DISTRIBUTION (after cleaning)",
        f"  'no'  : {vc.get('no',0):>6,}  ({vc_pct.get('no',0):.2f}%)",
        f"  'yes' : {vc.get('yes',0):>6,}  ({vc_pct.get('yes',0):.2f}%)",
        "",
    ]

    # New engineered columns
    lines += [
        "NEW / MODIFIED COLUMNS",
        "  y_encoded             : int (0=no, 1=yes) — binary-encoded target",
        "  previously_contacted  : int (0/1) — 1 if pdays != 999",
        "  campaign              : capped at 99th percentile (max=7)",
        "  all categorical cols  : lowercased and stripped",
        "  dot columns renamed   : emp.var.rate->emp_var_rate etc.",
        "",
    ]

    # Per-step log
    lines.append("CLEANING STEPS")
    lines.append(f"  {'Step':<6} {'Rule':<45} {'Before':>7} {'After':>7} {'Removed':>8}")
    lines.append("  " + "-" * 75)
    for entry in log:
        lines.append(
            f"  {entry['step']:<6} {entry['rule']:<45} "
            f"{entry['rows_before']:>7,} {entry['rows_after']:>7,} "
            f"{entry['rows_removed']:>8,}"
        )
    lines.append("")

    # Decisions detail
    lines.append("CLEANING DECISIONS (detailed)")
    for entry in log:
        lines.append(f"  [{entry['step']:02d}] {entry['rule']}")
        lines.append(f"       Decision  : {entry['decision']}")
        lines.append(f"       Cols      : {entry['cols_affected']}")
        if entry["notes"]:
            lines.append(f"       Notes     : {entry['notes']}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute full cleaning pipeline. Returns (df_raw, df_clean)."""
    print("=" * 60)
    print("BankInsight AI - Phase 2: Data Cleaning Pipeline")
    print("=" * 60)

    print("\n[1] Loading raw data ...")
    df_raw = step01_load(RAW_CSV)
    print(f"    Raw shape: {df_raw.shape}")

    print("\n[2] Running cleaning steps ...")
    df = df_raw.copy()
    df = step02_remove_duplicates(df)
    df = step03_drop_duration_zero(df)
    df = step04_fix_dtypes(df)
    df = step05_handle_unknown_default(df)
    df = step06_handle_unknown_education(df)
    df = step07_handle_unknown_housing_loan(df)
    df = step08_handle_unknown_job_marital(df)
    df = step09_engineer_pdays_flag(df)
    df = step10_cap_campaign_outliers(df)
    df = step11_rename_dot_columns(df)
    df = step12_validate_ranges(df)
    df = step13_verify_target(df)
    df = step14_final_check(df)

    print(f"\n    Clean shape: {df.shape}")

    print("\n[3] Saving cleaned dataset ...")
    df.to_csv(OUT_CSV, index=False)
    print(f"    Written: {OUT_CSV}")

    print("\n[4] Saving cleaning log CSV ...")
    log_df = pd.DataFrame(_log)
    log_df.to_csv(os.path.join(RPT_DIR, "cleaning_log.csv"), index=False)

    print("\n[5] Building cleaning report ...")
    report = build_cleaning_report(df_raw, df, _log)
    rpt_path = os.path.join(RPT_DIR, "cleaning_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report.encode("ascii", errors="replace").decode("ascii"))

    return df_raw, df


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_pipeline()
