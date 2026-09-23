"""
BankInsight AI - Phase 5: Feature Engineering Pipeline
=======================================================
Reads  : data/processed/bank_marketing_clean.csv
Writes : data/processed/bank_marketing_features.csv
         reports/phase5/feature_dictionary.csv
         reports/phase5/feature_engineering_report.txt
         reports/phase5/feature_correlations.csv

Rules:
- 'duration' is EXCLUDED — target leakage.
- Collinear macro vars: keep euribor3m only (r=-0.308 with target);
  drop emp_var_rate, nr_employed, cons_price_idx.
- cons_conf_idx retained (r=+0.055, not collinear with others).
- All steps are documented with name / logic / reason / dtype.
- Raw CSV is never modified.
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_clean.csv")
OUT_CSV   = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_features.csv")
RPT_DIR   = os.path.join(BASE_DIR, "reports", "phase5")
os.makedirs(RPT_DIR, exist_ok=True)

# ── Feature registry ──────────────────────────────────────────────────────────
# Every feature added by this pipeline is registered here.
# Format: (name, logic, reason, dtype, source_cols)
_registry: list[dict] = []

def _reg(name: str, logic: str, reason: str, dtype: str, source: str) -> None:
    _registry.append({"feature": name, "logic": logic,
                      "reason": reason, "dtype": dtype, "source_cols": source})


# ════════════════════════════════════════════════════════════════════════════════
# STEP 01 — Drop leakage & high-collinearity features
# ════════════════════════════════════════════════════════════════════════════════
LEAKAGE_COLS    = ["duration"]           # unknown before call; confirmed leakage
COLLINEAR_DROPS = ["emp_var_rate",       # r=0.97 with euribor3m
                   "nr_employed",        # r=0.95 with euribor3m
                   "cons_price_idx"]     # r=0.78 with emp_var_rate
# Also drop raw string target and interim flags that will be replaced/kept clean
ADMIN_DROPS     = []                     # y and y_encoded retained; previously_contacted kept

DROP_COLS = LEAKAGE_COLS + COLLINEAR_DROPS + ADMIN_DROPS


# ════════════════════════════════════════════════════════════════════════════════
# STEP 02 — Binned / discretised features
# ════════════════════════════════════════════════════════════════════════════════

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin age into 6 meaningful bands identified in EDA (Fig 03).
    66+ and 18-25 are high-conversion outlier bands.
    """
    bins   = [17, 25, 35, 45, 55, 65, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels,
                             right=True, include_lowest=True)
    df["age_group"] = df["age_group"].astype(str)
    _reg("age_group",
         "pd.cut(age, [17,25,35,45,55,65,100])",
         "EDA (Fig 03) showed 18-25 and 66+ as high-conversion bands; "
         "binning captures non-linear age effect for tree models",
         "categorical (str)", "age")
    return df


def add_campaign_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group campaign contacts into operational tiers from SQL Q07.
    """
    def _tier(n):
        if n == 1:   return "first"
        if n <= 3:   return "optimal"
        if n <= 6:   return "diminishing"
        return "over_contacted"

    df["campaign_group"] = df["campaign"].apply(_tier)
    _reg("campaign_group",
         "1->'first', 2-3->'optimal', 4-6->'diminishing', >6->'over_contacted'",
         "SQL Q07: conversion drops steeply after 3 contacts; "
         "this ordinal tier is more informative than raw count for linear models",
         "categorical (str)", "campaign")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 03 — Previous-contact features
# ════════════════════════════════════════════════════════════════════════════════

def add_pdays_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 999 sentinel in pdays with NaN-equivalent (as -1 for integer storage).
    A real_pdays_days column holds the actual recency; -1 means never contacted.
    """
    df["pdays_actual"] = df["pdays"].where(df["pdays"] != 999, other=-1)
    _reg("pdays_actual",
         "pdays if pdays!=999 else -1",
         "Strips the 999 sentinel; -1 signals 'not previously contacted'. "
         "Range for real values: 0-27 days.",
         "int", "pdays")
    return df


def add_recency_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin real pdays recency into groups; 'none' for no prior contact.
    """
    def _recency(row):
        if row["pdays"] == 999:  return "no_prior"
        if row["pdays"] <= 3:    return "very_recent"
        if row["pdays"] <= 7:    return "recent"
        if row["pdays"] <= 14:   return "moderate"
        return "distant"

    df["recency_group"] = df.apply(_recency, axis=1)
    _reg("recency_group",
         "pdays=999->'no_prior', <=3->'very_recent', <=7->'recent', "
         "<=14->'moderate', >14->'distant'",
         "Converts raw pdays to a 5-level ordinal recency category; "
         "previously-contacted clients (3.3% of data) show 65%+ conversion",
         "categorical (str)", "pdays")
    return df


def add_previous_capped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap previous contacts at 3 (99th pct is 3; extreme values add noise).
    """
    cap = int(df["previous"].quantile(0.99))
    df["previous_capped"] = df["previous"].clip(upper=cap)
    _reg("previous_capped",
         f"previous.clip(upper={cap})",
         "previous is right-skewed (max=7, 99th pct=3); "
         "capping reduces outlier noise without losing information",
         "int", "previous")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 04 — Interaction / composite features
# ════════════════════════════════════════════════════════════════════════════════

def add_repeat_success(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag: client was previously contacted AND previous campaign succeeded.
    This is the single strongest predictor combination found in EDA/SQL.
    """
    df["repeat_success"] = (
        (df["previously_contacted"] == 1) & (df["poutcome"] == "success")
    ).astype(int)
    _reg("repeat_success",
         "(previously_contacted==1) & (poutcome=='success') -> 1, else 0",
         "SQL Q08: poutcome=success + previously_contacted = 65% conversion rate. "
         "This interaction isolates the highest-value segment as a binary flag.",
         "int (0/1)", "previously_contacted, poutcome")
    return df


def add_cellular_first_contact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag: cellular contact on the very first call of this campaign.
    Combines the two strongest positive campaign-process signals.
    """
    df["cellular_first"] = (
        (df["contact"] == "cellular") & (df["campaign"] == 1)
    ).astype(int)
    _reg("cellular_first",
         "(contact=='cellular') & (campaign==1) -> 1, else 0",
         "EDA Fig 06: cellular outperforms telephone (14.7% vs 5.2%); "
         "first contact outperforms repeats. Interaction captures ideal scenario.",
         "int (0/1)", "contact, campaign")
    return df


def add_high_value_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag: client in a high-conversion job (student/retired) AND no housing loan.
    Combines top job tier with lowest debt-burden signal from EDA Fig 02 & 05.
    """
    df["high_value_segment"] = (
        df["job"].isin(["student", "retired"]) & (df["housing"] == "no")
    ).astype(int)
    _reg("high_value_segment",
         "(job in ['student','retired']) & (housing=='no') -> 1, else 0",
         "Students and retirees convert at 25-31%; combining with no-housing-loan "
         "flag isolates the clearest high-value, low-debt-burden segment.",
         "int (0/1)", "job, housing")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 05 — Ordinal encoding of ordered categoricals
# ════════════════════════════════════════════════════════════════════════════════

EDU_ORDER = {
    "illiterate": 0, "basic.4y": 1, "basic.6y": 2,
    "basic.9y": 3,   "high.school": 4, "professional.course": 5,
    "university.degree": 6, "unknown": -1
}

MONTH_ORDER = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

DOW_ORDER = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}


def add_education_encoded(df: pd.DataFrame) -> pd.DataFrame:
    df["education_encoded"] = df["education"].map(EDU_ORDER)
    _reg("education_encoded",
         "illiterate=0 < basic.4y=1 < ... < university.degree=6; unknown=-1",
         "EDA Fig 04 confirmed a monotonic upward trend from basic to university; "
         "ordinal encoding preserves this ordering. -1 for unknown is a "
         "deliberate neutral signal (not imputed).",
         "int", "education")
    return df


def add_month_num(df: pd.DataFrame) -> pd.DataFrame:
    df["month_num"] = df["month"].map(MONTH_ORDER)
    _reg("month_num",
         "jan=1, feb=2, ..., dec=12",
         "Converts month string to calendar integer for linear models; "
         "tree models can use the string directly but benefit from numeric form.",
         "int", "month")
    return df


def add_dow_num(df: pd.DataFrame) -> pd.DataFrame:
    df["dow_num"] = df["day_of_week"].map(DOW_ORDER)
    _reg("dow_num",
         "mon=1, tue=2, wed=3, thu=4, fri=5",
         "Calendar ordering for linear models. EDA showed minimal variation "
         "across days (~1 pp spread) but the numeric encoding costs nothing.",
         "int", "day_of_week")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 06 — One-hot encoding of nominal categoricals
# ════════════════════════════════════════════════════════════════════════════════

ONEHOT_COLS = [
    "job",          # 12 levels — nominal
    "marital",      # 4 levels  — nominal
    "default",      # 3 levels  — nominal (yes/no/unknown)
    "housing",      # 3 levels  — nominal (yes/no/unknown)
    "loan",         # 3 levels  — nominal (yes/no/unknown)
    "contact",      # 2 levels  — nominal
    "poutcome",     # 3 levels  — nominal
]
# month, day_of_week are already ordinal-encoded above;
# education is already ordinal-encoded above.
# drop_first=True avoids dummy variable trap for logistic regression.

def add_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    before_cols = len(df.columns)
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True, dtype=int)
    new_ohe_cols = [c for c in df.columns
                    if any(c.startswith(base + "_") for base in ONEHOT_COLS)]
    _reg("OHE: " + ", ".join(ONEHOT_COLS),
         "pd.get_dummies(drop_first=True, dtype=int)",
         "Nominal categoricals have no meaningful order; one-hot encoding "
         "is required for logistic regression and distance-based models. "
         "drop_first=True prevents perfect multicollinearity.",
         "int (0/1) per dummy column",
         ", ".join(ONEHOT_COLS))
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 07 — Log transforms for right-skewed numeric features
# ════════════════════════════════════════════════════════════════════════════════

def add_log_campaign(df: pd.DataFrame) -> pd.DataFrame:
    df["log_campaign"] = np.log1p(df["campaign"])
    _reg("log_campaign",
         "log(1 + campaign)",
         "campaign is right-skewed (skewness~3.7 after capping); "
         "log1p compresses scale and improves linear model assumptions.",
         "float", "campaign")
    return df


def add_log_previous(df: pd.DataFrame) -> pd.DataFrame:
    df["log_previous"] = np.log1p(df["previous_capped"])
    _reg("log_previous",
         "log(1 + previous_capped)",
         "previous_capped is severely zero-inflated with rare high values; "
         "log1p brings the non-zero tail closer to the bulk.",
         "float", "previous_capped")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 08 — Economic cycle indicator (derived from euribor3m)
# ════════════════════════════════════════════════════════════════════════════════

def add_low_rate_env(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: euribor3m < 2.0 indicates a low-interest-rate environment.
    EDA Fig 09 showed subscription rates are markedly higher when euribor is low.
    Threshold 2.0 is the natural gap in the euribor3m distribution.
    """
    threshold = 2.0
    df["low_rate_env"] = (df["euribor3m"] < threshold).astype(int)
    _reg("low_rate_env",
         f"euribor3m < {threshold} -> 1, else 0",
         "EDA Fig 09: subscription rate doubles in low-rate environments "
         "(euribor<2). Binary flag captures this non-linear threshold effect "
         "for linear models. euribor3m is retained as continuous too.",
         "int (0/1)", "euribor3m")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# Pipeline orchestrator
# ════════════════════════════════════════════════════════════════════════════════

def run_pipeline(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    print("  Step 01: Drop leakage/collinear columns ...")
    dropped = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"           Dropped: {dropped}")

    print("  Step 02: Binned features ...")
    df = add_age_group(df)
    df = add_campaign_group(df)

    print("  Step 03: Previous-contact features ...")
    df = add_pdays_clean(df)
    df = add_recency_group(df)
    df = add_previous_capped(df)

    print("  Step 04: Interaction/composite features ...")
    df = add_repeat_success(df)
    df = add_cellular_first_contact(df)
    df = add_high_value_segment(df)

    print("  Step 05: Ordinal encoding ...")
    df = add_education_encoded(df)
    df = add_month_num(df)
    df = add_dow_num(df)

    print("  Step 06: One-hot encoding ...")
    df = add_one_hot(df)

    print("  Step 07: Log transforms ...")
    df = add_log_campaign(df)
    df = add_log_previous(df)

    print("  Step 08: Economic indicator ...")
    df = add_low_rate_env(df)

    return df


# ════════════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════════════

def validate(df_raw: pd.DataFrame, df_feat: pd.DataFrame) -> list[dict]:
    results = []

    def _chk(name: str, actual, expected, tol=0):
        passed = abs(actual - expected) <= tol if isinstance(actual, float) else actual == expected
        results.append({"check": name, "actual": actual,
                        "expected": expected, "passed": "PASS" if passed else "FAIL"})
        mark = "PASS" if passed else "FAIL"
        print(f"    [{mark}] {name}: {actual} (expected {expected})")

    _chk("Row count preserved", len(df_feat), len(df_raw))
    _chk("y_encoded sum preserved",
         int(df_feat["y_encoded"].sum()), int(df_raw["y_encoded"].sum()))

    # duration must be absent
    has_dur = "duration" in df_feat.columns
    results.append({"check": "duration excluded", "actual": has_dur,
                    "expected": False,
                    "passed": "PASS" if not has_dur else "FAIL"})
    print(f"    {'PASS' if not has_dur else 'FAIL'} duration excluded: {not has_dur}")

    # Collinear drops absent
    for col in COLLINEAR_DROPS:
        absent = col not in df_feat.columns
        results.append({"check": f"{col} excluded", "actual": not absent,
                        "expected": False,
                        "passed": "PASS" if absent else "FAIL"})
        print(f"    {'PASS' if absent else 'FAIL'} {col} excluded")

    # repeat_success only 0/1
    if "repeat_success" in df_feat.columns:
        ok = set(df_feat["repeat_success"].unique()).issubset({0, 1})
        results.append({"check": "repeat_success is 0/1", "actual": ok,
                        "expected": True, "passed": "PASS" if ok else "FAIL"})
        print(f"    {'PASS' if ok else 'FAIL'} repeat_success is 0/1")

    # No nulls introduced
    new_nulls = int(df_feat.isnull().sum().sum())
    results.append({"check": "no new nulls", "actual": new_nulls,
                    "expected": 0, "passed": "PASS" if new_nulls == 0 else "FAIL"})
    print(f"    {'PASS' if new_nulls==0 else 'FAIL'} no new nulls: {new_nulls}")

    # education_encoded values in expected set
    if "education_encoded" in df_feat.columns:
        valid_vals = set(EDU_ORDER.values())
        bad = df_feat[~df_feat["education_encoded"].isin(valid_vals)]
        results.append({"check": "education_encoded values valid",
                        "actual": len(bad), "expected": 0,
                        "passed": "PASS" if len(bad)==0 else "FAIL"})
        print(f"    {'PASS' if len(bad)==0 else 'FAIL'} education_encoded values valid: {len(bad)} bad rows")

    # y and y_encoded are consistent
    y_check = (df_feat["y_encoded"] == (df_feat["y"] == "yes").astype(int)).all()
    results.append({"check": "y/y_encoded consistent", "actual": bool(y_check),
                    "expected": True, "passed": "PASS" if y_check else "FAIL"})
    print(f"    {'PASS' if y_check else 'FAIL'} y/y_encoded consistent")

    return results


# ════════════════════════════════════════════════════════════════════════════════
# Report builder
# ════════════════════════════════════════════════════════════════════════════════

def build_report(df_raw: pd.DataFrame, df_feat: pd.DataFrame,
                 val_results: list[dict]) -> str:
    sep = "=" * 70
    lines = [sep, "BankInsight AI - Phase 5: Feature Engineering Report", sep, ""]

    lines += [
        "DATASET",
        f"  Input  : data/processed/bank_marketing_clean.csv  "
        f"({len(df_raw):,} rows x {df_raw.shape[1]} cols)",
        f"  Output : data/processed/bank_marketing_features.csv  "
        f"({len(df_feat):,} rows x {df_feat.shape[1]} cols)",
        "",
    ]

    # Columns removed
    lines += [
        "COLUMNS REMOVED",
        f"  Leakage   : {LEAKAGE_COLS}",
        f"  Collinear : {COLLINEAR_DROPS}",
        "  Rationale : emp_var_rate/euribor3m r=0.97; nr_employed/euribor3m r=0.95;",
        "              cons_price_idx/emp_var_rate r=0.78. Retained: euribor3m",
        "              (highest individual correlation with target, r=-0.308).",
        "",
    ]

    # Feature dictionary
    lines += ["FEATURE DICTIONARY", ""]
    lines.append(f"  {'Feature':<30} {'Type':<20} {'Source':<30}")
    lines.append("  " + "-" * 80)
    for r in _registry:
        lines.append(f"  {r['feature']:<30} {r['dtype']:<20} {r['source_cols']}")
    lines.append("")

    # Detailed entries
    lines.append("FEATURE DETAILS")
    for i, r in enumerate(_registry, 1):
        lines += [
            f"  [{i:02d}] {r['feature']}",
            f"       Logic  : {r['logic']}",
            f"       Reason : {r['reason']}",
            f"       Type   : {r['dtype']}",
            f"       Source : {r['source_cols']}",
            "",
        ]

    # Column summary
    num_orig   = df_raw.shape[1]
    num_final  = df_feat.shape[1]
    non_feat   = ["y", "y_encoded"]
    feat_cols  = [c for c in df_feat.columns if c not in non_feat]
    lines += [
        "COLUMN COUNTS",
        f"  Original columns     : {num_orig}",
        f"  Final columns        : {num_final}",
        f"  Feature columns      : {len(feat_cols)}  (excludes y, y_encoded)",
        "",
        "FINAL FEATURE LIST",
    ]
    for c in feat_cols:
        lines.append(f"  {c}")
    lines.append("")

    # Validation
    lines += [sep, "VALIDATION RESULTS", sep, ""]
    for v in val_results:
        lines.append(f"  [{v['passed']}] {v['check']:40s} actual={v['actual']}")
    all_pass = all(v["passed"] == "PASS" for v in val_results)
    lines += ["", f"  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}",
              sep]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("BankInsight AI - Phase 5: Feature Engineering")
    print("=" * 60)

    print("\n[1] Loading cleaned dataset ...")
    df_raw = pd.read_csv(CLEAN_CSV)
    print(f"    Shape: {df_raw.shape}")

    print("\n[2] Running feature engineering pipeline ...")
    df_feat = run_pipeline(df_raw)
    print(f"    Result shape: {df_feat.shape}")

    print("\n[3] Validating ...")
    val_results = validate(df_raw, df_feat)

    print("\n[4] Computing feature-target correlations ...")
    # Only numeric columns, exclude y_encoded itself
    numeric_cols = [c for c in df_feat.select_dtypes(include="number").columns
                    if c != "y_encoded"]
    corr_series = df_feat[numeric_cols].corrwith(df_feat["y_encoded"])
    corr_df = (corr_series.rename("pearson_r")
                           .reset_index()
                           .rename(columns={"index": "feature"})
                           .sort_values("pearson_r", key=abs, ascending=False))
    corr_df["abs_r"] = corr_df["pearson_r"].abs().round(4)
    corr_df["pearson_r"] = corr_df["pearson_r"].round(4)
    corr_df.to_csv(os.path.join(RPT_DIR, "feature_correlations.csv"), index=False)
    print("    Top 15 feature-target correlations:")
    print(corr_df.head(15).to_string(index=False))

    print("\n[5] Saving outputs ...")
    df_feat.to_csv(OUT_CSV, index=False)
    print(f"    Features CSV: {OUT_CSV}")

    feat_dict_df = pd.DataFrame(_registry)
    feat_dict_df.to_csv(os.path.join(RPT_DIR, "feature_dictionary.csv"), index=False)

    pd.DataFrame(val_results).to_csv(
        os.path.join(RPT_DIR, "validation_results.csv"), index=False)

    print("\n[6] Building report ...")
    report = build_report(df_raw, df_feat, val_results)
    with open(os.path.join(RPT_DIR, "feature_engineering_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(report)
    print(report.encode("ascii", errors="replace").decode("ascii"))

    print("\nPhase 5 complete. Outputs in:", RPT_DIR)
    print("Feature dataset:", OUT_CSV)


if __name__ == "__main__":
    main()
