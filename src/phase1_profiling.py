"""
BankInsight AI - Phase 1: Data Understanding & Profiling
Generates:
  - reports/phase1/data_dictionary.csv
  - reports/phase1/profiling_report.txt
  - reports/phase1/descriptive_stats.csv
  - reports/phase1/target_distribution.csv
  - reports/phase1/data_quality_summary.csv

Raw CSV is never modified.
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV    = os.path.join(BASE_DIR, "bank_marketing.csv")
OUT_DIR    = os.path.join(BASE_DIR, "reports", "phase1")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load ─────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ── 1. Column-level metadata ─────────────────────────────────────────────────
COLUMN_META = {
    # client attributes
    "age":            ("Numerical",    "Client age in years"),
    "job":            ("Categorical",  "Type of job"),
    "marital":        ("Categorical",  "Marital status"),
    "education":      ("Categorical",  "Highest education level"),
    "default":        ("Categorical",  "Has credit in default? (yes/no/unknown)"),
    "housing":        ("Categorical",  "Has housing loan? (yes/no/unknown)"),
    "loan":           ("Categorical",  "Has personal loan? (yes/no/unknown)"),
    # contact
    "contact":        ("Categorical",  "Contact communication type (cellular/telephone)"),
    "month":          ("Categorical",  "Last contact month of year"),
    "day_of_week":    ("Categorical",  "Last contact day of week"),
    "duration":       ("Numerical",    "Last contact duration in seconds (NOTE: unknown before call; kept for reference only)"),
    # campaign
    "campaign":       ("Numerical",    "Number of contacts performed during this campaign"),
    "pdays":          ("Numerical",    "Days since last contact from previous campaign (999 = not previously contacted)"),
    "previous":       ("Numerical",    "Number of contacts before this campaign"),
    "poutcome":       ("Categorical",  "Outcome of previous marketing campaign"),
    # socio-economic context
    "emp.var.rate":   ("Numerical",    "Employment variation rate (quarterly indicator)"),
    "cons.price.idx": ("Numerical",    "Consumer price index (monthly indicator)"),
    "cons.conf.idx":  ("Numerical",    "Consumer confidence index (monthly indicator)"),
    "euribor3m":      ("Numerical",    "Euribor 3-month rate (daily indicator)"),
    "nr.employed":    ("Numerical",    "Number of employees (quarterly indicator)"),
    # target
    "y":              ("Target/Binary","Has the client subscribed a term deposit? (yes/no)"),
}

KNOWN_UNKNOWN_SENTINEL = {"job", "marital", "education", "default", "housing", "loan"}
PDAYS_NO_CONTACT_SENTINEL = 999


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        col_type, description = COLUMN_META.get(col, ("Unknown", ""))
        dtype       = str(df[col].dtype)
        n_unique    = df[col].nunique()
        null_count  = int(df[col].isnull().sum())
        null_pct    = round(null_count / len(df) * 100, 2)

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if not is_numeric:
            sample_vals = ", ".join(str(v) for v in df[col].unique()[:8])
            min_val = max_val = mean_val = std_val = ""
        else:
            sample_vals = ""
            min_val  = round(float(df[col].min()),  4)
            max_val  = round(float(df[col].max()),  4)
            mean_val = round(float(df[col].mean()), 4)
            std_val  = round(float(df[col].std()),  4)

        # flag 'unknown' as a string category value
        has_unknown = ""
        if col in KNOWN_UNKNOWN_SENTINEL:
            has_unknown = "YES" if "unknown" in df[col].values else "no"

        rows.append({
            "column":      col,
            "dtype":       dtype,
            "category":    col_type,
            "description": description,
            "n_unique":    n_unique,
            "null_count":  null_count,
            "null_pct":    null_pct,
            "has_unknown_label": has_unknown,
            "min":         min_val,
            "max":         max_val,
            "mean":        mean_val,
            "std":         std_val,
            "sample_values": sample_vals,
        })
    return pd.DataFrame(rows)


# ── 2. Profiling report (text) ─────────────────────────────────────────────
def build_profiling_report(df: pd.DataFrame) -> str:
    lines = []
    sep   = "=" * 70

    lines += [sep, "BankInsight AI - Phase 1: Dataset Profiling Report", sep, ""]

    # Basic dimensions
    lines += [
        "DATASET DIMENSIONS",
        f"  Rows    : {len(df):,}",
        f"  Columns : {len(df.columns)}",
        "",
    ]

    # Column overview
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    lines += [
        "COLUMN TYPES",
        f"  Numerical  ({len(num_cols)}): {', '.join(num_cols)}",
        f"  Categorical({len(cat_cols)}): {', '.join(cat_cols)}",
        "",
    ]

    # Missing values
    null_total = df.isnull().sum().sum()
    lines += [
        "MISSING / NULL VALUES",
        f"  Total null cells : {null_total}",
        "  (No structural nulls detected — 'unknown' is encoded as a string label)",
        "",
    ]

    # Duplicates
    n_dupes = df.duplicated().sum()
    lines += [
        "DUPLICATE RECORDS",
        f"  Exact-duplicate rows : {n_dupes:,}",
        "",
    ]

    # Unique counts per column
    lines.append("UNIQUE VALUES PER COLUMN")
    for col in df.columns:
        lines.append(f"  {col:<20}: {df[col].nunique():>6,}")
    lines.append("")

    # Categorical value listings
    lines.append("CATEGORICAL COLUMN - ALL UNIQUE VALUES")
    for col in cat_cols:
        vals = sorted(df[col].unique().tolist())
        lines.append(f"  {col:<20}: {vals}")
    lines.append("")

    # Target variable
    vc = df["y"].value_counts()
    vc_pct = df["y"].value_counts(normalize=True).mul(100).round(2)
    lines += [
        "TARGET VARIABLE  (y - term deposit subscription)",
        f"  'no'  : {vc.get('no',0):>6,}  ({vc_pct.get('no',0):.2f}%)",
        f"  'yes' : {vc.get('yes',0):>6,}  ({vc_pct.get('yes',0):.2f}%)",
        f"  Imbalance ratio (no:yes) approx {vc.get('no',0)/vc.get('yes',1):.1f}:1",
        "",
    ]

    # Date/time fields
    lines += [
        "DATE / TIME FIELDS",
        "  No explicit date/timestamp columns.",
        "  'month' and 'day_of_week' encode campaign contact time as string categories.",
        "",
    ]

    # Suspicious / invalid values
    duration_zero = int((df["duration"] == 0).sum())
    pdays_sentinel = int((df["pdays"] == PDAYS_NO_CONTACT_SENTINEL).sum())
    campaign_outliers = int((df["campaign"] > df["campaign"].quantile(0.99)).sum())
    age_outliers = int((df["age"] > 80).sum())

    lines += [
        "SUSPICIOUS / INVALID VALUES",
        f"  duration == 0         : {duration_zero} rows  "
        "(call never happened — target leakage risk if duration kept)",
        f"  pdays == 999 sentinel : {pdays_sentinel:,} rows  "
        f"({pdays_sentinel/len(df)*100:.1f}% — means 'not previously contacted')",
        f"  campaign > 99th pct   : {campaign_outliers} rows  (extreme contact frequency)",
        f"  age > 80              : {age_outliers} rows  (plausible but sparse)",
        "",
    ]

    # 'unknown' encoded as string per column
    lines.append("'unknown' STRING-LABEL COUNTS (proxy missing values)")
    for col in KNOWN_UNKNOWN_SENTINEL:
        n = int((df[col] == "unknown").sum())
        pct = round(n / len(df) * 100, 2)
        lines.append(f"  {col:<20}: {n:>5,}  ({pct:.2f}%)")
    lines.append("")

    # Socio-economic feature note
    lines += [
        "SOCIO-ECONOMIC FEATURES NOTE",
        "  emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed",
        "  are time-varying macro indicators. Within this dataset they take a",
        "  limited number of distinct values (one per campaign period).",
    ]
    for col in ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]:
        lines.append(f"  {col:<20}: {df[col].nunique()} unique values")
    lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ── 3. Descriptive statistics ─────────────────────────────────────────────────
def build_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns.tolist()  # numeric only; fine
    stats = df[num_cols].describe().T
    stats["skewness"] = df[num_cols].skew().round(4)
    stats["kurtosis"] = df[num_cols].kurt().round(4)
    stats = stats.round(4)
    return stats


# ── 4. Target distribution ────────────────────────────────────────────────────
def build_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vc  = df["y"].value_counts()
    pct = df["y"].value_counts(normalize=True).mul(100).round(2)
    return pd.DataFrame({"count": vc, "percentage": pct})


# ── 5. Data-quality summary ───────────────────────────────────────────────────
def build_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "check": "Structural nulls",
        "finding": f"{df.isnull().sum().sum()} null cells",
        "severity": "None",
        "recommendation": "No action needed for structural nulls",
    })
    rows.append({
        "check": "Exact duplicate rows",
        "finding": f"{df.duplicated().sum()} duplicates",
        "severity": "Medium" if df.duplicated().sum() > 0 else "None",
        "recommendation": "Inspect and remove if confirmed true duplicates in Phase 3",
    })
    for col in KNOWN_UNKNOWN_SENTINEL:
        n = int((df[col] == "unknown").sum())
        pct = round(n / len(df) * 100, 2)
        severity = "High" if pct > 10 else ("Medium" if pct > 1 else "Low")
        rows.append({
            "check": f"'unknown' in '{col}'",
            "finding": f"{n} rows ({pct}%)",
            "severity": severity,
            "recommendation": f"Impute or treat as separate category in Phase 3",
        })
    rows.append({
        "check": "duration == 0",
        "finding": f"{int((df['duration']==0).sum())} rows",
        "severity": "Low",
        "recommendation": "Consider dropping; call never happened, target is always 'no'",
    })
    rows.append({
        "check": "pdays == 999 sentinel",
        "finding": f"{int((df['pdays']==999).sum()):,} rows ({int((df['pdays']==999).sum())/len(df)*100:.1f}%)",
        "severity": "Medium",
        "recommendation": "Engineer binary 'was_previously_contacted' flag in Phase 3",
    })
    rows.append({
        "check": "Class imbalance (target y)",
        "finding": f"no={df['y'].value_counts().get('no',0):,} yes={df['y'].value_counts().get('yes',0):,} ratio≈{df['y'].value_counts().get('no',0)/df['y'].value_counts().get('yes',1):.1f}:1",
        "severity": "High",
        "recommendation": "Use stratified splits, class-weight or SMOTE in Phase 4",
    })
    rows.append({
        "check": "duration — target leakage risk",
        "finding": "duration is unknown at call time; highly correlated with outcome",
        "severity": "High",
        "recommendation": "Exclude 'duration' from predictive model features in Phase 3/4",
    })
    rows.append({
        "check": "campaign outliers (>99th pct)",
        "finding": f"{int((df['campaign']>df['campaign'].quantile(0.99)).sum())} rows",
        "severity": "Low",
        "recommendation": "Cap or log-transform in Phase 3",
    })
    rows.append({
        "check": "Socio-economic feature cardinality",
        "finding": "Low unique counts (macro indicators constant per campaign period)",
        "severity": "Low",
        "recommendation": "Treat as continuous; check multicollinearity in Phase 3",
    })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Ensure stdout can handle UTF-8 on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading dataset ...")
    df = load_data(RAW_CSV)
    print(f"  Loaded: {len(df):,} rows x {len(df.columns)} columns")

    print("Building data dictionary ...")
    dd = build_data_dictionary(df)
    dd.to_csv(os.path.join(OUT_DIR, "data_dictionary.csv"), index=False)

    print("Building profiling report ...")
    report = build_profiling_report(df)
    with open(os.path.join(OUT_DIR, "profiling_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    # Print report using ascii-safe path for Windows console
    print(report.encode("ascii", errors="replace").decode("ascii"))

    print("Building descriptive statistics ...")
    stats = build_descriptive_stats(df)
    stats.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"))

    print("Building target distribution ...")
    target_dist = build_target_distribution(df)
    target_dist.to_csv(os.path.join(OUT_DIR, "target_distribution.csv"))

    print("Building data-quality summary ...")
    quality = build_quality_summary(df)
    quality.to_csv(os.path.join(OUT_DIR, "data_quality_summary.csv"), index=False)

    print("\nPhase 1 complete. Outputs written to:", OUT_DIR)
    return df, dd, stats, target_dist, quality


if __name__ == "__main__":
    main()
