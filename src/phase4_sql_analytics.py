"""
BankInsight AI - Phase 4: SQL Analytics Layer
==============================================
1. Creates SQLite database at data/db/bankinsight.db
2. Loads cleaned dataset into campaign_contacts table
3. Executes all SQL analytics queries
4. Validates key SQL results against Python/Pandas
5. Saves all query results as CSVs under reports/phase4/
6. Writes schema documentation and analytics report

Never modifies bank_marketing.csv.
"""

import os
import sys
import sqlite3
import textwrap
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_clean.csv")
DB_PATH   = os.path.join(BASE_DIR, "data", "db", "bankinsight.db")
SQL_DIR   = os.path.join(BASE_DIR, "src", "sql")
RPT_DIR   = os.path.join(BASE_DIR, "reports", "phase4")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)

# ── Validation log ─────────────────────────────────────────────────────────────
_validation_results: list[dict] = []

def _vlog(check: str, sql_val, py_val, passed: bool) -> None:
    _validation_results.append({
        "check":    check,
        "sql_val":  sql_val,
        "py_val":   py_val,
        "passed":   "PASS" if passed else "FAIL",
    })
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] {check}: SQL={sql_val}  PY={py_val}")


# ── Step 1: Build database ─────────────────────────────────────────────────────
def build_database(df: pd.DataFrame) -> sqlite3.Connection:
    """
    Drop and recreate bankinsight.db.
    Applies the schema from 01_schema.sql, then bulk-inserts cleaned data.
    """
    # Remove stale DB to ensure clean state
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Apply schema
    with open(os.path.join(SQL_DIR, "01_schema.sql"), "r") as f:
        conn.executescript(f.read())
    conn.commit()

    # Map DataFrame columns to table columns
    # The CSV uses 'default' (Python reserved word); table uses 'credit_default'
    insert_df = df.rename(columns={"default": "credit_default"})[
        ["age", "job", "marital", "education", "credit_default",
         "housing", "loan", "contact", "month", "day_of_week",
         "duration", "campaign", "pdays", "previous", "poutcome",
         "emp_var_rate", "cons_price_idx", "cons_conf_idx",
         "euribor3m", "nr_employed", "y", "y_encoded", "previously_contacted"]
    ]

    insert_df.to_sql("campaign_contacts", conn,
                     if_exists="append", index=False)
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM campaign_contacts").fetchone()[0]
    print(f"    Loaded {n:,} rows into campaign_contacts")
    return conn


# ── Step 2: Query executor ─────────────────────────────────────────────────────
def run_query(conn: sqlite3.Connection, sql_file: str,
              query_name: str = None,
              stmt_index: int = 0) -> pd.DataFrame:
    """
    Execute a SQL script file and return result as DataFrame.
    stmt_index selects which statement to return if file has multiple.
    """
    path = os.path.join(SQL_DIR, sql_file)
    with open(path, "r") as f:
        raw = f.read()

    # Split on semicolons (skip empty, skip comment-only blocks)
    stmts = [s.strip() for s in raw.split(";") if s.strip()
             and not all(line.startswith("--") for line in s.strip().splitlines())]

    # Execute all; return requested result
    result_df = None
    for i, stmt in enumerate(stmts):
        try:
            cursor = conn.execute(stmt)
            if cursor.description:  # is a SELECT
                cols = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                if i == stmt_index:
                    result_df = pd.DataFrame(rows, columns=cols)
        except sqlite3.OperationalError as e:
            print(f"    [SQL ERROR] {sql_file} stmt {i}: {e}")
            raise

    if result_df is None:
        result_df = pd.DataFrame()
    return result_df


# ── Step 3: Validation ─────────────────────────────────────────────────────────
def validate_against_pandas(conn: sqlite3.Connection,
                             df: pd.DataFrame) -> None:
    print("\n  Validation checks (SQL vs Pandas):")

    # V1: Total client count
    sql_total = conn.execute(
        "SELECT COUNT(*) FROM campaign_contacts").fetchone()[0]
    py_total = len(df)
    _vlog("Total row count", sql_total, py_total, sql_total == py_total)

    # V2: Subscription rate
    sql_rate = conn.execute(
        "SELECT ROUND(100.0*SUM(y_encoded)/COUNT(*),2) FROM campaign_contacts"
    ).fetchone()[0]
    py_rate = round(df["y_encoded"].mean() * 100, 2)
    _vlog("Subscription rate %", sql_rate, py_rate, abs(sql_rate - py_rate) < 0.01)

    # V3: Student conversion rate
    sql_student = conn.execute(
        "SELECT ROUND(100.0*SUM(y_encoded)/COUNT(*),2) FROM campaign_contacts "
        "WHERE job='student'"
    ).fetchone()[0]
    py_student = round(
        df[df["job"] == "student"]["y_encoded"].mean() * 100, 2)
    _vlog("Student conv rate %", sql_student, py_student,
          abs(sql_student - py_student) < 0.01)

    # V4: Cellular count
    sql_cell = conn.execute(
        "SELECT COUNT(*) FROM campaign_contacts WHERE contact='cellular'"
    ).fetchone()[0]
    py_cell = int((df["contact"] == "cellular").sum())
    _vlog("Cellular contact count", sql_cell, py_cell, sql_cell == py_cell)

    # V5: poutcome=success conversion
    sql_succ = conn.execute(
        "SELECT ROUND(100.0*SUM(y_encoded)/COUNT(*),2) FROM campaign_contacts "
        "WHERE poutcome='success'"
    ).fetchone()[0]
    py_succ = round(
        df[df["poutcome"] == "success"]["y_encoded"].mean() * 100, 2)
    _vlog("poutcome=success conv %", sql_succ, py_succ,
          abs(sql_succ - py_succ) < 0.01)

    # V6: Average call duration
    sql_dur = conn.execute(
        "SELECT ROUND(AVG(duration),2) FROM campaign_contacts"
    ).fetchone()[0]
    py_dur = round(float(df["duration"].mean()), 2)
    _vlog("Avg call duration (sec)", sql_dur, py_dur,
          abs(sql_dur - py_dur) < 0.05)

    # V7: No rows with duration == 0 (cleaning verified)
    sql_d0 = conn.execute(
        "SELECT COUNT(*) FROM campaign_contacts WHERE duration=0"
    ).fetchone()[0]
    _vlog("duration=0 rows (must be 0)", sql_d0, 0, sql_d0 == 0)

    # V8: Duplicate check
    sql_dupes = conn.execute(
        """SELECT COUNT(*) FROM (
             SELECT age,job,marital,education,credit_default,housing,loan,
                    contact,month,day_of_week,duration,campaign,pdays,
                    previous,poutcome,emp_var_rate,cons_price_idx,
                    cons_conf_idx,euribor3m,nr_employed,y
             FROM campaign_contacts
             GROUP BY age,job,marital,education,credit_default,housing,loan,
                      contact,month,day_of_week,duration,campaign,pdays,
                      previous,poutcome,emp_var_rate,cons_price_idx,
                      cons_conf_idx,euribor3m,nr_employed,y
             HAVING COUNT(*) > 1)"""
    ).fetchone()[0]
    _vlog("Duplicate rows in DB (must be 0)", sql_dupes, 0, sql_dupes == 0)


# ── Step 4: Report builder ─────────────────────────────────────────────────────
def build_report(results: dict[str, pd.DataFrame]) -> str:
    sep = "=" * 70
    lines = [sep, "BankInsight AI - Phase 4: SQL Analytics Report", sep, ""]

    lines += [
        "DATABASE",
        f"  File   : data/db/bankinsight.db",
        f"  Engine : SQLite {sqlite3.sqlite_version}",
        f"  Table  : campaign_contacts",
        "",
        "NOTE ON 'balance' COLUMN",
        "  The 'balance' column is NOT present in this dataset.",
        "  It exists only in the older UCI Bank Marketing reduced dataset (v1).",
        "  This project uses the full dataset (v2, 41,188 rows / 20 features)",
        "  which replaces 'balance' with five socio-economic macro indicators.",
        "",
    ]

    for qname, df in results.items():
        lines += [f"--- {qname} ---", df.to_string(index=False), ""]

    lines += [sep, "VALIDATION RESULTS", sep, ""]
    for v in _validation_results:
        lines.append(f"  [{v['passed']}] {v['check']:40s} SQL={v['sql_val']}  PY={v['py_val']}")

    all_pass = all(v["passed"] == "PASS" for v in _validation_results)
    lines += ["",
              f"  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}",
              sep]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("BankInsight AI - Phase 4: SQL Analytics")
    print("=" * 60)

    # Load cleaned data
    print("\n[1] Loading cleaned dataset ...")
    df = pd.read_csv(CLEAN_CSV)
    print(f"    Shape: {df.shape}")

    # Build database
    print("\n[2] Building SQLite database ...")
    conn = build_database(df)

    # Run validation
    print("\n[3] Validating SQL vs Pandas ...")
    validate_against_pandas(conn, df)

    # Execute and save all analytics queries
    print("\n[4] Running analytics queries ...")

    queries = [
        ("02_overview.sql",              "q01_overview",              0),
        ("03_conversion_by_job.sql",     "q02_conversion_by_job",     0),
        ("04_conversion_by_education.sql","q03_conversion_by_education",0),
        ("05_conversion_by_marital.sql", "q04_conversion_by_marital", 0),
        ("06_conversion_by_housing_loan.sql","q05_conversion_housing_loan",0),
        ("07_conversion_by_contact.sql", "q06_conversion_by_contact", 0),
        ("08_conversion_by_campaign_count.sql","q07_campaign_count",  0),
        ("09_conversion_by_poutcome.sql","q08_conversion_poutcome",   0),
        ("10_duration_contact_stats.sql","q09_duration_by_outcome",   0),
        ("10_duration_contact_stats.sql","q09_duration_buckets",      1),
        ("11_customer_segment_analysis.sql","q10_customer_segments",  0),
        ("12_temporal_patterns.sql",     "q11_temporal_patterns",     0),
    ]

    results: dict[str, pd.DataFrame] = {}
    for sql_file, label, stmt_idx in queries:
        try:
            res = run_query(conn, sql_file, label, stmt_index=stmt_idx)
            results[label] = res
            out_path = os.path.join(RPT_DIR, f"{label}.csv")
            res.to_csv(out_path, index=False)
            print(f"    {label}: {len(res)} rows -> {os.path.basename(out_path)}")
        except Exception as e:
            print(f"    [ERROR] {label}: {e}")

    # Print selected key results
    print("\n[5] Key Results Preview:")
    for key in ["q01_overview", "q02_conversion_by_job", "q08_conversion_poutcome"]:
        if key in results:
            print(f"\n  {key}:")
            print(textwrap.indent(results[key].to_string(index=False), "    "))

    # Build and save report
    print("\n[6] Building report ...")
    report = build_report(results)
    rpt_path = os.path.join(RPT_DIR, "sql_analytics_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Save validation CSV
    val_df = pd.DataFrame(_validation_results)
    val_df.to_csv(os.path.join(RPT_DIR, "validation_results.csv"), index=False)

    # Print report (ASCII-safe for Windows console)
    print(report.encode("ascii", errors="replace").decode("ascii"))

    conn.close()
    print("\nPhase 4 complete. Outputs in:", RPT_DIR)
    print("Database at:", DB_PATH)


if __name__ == "__main__":
    main()
