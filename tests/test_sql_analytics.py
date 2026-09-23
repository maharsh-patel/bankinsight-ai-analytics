import os
import sys
import sqlite3
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
import phase4_sql_analytics as p4

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH   = os.path.join(BASE_DIR, "data", "db", "bankinsight.db")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_clean.csv")
SQL_DIR   = os.path.join(BASE_DIR, "src", "sql")


def test_database_exists_and_valid():
    assert os.path.exists(DB_PATH), f"Database missing at {DB_PATH}"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM campaign_contacts")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 41172, f"Expected 41172 rows in campaign_contacts, got {count}"


def test_sql_query_files_exist():
    sql_files = [f for f in os.listdir(SQL_DIR) if f.endswith(".sql")]
    assert len(sql_files) >= 12, f"Expected at least 12 SQL query files, found {len(sql_files)}"
    assert "01_schema.sql" in sql_files
    assert "02_overview.sql" in sql_files
    assert "12_temporal_patterns.sql" in sql_files


def test_sql_queries_execution():
    conn = sqlite3.connect(DB_PATH)
    
    # Overview Q02
    df_ov = p4.run_query(conn, "02_overview.sql")
    assert not df_ov.empty
    assert "total_clients" in df_ov.columns
    assert int(df_ov["total_clients"].iloc[0]) == 41172
    assert int(df_ov["total_subscribed"].iloc[0]) == 4639
    
    # Conversion by job Q03
    df_job = p4.run_query(conn, "03_conversion_by_job.sql")
    assert not df_job.empty
    assert "job" in df_job.columns
    assert "conv_rate_pct" in df_job.columns
    assert len(df_job) >= 10
    
    # Conversion by education Q04
    df_edu = p4.run_query(conn, "04_conversion_by_education.sql")
    assert not df_edu.empty
    
    # Temporal Q12
    df_temp = p4.run_query(conn, "12_temporal_patterns.sql")
    assert not df_temp.empty

    conn.close()


def test_sql_vs_pandas_kpi_consistency():
    df_clean = pd.read_csv(CLEAN_CSV)
    py_total = len(df_clean)
    py_subscribed = int((df_clean["y_encoded"] == 1).sum())
    py_rate = round(py_subscribed / py_total * 100, 2)
    
    conn = sqlite3.connect(DB_PATH)
    df_ov = p4.run_query(conn, "02_overview.sql")
    conn.close()
    
    sql_total = int(df_ov["total_clients"].iloc[0])
    sql_subscribed = int(df_ov["total_subscribed"].iloc[0])
    sql_rate = float(df_ov["subscription_rate_pct"].iloc[0])
    
    assert py_total == sql_total
    assert py_subscribed == sql_subscribed
    assert pytest.approx(py_rate, abs=0.01) == sql_rate
