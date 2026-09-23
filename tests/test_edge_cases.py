import os
import sys
import tempfile
import sqlite3
import joblib
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(__file__), "..", "src")))
import phase2_cleaning as p2
import phase5_feature_engineering as p5
import phase6_modelling as p6
import phase4_sql_analytics as p4
import phase8_ai_engine as p8

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV  = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_features.csv")
RF_MODEL  = os.path.join(BASE_DIR, "models", "rf_pipeline.joblib")
DB_PATH   = os.path.join(BASE_DIR, "data", "db", "bankinsight.db")


# ── 1. Edge Case: Missing values / NaNs ───────────────────────────────────────
def test_edge_case_missing_values():
    df_clean = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "bank_marketing_clean.csv")).head(50).copy()
    # Inject NaNs in non-critical columns
    df_clean.loc[0:5, "age"] = np.nan
    df_clean.loc[10:15, "job"] = np.nan
    
    # Check that age grouping handles NaNs without crashing
    df_age = p5.add_age_group(df_clean.copy())
    assert df_age["age_group"].isna().sum() > 0 or (df_age["age_group"] == "nan").sum() > 0


# ── 2. Edge Case: Invalid filters & empty dataframes ──────────────────────────
def test_edge_case_empty_results():
    conn = sqlite3.connect(DB_PATH)
    # Query with filter matching 0 rows
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM campaign_contacts WHERE age > 200")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 0


def test_edge_case_invalid_filters():
    df_clean = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "bank_marketing_clean.csv"))
    # Filter for non-existent job category
    filtered = df_clean[df_clean["job"] == "astronaut"]
    assert len(filtered) == 0
    # Operations on empty filtered dataframe should not raise UnboundLocalError or ZeroDivisionError
    rate = (filtered["y_encoded"] == 1).mean() if len(filtered) > 0 else 0.0
    assert rate == 0.0


# ── 3. Edge Case: Model Errors ────────────────────────────────────────────────
def test_edge_case_model_missing_columns():
    rf_pipe = joblib.load(RF_MODEL)
    df_feat = pd.read_csv(FEAT_CSV).head(10)
    feat_cols = p6.get_feature_cols(df_feat)
    
    # Drop one required feature column
    bad_x = df_feat[feat_cols].drop(columns=[feat_cols[0]])
    
    with pytest.raises(ValueError):
        rf_pipe.predict(bad_x)


def test_edge_case_corrupted_model_file():
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp.write(b"CORRUPTED_MODEL_BYTES")
        tmp_path = tmp.name
        
    try:
        with pytest.raises(Exception):
            joblib.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── 4. Edge Case: Database Errors ─────────────────────────────────────────────
def test_edge_case_nonexistent_database():
    non_existent_db = os.path.join(BASE_DIR, "data", "db", "non_existent.db")
    if os.path.exists(non_existent_db):
        os.remove(non_existent_db)
        
    conn = sqlite3.connect(non_existent_db)
    cur = conn.cursor()
    with pytest.raises(sqlite3.OperationalError):
        cur.execute("SELECT * FROM non_existent_table")
    conn.close()
    if os.path.exists(non_existent_db):
        os.remove(non_existent_db)


# ── 5. Edge Case: AI API Unavailable ──────────────────────────────────────────
def test_edge_case_ai_unavailable_fallback():
    # Save original env
    orig_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = ""
    
    try:
        assert p8.ai_available() is False
        ctx = "FACT: Overall conversion rate is 11.27%."
        res = p8.ask("What is the conversion rate?", ctx)
        assert res["source"] == "fallback"
        assert "11.27%" in res["answer"]
    finally:
        if orig_key is not None:
            os.environ["OPENAI_API_KEY"] = orig_key
