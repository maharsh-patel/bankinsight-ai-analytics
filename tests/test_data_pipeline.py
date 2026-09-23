import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import phase1_profiling as p1
import phase2_cleaning as p2
import phase5_feature_engineering as p5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV  = os.path.join(BASE_DIR, "bank_marketing.csv")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_clean.csv")
FEAT_CSV  = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_features.csv")


def test_raw_data_file_exists():
    assert os.path.exists(RAW_CSV), f"Raw dataset missing at {RAW_CSV}"
    assert os.path.getsize(RAW_CSV) > 0, "Raw dataset is empty"


def test_phase1_profiling_dictionary():
    df = p1.load_data(RAW_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 41188
    dict_df = p1.build_data_dictionary(df)
    assert len(dict_df) == 21
    assert set(dict_df["column"]) == set(df.columns)
    assert "has_unknown_label" in dict_df.columns


def test_phase2_cleaning_steps():
    df_raw = p1.load_data(RAW_CSV)
    
    # Step 1: Load
    df = p2.step01_load(RAW_CSV)
    assert len(df) == 41188
    
    # Step 2: Remove duplicates
    df_no_dup = p2.step02_remove_duplicates(df.copy())
    assert len(df_no_dup) == 41176  # 12 duplicates removed
    
    # Step 3: Drop duration zero
    df_no_zero = p2.step03_drop_duration_zero(df_no_dup.copy())
    assert len(df_no_zero) == 41172 # 4 duration=0 removed
    
    # Step 4 & 5: Dtypes & sentinels
    df_fixed = p2.step04_fix_dtypes(df_no_zero.copy())
    assert "y_encoded" in df_fixed.columns
    assert set(df_fixed["y_encoded"].unique()) == {0, 1}
    
    df_sentinel = p2.step09_engineer_pdays_flag(df_fixed.copy())
    assert "previously_contacted" in df_sentinel.columns
    assert set(df_sentinel["previously_contacted"].unique()) == {0, 1}


def test_phase2_processed_file():
    assert os.path.exists(CLEAN_CSV), f"Clean CSV missing at {CLEAN_CSV}"
    df_clean = pd.read_csv(CLEAN_CSV)
    assert len(df_clean) == 41172
    assert "y_encoded" in df_clean.columns
    assert df_clean["duration"].min() > 0
    assert df_clean["y_encoded"].isin([0, 1]).all()


def test_phase5_feature_engineering_transforms():
    df_clean = pd.read_csv(CLEAN_CSV)
    
    # Age group
    df_age = p5.add_age_group(df_clean.copy())
    assert "age_group" in df_age.columns
    assert set(df_age["age_group"].unique()).issubset({"18-25", "26-35", "36-45", "46-55", "56-65", "66+"})
    
    # Campaign group
    df_camp = p5.add_campaign_group(df_clean.copy())
    assert "campaign_group" in df_camp.columns
    assert set(df_camp["campaign_group"].unique()).issubset({"first", "optimal", "diminishing", "over_contacted"})
    
    # Pdays clean & recency group
    df_pdays = p5.add_pdays_clean(df_clean.copy())
    assert "pdays_actual" in df_pdays.columns
    df_rec = p5.add_recency_group(df_pdays)
    assert "recency_group" in df_rec.columns


def test_phase5_feature_processed_file():
    assert os.path.exists(FEAT_CSV), f"Feature CSV missing at {FEAT_CSV}"
    df_feat = pd.read_csv(FEAT_CSV)
    assert len(df_feat) == 41172
    # Ensure duration and leakage features are dropped
    assert "duration" not in df_feat.columns
    assert "emp_var_rate" not in df_feat.columns
    assert "nr_employed" not in df_feat.columns
    assert "cons_price_idx" not in df_feat.columns
    # Ensure euribor3m is retained
    assert "euribor3m" in df_feat.columns
    assert "y_encoded" in df_feat.columns
