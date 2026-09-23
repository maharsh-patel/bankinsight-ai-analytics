import os
import sys
import joblib
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
import phase6_modelling as p6

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV  = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_features.csv")
LR_MODEL  = os.path.join(BASE_DIR, "models", "lr_pipeline.joblib")
RF_MODEL  = os.path.join(BASE_DIR, "models", "rf_pipeline.joblib")
EVAL_CSV  = os.path.join(BASE_DIR, "reports", "phase6", "evaluation_results.csv")


def test_model_files_exist():
    assert os.path.exists(LR_MODEL), f"Logistic Regression pipeline missing at {LR_MODEL}"
    assert os.path.exists(RF_MODEL), f"Random Forest pipeline missing at {RF_MODEL}"
    assert os.path.exists(EVAL_CSV), f"Evaluation CSV missing at {EVAL_CSV}"


def test_model_loading_and_prediction():
    lr_pipe = joblib.load(LR_MODEL)
    rf_pipe = joblib.load(RF_MODEL)
    
    df_feat = pd.read_csv(FEAT_CSV)
    feat_cols = p6.get_feature_cols(df_feat)
    
    sample_x = df_feat[feat_cols].head(10)
    
    # Predict binary labels
    lr_preds = lr_pipe.predict(sample_x)
    rf_preds = rf_pipe.predict(sample_x)
    
    assert len(lr_preds) == 10
    assert len(rf_preds) == 10
    assert set(lr_preds).issubset({0, 1})
    assert set(rf_preds).issubset({0, 1})
    
    # Predict probabilities
    lr_probs = lr_pipe.predict_proba(sample_x)[:, 1]
    rf_probs = rf_pipe.predict_proba(sample_x)[:, 1]
    
    assert (lr_probs >= 0.0).all() and (lr_probs <= 1.0).all()
    assert (rf_probs >= 0.0).all() and (rf_probs <= 1.0).all()


def test_model_evaluation_metrics():
    eval_df = pd.read_csv(EVAL_CSV)
    assert not eval_df.empty
    assert "model" in eval_df.columns
    assert "f1" in eval_df.columns
    assert "roc_auc" in eval_df.columns
    assert "precision" in eval_df.columns
    assert "recall" in eval_df.columns
    
    # Check that M2 (RandomForest) achieves expected performance
    rf_row = eval_df[eval_df["model"].str.contains("RandomForest")]
    assert not rf_row.empty
    roc_auc = float(rf_row["roc_auc"].iloc[0])
    assert roc_auc > 0.70, f"Random Forest ROC-AUC low: {roc_auc}"


def test_explainability_feature_importances():
    rf_pipe = joblib.load(RF_MODEL)
    rf_clf  = rf_pipe.named_steps["clf"]
    
    df_feat   = pd.read_csv(FEAT_CSV)
    feat_cols = p6.get_feature_cols(df_feat)
    
    assert hasattr(rf_clf, "feature_importances_")
    importances = rf_clf.feature_importances_
    assert len(importances) == len(feat_cols)
    assert pytest.approx(sum(importances), abs=1e-4) == 1.0
    
    # Top features check
    top_idx = np.argsort(importances)[::-1][:5]
    top_features = [feat_cols[i] for i in top_idx]
    assert len(top_features) == 5
