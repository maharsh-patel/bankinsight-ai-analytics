import os
import sys
import json
import pytest

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "dashboard", "dashboard_data.json")
HTML_PATH = os.path.join(BASE_DIR, "dashboard", "index.html")


def test_dashboard_file_existence():
    assert os.path.exists(JSON_PATH), f"dashboard_data.json missing at {JSON_PATH}"
    assert os.path.exists(HTML_PATH), f"index.html missing at {HTML_PATH}"


def test_dashboard_json_structure():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    required_keys = [
        "kpis", "by_job", "by_education", "by_marital",
        "by_housing_loan", "by_contact", "by_campaign_count",
        "by_poutcome", "by_duration_bucket", "by_duration_outcome",
        "by_temporal", "segments", "model_eval", "model_cv", "ai_insights"
    ]
    
    for key in required_keys:
        assert key in data, f"Key '{key}' missing from dashboard_data.json"
        
    kpis = data["kpis"]
    assert kpis["total_clients"] == 41172
    assert kpis["total_subscribed"] == 4639
    assert pytest.approx(kpis["subscription_rate"], abs=0.01) == 11.27
    assert kpis["balance_available"] is False


def test_dashboard_ai_insights_non_empty():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    insights = data["ai_insights"]
    assert isinstance(insights, list)
    assert len(insights) >= 5
    for item in insights:
        assert "headline" in item
        assert "detail" in item
        assert "evidence" in item


def test_dashboard_html_content():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
        
    assert "<!DOCTYPE html>" in html
    assert "BankInsight AI" in html
    assert "echarts" in html.lower()
    assert "dashboard_data.json" in html or "fetch" in html
