import os
import sys
import re
import pytest
import importlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")


def get_src_python_files():
    py_files = []
    for root, _, files in os.walk(SRC_DIR):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    return py_files


def test_no_hardcoded_absolute_paths():
    """Ensure source files use relative BASE_DIR paths instead of hardcoded drive letters."""
    pattern = re.compile(r'([A-Z]:[/\\](?:[A-Za-z0-9_-]+[/\\])+[^\s"\'\)]*|/home/[^\n"\']+|/Users/[^\n"\']+)')
    py_files = get_src_python_files()
    
    violations = []
    for filepath in py_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                # Ignore comments
                if line.strip().startswith("#"):
                    continue
                match = pattern.search(line)
                if match:
                    violations.append(f"{os.path.basename(filepath)}:{line_idx} -> {match.group(0)}")
                    
    assert not violations, f"Found hardcoded absolute paths:\n" + "\n".join(violations)


def test_no_exposed_api_secrets():
    """Ensure no real OpenAI API keys are committed in code or env files."""
    secret_pattern = re.compile(r'sk-[a-zA-Z0-9]{32,}')
    py_files = get_src_python_files()
    
    violations = []
    for filepath in py_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                match = secret_pattern.search(line)
                if match:
                    violations.append(f"{os.path.basename(filepath)}:{line_idx}")
                    
    assert not violations, f"Found potential exposed secrets:\n" + "\n".join(violations)


def test_src_modules_import_cleanly():
    """Verify that all src modules can be imported without raising ImportError."""
    sys.path.insert(0, SRC_DIR)
    src_modules = [
        "phase1_profiling",
        "phase2_cleaning",
        "phase3_eda",
        "phase4_sql_analytics",
        "phase5_feature_engineering",
        "phase6_modelling",
        "phase7_dashboard_data",
        "phase8_ai_context",
        "phase8_ai_engine",
        "phase8_ai_server",
    ]
    
    imported = []
    for mod_name in src_modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None
        imported.append(mod_name)
        
    assert len(imported) == len(src_modules)


def test_reproducibility_seed_configuration():
    """Verify that modeling module specifies a fixed random seed."""
    import phase6_modelling as p6
    assert hasattr(p6, "RANDOM_STATE")
    assert p6.RANDOM_STATE == 42
