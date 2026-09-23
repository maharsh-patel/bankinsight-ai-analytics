import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from phase8_ai_context import get_context
from phase8_ai_engine  import ask, ai_available, _build_fallback_library


def test_ai_context_generation():
    ctx = get_context()
    assert isinstance(ctx, str)
    assert len(ctx) > 1000
    assert "41,172" in ctx or "41172" in ctx
    assert "subscription rate" in ctx.lower() or "conversion" in ctx.lower()


def test_ai_engine_ask_fallback_mode():
    ctx = get_context()
    # Test query under fallback (or live if key present)
    res = ask("Which customer groups had higher conversion rates?", ctx)
    assert isinstance(res, dict)
    assert "answer" in res
    assert "source" in res
    assert len(res["answer"]) > 50
    assert res["source"] in ("ai", "fallback")


def test_ai_fallback_library():
    ctx = get_context()
    lib = _build_fallback_library(ctx)
    assert "conversion_rate" in lib
    assert "customer_groups" in lib
    assert "campaign" in lib
    assert "model" in lib
    assert "FACT:" in lib["conversion_rate"]


def test_ai_availability_flag():
    avail = ai_available()
    assert isinstance(avail, bool)
    # If no key, should be False
    if not os.getenv("OPENAI_API_KEY"):
        assert avail is False
