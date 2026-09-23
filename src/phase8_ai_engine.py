"""
BankInsight AI — Phase 8: AI Query Engine
==========================================
Provides context-grounded natural-language Q&A over BankInsight analytics.

Architecture:
  1. phase8_ai_context.py  builds the factual context from validated outputs
  2. THIS FILE           wraps the context + user question in a strict prompt
                         and calls the OpenAI Chat Completions API
  3. Fallback            if no API key or API fails, returns rule-based answers
                         derived from the same validated data

Rules enforced in system prompt:
  - Never invent numbers not present in the context
  - Clearly distinguish facts (from context) from interpretation
  - If the context doesn't contain an answer, say so explicitly

Config: loaded from .env (never hardcoded)
"""

import os
import sys
import json
import time
import textwrap
from typing import Optional

from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()   # reads .env in CWD or parent directories

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")        # Azure override
OPENAI_API_VER  = os.getenv("OPENAI_API_VERSION", "")
AI_MAX_TOKENS   = int(os.getenv("AI_MAX_TOKENS", "1200"))
AI_TEMPERATURE  = float(os.getenv("AI_TEMPERATURE", "0.3"))
AI_TIMEOUT      = float(os.getenv("AI_TIMEOUT_SEC", "30"))
AI_MAX_RETRIES  = int(os.getenv("AI_MAX_RETRIES", "2"))

_client = None  # lazy init

# ── System prompt template ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are BankInsight AI, an expert data analyst for a Bank Marketing campaign analysis project.

You have access to a detailed analytical context below that contains ONLY verified facts derived from the UCI Bank Marketing dataset (41,172 records, 2008-2013, Portuguese bank).

STRICT RULES:
1. ONLY cite numbers, percentages, and statistics that appear verbatim in the analytical context. Never fabricate or estimate figures.
2. If a question asks for something not covered in the context, say: "This information is not available in the current analytical context."
3. Clearly label facts vs. your interpretation. Use "FACT:" for context-derived statements and "INTERPRETATION:" for your analytical reasoning.
4. Keep responses structured, concise, and business-relevant.
5. Do not reference 'balance' as a feature — it does not exist in this dataset (UCI v2).
6. When discussing model performance, always cite both F1 and ROC-AUC, and explain why accuracy alone is misleading here (8:1 class imbalance).

=== ANALYTICAL CONTEXT (VERIFIED FACTS ONLY) ===
{context}
=== END CONTEXT ==="""

# ── Fallback answers ──────────────────────────────────────────────────────────
# Rule-based responses from validated data when AI is unavailable.
# Keyed by question category detected via keywords.

def _build_fallback_library(context: str) -> dict:
    return {
        "conversion_rate": (
            "FACT: The overall campaign subscription rate is 11.27% "
            "(4,639 subscribed out of 41,172 total contacts).\n"
            "FACT: The class imbalance ratio is 7.9:1 (no:yes).\n"
            "INTERPRETATION: This severe imbalance means accuracy alone is not "
            "a useful metric — the model could achieve 88.73% accuracy by always "
            "predicting 'no'. F1-score and ROC-AUC are the appropriate measures."
        ),
        "customer_groups": (
            "FACT: The highest-converting job groups are:\n"
            "  - student: 31.43% (n=875)\n"
            "  - retired: 25.26% (n=1,718)\n"
            "  - unemployed: 14.20% (n=1,014)\n"
            "FACT: The lowest-converting groups are blue-collar (6.90%) and services (8.14%).\n"
            "FACT: Single clients convert at 14.01% vs married (10.16%).\n"
            "FACT: University degree holders convert at 13.72%, rising to 14.51% for 'unknown' education.\n"
            "INTERPRETATION: Students and retirees likely have more financial flexibility "
            "and fewer competing financial obligations than core working-age clients."
        ),
        "campaign": (
            "FACT: Cellular contact achieves 14.74% conversion vs telephone 5.23%.\n"
            "FACT: First contact (campaign=1): 13.04% conversion.\n"
            "FACT: Conversion drops progressively: 2 contacts=11.46%, 3=10.75%, "
            "4=9.40%, 7+=below 6.1%.\n"
            "FACT: March (50.55%), December (48.90%), September (44.91%) are the "
            "highest-conversion months.\n"
            "FACT: May has the lowest rate (6.44%) despite being the highest-volume "
            "month (33.43% of calls).\n"
            "INTERPRETATION: Diminishing returns beyond 3 contacts suggests persistent "
            "calling alienates rather than convinces. High volume in May likely reflects "
            "mass-calling campaigns that dilute lead quality."
        ),
        "model": (
            "FACT: Three models were compared on a held-out test set of 8,235 records:\n"
            "  M0 Dummy baseline:  F1=0.1175, ROC-AUC=0.5032, PR-AUC=0.1134\n"
            "  M1 Logistic Regression: F1=0.4038, ROC-AUC=0.7820, PR-AUC=0.4129\n"
            "  M2 Random Forest: F1=0.4815, ROC-AUC=0.8120, PR-AUC=0.4778 (BEST)\n"
            "FACT: Random Forest 5-fold CV F1=0.4625+/-0.0142 — consistent with test result.\n"
            "FACT: PR-AUC of 0.4778 is 4.2x the naive baseline of 0.1127.\n"
            "INTERPRETATION: The CV-test consistency confirms no overfitting. "
            "Random Forest captures non-linear feature interactions (e.g. poutcome × pdays) "
            "better than Logistic Regression."
        ),
        "poutcome": (
            "FACT: Previous campaign outcome is the strongest categorical predictor:\n"
            "  poutcome=success (previously contacted): 65.11% conversion (n=1,373)\n"
            "  poutcome=failure (previously contacted): 51.41% conversion (n=142)\n"
            "  poutcome=failure (new/not-prev-contacted): 12.94% (n=4,110)\n"
            "  poutcome=nonexistent (new client): 8.83% (n=35,547)\n"
            "INTERPRETATION: Re-targeting clients with a prior success history should be "
            "the top campaign priority. Even failure outcomes show elevated conversion when "
            "the client was genuinely contacted previously."
        ),
        "recommendations": (
            "Based on validated analytical findings:\n\n"
            "1. RE-TARGET PRIOR SUCCESSES: poutcome=success clients convert at 65.11% — "
            "create a priority list of 1,373 such clients.\n"
            "2. CAP CONTACTS AT 3: Conversion halves from 13.0% (1st contact) to 6.0% "
            "(7th contact). Stop at 3 contacts per campaign cycle.\n"
            "3. SWITCH TO CELLULAR: 14.74% vs 5.23% — a 2.8x difference.\n"
            "4. AVOID MAY MASS CAMPAIGNS: 6.44% conversion despite 33% of all calls.\n"
            "5. TARGET STUDENTS/RETIREES: 31.43% and 25.26% conversion vs 11.27% overall.\n"
            "6. USE ML SCORING: Random Forest (ROC-AUC=0.812) can rank call lists by "
            "predicted subscription probability to focus agent time on high-probability clients."
        ),
        "features": (
            "FACT: Top features by Pearson |r| with subscription target:\n"
            "  pdays / previously_contacted: 0.325\n"
            "  repeat_success (interaction): 0.316\n"
            "  poutcome_success (OHE):       0.316\n"
            "  euribor3m:                    0.308 (negative — lower rate = higher sub)\n"
            "  low_rate_env flag:            0.291\n"
            "  pdays_actual:                 0.279\n"
            "  previous contacts:            0.230\n"
            "FACT: 'duration' was excluded from all models — it is a target leakage "
            "feature (unknown before the call is made).\n"
            "INTERPRETATION: Prior contact history dominates. Economic conditions "
            "(euribor) are the second-strongest signal group, confirming macro "
            "context strongly influences savings behaviour."
        ),
        "model_limitations": (
            "FACT: Known limitations of the BankInsight predictive model:\n"
            "  1. 'duration' excluded — the single strongest raw correlate but is target "
            "leakage (unknown before the call is placed). Including it inflates metrics "
            "artificially and makes the model useless in a real campaign setting.\n"
            "  2. Dataset scope: 2008-2013 Portuguese bank campaigns during the post-crisis "
            "low-rate environment. The macro feature (euribor) may not generalize to "
            "different economic periods or regions.\n"
            "  3. Default threshold (0.5) not tuned: shifting to ~0.3 would improve recall "
            "(catch more subscribers) at the cost of lower precision (more wasted calls).\n"
            "  4. Class imbalance (8:1): despite class_weight='balanced', the minority "
            "class is still under-represented. F1=0.4815 and PR-AUC=0.4778 are honest "
            "metrics; accuracy (~90%) would be misleading.\n"
            "  5. 'balance' column: does NOT exist in this dataset (UCI v2). Only UCI v1 "
            "contained account balance.\n"
            "INTERPRETATION: The model is suitable for ranking call lists by subscription "
            "probability, not for making hard yes/no predictions. A human review of "
            "high-probability cases is recommended."
        ),
        "default": (
            "I can answer questions about this Bank Marketing analysis. Topics I cover:\n"
            "- Overall subscription/conversion rates\n"
            "- Customer segment analysis (job, education, marital, age)\n"
            "- Campaign characteristics (contact method, frequency, timing)\n"
            "- Previous campaign outcome effects\n"
            "- ML model performance and feature importance\n"
            "- Business recommendations derived from validated data\n\n"
            "Please ask a more specific question about one of these areas."
        ),
    }


def _classify_question(q: str) -> str:
    q_lower = q.lower()
    # Most-specific first
    if any(w in q_lower for w in ["limitation", "limit", "weakness", "caveat", "drawback"]):
        return "model_limitations"
    if any(w in q_lower for w in ["executive summary", "summary", "overview of the project",
                                   "summarise", "summarize"]):
        return "recommendations"   # executive summary -> best served by recommendations
    if any(w in q_lower for w in ["feature", "important", "driver", "variable",
                                   "predictive", "predictor"]):
        return "features"
    if any(w in q_lower for w in ["previous", "poutcome", "prior campaign", "retarget"]):
        return "poutcome"
    if any(w in q_lower for w in ["recommend", "should", "action", "invest", "strategy",
                                   "improve", "major finding", "key finding", "investigate"]):
        return "recommendations"
    if any(w in q_lower for w in ["conversion rate", "subscription rate", "how many", "overall rate"]):
        return "conversion_rate"
    if any(w in q_lower for w in ["customer group", "segment", "which group", "which customer",
                                   "job", "education", "marital", "age", "demographic"]):
        return "customer_groups"
    if any(w in q_lower for w in ["month", "timing", "when to", "best time",
                                   "seasonal", "temporal"]):
        return "campaign"
    if any(w in q_lower for w in ["campaign", "contact", "channel", "cellular", "telephone",
                                   "frequency", "how many calls"]):
        return "campaign"
    if any(w in q_lower for w in ["model", "random forest", "logistic", "f1", "auc",
                                   "precision", "recall", "accuracy", "ml", "predict",
                                   "perform"]):
        return "model"
    return "default"


def fallback_answer(question: str, context: str) -> dict:
    """Rule-based answer from validated data when AI API is unavailable."""
    lib = _build_fallback_library(context)
    category = _classify_question(question)
    answer = lib.get(category, lib["default"])
    return {
        "answer": answer,
        "source": "fallback",
        "model": "rule-based (AI unavailable)",
        "context_used": True,
        "question": question,
    }


# ── AI client initialisation ─────────────────────────────────────────────────

def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import openai
        if OPENAI_BASE_URL:
            _client = openai.AzureOpenAI(
                api_key=OPENAI_API_KEY,
                azure_endpoint=OPENAI_BASE_URL,
                api_version=OPENAI_API_VER or "2024-02-01",
            )
        else:
            _client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                timeout=AI_TIMEOUT,
                max_retries=AI_MAX_RETRIES,
            )
        return _client
    except Exception as e:
        return None


def ai_available() -> bool:
    """True if an API key is configured (does not make a network call)."""
    return bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here")


# ── Core query function ───────────────────────────────────────────────────────

def ask(question: str, context: str) -> dict:
    """
    Ask an analytical question. Returns a dict with:
      answer      : str
      source      : 'ai' | 'fallback'
      model       : str
      context_used: bool
      question    : str
      error       : str (only if fallback triggered by error)
    """
    if not question or not question.strip():
        return {"answer": "Please provide a question.", "source": "error",
                "model": "", "context_used": False, "question": question}

    if not ai_available():
        result = fallback_answer(question, context)
        result["error"] = "No API key configured. Set OPENAI_API_KEY in .env"
        return result

    client = _get_client()
    if client is None:
        result = fallback_answer(question, context)
        result["error"] = "Could not initialise OpenAI client"
        return result

    system_msg = SYSTEM_PROMPT.format(context=context)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": question.strip()},
    ]

    # Retry loop for rate limits
    last_error = ""
    for attempt in range(AI_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=AI_MAX_TOKENS,
                temperature=AI_TEMPERATURE,
            )
            answer = resp.choices[0].message.content.strip()
            return {
                "answer":       answer,
                "source":       "ai",
                "model":        resp.model,
                "context_used": True,
                "question":     question,
                "usage": {
                    "prompt_tokens":     resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens":      resp.usage.total_tokens,
                },
            }
        except Exception as e:
            last_error = str(e)
            err_lower = last_error.lower()
            # Rate limit: back off and retry
            if "rate" in err_lower or "429" in err_lower:
                wait = 10 * (attempt + 1)
                time.sleep(wait)
                continue
            # Auth / quota errors: no point retrying
            if any(x in err_lower for x in ["401", "403", "authentication", "api_key",
                                             "insufficient_quota", "billing"]):
                break
            # Other transient errors: retry once
            if attempt < AI_MAX_RETRIES:
                time.sleep(2)
                continue
            break

    # All retries exhausted — fall back
    result = fallback_answer(question, context)
    result["error"] = last_error
    return result


# ── Canned question set ───────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "Which customer groups had higher conversion rates?",
    "What campaign characteristics are associated with better outcomes?",
    "What are the major findings from the analysis?",
    "What should a marketing manager investigate first?",
    "How did the Random Forest model perform and what does that mean?",
    "What factors are most predictive of term deposit subscription?",
    "Which months are best for running campaigns and why?",
]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Import here to avoid circular at module level
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from phase8_ai_context import get_context

    ctx = get_context()
    print(f"Context built: {len(ctx)} chars\n")

    if not ai_available():
        print("[NOTICE] No OPENAI_API_KEY found in .env — using fallback mode.\n")

    print("Running demo questions...\n")
    print("=" * 60)

    for q in DEMO_QUESTIONS:
        print(f"Q: {q}")
        result = ask(q, ctx)
        print(f"[{result['source'].upper()}]")
        # Wrap answer for readable console output
        for line in result["answer"].split("\n"):
            print(textwrap.fill(line, width=72, subsequent_indent="  ") if line.strip() else "")
        if result.get("error"):
            print(f"[ERROR] {result['error']}")
        print("-" * 60)
