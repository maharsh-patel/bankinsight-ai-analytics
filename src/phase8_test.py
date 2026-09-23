"""
BankInsight AI — Phase 8: AI Test Harness
==========================================
Tests the AI layer with real project questions.
Works in both AI mode (API key present) and fallback mode.

Usage:
    python src/phase8_test.py
    python src/phase8_test.py --question "Which job types convert best?"
"""

import os
import sys
import json
import argparse
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase8_ai_context import get_context
from phase8_ai_engine  import ask, ai_available, DEMO_QUESTIONS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RPT_DIR  = os.path.join(BASE_DIR, "reports", "phase8")
os.makedirs(RPT_DIR, exist_ok=True)


TEST_QUESTIONS = [
    # Specified in the prompt
    "Which customer groups had higher conversion rates?",
    "What campaign characteristics are associated with better outcomes?",
    "What are the major findings from the analysis?",
    "What should a marketing manager investigate?",
    # Extended set
    "How did the Random Forest model perform?",
    "What factors are most predictive of term deposit subscription?",
    "Which months are best for running campaigns and why?",
    "What is the effect of previous campaign outcome on subscription?",
    "What are the limitations of the predictive model?",
    "Give me an executive summary of the BankInsight project.",
]


def run_tests(questions: list[str]) -> list[dict]:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 66)
    print("BankInsight AI - Phase 8: AI Layer Test")
    print("=" * 66)

    ctx = get_context()
    print(f"Context: {len(ctx)} chars / ~{len(ctx)//4} estimated tokens")
    print(f"AI mode: {'LIVE (OpenAI)' if ai_available() else 'FALLBACK (rule-based)'}")
    print()

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i:02d}/{len(questions):02d}] Q: {q}")
        result = ask(q, ctx)
        result["test_num"] = i

        source_tag = f"[{result['source'].upper()}]"
        if result.get("error") and result["source"] == "ai":
            source_tag += " [ERROR]"

        print(f"       {source_tag}")
        # Print answer (word-wrapped for console)
        for line in result["answer"].split("\n"):
            wrapped = textwrap.fill(line, width=68, subsequent_indent="       ") if line.strip() else ""
            print("       " + wrapped if wrapped else "")

        if result.get("error") and result["source"] == "fallback":
            print(f"       [FALLBACK REASON] {result['error'][:80]}")
        if result.get("usage"):
            u = result["usage"]
            print(f"       [TOKENS] prompt={u['prompt_tokens']} completion={u['completion_tokens']} total={u['total_tokens']}")
        print("-" * 66)
        results.append(result)

    return results


def save_results(results: list[dict]) -> None:
    # Save JSON
    out_json = os.path.join(RPT_DIR, "ai_test_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_json}")

    # Save readable text report
    out_txt = os.path.join(RPT_DIR, "ai_test_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("BankInsight AI - Phase 8: AI Test Report\n")
        f.write("=" * 66 + "\n\n")
        f.write(f"Total questions: {len(results)}\n")
        ai_count  = sum(1 for r in results if r["source"] == "ai")
        fb_count  = sum(1 for r in results if r["source"] == "fallback")
        f.write(f"AI responses:    {ai_count}\n")
        f.write(f"Fallback:        {fb_count}\n\n")
        f.write("=" * 66 + "\n\n")
        for r in results:
            f.write(f"Q{r['test_num']:02d}: {r['question']}\n")
            f.write(f"Source: {r['source']}\n")
            f.write(f"Answer:\n{r['answer']}\n")
            if r.get("usage"):
                u = r["usage"]
                f.write(f"Tokens: {u['total_tokens']}\n")
            f.write("-" * 66 + "\n\n")
    print(f"Report saved:  {out_txt}")

    # Summary
    print(f"\nSummary: {len(results)} questions | "
          f"AI={sum(1 for r in results if r['source']=='ai')} | "
          f"Fallback={sum(1 for r in results if r['source']=='fallback')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BankInsight AI test harness")
    parser.add_argument("--question", "-q", type=str, default="",
                        help="Ask a single custom question")
    args = parser.parse_args()

    if args.question:
        ctx = get_context()
        print(f"AI mode: {'LIVE' if ai_available() else 'FALLBACK'}\n")
        result = ask(args.question, ctx)
        print(f"[{result['source'].upper()}]")
        print(result["answer"])
        if result.get("error"):
            print(f"\n[FALLBACK REASON] {result['error']}")
    else:
        results = run_tests(TEST_QUESTIONS)
        save_results(results)
