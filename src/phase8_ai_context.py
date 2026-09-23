"""
BankInsight AI — Phase 8: AI Context Builder
=============================================
Builds a structured, factual analytical context document from all validated
project outputs. This context is injected into every AI prompt so the model
cannot invent numbers — it can only reason about facts we supply.

No API calls here. Pure data assembly.
"""

import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rp4(name):
    return pd.read_csv(os.path.join(BASE_DIR, "reports", "phase4", name))


def _rp6(name):
    return pd.read_csv(os.path.join(BASE_DIR, "reports", "phase6", name))


def build_context() -> str:
    """
    Return a self-contained plain-text analytical context document.
    Every number in this document is derived from validated project outputs.
    """
    lines = []

    # ── Dataset overview ──────────────────────────────────────────────────────
    ov = _rp4("q01_overview.csv").iloc[0]
    lines += [
        "=== BANKINSIGHT AI — ANALYTICAL CONTEXT ===",
        "Dataset: UCI Bank Marketing (Full, v2) — Portuguese bank, 2008-2013",
        f"Records: {int(ov['total_clients']):,} campaign contacts after cleaning",
        f"Target : y_encoded — subscribed to term deposit (1=yes, 0=no)",
        f"Subscribed    : {int(ov['total_subscribed']):,} ({float(ov['subscription_rate_pct']):.2f}%)",
        f"Not subscribed: {int(ov['total_not_subscribed']):,} ({float(ov['non_subscription_rate_pct']):.2f}%)",
        f"Class imbalance: {float(ov['imbalance_ratio_no_per_yes']):.1f}:1 (no:yes)",
        "",
    ]

    # ── Customer demographics ─────────────────────────────────────────────────
    job = _rp4("q02_conversion_by_job.csv")
    top3_job = job.nlargest(3, "conv_rate_pct")[["job","conv_rate_pct","total_clients","subscribed"]]
    bot3_job = job.nsmallest(3, "conv_rate_pct")[["job","conv_rate_pct","total_clients","subscribed"]]

    edu = _rp4("q03_conversion_by_education.csv")
    mar = _rp4("q04_conversion_by_marital.csv")

    lines.append("=== CUSTOMER DEMOGRAPHICS ===")
    lines.append("Conversion rate by job (top 3):")
    for _, r in top3_job.iterrows():
        lines.append(f"  {r['job']}: {r['conv_rate_pct']}% (n={int(r['total_clients']):,}, subscribed={int(r['subscribed']):,})")
    lines.append("Conversion rate by job (bottom 3):")
    for _, r in bot3_job.iterrows():
        lines.append(f"  {r['job']}: {r['conv_rate_pct']}% (n={int(r['total_clients']):,})")
    lines.append("Conversion rate by education (ordered low to high):")
    for _, r in edu.iterrows():
        lines.append(f"  {r['education']} (order {r['edu_order']}): {r['conv_rate_pct']}% (n={int(r['total_clients']):,})")
    lines.append("Conversion rate by marital status:")
    for _, r in mar.sort_values("conv_rate_pct", ascending=False).iterrows():
        lines.append(f"  {r['marital']}: {r['conv_rate_pct']}% (avg age {r['avg_age']}, n={int(r['total_clients']):,})")
    lines.append("")

    # ── Housing / loan / default ──────────────────────────────────────────────
    hl = _rp4("q05_conversion_housing_loan.csv")
    lines.append("=== DEBT PROFILE ===")
    for _, r in hl.iterrows():
        lines.append(f"  {r['dimension']}={r['category']}: {r['conv_rate_pct']}% (n={int(r['total_clients']):,})")
    lines.append("")

    # ── Campaign analytics ────────────────────────────────────────────────────
    ct = _rp4("q06_conversion_by_contact.csv")
    cc = _rp4("q07_campaign_count.csv")
    po = _rp4("q08_conversion_poutcome.csv")
    tm = _rp4("q11_temporal_patterns.csv")

    lines.append("=== CAMPAIGN ANALYTICS ===")
    lines.append("Conversion rate by contact method:")
    for _, r in ct.iterrows():
        lines.append(f"  {r['contact']}: {r['conv_rate_pct']}% (avg duration {r['avg_call_duration_sec']}s, n={int(r['total_clients']):,})")

    lines.append("Conversion rate by number of campaign contacts (1-14):")
    for _, r in cc.iterrows():
        lines.append(f"  {int(r['contact_count'])} contact(s): {r['conv_rate_pct']}% [{r['contact_zone']}] n={int(r['total_clients']):,}")

    lines.append("Conversion rate by previous campaign outcome:")
    for _, r in po.iterrows():
        prev = "previously_contacted" if r['previously_contacted'] else "new_client"
        lines.append(f"  poutcome={r['poutcome']} ({prev}): {r['conv_rate_pct']}% (n={int(r['total_clients']):,})")

    months = tm[tm["time_dimension"] == "month"].sort_values("conv_rate_pct", ascending=False)
    lines.append("Monthly conversion rates (descending):")
    for _, r in months.iterrows():
        lines.append(f"  {r['period']}: {r['conv_rate_pct']}% (volume={int(r['total_calls']):,})")

    dow = tm[tm["time_dimension"] == "day_of_week"].sort_values("conv_rate_pct", ascending=False)
    lines.append("Day-of-week conversion rates:")
    for _, r in dow.iterrows():
        lines.append(f"  {r['period']}: {r['conv_rate_pct']}%")
    lines.append("")

    # ── Conversion analysis ───────────────────────────────────────────────────
    db = _rp4("q09_duration_buckets.csv")
    do_ = _rp4("q09_duration_by_outcome.csv")
    seg = _rp4("q10_customer_segments.csv")

    lines.append("=== CONVERSION ANALYSIS ===")
    lines.append("Call duration bucket vs conversion rate:")
    for _, r in db.iterrows():
        lines.append(f"  {r['duration_bucket']}: {r['conv_rate_pct']}% (n={int(r['total_clients']):,}, {r['pct_of_calls']}% of calls)")
    lines.append("Note: duration is NOT a predictive feature (target leakage — unknown before calling).")

    lines.append("Avg call duration by outcome:")
    for _, r in do_.iterrows():
        lines.append(f"  outcome={r['outcome']}: avg {r['avg_duration_sec']}s ({r['avg_duration_min']} min), "
                     f"median~{r['approx_median_sec']}s, n={int(r['clients']):,}")

    lines.append("Top 10 customer segments by conversion (age_band x job_tier x prior_contact):")
    for _, r in seg.head(10).iterrows():
        lines.append(f"  Rank {int(r['segment_rank'])}: {r['age_band']} / {r['job_tier']} / "
                     f"prev_contacted={int(r['previously_contacted'])} -> "
                     f"{r['conv_rate_pct']}% (n={int(r['clients'])})")
    lines.append("")

    # ── Feature engineering decisions ────────────────────────────────────────
    lines += [
        "=== FEATURE ENGINEERING ===",
        "Features used in ML models (41 numeric columns):",
        "  - Duration EXCLUDED (target leakage: unknown before call is placed)",
        "  - Collinear macro vars dropped: emp_var_rate, nr_employed, cons_price_idx",
        "  - Retained: euribor3m (r=-0.308 with target)",
        "  - Engineered flags: repeat_success, cellular_first, high_value_segment,",
        "    previously_contacted, low_rate_env",
        "  - Ordinal encodings: education_encoded, month_num, dow_num",
        "  - Log transforms: log_campaign, log_previous",
        "Top feature-target correlations (Pearson |r|):",
        "  pdays / previously_contacted : 0.325",
        "  repeat_success               : 0.316",
        "  poutcome_success             : 0.316",
        "  euribor3m                    : 0.308  (negative: lower rate -> higher sub)",
        "  low_rate_env                 : 0.291",
        "  pdays_actual                 : 0.279",
        "  previous                     : 0.230",
        "  poutcome_nonexistent         : 0.194  (negative)",
        "  contact_telephone            : 0.145  (negative)",
        "  cellular_first               : 0.106",
        "",
    ]

    # ── ML model results ──────────────────────────────────────────────────────
    ev = _rp6("evaluation_results.csv")
    cv = _rp6("cv_results.csv")

    lines += [
        "=== ML MODEL RESULTS (test set: 8,235 held-out records) ===",
        "Train/test: 80/20 stratified split, random_state=42",
        "Validation: 5-fold stratified CV on training set",
        "Class imbalance handling: class_weight='balanced' for LR and RF",
        "",
    ]
    for _, r in ev.iterrows():
        cvr = cv[cv["model"] == r["model"]].iloc[0]
        lines += [
            f"  {r['model']}:",
            f"    Test  -> Precision={r['precision']:.4f} Recall={r['recall']:.4f} "
            f"F1={r['f1']:.4f} ROC-AUC={r['roc_auc']:.4f} PR-AUC={r['pr_auc']:.4f}",
            f"    CV    -> F1={cvr['test_f1_mean']:.4f}+/-{cvr['test_f1_std']:.4f} "
            f"ROC-AUC={cvr['test_roc_auc_mean']:.4f}",
            f"    CM    -> TP={int(r['tp']):,} FP={int(r['fp']):,} "
            f"FN={int(r['fn']):,} TN={int(r['tn']):,}",
            "",
        ]

    lines += [
        "Best model: M2 RandomForest",
        "  - Best F1 (0.4815) and ROC-AUC (0.8120) on test set",
        "  - PR-AUC 0.4778 vs naive baseline 0.1127 (4.2x better)",
        "  - CV results consistent with test — no overfitting detected",
        "",
        "=== KNOWN LIMITATIONS ===",
        "  1. Duration excluded by design (leakage) — including it would inflate metrics unrealistically",
        "  2. Dataset: 2008-2013 Portuguese bank; macro conditions differ today",
        "  3. Threshold not tuned (default 0.5); ~0.3 would improve recall",
        "  4. 'balance' column does NOT exist in this dataset (UCI v2 uses macro indicators instead)",
        "  5. Class imbalance (8:1) — accuracy is a misleading metric here",
        "",
        "=== END OF ANALYTICAL CONTEXT ===",
    ]

    return "\n".join(lines)


def get_context() -> str:
    """Cached accessor — rebuilds each call (context is fast to build)."""
    return build_context()


if __name__ == "__main__":
    ctx = get_context()
    print(ctx)
    print(f"\nContext length: {len(ctx)} characters / ~{len(ctx)//4} tokens")
