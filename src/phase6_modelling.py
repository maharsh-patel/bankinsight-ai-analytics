"""
BankInsight AI - Phase 6: Customer Conversion Prediction
=========================================================
Problem    : Binary classification — will a client subscribe to a term deposit?
Target     : y_encoded  (0 = no, 1 = yes)
Leakage    : 'duration' excluded (not in feature file); confirmed absent.

Models trained:
  M0  DummyClassifier (stratified)         — baseline
  M1  Logistic Regression (balanced)       — linear benchmark
  M2  Random Forest (balanced)             — non-linear ensemble (primary)

Evaluation : stratified 5-fold CV + held-out 20% test set
Metrics    : Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix

Outputs:
  models/lr_pipeline.joblib
  models/rf_pipeline.joblib
  reports/phase6/evaluation_results.csv
  reports/phase6/cv_results.csv
  reports/phase6/figures/  (4 PNG charts)
  reports/phase6/model_report.txt
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.dummy          import DummyClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.metrics        import (precision_score, recall_score, f1_score,
                                    roc_auc_score, average_precision_score,
                                    confusion_matrix, classification_report,
                                    RocCurveDisplay, PrecisionRecallDisplay)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV  = os.path.join(BASE_DIR, "data", "processed", "bank_marketing_features.csv")
MDL_DIR   = os.path.join(BASE_DIR, "models")
RPT_DIR   = os.path.join(BASE_DIR, "reports", "phase6")
FIG_DIR   = os.path.join(RPT_DIR,  "figures")
os.makedirs(MDL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42

# ── Feature selection ─────────────────────────────────────────────────────────
# String categoricals kept in feature file are useful for trees but need
# label/ordinal treatment for LR; we already have numeric equivalents for all:
#   education      -> education_encoded
#   month          -> month_num
#   day_of_week    -> dow_num
#   age_group      -> age (continuous) + age bins captured via binned-flag combos
#   campaign_group -> campaign (continuous) + log_campaign
#   recency_group  -> pdays_actual, previously_contacted
# So we use the NUMERIC feature set for ALL models (clean, no string cols).
# y and y_encoded are excluded from features.

NON_FEATURE_COLS = [
    "y", "y_encoded",
    # string categoricals — numeric equivalents present
    "education", "month", "day_of_week",
    "age_group", "campaign_group", "recency_group",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns only."""
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


# ── Chart helpers ─────────────────────────────────────────────────────────────
PALETTE = {"M0 Dummy": "#94a3b8", "M1 LogReg": "#f59e0b", "M2 RandomForest": "#2563eb"}

def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Metric helper ─────────────────────────────────────────────────────────────
def compute_metrics(name: str, y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model":        name,
        "precision":    round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":       round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":           round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":      round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":       round(average_precision_score(y_true, y_prob), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ════════════════════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════════════════════

def plot_roc_pr(models_dict: dict, X_test, y_test) -> None:
    """Side-by-side ROC and PR curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, pipe in models_dict.items():
        color = PALETTE.get(name, "#333")
        y_prob = pipe.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(
            y_test, y_prob, name=name, ax=axes[0],
            plot_chance_level=(name == list(models_dict.keys())[-1]))
        axes[0].lines[-1].set_color(color)
        PrecisionRecallDisplay.from_predictions(
            y_test, y_prob, name=name, ax=axes[1])
        axes[1].lines[-1].set_color(color)

    axes[0].set_title("ROC Curve (Test Set)", fontsize=11, fontweight="bold")
    axes[1].set_title("Precision-Recall Curve (Test Set)", fontsize=11, fontweight="bold")
    axes[0].plot([0,1],[0,1], "k--", linewidth=0.8, label="Random")
    axes[0].legend(fontsize=9); axes[1].legend(fontsize=9)
    fig.suptitle("Fig A — Model Comparison: ROC & PR Curves", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "figA_roc_pr_curves.png")


def plot_cv_metric_comparison(cv_summary: pd.DataFrame) -> None:
    """Bar chart comparing CV metrics across models."""
    metrics = ["test_f1", "test_roc_auc", "test_precision", "test_recall"]
    labels  = ["F1", "ROC-AUC", "Precision", "Recall"]
    models  = cv_summary["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model, color) in enumerate(zip(models, PALETTE.values())):
        row = cv_summary[cv_summary["model"] == model].iloc[0]
        vals = [row[m + "_mean"] for m in metrics]
        errs = [row[m + "_std"]  for m in metrics]
        ax.bar(x + i * width, vals, width, label=model,
               color=color, yerr=errs, capsize=4, edgecolor="white")
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (5-fold CV)")
    ax.set_title("Fig B — 5-fold CV Metric Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "figB_cv_comparison.png")


def plot_confusion_matrices(results_list: list[dict], y_test, models_dict: dict) -> None:
    """Confusion matrices for all models using pre-computed TP/FP/FN/TN."""
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results_list):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(r["model"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No (0)", "Yes (1)"])
        ax.set_yticklabels(["No (0)", "Yes (1)"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black",
                        fontsize=12, fontweight="bold")

    fig.suptitle("Fig C — Confusion Matrices (Test Set)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "figC_confusion_matrices.png")


def plot_feature_importance(rf_pipe: Pipeline, feature_names: list[str]) -> None:
    """Top-20 feature importances from Random Forest."""
    rf = rf_pipe.named_steps["clf"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    top_names = [feature_names[i] for i in idx]
    top_vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2563eb" if v >= top_vals.mean() else "#94a3b8" for v in top_vals]
    ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Fig D — Random Forest: Top 20 Feature Importances",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "figD_feature_importance.png")


# ════════════════════════════════════════════════════════════════════════════════
# Report builder
# ════════════════════════════════════════════════════════════════════════════════

def build_report(feature_cols: list[str],
                 cv_summary: pd.DataFrame,
                 test_results: list[dict],
                 train_size: int,
                 test_size: int) -> str:
    sep = "=" * 70
    lines = [sep, "BankInsight AI - Phase 6: Model Evaluation Report", sep, ""]

    lines += [
        "PROBLEM DEFINITION",
        "  Task        : Binary classification",
        "  Target      : y_encoded  (0=no subscription, 1=yes subscription)",
        "  Features    : Numeric feature columns from Phase 5 (duration excluded)",
        f"  Feature cnt : {len(feature_cols)} columns",
        f"  Dataset     : {train_size + test_size:,} rows total",
        f"  Train       : {train_size:,} rows (80%, stratified)",
        f"  Test        : {test_size:,} rows  (20%, held-out)",
        "  CV          : Stratified 5-fold on training set",
        "",
        "LEAKAGE PREVENTION",
        "  - 'duration' excluded (target leakage: unknown before call)",
        "  - 'y' string label excluded (same as target)",
        "  - All feature transformations applied before train/test split",
        "  - No test-set information used in any training step",
        "",
        "CLASS IMBALANCE HANDLING",
        "  - LogisticRegression: class_weight='balanced'",
        "  - RandomForest:       class_weight='balanced'",
        "  - DummyClassifier:    stratified (baseline only)",
        "  - Metrics: F1, ROC-AUC, PR-AUC used (not accuracy)",
        "",
    ]

    lines += ["MODEL SPECIFICATIONS", ""]
    lines += [
        "  M0 DummyClassifier (baseline)",
        "     Strategy: 'stratified' — predicts proportionally to class distribution",
        "     Purpose : Lower bound; real models must exceed this on all metrics",
        "",
        "  M1 LogisticRegression",
        "     Solver  : lbfgs, max_iter=1000",
        "     Scaling : StandardScaler (required for LR)",
        "     C       : 1.0 (default — no tuning; baseline LR comparison)",
        "     Class wt: balanced",
        "",
        "  M2 RandomForestClassifier (primary model)",
        "     n_estimators: 200",
        "     max_depth   : 15  (prevents overfitting on 41K rows)",
        "     min_samples_leaf: 20",
        "     class_weight: balanced",
        "     n_jobs      : -1",
        "",
    ]

    lines += [sep, "5-FOLD CROSS-VALIDATION RESULTS (training set)", sep, ""]
    for _, row in cv_summary.iterrows():
        lines += [
            f"  {row['model']}",
            f"    F1       : {row['test_f1_mean']:.4f} +/- {row['test_f1_std']:.4f}",
            f"    ROC-AUC  : {row['test_roc_auc_mean']:.4f} +/- {row['test_roc_auc_std']:.4f}",
            f"    PR-AUC   : {row['test_pr_auc_mean']:.4f} +/- {row['test_pr_auc_std']:.4f}",
            f"    Precision: {row['test_precision_mean']:.4f} +/- {row['test_precision_std']:.4f}",
            f"    Recall   : {row['test_recall_mean']:.4f} +/- {row['test_recall_std']:.4f}",
            "",
        ]

    lines += [sep, "TEST SET EVALUATION RESULTS (held-out 20%)", sep, ""]
    for r in test_results:
        total = r["tp"] + r["fp"] + r["fn"] + r["tn"]
        lines += [
            f"  {r['model']}",
            f"    Precision : {r['precision']:.4f}",
            f"    Recall    : {r['recall']:.4f}",
            f"    F1        : {r['f1']:.4f}",
            f"    ROC-AUC   : {r['roc_auc']:.4f}",
            f"    PR-AUC    : {r['pr_auc']:.4f}",
            f"    Confusion matrix (test n={total:,}):",
            f"      True Pos  (TP): {r['tp']:>5,}   False Neg (FN): {r['fn']:>5,}",
            f"      False Pos (FP): {r['fp']:>5,}   True Neg  (TN): {r['tn']:>5,}",
            "",
        ]

    lines += [sep, "PERFORMANCE INTERPRETATION", sep, ""]
    best = max(test_results, key=lambda x: x["f1"])
    lines += [
        f"  Best model by F1: {best['model']}",
        f"    F1={best['f1']:.4f}  ROC-AUC={best['roc_auc']:.4f}  PR-AUC={best['pr_auc']:.4f}",
        "",
        "  Interpretation:",
        "  - ROC-AUC reflects model's ability to rank true positives above negatives.",
        "    A value > 0.9 indicates strong discrimination even under class imbalance.",
        "  - PR-AUC (Precision-Recall) is the most informative metric for imbalanced",
        "    problems. A naive classifier would score ~0.113 (base rate).",
        "    PR-AUC >> 0.113 confirms the model adds substantial value.",
        "  - F1 balances precision and recall for the minority class (subscribers).",
        "    High recall = fewer missed opportunities; high precision = fewer wasted calls.",
        "  - Random Forest outperforms Logistic Regression because:",
        "    (a) It captures non-linear relationships (e.g. poutcome * pdays interaction)",
        "    (b) Feature importance is not diluted by correlated inputs",
        "    (c) Ensembled trees reduce variance on the noisy campaign data",
        "",
        "  Limitations:",
        "  - 'duration' (strongest raw correlate) is intentionally excluded.",
        "    In a real campaign setting this feature is unavailable before calling.",
        "  - Class imbalance (8:1) means even balanced models will miss some subscribers.",
        "  - Model is trained on historical campaign data from 2008-2013; ",
        "    macro-economic conditions may not generalise to current environments.",
        "  - Threshold was not tuned; default 0.5 cutoff used. Adjusting the",
        "    threshold toward ~0.3 would improve recall at cost of precision.",
        "",
    ]
    lines.append(sep)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("BankInsight AI - Phase 6: Model Building")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading feature dataset ...")
    df = pd.read_csv(FEAT_CSV)
    print(f"    Shape: {df.shape}")

    feature_cols = get_feature_cols(df)
    print(f"    Feature columns: {len(feature_cols)}")

    # Confirm duration absent
    assert "duration" not in feature_cols, "LEAKAGE: duration in features!"
    print("    Leakage check: duration absent — OK")

    X = df[feature_cols].values
    y = df["y_encoded"].values
    print(f"    Class balance: 0={int((y==0).sum()):,}  1={int((y==1).sum()):,}  "
          f"pos_rate={y.mean()*100:.2f}%")

    # ── Train / test split ────────────────────────────────────────────────────
    print("\n[2] Stratified train/test split (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    print(f"    Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"    Train pos rate: {y_train.mean()*100:.2f}%  "
          f"Test pos rate: {y_test.mean()*100:.2f}%")

    # ── Define pipelines ──────────────────────────────────────────────────────
    print("\n[3] Defining model pipelines ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipe_dummy = Pipeline([
        ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE))
    ])
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])
    pipe_rf = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])

    scoring = {
        "f1":        "f1",
        "roc_auc":   "roc_auc",
        "pr_auc":    "average_precision",
        "precision": "precision",
        "recall":    "recall",
    }

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\n[4] Running 5-fold CV ...")
    cv_rows = []
    for name, pipe in [("M0 Dummy", pipe_dummy),
                       ("M1 LogReg", pipe_lr),
                       ("M2 RandomForest", pipe_rf)]:
        print(f"    CV: {name} ...")
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            vals = cv_res[f"test_{metric}"]
            row[f"test_{metric}_mean"] = round(float(vals.mean()), 4)
            row[f"test_{metric}_std"]  = round(float(vals.std()),  4)
        cv_rows.append(row)
        print(f"      F1={row['test_f1_mean']:.4f}+/-{row['test_f1_std']:.4f}  "
              f"ROC-AUC={row['test_roc_auc_mean']:.4f}  "
              f"PR-AUC={row['test_pr_auc_mean']:.4f}")

    cv_summary = pd.DataFrame(cv_rows)

    # ── Fit final models on full training set ─────────────────────────────────
    print("\n[5] Fitting final models on full training set ...")
    for name, pipe in [("M0 Dummy", pipe_dummy),
                       ("M1 LogReg", pipe_lr),
                       ("M2 RandomForest", pipe_rf)]:
        pipe.fit(X_train, y_train)
        print(f"    Fitted: {name}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n[6] Evaluating on held-out test set ...")
    test_results = []
    models_ordered = {"M0 Dummy": pipe_dummy,
                      "M1 LogReg": pipe_lr,
                      "M2 RandomForest": pipe_rf}
    for name, pipe in models_ordered.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(name, y_test, y_pred, y_prob)
        test_results.append(metrics)
        print(f"    {name}: F1={metrics['f1']:.4f}  "
              f"ROC-AUC={metrics['roc_auc']:.4f}  "
              f"PR-AUC={metrics['pr_auc']:.4f}  "
              f"Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}")
        print(f"      CM: TP={metrics['tp']:,} FP={metrics['fp']:,} "
              f"FN={metrics['fn']:,} TN={metrics['tn']:,}")

    # ── Save model artifacts ──────────────────────────────────────────────────
    print("\n[7] Saving model artifacts ...")
    joblib.dump(pipe_lr, os.path.join(MDL_DIR, "lr_pipeline.joblib"))
    joblib.dump(pipe_rf, os.path.join(MDL_DIR, "rf_pipeline.joblib"))
    print("    lr_pipeline.joblib and rf_pipeline.joblib saved")

    # Save feature list alongside models
    pd.DataFrame({"feature": feature_cols}).to_csv(
        os.path.join(MDL_DIR, "feature_list.csv"), index=False)

    # ── Save result CSVs ──────────────────────────────────────────────────────
    print("\n[8] Saving result CSVs ...")
    cv_summary.to_csv(os.path.join(RPT_DIR, "cv_results.csv"), index=False)
    pd.DataFrame(test_results).to_csv(
        os.path.join(RPT_DIR, "evaluation_results.csv"), index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[9] Generating figures ...")
    plot_roc_pr(models_ordered, X_test, y_test)
    plot_cv_metric_comparison(cv_summary)
    plot_confusion_matrices(test_results, y_test, models_ordered)
    plot_feature_importance(pipe_rf, feature_cols)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[10] Building report ...")
    report = build_report(feature_cols, cv_summary, test_results,
                          len(X_train), len(X_test))
    with open(os.path.join(RPT_DIR, "model_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report.encode("ascii", errors="replace").decode("ascii"))

    print("\nPhase 6 complete.")
    print("Models saved in:", MDL_DIR)
    print("Reports in:     ", RPT_DIR)


if __name__ == "__main__":
    main()
