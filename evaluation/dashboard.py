"""
evaluation/dashboard.py

Visualise evaluation results and export error analysis.

Outputs
-------
evaluation/eval_dashboard.png  — dual bar chart: Token F1 and Exact Match
                                  by clause category
evaluation/error_analysis.csv  — low-F1 cases (F1 < 0.2) for qualitative review

Input
-----
evaluation/eval_results.csv    — written by generation_eval.py

Run
---
    python evaluation/dashboard.py
    python evaluation/dashboard.py --results evaluation/eval_results.csv
                                   --f1-threshold 0.2
                                   --error-limit 20
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")   # headless — works in terminal environments without a display

import matplotlib.pyplot as plt
import pandas as pd


# ── colour palette (colourblind-friendly) ────────────────────────────────────
F1_BAR_COLOUR  = "#5DCAA5"   # teal-green
EM_BAR_COLOUR  = "#AFA9EC"   # soft purple
MEAN_LINE_CLR  = "#D85A30"   # warm orange


def build_dashboard(
    results_path:  str   = "evaluation/eval_results.csv",
    output_path:   str   = "evaluation/eval_dashboard.png",
    error_path:    str   = "evaluation/error_analysis.csv",
    f1_threshold:  float = 0.2,
    error_limit:   int   = 20,
) -> None:
    """
    Build evaluation dashboard and error analysis CSV.

    Parameters
    ----------
    results_path  : path to eval_results.csv
    output_path   : where to save the PNG
    error_path    : where to save low-F1 cases
    f1_threshold  : F1 below this value is considered a failure
    error_limit   : max rows in error_analysis.csv
    """
    df = pd.read_csv(results_path)

    # ── per-category metrics ──────────────────────────────────────────────────
    cat_metrics = (
        df.groupby("category")[["em", "f1"]]
        .mean()
        .sort_values("f1")
    )

    mean_f1 = cat_metrics["f1"].mean()
    mean_em = cat_metrics["em"].mean()

    # ── figure layout ─────────────────────────────────────────────────────────
    n_cats   = len(cat_metrics)
    fig_h    = max(8, n_cats * 0.35)   # scale height with category count
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_h))

    # ── left panel: Token F1 ──────────────────────────────────────────────────
    cat_metrics["f1"].plot(
        kind="barh", ax=axes[0], color=F1_BAR_COLOUR, edgecolor="white",
    )
    axes[0].axvline(
        mean_f1, color=MEAN_LINE_CLR, linestyle="--", linewidth=1.5,
        label=f"Mean F1: {mean_f1:.2f}",
    )
    axes[0].set_title("Token F1 by clause category", fontsize=13, pad=12)
    axes[0].set_xlabel("F1 score", fontsize=11)
    axes[0].set_xlim(0, 1.05)
    axes[0].legend(fontsize=10)
    axes[0].tick_params(axis="y", labelsize=9)

    # annotate each bar
    for bar, val in zip(axes[0].patches, cat_metrics["f1"]):
        axes[0].text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left", fontsize=8,
        )

    # ── right panel: Exact Match ──────────────────────────────────────────────
    cat_metrics["em"].plot(
        kind="barh", ax=axes[1], color=EM_BAR_COLOUR, edgecolor="white",
    )
    axes[1].axvline(
        mean_em, color=MEAN_LINE_CLR, linestyle="--", linewidth=1.5,
        label=f"Mean EM: {mean_em:.2f}",
    )
    axes[1].set_title("Exact Match by clause category", fontsize=13, pad=12)
    axes[1].set_xlabel("EM score", fontsize=11)
    axes[1].set_xlim(0, 1.05)
    axes[1].legend(fontsize=10)
    axes[1].tick_params(axis="y", labelsize=9)

    for bar, val in zip(axes[1].patches, cat_metrics["em"]):
        axes[1].text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left", fontsize=8,
        )

    # ── overall title and layout ──────────────────────────────────────────────
    n_cases = len(df)
    plt.suptitle(
        f"CUAD RAG Chatbot — Evaluation Results\n"
        f"({n_cases} cases · Overall F1 {mean_f1:.3f} · EM {mean_em:.3f})",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved to {output_path}")
    plt.close()

    # ── error analysis ────────────────────────────────────────────────────────
    low_f1 = (
        df[df["f1"] < f1_threshold]
        [["contract", "category", "question", "ground_truth", "prediction", "f1", "em"]]
        .sort_values("f1")
        .head(error_limit)
    )
    low_f1.to_csv(error_path, index=False)
    print(
        f"Error analysis saved to {error_path}  "
        f"({len(low_f1)} cases with F1 < {f1_threshold})"
    )

    # ── console summary table ─────────────────────────────────────────────────
    print()
    print(f"{'Category':<42}  {'F1':>6}  {'EM':>6}")
    print("-" * 58)
    for cat, row in cat_metrics.iterrows():
        print(f"{cat:<42}  {row['f1']:>6.3f}  {row['em']:>6.3f}")
    print("-" * 58)
    print(f"{'OVERALL':<42}  {mean_f1:>6.3f}  {mean_em:>6.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUAD evaluation dashboard")
    parser.add_argument("--results",       default="evaluation/eval_results.csv")
    parser.add_argument("--output",        default="evaluation/eval_dashboard.png")
    parser.add_argument("--errors",        default="evaluation/error_analysis.csv")
    parser.add_argument("--f1-threshold",  type=float, default=0.2)
    parser.add_argument("--error-limit",   type=int,   default=20)
    args = parser.parse_args()

    build_dashboard(
        results_path=args.results,
        output_path=args.output,
        error_path=args.errors,
        f1_threshold=args.f1_threshold,
        error_limit=args.error_limit,
    )
