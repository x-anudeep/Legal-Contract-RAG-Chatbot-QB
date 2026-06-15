"""
evaluation/generation_eval.py

End-to-end generation evaluation against Person B's /chat endpoint.

Scores
------
EM  (exact match)  — 1 if normalised prediction == normalised ground truth
F1  (token F1)     — token-level overlap, SQuAD style

Person B's API must be running at http://localhost:8000 with:
    POST /chat
    { "question": "...", "clause_filter": "..." }
    → { "answer": "..." }

Run
---
    # Quick smoke test — 10 cases
    python evaluation/generation_eval.py --sample 10

    # Standard eval — 100 cases (default)
    python evaluation/generation_eval.py

    # Full eval — all test cases (expensive / slow)
    python evaluation/generation_eval.py --sample 0
"""

import argparse
import time
import requests
import pandas as pd

from scorer import exact_match, token_f1


def _chat(api_url: str, question: str, clause_filter: str) -> str:
    """
    Call Person B's /chat endpoint and return the answer string.
    Returns empty string on any error.
    """
    try:
        resp = requests.post(
            f"{api_url}/chat",
            json={"question": question, "clause_filter": clause_filter},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception as exc:
        print(f"  [WARN] chat failed: {exc}")
        return ""


def run_generation_eval(
    test_set_path: str = "evaluation/eval_test_set.csv",
    output_path:   str = "evaluation/eval_results.csv",
    api_url:       str = "http://localhost:8000",
    sample_n:      int = 100,
    random_state:  int = 42,
) -> pd.DataFrame:
    """
    Run generation evaluation, write per-case results to CSV, print summary.

    Parameters
    ----------
    test_set_path : path to eval_test_set.csv
    output_path   : where to write eval_results.csv
    api_url       : base URL of Person B's FastAPI service
    sample_n      : number of test cases to evaluate (0 = all)
    random_state  : seed for reproducible sampling

    Returns
    -------
    DataFrame with one row per evaluated case
    """
    test = pd.read_csv(test_set_path)

    if sample_n and sample_n < len(test):
        sample = test.sample(n=sample_n, random_state=random_state)
        print(f"Sampled {sample_n} / {len(test)} rows for generation eval")
    else:
        sample = test
        print(f"Evaluating all {len(test)} rows")

    results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        pred = _chat(api_url, row["question"], row["clause_category"])

        em = exact_match(pred, row["ground_truth"])
        f1 = token_f1(pred,   row["ground_truth"])

        results.append({
            "contract":     row["contract_name"],
            "category":     row["clause_category"],
            "question":     row["question"],
            "ground_truth": row["ground_truth"],
            "prediction":   pred,
            "em":           em,
            "f1":           f1,
        })

        if (i + 1) % 10 == 0:
            so_far = pd.DataFrame(results)
            print(
                f"  [{i+1:>5}/{len(sample)}]  "
                f"EM: {so_far['em'].mean():.3f}   F1: {so_far['f1'].mean():.3f}"
            )

        time.sleep(0.5)   # avoid rate limiting

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print(f"  Overall EM : {results_df['em'].mean():.3f}")
    print(f"  Overall F1 : {results_df['f1'].mean():.3f}")
    print(f"  Cases      : {len(results_df)}")
    print("=" * 55)
    print()
    print("F1 by clause category (ascending):")
    cat_summary = (
        results_df.groupby("category")[["em", "f1"]]
        .mean()
        .sort_values("f1")
    )
    print(cat_summary.to_string(float_format="{:.3f}".format))
    print()
    print(f"Results written to {output_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUAD generation evaluation")
    parser.add_argument("--test-set", default="evaluation/eval_test_set.csv")
    parser.add_argument("--output",   default="evaluation/eval_results.csv")
    parser.add_argument("--api-url",  default="http://localhost:8000")
    parser.add_argument(
        "--sample", type=int, default=100,
        help="Number of cases to evaluate (0 = all, default 100)",
    )
    args = parser.parse_args()

    run_generation_eval(
        test_set_path=args.test_set,
        output_path=args.output,
        api_url=args.api_url,
        sample_n=args.sample,
    )
