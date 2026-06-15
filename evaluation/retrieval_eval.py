"""
evaluation/retrieval_eval.py

Measures retrieval quality against Person B's /retrieve endpoint.

Metrics
-------
Recall@5 : fraction of test cases where at least one ground-truth chunk
           appears in the top-5 retrieved results.
MRR      : mean reciprocal rank of the first relevant result (across top-5).

Input
-----
evaluation/eval_test_set.csv   — built by build_test_set.py
                                 must contain columns:
                                     question, clause_category, chunk_id

Person B's API must be running at http://localhost:8000 with a /retrieve
endpoint that accepts:
    POST /retrieve
    { "question": "...", "clause_filter": "...", "top_k": 5 }
and returns:
    { "results": [ { "chunk_id": <int>, ... }, ... ] }

Run
---
    python evaluation/retrieval_eval.py
    python evaluation/retrieval_eval.py --test-set evaluation/eval_test_set.csv
                                        --api-url http://localhost:8000
                                        --top-k 5
                                        --sample 200
"""

import argparse
import time
import requests
import pandas as pd


def _retrieve(api_url: str, question: str, clause_filter: str, top_k: int) -> list[int]:
    """
    Call Person B's /retrieve endpoint and return a list of chunk_ids
    in ranked order (most relevant first).
    Returns an empty list on any error.
    """
    try:
        resp = requests.post(
            f"{api_url}/retrieve",
            json={"question": question, "clause_filter": clause_filter, "top_k": top_k},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [r["chunk_id"] for r in results]
    except Exception as exc:
        print(f"  [WARN] retrieve failed: {exc}")
        return []


def run_retrieval_eval(
    test_set_path: str = "evaluation/eval_test_set.csv",
    api_url:       str = "http://localhost:8000",
    top_k:         int = 5,
    sample_n:      int | None = None,
    random_state:  int = 42,
) -> dict:
    """
    Run retrieval evaluation and print Recall@k and MRR.

    Parameters
    ----------
    test_set_path : path to eval_test_set.csv
    api_url       : base URL of Person B's FastAPI service
    top_k         : number of results to request and evaluate against
    sample_n      : if set, evaluate on a random sample of this size
    random_state  : seed for reproducible sampling

    Returns
    -------
    dict with keys: recall, mrr, n_evaluated
    """
    test = pd.read_csv(test_set_path)

    if sample_n is not None and sample_n < len(test):
        test = test.sample(n=sample_n, random_state=random_state)
        print(f"Sampled {sample_n} rows from {test_set_path}")
    else:
        print(f"Evaluating all {len(test)} rows in {test_set_path}")

    recall_results: list[int]   = []
    mrr_results:    list[float] = []

    for i, (_, row) in enumerate(test.iterrows()):
        gt_chunk_id  = int(row["chunk_id"])
        retrieved_ids = _retrieve(
            api_url,
            question=row["question"],
            clause_filter=row["clause_category"],
            top_k=top_k,
        )

        # Recall@k
        hit = int(gt_chunk_id in retrieved_ids)
        recall_results.append(hit)

        # MRR
        mrr_score = 0.0
        for rank, rid in enumerate(retrieved_ids):
            if rid == gt_chunk_id:
                mrr_score = 1.0 / (rank + 1)
                break
        mrr_results.append(mrr_score)

        if (i + 1) % 50 == 0:
            running_recall = sum(recall_results) / len(recall_results)
            running_mrr    = sum(mrr_results)    / len(mrr_results)
            print(
                f"  [{i+1:>5}/{len(test)}]  "
                f"Recall@{top_k}: {running_recall:.3f}   MRR: {running_mrr:.3f}"
            )

        time.sleep(0.1)   # polite rate limit

    n = len(recall_results)
    final_recall = sum(recall_results) / n
    final_mrr    = sum(mrr_results)    / n

    print()
    print("=" * 45)
    print(f"  Recall@{top_k}  : {final_recall:.3f}")
    print(f"  MRR        : {final_mrr:.3f}")
    print(f"  Evaluated  : {n} cases")
    print("=" * 45)

    if final_recall < 0.6:
        print(
            "\n[WARNING] Recall@5 < 0.6 — retrieval quality is low.\n"
            "Bring this finding back to Person A before running generation_eval.py.\n"
            "Consider: smaller chunk size, different embedding model, or hybrid BM25+vector."
        )

    return {"recall": final_recall, "mrr": final_mrr, "n_evaluated": n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUAD retrieval evaluation")
    parser.add_argument("--test-set",    default="evaluation/eval_test_set.csv")
    parser.add_argument("--api-url",     default="http://localhost:8000")
    parser.add_argument("--top-k",       type=int, default=5)
    parser.add_argument("--sample",      type=int, default=None,
                        help="Evaluate on a random sample of N rows")
    args = parser.parse_args()

    run_retrieval_eval(
        test_set_path=args.test_set,
        api_url=args.api_url,
        top_k=args.top_k,
        sample_n=args.sample,
    )
