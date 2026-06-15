"""
evaluation/ragas_eval.py

RAGAS evaluation of the CUAD RAG chatbot.

Metrics
-------
faithfulness      — Are all claims in the answer supported by the retrieved contexts?
answer_relevancy  — Is the answer relevant to the question asked?
context_recall    — Does the retrieved context cover the ground truth?

Input
-----
evaluation/eval_results.csv   — written by generation_eval.py
                                 must contain columns:
                                     question, prediction, ground_truth

Note: RAGAS uses an LLM internally (OpenAI by default).
Set OPENAI_API_KEY before running.

Run
---
    export OPENAI_API_KEY="your-key-here"
    python evaluation/ragas_eval.py

    # Run on a smaller slice for cost control
    python evaluation/ragas_eval.py --sample 50
"""

import argparse
import os
import pandas as pd

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset


def run_ragas_eval(
    results_path: str = "evaluation/eval_results.csv",
    sample_n:     int | None = None,
    random_state: int = 42,
) -> dict:
    """
    Run RAGAS evaluation on generation results.

    Parameters
    ----------
    results_path : path to eval_results.csv from generation_eval.py
    sample_n     : if set, evaluate a random sample of this size
    random_state : seed for reproducible sampling

    Returns
    -------
    dict of RAGAS metric scores
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. RAGAS uses OpenAI LLMs internally.\n"
            "Run: export OPENAI_API_KEY='your-key-here'"
        )

    df = pd.read_csv(results_path)

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state)
        print(f"Sampled {sample_n} rows from {results_path}")
    else:
        print(f"Running RAGAS on all {len(df)} rows in {results_path}")

    # RAGAS expects:
    #   question     – the user question
    #   answer       – the model's prediction
    #   contexts     – list of retrieved context strings (we use ground_truth as proxy
    #                  since eval_results.csv does not store raw retrieved passages)
    #   ground_truth – the reference answer
    #
    # If Person B's API returns retrieved passages, replace the contexts column below
    # with the actual retrieved text for a more accurate faithfulness score.
    ragas_dataset = Dataset.from_dict({
        "question":     df["question"].tolist(),
        "answer":       df["prediction"].fillna("").tolist(),
        "contexts":     [[gt] for gt in df["ground_truth"].fillna("").tolist()],
        "ground_truth": df["ground_truth"].fillna("").tolist(),
    })

    print("Running RAGAS evaluation (this calls OpenAI — may take a few minutes)…")
    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )

    print()
    print("=" * 45)
    print("  RAGAS scores")
    print("=" * 45)
    print(f"  Faithfulness     : {result['faithfulness']:.3f}")
    print(f"  Answer Relevancy : {result['answer_relevancy']:.3f}")
    print(f"  Context Recall   : {result['context_recall']:.3f}")
    print("=" * 45)
    print()
    print("Interpretation:")
    print("  faithfulness    < 0.7  → model is hallucinating; tighten the prompt")
    print("  answer_relevancy< 0.7  → answers are off-topic; check clause_filter logic")
    print("  context_recall  < 0.7  → retriever is missing relevant passages")

    return dict(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUAD RAGAS evaluation")
    parser.add_argument("--results", default="evaluation/eval_results.csv")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Evaluate on a random sample of N rows (default: all)",
    )
    args = parser.parse_args()

    run_ragas_eval(results_path=args.results, sample_n=args.sample)
