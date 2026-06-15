"""
evaluation/scorer.py

Scoring helpers used by generation_eval.py and retrieval_eval.py.
Implements SQuAD-style exact match and token-level F1.
"""

import re
import string
from collections import Counter


# ── text normalisation (matches SQuAD official eval) ─────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── core metrics ──────────────────────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> int:
    """Return 1 if normalised prediction == normalised ground truth, else 0."""
    if not isinstance(prediction, str):
        prediction = ""
    if not isinstance(ground_truth, str):
        ground_truth = ""
    return int(_normalize(prediction) == _normalize(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 between prediction and ground truth (SQuAD style).

    Tokenises by whitespace after normalisation and computes the overlap
    between the two bags of tokens. Returns 0.0 when either string is empty
    after normalisation.
    """
    if not isinstance(prediction, str):
        prediction = ""
    if not isinstance(ground_truth, str):
        ground_truth = ""

    pred_tokens = _normalize(prediction).split()
    gt_tokens   = _normalize(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return f1


# ── batch helpers ─────────────────────────────────────────────────────────────

def score_batch(predictions: list[str], ground_truths: list[str]) -> dict:
    """
    Score a list of (prediction, ground_truth) pairs.

    Returns a dict with keys:
        em_scores   – list of per-example EM (0 or 1)
        f1_scores   – list of per-example F1 (float)
        mean_em     – float
        mean_f1     – float
    """
    assert len(predictions) == len(ground_truths), \
        "predictions and ground_truths must have the same length"

    em_scores = [exact_match(p, g) for p, g in zip(predictions, ground_truths)]
    f1_scores = [token_f1(p, g)    for p, g in zip(predictions, ground_truths)]

    return {
        "em_scores": em_scores,
        "f1_scores": f1_scores,
        "mean_em":   sum(em_scores) / len(em_scores),
        "mean_f1":   sum(f1_scores) / len(f1_scores),
    }


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("Nevada",                   "Nevada",                 1,   1.0),
        ("the state of nevada",      "Nevada",                 0,   0.4),   # partial token overlap after norm
        ("Yes",                      "No",                     0,   0.0),
        ("",                         "California",             0,   0.0),
        ("5/8/2014",                 "5/8/2014",               1,   1.0),
        ("California and Delaware",  "California",             0,   0.5),   # partial overlap
    ]

    print(f"{'Prediction':<30} {'Ground Truth':<20} {'EM':>4} {'F1':>6}  {'EM ok':>5}  {'F1 ok':>5}")
    print("-" * 80)
    all_pass = True
    for pred, gt, expected_em, expected_f1 in cases:
        em  = exact_match(pred, gt)
        f1  = token_f1(pred, gt)
        ok_em = em == expected_em
        ok_f1 = abs(f1 - expected_f1) < 1e-6
        status_em = "PASS" if ok_em else "FAIL"
        status_f1 = "PASS" if ok_f1 else "FAIL"
        print(f"{pred!r:<30} {gt!r:<20} {em:>4} {f1:>6.3f}  {status_em:>5}  {status_f1:>5}")
        if not (ok_em and ok_f1):
            all_pass = False

    print()
    print("All tests passed." if all_pass else "SOME TESTS FAILED.")
