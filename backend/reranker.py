import os
from math import exp
from typing import Any

from .retriever import tokenize


STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "clause",
    "contract",
    "could",
    "did",
    "do",
    "does",
    "either",
    "for",
    "has",
    "have",
    "include",
    "includes",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "party",
    "parties",
    "the",
    "this",
    "to",
    "what",
    "without",
}

TERM_NORMALIZATIONS = {
    "assignable": "assign",
    "assigned": "assign",
    "assignment": "assign",
    "assignments": "assign",
    "assigning": "assign",
    "capped": "cap",
    "caps": "cap",
    "governed": "govern",
    "governing": "govern",
    "laws": "law",
    "liabilities": "liability",
    "solicitation": "solicit",
    "soliciting": "solicit",
    "terminated": "terminate",
    "terminates": "terminate",
    "terminating": "terminate",
    "termination": "terminate",
    "transferable": "transfer",
    "transferred": "transfer",
    "transferring": "transfer",
}


def _normalize_term(term: str) -> str:
    return TERM_NORMALIZATIONS.get(term, term)


def _content_terms(text: str) -> set[str]:
    return {
        _normalize_term(term)
        for term in tokenize(text)
        if term not in STOPWORDS and len(term) > 1
    }


def _simple_score(query: str, chunk: dict[str, Any]) -> tuple[float, float, str]:
    query_terms = _content_terms(query)
    text_terms = _content_terms(chunk.get("text", ""))
    category_terms = _content_terms(" ".join(chunk.get("clause_categories", [])))
    answer_terms = _content_terms(" ".join(chunk.get("answers", [])))

    text_matches = query_terms & text_terms
    category_matches = query_terms & category_terms
    answer_matches = query_terms & answer_terms

    raw_score = (
        len(text_matches)
        + 2 * len(category_matches)
        + len(answer_matches)
    )

    if not query_terms:
        return 0.0, 0.0, "No meaningful query terms were available to score this source."

    max_possible_score = len(query_terms) * 4
    score_10 = min(10.0, (raw_score / max_possible_score) * 10)

    matched_terms = sorted(text_matches | category_matches | answer_matches)
    if matched_terms:
        reason = "Matched query terms: " + ", ".join(matched_terms) + "."
    else:
        reason = "0.00/10 because no meaningful query terms matched this source."

    return float(raw_score), round(score_10, 2), reason


def _cross_encoder_score_to_10(raw_score: float) -> float:
    return round(10 / (1 + exp(-raw_score)), 2)


def rerank(query: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    mode = os.getenv("RERANKER_MODE", "simple").lower()
    if mode != "cross-encoder" or not chunks:
        scored_chunks = [
            (_simple_score(query, chunk), chunk)
            for chunk in chunks
        ]
        ranked = sorted(scored_chunks, key=lambda item: item[0][0], reverse=True)
        return [
            {
                **chunk,
                "rerank_score": score_10,
                "rerank_score_raw": raw_score,
                "rerank_score_scale": "0-10",
                "rerank_reason": reason,
            }
            for (raw_score, score_10, reason), chunk in ranked[:top_k]
        ]

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda item: item[0], reverse=True)
    return [
        {
            **chunk,
            "rerank_score": _cross_encoder_score_to_10(float(score)),
            "rerank_score_raw": float(score),
            "rerank_score_scale": "0-10",
            "rerank_reason": (
                "Cross-encoder relevance score converted to a 0-10 scale; "
                f"raw model score: {float(score):.4f}."
            ),
        }
        for score, chunk in ranked[:top_k]
    ]
