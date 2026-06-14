import os
from typing import Any

from .retriever import tokenize


def _simple_score(query: str, chunk: dict[str, Any]) -> int:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(chunk.get("text", "")))
    category_terms = set(tokenize(" ".join(chunk.get("clause_categories", []))))
    return len(query_terms & text_terms) + 2 * len(query_terms & category_terms)


def rerank(query: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    mode = os.getenv("RERANKER_MODE", "simple").lower()
    if mode != "cross-encoder" or not chunks:
        return sorted(chunks, key=lambda chunk: _simple_score(query, chunk), reverse=True)[:top_k]

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]
