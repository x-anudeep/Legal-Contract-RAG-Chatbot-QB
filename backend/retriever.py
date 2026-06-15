import os
import pickle
import re
from pathlib import Path
from typing import Any


MOCK_CHUNKS: list[dict[str, Any]] = [
    {
        "chunk_id": 0,
        "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of Nevada.",
        "contract_name": "SampleCo_License_2019",
        "chunk_index": 0,
        "clause_categories": ["Governing Law"],
        "answers": ["Nevada"],
    },
    {
        "chunk_id": 1,
        "text": "Either party may terminate this Agreement for convenience upon thirty (30) days prior written notice to the other party.",
        "contract_name": "SampleCo_License_2019",
        "chunk_index": 1,
        "clause_categories": ["Termination for Convenience", "Notice to Terminate Renewal"],
        "answers": ["Yes", "30 days"],
    },
    {
        "chunk_id": 2,
        "text": "During the term of this Agreement and for two years thereafter, neither party shall solicit or hire employees of the other party.",
        "contract_name": "AcmeCorp_Service_2020",
        "chunk_index": 0,
        "clause_categories": ["No-Solicit of Employees"],
        "answers": ["Yes"],
    },
    {
        "chunk_id": 3,
        "text": "Customer may not assign this Agreement without Vendor's prior written consent, except to an affiliate or successor in a merger.",
        "contract_name": "Northstar_SaaS_2021",
        "chunk_index": 0,
        "clause_categories": ["Anti-Assignment", "Change of Control"],
        "answers": ["Yes", "Yes"],
    },
    {
        "chunk_id": 4,
        "text": "Vendor grants Customer a non-exclusive, non-transferable license to access and use the software solely for internal business purposes.",
        "contract_name": "Northstar_SaaS_2021",
        "chunk_index": 1,
        "clause_categories": ["License Grant", "Non-Transferable License"],
        "answers": ["Yes", "Yes"],
    },
    {
        "chunk_id": 5,
        "text": "Except for confidentiality obligations and indemnification claims, each party's aggregate liability is capped at the fees paid in the twelve months preceding the claim.",
        "contract_name": "BlueRiver_MasterServices_2022",
        "chunk_index": 0,
        "clause_categories": ["Cap On Liability", "Uncapped Liability", "Indemnification"],
        "answers": ["Yes", "Yes", "Yes"],
    },
    {
        "chunk_id": 6,
        "text": "The Agreement begins on May 8, 2014 and continues for an initial term of three years unless earlier terminated.",
        "contract_name": "HelioSupply_Distribution_2014",
        "chunk_index": 0,
        "clause_categories": ["Effective Date", "Expiration Date"],
        "answers": ["May 8, 2014", "Three years after May 8, 2014"],
    },
    {
        "chunk_id": 7,
        "text": "The parties shall maintain commercially reasonable insurance coverage, including general liability and workers compensation insurance.",
        "contract_name": "BlueRiver_MasterServices_2022",
        "chunk_index": 1,
        "clause_categories": ["Insurance"],
        "answers": ["Yes"],
    },
    {
        "chunk_id": 8,
        "text": "All disputes arising under this Agreement shall be resolved by binding arbitration in San Francisco, California.",
        "contract_name": "AcmeCorp_Service_2020",
        "chunk_index": 1,
        "clause_categories": ["Dispute Resolution"],
        "answers": ["Binding arbitration in San Francisco, California"],
    },
    {
        "chunk_id": 9,
        "text": "All intellectual property created specifically for Customer under a statement of work shall be assigned to Customer upon full payment.",
        "contract_name": "BlueRiver_MasterServices_2022",
        "chunk_index": 2,
        "clause_categories": ["IP Ownership Assignment"],
        "answers": ["Yes"],
    },
]


def normalize_clause_category(category: str | None) -> str | None:
    if not category:
        return None
    return category.split("__", 1)[-1].strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _lexical_score(query: str, chunk: dict[str, Any]) -> float:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(chunk.get("text", "")))
    category_terms = set(tokenize(" ".join(chunk.get("clause_categories", []))))
    answer_terms = set(tokenize(" ".join(chunk.get("answers", []))))

    score = len(query_terms & text_terms)
    score += 2 * len(query_terms & category_terms)
    score += len(query_terms & answer_terms)
    return float(score)


def _with_preview(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "contract_name": chunk.get("contract_name"),
        "chunk_index": chunk.get("chunk_index"),
        "clause_categories": [
            normalize_clause_category(c) or c for c in chunk.get("clause_categories", [])
        ],
        "answers": chunk.get("answers", []),
        "text": chunk.get("text", ""),
        "text_preview": chunk.get("text", "")[:300],
    }


def _matches_clause_filter(chunk: dict[str, Any], clause_filter: str | None) -> bool:
    normalized_filter = normalize_clause_category(clause_filter)
    if not normalized_filter:
        return True

    categories = [
        normalize_clause_category(category)
        for category in chunk.get("clause_categories", [])
    ]
    return normalized_filter in categories


def mock_retrieve(query: str, clause_filter: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
    candidates = MOCK_CHUNKS

    if clause_filter:
        filtered = [
            chunk
            for chunk in MOCK_CHUNKS
            if _matches_clause_filter(chunk, clause_filter)
        ]
        candidates = filtered or MOCK_CHUNKS

    ranked = sorted(
        candidates,
        key=lambda chunk: (_lexical_score(query, chunk), -int(chunk["chunk_id"])),
        reverse=True,
    )
    return [_with_preview(chunk) for chunk in ranked[:top_k]]


def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class RealRetriever:
    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
        self.client = QdrantClient(url=qdrant_url)
        self.model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))

        bm25_path = Path(os.getenv("BM25_INDEX_PATH", "ingestion/bm25_index.pkl"))
        with bm25_path.open("rb") as f:
            bm25_data = pickle.load(f)

        self.bm25: BM25Okapi = bm25_data["bm25"]
        self.chunk_ids: list[int] = bm25_data["chunk_ids"]

    def retrieve(self, query: str, clause_filter: str | None = None, top_k: int = 20) -> list[dict[str, Any]]:
        q_vec = self.model.encode(query, normalize_embeddings=True).tolist()
        candidate_limit = max(top_k * 5, 50)

        dense_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=q_vec,
            limit=candidate_limit,
        )
        dense_ids = [int(hit.id) for hit in dense_hits]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ids = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:candidate_limit]
        bm25_ids = [int(self.chunk_ids[index]) for index in bm25_ids]

        fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])[:candidate_limit]
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=fused_ids,
            with_payload=True,
        )

        chunks_by_id = {}
        for result in results:
            payload = result.payload or {}
            payload["chunk_id"] = int(result.id)
            chunks_by_id[int(result.id)] = _with_preview(payload)

        chunks = [
            chunks_by_id[chunk_id]
            for chunk_id in fused_ids
            if chunk_id in chunks_by_id
        ]

        filtered_chunks = [
            chunk for chunk in chunks if _matches_clause_filter(chunk, clause_filter)
        ]
        return (filtered_chunks or chunks)[:top_k]


_REAL_RETRIEVER: RealRetriever | None = None


def retrieve(query: str, clause_filter: str | None = None, top_k: int = 20) -> list[dict[str, Any]]:
    mode = os.getenv("RETRIEVER_MODE", "mock").lower()
    if mode != "real":
        return mock_retrieve(query, clause_filter, top_k)

    global _REAL_RETRIEVER
    if _REAL_RETRIEVER is None:
        _REAL_RETRIEVER = RealRetriever()
    return _REAL_RETRIEVER.retrieve(query, clause_filter, top_k)
