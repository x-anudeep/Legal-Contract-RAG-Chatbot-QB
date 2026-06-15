import os
import pickle
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cuad_contracts")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDINGS_CACHE_PATH = Path(os.getenv("EMBEDDINGS_CACHE_PATH", "ingestion/embeddings_cache.pkl"))
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "256"))


def normalize_clause_category(category: str | None) -> str | None:
    if not category:
        return None
    return category.split("__", 1)[-1].strip()


def payload_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    raw_categories = chunk.get("clause_categories", [])
    return {
        "contract_name": chunk.get("contract_name"),
        "chunk_index": chunk.get("chunk_index"),
        "text": chunk.get("text", ""),
        "clause_categories": [
            normalize_clause_category(category) or category
            for category in raw_categories
        ],
        "raw_clause_categories": raw_categories,
        "answers": chunk.get("answers", []),
    }


def load_qdrant_from_cache() -> None:
    with EMBEDDINGS_CACHE_PATH.open("rb") as f:
        chunks = pickle.load(f)

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=index,
            vector=chunk["embedding"],
            payload=payload_from_chunk(chunk),
        )
        for index, chunk in enumerate(chunks)
    ]

    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start : start + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        end = min(start + BATCH_SIZE, len(points))
        print(f"Uploaded {end}/{len(points)} points")

    collection = client.get_collection(COLLECTION_NAME)
    print(f"Loaded {collection.points_count} points into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    load_qdrant_from_cache()
