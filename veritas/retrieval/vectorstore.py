"""Dense vector store.

Default is an exact in-memory cosine index (fine for a document or a corpus of
them). Qdrant and Pinecone adapters implement the same ``add`` / ``search``
surface for hosted, scalable similarity search.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from ..embeddings import cosine
from ..models import Chunk


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query_vec: Sequence[float], k: int = 10) -> list[tuple[Chunk, float]]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add(self, chunks: Sequence[Chunk]) -> None:
        self._chunks.extend(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query_vec: Sequence[float], k: int = 10) -> list[tuple[Chunk, float]]:
        scored = [(c, cosine(query_vec, c.embedding)) for c in self._chunks if c.embedding]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]


class QdrantVectorStore:  # pragma: no cover - requires qdrant
    def __init__(self, url: str, collection: str = "veritas", dim: int = 256) -> None:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        self._client = QdrantClient(url=url)
        self._collection = collection
        self._n = 0
        self._by_id: dict[int, Chunk] = {}
        try:
            self._client.get_collection(collection)
        except Exception:
            self._client.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def add(self, chunks):
        from qdrant_client.models import PointStruct  # type: ignore

        points = []
        for c in chunks:
            self._n += 1
            self._by_id[self._n] = c
            points.append(PointStruct(id=self._n, vector=list(c.embedding), payload={"chunk_id": c.id}))
        self._client.upsert(self._collection, points=points)

    def search(self, query_vec, k=10):
        res = self._client.search(self._collection, query_vector=list(query_vec), limit=k)
        return [(self._by_id[p.id], float(p.score)) for p in res if p.id in self._by_id]


def build_vector_store(backend: str = "memory", url: str = "", dim: int = 256) -> VectorStore:
    if backend == "qdrant" and url:
        return QdrantVectorStore(url, dim=dim)
    return InMemoryVectorStore()
