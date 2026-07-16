"""Hybrid retrieval with Reciprocal Rank Fusion.

Runs dense (vector) and sparse (BM25) retrieval independently, then fuses their
ranked lists with Reciprocal Rank Fusion:

    RRF(d) = Σ_{i ∈ retrievers} 1 / (k + rank_i(d))

RRF combines rankings without needing to calibrate incompatible score scales, and
reliably lifts the chunks that *both* meaning and keywords agree on. ``k`` is a
smoothing constant (default 60).
"""

from __future__ import annotations

from typing import Sequence

from ..embeddings import Embedder
from ..models import Chunk, Retrieved
from .bm25 import BM25Index
from .vectorstore import VectorStore


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    """rankings: {retriever_name: [chunk_id ordered best-first]} -> {chunk_id: rrf_score}."""
    scores: dict[str, float] = {}
    for ids in rankings.values():
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 bm25: BM25Index | None = None, rrf_k: int = 60, candidate_k: int = 20) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25 or BM25Index()
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self._by_id: dict[str, Chunk] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            if not c.embedding:
                c.embedding = self.embedder.embed(c.text)
            self._by_id[c.id] = c
        self.vector_store.add(chunks)
        self.bm25.add(chunks)

    def retrieve(self, query: str, top_k: int = 6) -> list[Retrieved]:
        qvec = self.embedder.embed(query)
        dense = self.vector_store.search(qvec, k=self.candidate_k)
        sparse = self.bm25.search(query, k=self.candidate_k)

        rankings = {
            "dense": [c.id for c, _ in dense],
            "bm25": [c.id for c, _ in sparse],
        }
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)

        # track which retrievers surfaced each chunk (for transparency)
        surfaced: dict[str, list[str]] = {}
        for name, ids in rankings.items():
            for cid in ids:
                surfaced.setdefault(cid, []).append(name)

        ranked = sorted(fused.items(), key=lambda p: p[1], reverse=True)[:top_k]
        out: list[Retrieved] = []
        for rank, (cid, score) in enumerate(ranked, start=1):
            chunk = self._by_id.get(cid)
            if chunk is None:
                continue
            out.append(Retrieved(chunk=chunk, score=score, rank=rank,
                                 sources=surfaced.get(cid, [])))
        return out
