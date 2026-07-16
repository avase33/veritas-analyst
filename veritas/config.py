"""Central configuration resolved from the environment.

Offline-first defaults keep everything in-process (mock LLM, hashing embeddings,
in-memory hybrid index). Point adapters at OpenAI/Anthropic and Pinecone/Qdrant
for production via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Embeddings: hashing | openai
    embedding_backend: str = "hashing"
    embedding_dim: int = 256
    openai_embedding_model: str = "text-embedding-3-large"

    # LLM: mock | anthropic | openai
    llm_backend: str = "mock"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    openai_model: str = "gpt-4o"

    # Vector store: memory | qdrant | pinecone
    vector_backend: str = "memory"
    qdrant_url: str = "http://localhost:6333"
    pinecone_index: str = "veritas"

    # Chunking
    chunk_tokens: int = 180          # approx words per chunk
    chunk_overlap: int = 40
    contextual_headers: bool = True  # prepend doc title + section header to chunks

    # Retrieval
    top_k: int = 6                   # chunks passed to the synthesizer
    rrf_k: int = 60                  # Reciprocal Rank Fusion constant
    candidate_k: int = 20            # candidates fetched per retriever before fusion

    # Grounding (anti-hallucination): min fraction of query content terms that
    # must appear in a retrieved chunk, else refuse to answer.
    min_overlap: float = 0.12

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.environ.get
        llm = g("VERITAS_LLM") or (
            "anthropic" if g("ANTHROPIC_API_KEY") else ("openai" if g("OPENAI_API_KEY") else "mock"))
        emb = g("VERITAS_EMBEDDINGS") or ("openai" if g("OPENAI_API_KEY") else "hashing")
        return cls(
            embedding_backend=emb,
            embedding_dim=int(g("VERITAS_EMBED_DIM", "256")),
            llm_backend=llm,
            vector_backend=g("VERITAS_VECTOR", "memory"),
            qdrant_url=g("QDRANT_URL", "http://localhost:6333"),
            pinecone_index=g("PINECONE_INDEX", "veritas"),
            chunk_tokens=int(g("VERITAS_CHUNK_TOKENS", "180")),
            chunk_overlap=int(g("VERITAS_CHUNK_OVERLAP", "40")),
            contextual_headers=g("VERITAS_CONTEXTUAL", "true").lower() in ("1", "true", "yes"),
            top_k=int(g("VERITAS_TOP_K", "6")),
            rrf_k=int(g("VERITAS_RRF_K", "60")),
            candidate_k=int(g("VERITAS_CANDIDATE_K", "20")),
            min_overlap=float(g("VERITAS_MIN_OVERLAP", "0.12")),
        )
