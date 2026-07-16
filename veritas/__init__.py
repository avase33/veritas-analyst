"""Veritas — Domain-Specific AI Analyst (Multi-Agent RAG).

Upload dense documents (financial filings, contracts, research) and ask complex
questions answered *strictly* from the provided context. A router delegates to
specialised agents — data retrieval, math/calculator, semantic search — and a
synthesizer composes a cited answer, refusing when the evidence isn't there.

Retrieval is hybrid: dense vector search + BM25 keyword search fused with
Reciprocal Rank Fusion. Offline-first: deterministic mock LLM + hashing
embeddings + in-memory vector store mean the whole system runs, tests and
benchmarks with zero external services; OpenAI/Anthropic/Pinecone/Qdrant adapters
wire in for production.
"""

from .version import __version__
from .config import Settings
from .engine import Analyst

__all__ = ["__version__", "Settings", "Analyst"]
