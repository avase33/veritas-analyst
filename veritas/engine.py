"""Analyst — the engine wiring ingestion, hybrid retrieval and the agent graph.

    ingest(document) -> chunk (contextual) -> embed -> index (vector + BM25)
    ask(query)       -> router -> specialists -> math -> synthesizer (cited)

One object is shared by the CLI, API and tests. Offline it needs no keys and no
servers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .agents.graph import AgentGraph
from .config import Settings
from .embeddings import build_embedder
from .ingestion.chunking import chunk_document
from .ingestion.loader import load_document, load_text
from .llm import build_llm
from .logging_setup import get_logger
from .models import Answer, Document
from .retrieval.bm25 import BM25Index
from .retrieval.hybrid import HybridRetriever
from .retrieval.vectorstore import build_vector_store

log = get_logger("engine")


@dataclass
class IngestReport:
    doc_id: str
    title: str
    chunks: int
    pages: int


class Analyst:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        s = self.settings
        self.embedder = build_embedder(s.embedding_backend, s.embedding_dim, s.openai_embedding_model)
        self.vector_store = build_vector_store(s.vector_backend, s.qdrant_url, s.embedding_dim)
        self.retriever = HybridRetriever(self.embedder, self.vector_store, BM25Index(),
                                         rrf_k=s.rrf_k, candidate_k=s.candidate_k)
        self.llm = build_llm(s.llm_backend, s.anthropic_model, s.openai_model)
        self.graph = AgentGraph(self.retriever, self.llm, s.llm_backend, s)
        self.documents: dict[str, Document] = {}
        self._chunks = 0

    # ---- ingestion ------------------------------------------------------

    def ingest(self, doc: Document) -> IngestReport:
        chunks = chunk_document(doc, self.settings)
        self.retriever.index(chunks)
        self.documents[doc.id] = doc
        self._chunks += len(chunks)
        pages = max((c.page for c in chunks), default=0)
        log.info("ingested '%s': %d chunks across %d pages", doc.title, len(chunks), pages)
        return IngestReport(doc_id=doc.id, title=doc.title, chunks=len(chunks), pages=pages)

    def ingest_text(self, text: str, title: str = "Untitled") -> IngestReport:
        return self.ingest(load_text(text, title=title))

    def ingest_file(self, path: str, title: Optional[str] = None) -> IngestReport:
        return self.ingest(load_document(path, title=title))

    # ---- querying -------------------------------------------------------

    def ask(self, query: str) -> Answer:
        return self.graph.run(query)

    def stats(self) -> dict:
        return {"documents": len(self.documents), "chunks": self._chunks,
                "vector_store": len(self.vector_store) if hasattr(self.vector_store, "__len__") else None,
                "bm25_docs": len(self.retriever.bm25)}
