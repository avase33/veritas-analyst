from veritas.config import Settings
from veritas.ingestion.chunking import chunk_document
from veritas.ingestion.loader import load_text
from veritas.embeddings import HashingEmbedder
from veritas.retrieval.bm25 import BM25Index
from veritas.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from veritas.retrieval.vectorstore import InMemoryVectorStore
from veritas.mockdata import ANNUAL_REPORT


def test_chunking_tracks_pages_and_sections():
    doc = load_text(ANNUAL_REPORT, title="Report")
    chunks = chunk_document(doc, Settings())
    assert len(chunks) > 3
    pages = {c.page for c in chunks}
    assert max(pages) >= 4
    # a risk-factor chunk should be on page 4 and carry that section
    risk = [c for c in chunks if "liability" in c.body.lower()]
    assert risk and risk[0].page == 4


def test_contextual_header_prepended():
    doc = load_text(ANNUAL_REPORT, title="Northwind Report")
    chunks = chunk_document(doc, Settings(contextual_headers=True))
    # the embedded text carries title/section context, body does not
    c = next(c for c in chunks if "Revenue" in c.section or "revenue" in c.body.lower())
    assert "Northwind Report" in c.text
    assert "Northwind Report" not in c.body


def test_bm25_ranks_exact_terms():
    doc = load_text(ANNUAL_REPORT, title="R")
    chunks = chunk_document(doc, Settings())
    bm25 = BM25Index()
    bm25.add(chunks)
    res = bm25.search("deadlock indemnification liability", k=3)
    assert res
    assert any("liab" in c.body.lower() for c, _ in res)


def test_rrf_fusion_math():
    rankings = {"a": ["x", "y", "z"], "b": ["y", "x"]}
    scores = reciprocal_rank_fusion(rankings, k=60)
    # y appears at ranks 2 and 1; x at 1 and 2 -> equal; both above z
    assert scores["x"] > scores["z"]
    assert abs(scores["x"] - scores["y"]) < 1e-9


def test_hybrid_retrieval_finds_relevant_chunk():
    doc = load_text(ANNUAL_REPORT, title="R")
    chunks = chunk_document(doc, Settings())
    r = HybridRetriever(HashingEmbedder(128), InMemoryVectorStore(), BM25Index())
    r.index(chunks)
    hits = r.retrieve("what are the primary risk factors", top_k=5)
    assert hits
    assert any("risk" in h.chunk.body.lower() or "liability" in h.chunk.body.lower() for h in hits)
    # each hit records which retrievers surfaced it
    assert all(h.sources for h in hits)
