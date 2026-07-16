# Architecture

Veritas is a **multi-agent RAG** analyst: it answers questions strictly from
uploaded documents, delegating work to specialised agents and citing pages.

## Data flow

```
Phase A — Ingestion
   PDF/TXT/MD ─► parse (page-aware) ─► contextual chunking ─► embed ─► index
                                                                  │
                                       ┌──────────────────────────┴───────┐
                                       │  vector store        BM25 index   │
                                       └───────────────────────────────────┘

Phase B — Multi-agent retrieval & generation
   query ─► Router ─► sub-questions
                       ├─ numeric  ─► Data-Retrieval agent ─► Math agent
                       └─ qualitative ─► Semantic-Search agent
                                   │
                       hybrid retrieval (dense + BM25) fused by RRF
                                   │
                       Grounding gate (refuse if unsupported)
                                   │
                       Synthesizer ─► cited answer ("… (Page 14)")
```

## Ingestion

* **Page-aware parsing** — `.txt/.md` (native) and `.pdf` (pypdf) are normalised
  so every chunk knows its page number — the basis for citations.
* **Contextual chunking** — overlapping windows, but the document title and the
  current **section header** are prepended to each chunk before embedding, so an
  orphaned fact like *"Revenue was up 15%."* is retrievable in context. This is
  the single biggest quality lever for messy documents.

## Hybrid retrieval + RRF

Dense vector search captures meaning; **BM25** (implemented from scratch) captures
exact terms an analyst cares about. The two ranked lists are merged with
**Reciprocal Rank Fusion**:

```
RRF(d) = Σ_i 1 / (k + rank_i(d))        (k = 60)
```

RRF fuses rankings without calibrating incompatible score scales and lifts the
chunks both retrievers agree on.

## Multi-agent orchestration (LangGraph-style)

A stateful graph routes each sub-question to a specialist:

* **Router** decomposes compound questions and classifies each part
  (numeric → data-retrieval + math; qualitative → semantic-search).
* **Data-Retrieval agent** pulls passages and extracts concrete figures.
* **Math agent** computes growth/total/average/difference/ratio from the extracted
  figures, deterministically, when the intent is clear.
* **Semantic-Search agent** pulls the passages that explain qualitative asks.
* **Synthesizer** composes the final, cited answer.

## The engineering differentiators

* **Anti-hallucination grounding** — before answering, the system checks the
  retrieved evidence actually overlaps the question; if not, it returns exactly
  *"I cannot find this information in the provided document."* rather than guessing.
* **Contextual chunking** (above) so facts aren't orphaned from their context.
* **Citations** — page numbers are carried as chunk metadata from ingestion
  through retrieval to the final answer.

## Offline-first

The default embedder (hashing), LLM (deterministic extractive synthesizer) and
vector store (in-memory) mean the whole system runs, tests and benchmarks with
**zero external services**. Set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and
`VERITAS_VECTOR=qdrant|pinecone` to switch on production models and stores — the
agent graph is unchanged.
