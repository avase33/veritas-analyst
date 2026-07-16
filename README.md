<div align="center">

# Veritas

### Domain-Specific AI Analyst — Multi-Agent RAG

Upload dense documents (financial filings, contracts, research) and ask complex
questions answered **strictly from the provided context**. A router delegates to
specialist agents — data retrieval, math, semantic search — and a synthesizer
returns a **cited** answer, refusing when the evidence isn't there.

[![CI](https://github.com/avase33/veritas-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/avase33/veritas-analyst/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## The problem

General LLMs hallucinate and can't see your private documents. Basic RAG fails
when a question needs you to *connect the dots* across pages or combine different
kinds of data. Veritas is a **multi-agent RAG** system: instead of one model
doing everything, a router splits the question and hands each part to a specialist.

```
query → Router → { data-retrieval + math | semantic-search } → Synthesizer (cited)
                         ▲ hybrid retrieval: dense + BM25, fused by RRF
                         ▲ grounding gate: refuse if unsupported
```

## What's implemented (and actually runs)

Everything is **pure Python** and runs with **zero external services** offline:

- **Contextual chunking** — overlapping windows with the document title + section
  header prepended before embedding, so orphaned facts ("Revenue was up 15%") stay
  retrievable. Page numbers are tracked for citations.
- **Hybrid retrieval + RRF** — dense vector search **and** BM25 keyword search
  (both from scratch), fused with Reciprocal Rank Fusion (`RRF(d)=Σ 1/(k+rank)`).
- **Multi-agent orchestration** — a LangGraph-style stateful graph: **router**
  decomposes compound questions; **data-retrieval** + **math** agents handle
  numbers (growth, totals, ratios — parsing "from A to B" so growth isn't
  reversed); **semantic-search** handles qualitative asks; **synthesizer** writes
  the cited answer.
- **Anti-hallucination grounding** — refuses with *"I cannot find this information
  in the provided document."* when retrieval doesn't support the question.
- **Citations** — every claim carries a page number, carried as metadata from
  ingestion through to the answer.
- **FastAPI backend + Next.js frontend** — a full decoupled web app with a chat UI.

Set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and `VERITAS_VECTOR=qdrant|pinecone`
to switch to real models and hosted vector stores — the agent graph is unchanged.

## Quickstart (no keys, no servers)

```bash
pip install -e .
veritas demo
```

Real output (mock/offline mode):

```
Q: What was the total revenue growth in Q3, and what are the primary risk factors?
   [grounded]  agents: router, data_retrieval, math, semantic_search, synthesizer
A: Computed change from 112.80 to 138.90 = +23.1%. (Page 2) Year over year, Q3
   revenue increased from $96.4 million in the prior-year quarter. (Page 3) The
   company faces several primary risk factors... supply chain exposure... product
   liability... (Page 4)
   Sources: Page 2; Page 3; Page 4

Q: Who is the current CEO and what is their salary?
   [REFUSED (not in document)]
A: I cannot find this information in the provided document.
```

Ask against your own file:

```bash
veritas ask "What is the limitation of liability?" --file contract.pdf   # needs [pdf] extra
```

## Run the web app

```bash
pip install -e ".[serve]"
veritas serve                     # http://localhost:8000  (chat UI + API)
```

```
POST /api/upload   ingest a document
POST /api/chat     ask -> { answer, grounded, citations[], agents_used[] }
GET  /api/stats
```

Next.js frontend (Vercel-ready) lives in [`frontend/`](frontend/):

```bash
cd frontend && npm install && npm run dev    # proxies /api to the backend
```

## Full stack (Docker Compose)

```bash
docker compose up --build          # backend + Qdrant + Next.js frontend
```

Deploy: **backend → Render** (`render.yaml`), **frontend → Vercel**
(`frontend/vercel.json`).

## Repository layout

```
veritas/
  ingestion/     page-aware loader (txt/md/pdf) + contextual chunking
  embeddings.py  hashing embedder (+ OpenAI adapter)
  retrieval/     vector store, BM25, hybrid RRF fusion
  agents/        router, specialists (data/math/semantic), synthesizer, graph
  grounding.py   anti-hallucination gate       |  llm.py  mock/Anthropic/OpenAI
  engine.py      Analyst façade                |  serving/  FastAPI + chat UI
frontend/        Next.js + TypeScript chat frontend
docker-compose.yml, Dockerfile, render.yaml, .github/workflows/ci.yml
```

## Development

```bash
pip install -e ".[serve,dev]"
pytest --cov=veritas
ruff check veritas scripts
python verify_veritas.py       # offline end-to-end self-check
```

## License

MIT © Akhil Vase
