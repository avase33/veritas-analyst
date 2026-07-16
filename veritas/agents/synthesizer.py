"""Synthesizer agent.

Merges the specialists' findings into a single, cited answer. Offline it composes
the answer *extractively* — selecting the most on-topic sentences from the
retrieved passages and attaching a page citation to each — which guarantees the
answer is grounded and never invents facts. With a real LLM backend it instead
builds a strict RAG prompt (numbered context + page metadata + the
anti-hallucination system prompt) and delegates generation, still returning the
page citations from the retrieved chunks.
"""

from __future__ import annotations

from ..config import Settings
from ..grounding import REFUSAL, check_grounding
from ..llm import RAG_SYSTEM, LLM
from ..models import AgentResult, Answer, Citation, Retrieved
from ..text_utils import sentences, tokenize


class Synthesizer:
    name = "synthesizer"

    def __init__(self, llm: LLM, backend: str, settings: Settings) -> None:
        self.llm = llm
        self.backend = backend
        self.settings = settings

    def synthesize(self, query: str, subqueries: list[tuple[str, AgentResult]],
                   math_results: list[AgentResult]) -> Answer:
        # aggregate retrieved chunks across sub-questions, best score wins
        agg: dict[str, Retrieved] = {}
        for _, res in subqueries:
            for r in res.data.get("retrieved", []):
                cur = agg.get(r.chunk.id)
                if cur is None or r.score > cur.score:
                    agg[r.chunk.id] = r
        retrieved = sorted(agg.values(), key=lambda r: r.score, reverse=True)

        verdict = check_grounding(query, retrieved, self.settings.min_overlap)
        if not verdict.grounded:
            return Answer(query=query, text=REFUSAL, grounded=False, retrieved=retrieved)

        if self.backend == "mock":
            text, citations = self._extractive(query, subqueries, math_results)
        else:
            text, citations = self._llm(query, retrieved, math_results)

        return Answer(query=query, text=text, grounded=True, citations=citations,
                      retrieved=retrieved)

    # ---- offline extractive synthesis ----------------------------------

    def _extractive(self, query, subqueries, math_results):
        parts: list[str] = []
        citations: list[Citation] = []
        seen_sent: set[str] = set()

        for m in math_results:
            pages = m.data.get("pages", [])
            cite = f" (Page {pages[0]})" if pages else ""
            parts.append(m.output + cite)

        for subq, res in subqueries:
            picked = self._top_sentences(subq, res.data.get("retrieved", []), limit=2)
            for sent, r in picked:
                key = sent.lower()[:80]
                if key in seen_sent:
                    continue
                seen_sent.add(key)
                parts.append(f"{sent} (Page {r.chunk.page})")
                citations.append(Citation(doc_title=r.chunk.doc_title, page=r.chunk.page,
                                          section=r.chunk.section, quote=sent[:200]))

        if not parts:
            return REFUSAL, []
        return " ".join(parts), _dedup_citations(citations)

    def _top_sentences(self, subquery, retrieved, limit=2):
        q = set(tokenize(subquery))
        scored = []
        for r in retrieved[:5]:
            for sent in sentences(r.chunk.body):
                overlap = len(q & set(tokenize(sent)))
                if overlap > 0:
                    scored.append((overlap, len(sent), sent.strip(), r))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [(s, r) for _, _, s, r in scored[:limit]]

    # ---- real-LLM synthesis --------------------------------------------

    def _llm(self, query, retrieved, math_results):  # pragma: no cover - needs API
        ctx_lines = []
        for i, r in enumerate(retrieved, 1):
            ctx_lines.append(f"[{i}] (Page {r.chunk.page}"
                             + (f", {r.chunk.section}" if r.chunk.section else "") + f") {r.chunk.body}")
        math_note = ""
        if math_results:
            math_note = "\nPre-computed figures: " + "; ".join(m.output for m in math_results)
        prompt = (f"Context passages:\n" + "\n\n".join(ctx_lines) + math_note
                  + f"\n\nQuestion: {query}\n\nAnswer with page citations:")
        text = self.llm.complete(RAG_SYSTEM, prompt).strip() or REFUSAL
        citations = _dedup_citations([
            Citation(doc_title=r.chunk.doc_title, page=r.chunk.page,
                     section=r.chunk.section, quote=r.chunk.body[:200]) for r in retrieved[:4]])
        return text, citations


def _dedup_citations(citations: list[Citation]) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for c in citations:
        key = (c.doc_title, c.page)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
