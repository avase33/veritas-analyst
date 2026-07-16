"""Router agent.

Decomposes a possibly-compound question into sub-questions and routes each to the
right specialist:

* numeric / financial intent ("revenue growth", "how much", "total", "ratio") ->
  the **data-retrieval** agent (which also invokes the math agent);
* qualitative intent ("risks", "why", "summarise", "liabilities") -> the
  **semantic-search** agent.

Both specialists share the hybrid retriever; the distinction is what they *do*
with the chunks. Routing is deterministic keyword logic offline; an LLM classifier
can replace it via the same interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMERIC_CUES = ("revenue", "growth", "grew", "increase", "decrease", "total", "sum",
                 "average", "how much", "how many", "percent", "percentage", "ratio",
                 "margin", "ebitda", "profit", "loss", "cost", "price", "amount",
                 "calculate", "difference", "compare", "change", "rate", "number of",
                 "figure", "value", "$", "count")
_QUALITATIVE_CUES = ("risk", "liabilit", "exposure", "why", "explain", "summar", "describe",
                     "factor", "concern", "strateg", "outlook", "reason", "impact", "overview",
                     "what are", "list", "context")

_SPLIT = re.compile(r"\s+and\s+|;\s*|\?\s*", re.I)


@dataclass
class SubQuery:
    text: str
    agent: str          # "data_retrieval" | "semantic_search"


@dataclass
class RoutePlan:
    subqueries: list[SubQuery] = field(default_factory=list)

    @property
    def agents(self) -> list[str]:
        seen: list[str] = []
        for sq in self.subqueries:
            if sq.agent not in seen:
                seen.append(sq.agent)
        seen.append("synthesizer")
        return seen


class Router:
    def classify(self, text: str) -> str:
        low = text.lower()
        num = sum(1 for c in _NUMERIC_CUES if c in low)
        qual = sum(1 for c in _QUALITATIVE_CUES if c in low)
        return "data_retrieval" if num >= qual and num > 0 else "semantic_search"

    def plan(self, query: str) -> RoutePlan:
        clauses = [c.strip(" .,") for c in _SPLIT.split(query) if c.strip(" .,")]
        if not clauses:
            clauses = [query]
        # merge clauses that are too short to be standalone questions
        merged: list[str] = []
        for c in clauses:
            if len(c.split()) <= 2 and merged:
                merged[-1] = merged[-1] + " " + c
            else:
                merged.append(c)
        subs = [SubQuery(text=c, agent=self.classify(c)) for c in merged]
        return RoutePlan(subqueries=subs)
