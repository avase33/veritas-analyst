"""Anti-hallucination grounding gate.

Before the synthesizer speaks, we check that retrieval actually found relevant
evidence. The gate measures how much of the query's *content vocabulary* is
present in the retrieved passages (a scale-stable signal, unlike raw fusion
scores). If no retrieved chunk shares enough terms with the query, the system
refuses with a fixed message instead of inventing an answer — the single most
important guardrail for a document analyst.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import Retrieved
from .text_utils import tokenize

REFUSAL = "I cannot find this information in the provided document."


@dataclass
class GroundingVerdict:
    grounded: bool
    reason: str = ""
    overlap: float = 0.0


def check_grounding(query: str, retrieved: Sequence[Retrieved], min_overlap: float) -> GroundingVerdict:
    if not retrieved:
        return GroundingVerdict(False, "no chunks retrieved")

    q_terms = set(tokenize(query))
    if not q_terms:
        return GroundingVerdict(True, "no content terms in query", 1.0)

    best_overlap = 0.0
    matched: set[str] = set()
    for r in retrieved:
        c_terms = set(tokenize(r.chunk.body))
        hit = q_terms & c_terms
        matched |= hit
        best_overlap = max(best_overlap, len(hit) / len(q_terms))

    # A single incidental word (e.g. "annual" matching a title) must not ground an
    # answer: require at least two distinct query terms to appear, unless the query
    # itself has only one content term.
    min_terms = 1 if len(q_terms) <= 1 else 2
    if len(matched) < min_terms:
        return GroundingVerdict(False, f"only {len(matched)} query terms found", best_overlap)
    if best_overlap < min_overlap:
        return GroundingVerdict(False, f"query-term overlap {best_overlap:.2f} < {min_overlap}", best_overlap)
    return GroundingVerdict(True, "grounded", best_overlap)
