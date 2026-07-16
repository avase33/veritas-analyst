"""Specialist agents: semantic search, data retrieval, and math.

* **SemanticSearchAgent** — retrieves the passages most relevant to a qualitative
  sub-question (risks, explanations, summaries).
* **DataRetrievalAgent** — retrieves passages and extracts the concrete numeric
  figures (with their surrounding context) needed to answer a quantitative
  sub-question.
* **MathAgent** — given extracted figures and an arithmetic intent (growth,
  total, average, difference, ratio), computes the result deterministically. It
  only fires when it can compute confidently; otherwise the extractive answer
  stands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..models import AgentResult, Retrieved
from ..retrieval.hybrid import HybridRetriever
from ..text_utils import find_numbers, sentences, tokenize


@dataclass
class Figure:
    value: float
    raw: str
    page: int
    context: str
    is_percent: bool = False

    def to_dict(self) -> dict:
        return {"value": self.value, "raw": self.raw, "page": self.page,
                "is_percent": self.is_percent}


def _to_float(raw: str) -> Optional[float]:
    s = raw.replace(",", "").replace("$", "").strip()
    pct = s.endswith("%")
    s = s.rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


class SemanticSearchAgent:
    name = "semantic_search"

    def run(self, subquery: str, retriever: HybridRetriever, top_k: int = 6) -> AgentResult:
        retrieved = retriever.retrieve(subquery, top_k=top_k)
        snippet = _best_sentence(subquery, retrieved)
        res = AgentResult(agent=self.name, output=snippet)
        res.data["retrieved"] = retrieved
        return res


class DataRetrievalAgent:
    name = "data_retrieval"

    def run(self, subquery: str, retriever: HybridRetriever, top_k: int = 6) -> AgentResult:
        retrieved = retriever.retrieve(subquery, top_k=top_k)
        figures: list[Figure] = []
        for r in retrieved[:4]:
            for sent in sentences(r.chunk.body):
                for raw in find_numbers(sent):
                    val = _to_float(raw)
                    if val is None:
                        continue
                    figures.append(Figure(value=val, raw=raw, page=r.chunk.page,
                                          context=sent.strip(),
                                          is_percent=raw.strip().endswith("%")))
        res = AgentResult(agent=self.name, output=_best_sentence(subquery, retrieved))
        res.data["retrieved"] = retrieved
        res.data["figures"] = figures
        return res


class MathAgent:
    name = "math"

    _OPS = {
        "growth": ("growth", "grew", "increase", "change", "rose", "up by", "decline", "decrease"),
        "total": ("total", "sum", "combined", "altogether"),
        "average": ("average", "mean", "per "),
        "difference": ("difference", "more than", "less than", "gap", "versus", "vs"),
        "ratio": ("ratio", "margin", "per dollar", "divided by"),
    }

    def _intent(self, query: str) -> Optional[str]:
        low = query.lower()
        for op, cues in self._OPS.items():
            if any(c in low for c in cues):
                return op
        return None

    @staticmethod
    def _ordered_from_to(query: str, figures: list["Figure"]) -> Optional[tuple[float, float]]:
        """Find (start, end) from a 'from A to B' / 'to B from A' phrasing.

        When several sentences use that phrasing (e.g. gross margin *and* revenue),
        prefer the one whose context best overlaps the question, so 'revenue growth
        in Q3' picks the revenue sentence rather than the margin sentence.
        """
        num = r"\$?([\d,]+(?:\.\d+)?)"
        q_terms = set(tokenize(query))
        best: Optional[tuple[float, float]] = None
        best_overlap = -1
        for f in figures:
            ctx = f.context.lower()
            pair = None
            m = re.search(r"\bfrom\s+" + num + r".*?\bto\s+" + num, ctx)
            if m:
                a, b = _to_float(m.group(1)), _to_float(m.group(2))
                if a is not None and b is not None:
                    pair = (a, b)          # from A ... to B  => start A, end B
            if pair is None:
                m = re.search(r"\bto\s+" + num + r".*?\bfrom\s+" + num, ctx)
                if m:
                    a, b = _to_float(m.group(1)), _to_float(m.group(2))
                    if a is not None and b is not None:
                        pair = (b, a)      # to B ... from A  => start A, end B
            if pair is None:
                continue
            overlap = len(q_terms & set(tokenize(f.context)))
            if overlap > best_overlap:
                best, best_overlap = pair, overlap
        return best

    def compute(self, query: str, figures: list[Figure]) -> Optional[AgentResult]:
        op = self._intent(query)
        if op is None or not figures:
            return None
        # prefer non-percent figures for arithmetic; keep order of appearance
        nums = [f for f in figures if not f.is_percent]

        if op == "growth":
            pair = self._ordered_from_to(query, figures) or (
                (nums[0].value, nums[1].value) if len(nums) >= 2 else None)
            if pair is None or pair[0] == 0:
                return None
            start, end = pair
            g = (end - start) / abs(start) * 100.0
            out = f"Computed change from {start:,.2f} to {end:,.2f} = {g:+.1f}%."
            return _math_result(out, {"operation": "growth", "result": round(g, 2)}, nums[:2])

        if op in ("total", "average") and len(nums) >= 2:
            vals = [f.value for f in nums]
            total = sum(vals)
            if op == "total":
                out = f"Computed total = {total:,.2f} from {len(vals)} figures."
                data = {"operation": "total", "result": total}
            else:
                out = f"Computed average = {total/len(vals):,.2f} across {len(vals)} figures."
                data = {"operation": "average", "result": total / len(vals)}
            return _math_result(out, data, nums)
        if op in ("difference", "ratio") and len(nums) >= 2:
            a, b = nums[0].value, nums[1].value
            if op == "difference":
                out = f"Computed difference = {abs(a - b):,.2f} ({a:,.2f} vs {b:,.2f})."
                data = {"operation": "difference", "result": abs(a - b)}
            else:  # ratio
                if b == 0:
                    return None
                out = f"Computed ratio = {a / b:,.3f} ({a:,.2f} / {b:,.2f})."
                data = {"operation": "ratio", "result": a / b}
            return _math_result(out, data, nums[:2])
        return None


def _math_result(out: str, data: dict, used: list[Figure]) -> AgentResult:
    r = AgentResult(agent="math", output=out, data={**data, "used": [f.to_dict() for f in used]})
    r.data["pages"] = sorted({f.page for f in used})
    return r


def _best_sentence(query: str, retrieved: list[Retrieved]) -> str:
    q = set(tokenize(query))
    best, best_score = "", -1
    for r in retrieved[:5]:
        for sent in sentences(r.chunk.body):
            score = len(q & set(tokenize(sent)))
            if score > best_score:
                best, best_score = sent.strip(), score
    return best
