"""Stateful multi-agent graph (LangGraph-style).

Orchestrates the agents as a small state machine:

    route ─► for each sub-question: (data_retrieval | semantic_search)
          ─► math (if numeric figures were extracted)
          ─► synthesize (grounded, cited)

State flows through as an :class:`GraphState`; nodes are pure functions of that
state, mirroring how a LangGraph graph transitions. Swapping the deterministic
router/synthesizer for LLM-backed ones doesn't change the graph.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import Settings
from ..llm import LLM
from ..models import AgentResult, Answer
from ..retrieval.hybrid import HybridRetriever
from .router import Router
from .specialists import DataRetrievalAgent, MathAgent, SemanticSearchAgent
from .synthesizer import Synthesizer


@dataclass
class GraphState:
    query: str
    subquery_results: list[tuple[str, AgentResult]] = field(default_factory=list)
    math_results: list[AgentResult] = field(default_factory=list)
    agents_used: list[str] = field(default_factory=list)


class AgentGraph:
    def __init__(self, retriever: HybridRetriever, llm: LLM, backend: str,
                 settings: Settings) -> None:
        self.retriever = retriever
        self.settings = settings
        self.router = Router()
        self.semantic = SemanticSearchAgent()
        self.data = DataRetrievalAgent()
        self.math = MathAgent()
        self.synthesizer = Synthesizer(llm, backend, settings)

    def run(self, query: str) -> Answer:
        t0 = time.perf_counter()
        state = GraphState(query=query)

        plan = self.router.plan(query)
        state.agents_used.append("router")

        for sub in plan.subqueries:
            if sub.agent == "data_retrieval":
                res = self.data.run(sub.text, self.retriever, top_k=self.settings.top_k)
                state.subquery_results.append((sub.text, res))
                _add(state.agents_used, "data_retrieval")
                figures = res.data.get("figures", [])
                math_res = self.math.compute(sub.text, figures)
                if math_res is not None:
                    state.math_results.append(math_res)
                    _add(state.agents_used, "math")
            else:
                res = self.semantic.run(sub.text, self.retriever, top_k=self.settings.top_k)
                state.subquery_results.append((sub.text, res))
                _add(state.agents_used, "semantic_search")

        answer = self.synthesizer.synthesize(query, state.subquery_results, state.math_results)
        _add(state.agents_used, "synthesizer")
        answer.agents_used = state.agents_used
        answer.latency_ms = (time.perf_counter() - t0) * 1000.0
        return answer


def _add(lst: list[str], name: str) -> None:
    if name not in lst:
        lst.append(name)
