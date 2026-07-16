import pytest

from veritas.config import Settings
from veritas.engine import Analyst
from veritas.agents.router import Router
from veritas.agents.specialists import MathAgent, Figure
from veritas.grounding import REFUSAL
from veritas.mockdata import sample_documents


@pytest.fixture(scope="module")
def analyst():
    a = Analyst(Settings())
    for doc in sample_documents():
        a.ingest(doc)
    return a


def test_router_decomposes_and_classifies():
    plan = Router().plan("What was the total revenue growth in Q3, and what are the primary risk factors?")
    assert len(plan.subqueries) == 2
    agents = {sq.agent for sq in plan.subqueries}
    assert "data_retrieval" in agents and "semantic_search" in agents
    assert "synthesizer" in plan.agents


def test_math_agent_growth_respects_from_to_order():
    figs = [Figure(value=138.9, raw="138.9", page=3,
                   context="Third quarter revenue grew to 138.9 million from 112.8 million."),
            Figure(value=112.8, raw="112.8", page=3, context="")]
    res = MathAgent().compute("what was the revenue growth", figs)
    assert res is not None
    assert res.data["operation"] == "growth"
    assert 20 < res.data["result"] < 26     # +23.1%, not the reversed -18.8%


def test_answer_is_grounded_and_cited(analyst):
    ans = analyst.ask("What was the total revenue growth in Q3, and what are the primary risk factors?")
    assert ans.grounded
    assert ans.citations and all(c.page > 0 for c in ans.citations)
    assert "data_retrieval" in ans.agents_used and "semantic_search" in ans.agents_used
    low = ans.text.lower()
    assert "risk" in low or "liability" in low or "exposure" in low


def test_refuses_when_not_in_document(analyst):
    ans = analyst.ask("Who is the current CEO and what is their annual salary?")
    assert not ans.grounded
    assert ans.text == REFUSAL
    assert ans.citations == []


def test_liability_question_cites_agreement(analyst):
    ans = analyst.ask("What is the limitation of liability in the services agreement?")
    assert ans.grounded
    assert "liability" in ans.text.lower()
    assert any("Services Agreement" in c.doc_title for c in ans.citations)


def test_cash_question_returns_figure(analyst):
    ans = analyst.ask("How much cash did the company end the year with?")
    assert ans.grounded
    assert "210" in ans.text          # $210.0 million
