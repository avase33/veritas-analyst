"""LLM providers.

The synthesizer needs a text generator that is *grounded*: it must answer only
from the supplied context and cite pages. The default :class:`MockLLM` is a
deterministic **extractive** generator (selects and stitches the most relevant
context sentences, adds citations) so the whole pipeline runs and is testable
offline with no keys. :class:`AnthropicLLM` / :class:`OpenAILLM` call the real
models with the same strict system prompt.
"""

from __future__ import annotations

from typing import Protocol

RAG_SYSTEM = (
    "You are a meticulous domain analyst. Answer the user's question STRICTLY using "
    "the provided context passages. Cite the page number for every claim like "
    "'(Page N)'. If the answer is not present in the context, reply exactly: "
    "'I cannot find this information in the provided document.' Never use outside knowledge."
)


class LLM(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class MockLLM:
    """Deterministic — synthesis is handled extractively by the Synthesizer."""

    name = "mock"

    def complete(self, system: str, prompt: str) -> str:
        return ""


class AnthropicLLM:  # pragma: no cover - optional dep
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-sonnet-latest", api_key: str | None = None) -> None:
        import anthropic  # type: ignore

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAILLM:  # pragma: no cover - optional dep
    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model

    def complete(self, system: str, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model, messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""


def build_llm(backend: str = "mock", anthropic_model: str = "claude-3-5-sonnet-latest",
              openai_model: str = "gpt-4o") -> LLM:
    if backend == "anthropic":
        return AnthropicLLM(anthropic_model)
    if backend == "openai":
        return OpenAILLM(openai_model)
    return MockLLM()
