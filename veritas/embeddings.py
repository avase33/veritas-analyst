"""Text embeddings.

Default is a deterministic hashing embedder (the "hashing trick") over content
tokens + bigrams with sublinear tf weighting, L2-normalised — no model download,
reproducible everywhere, good enough for dense semantic retrieval in the offline
pipeline. Set ``VERITAS_EMBEDDINGS=openai`` (with ``OPENAI_API_KEY``) to use
``text-embedding-3-large`` via :class:`OpenAIEmbedder`.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence

from .text_utils import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


def _h(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")


def l2(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else vec


class HashingEmbedder:
    def __init__(self, dim: int = 256, use_bigrams: bool = True) -> None:
        self.dim = dim
        self.use_bigrams = use_bigrams

    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        if self.use_bigrams:
            toks = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        return toks

    def embed(self, text: str) -> list[float]:
        counts: dict[int, float] = {}
        for tok in self._features(text):
            h = _h(tok)
            counts[h] = counts.get(h, 0.0) + 1.0
        vec = [0.0] * self.dim
        for h, c in counts.items():
            sign = 1.0 if (h >> 61) & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(c))
        return l2(vec)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class OpenAIEmbedder:  # pragma: no cover - optional dep
    def __init__(self, model: str = "text-embedding-3-large", api_key: str | None = None) -> None:
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model
        self.dim = 3072

    def embed(self, text: str) -> list[float]:
        r = self._client.embeddings.create(model=self._model, input=text)
        return l2(list(r.data[0].embedding))

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        r = self._client.embeddings.create(model=self._model, input=list(texts))
        return [l2(list(d.embedding)) for d in r.data]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))   # inputs L2-normalised


def build_embedder(backend: str = "hashing", dim: int = 256,
                   openai_model: str = "text-embedding-3-large") -> Embedder:
    if backend == "openai":
        return OpenAIEmbedder(openai_model)
    return HashingEmbedder(dim=dim)
