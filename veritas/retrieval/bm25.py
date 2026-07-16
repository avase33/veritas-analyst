"""BM25 keyword search (Okapi BM25), implemented from scratch.

Dense vectors capture meaning but miss exact terms — an analyst asking about a
specific line item ("EBITDA", a statute number, a defined term) needs lexical
precision. BM25 provides it. This is the keyword arm of the hybrid retriever.

Score for query q over document d:

    score(q, d) = Σ_t idf(t) · ( f(t,d) · (k1+1) ) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )

with idf(t) = ln( (N - n_t + 0.5)/(n_t + 0.5) + 1 ).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from ..models import Chunk
from ..text_utils import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0

    def add(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            toks = tokenize(c.body)
            tf = Counter(toks)
            self._chunks.append(c)
            self._tf.append(tf)
            self._len.append(len(toks))
            for term in tf:
                self._df[term] += 1
        total = sum(self._len)
        self._avgdl = total / len(self._len) if self._len else 0.0

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        if n == 0:
            return 0.0
        N = len(self._chunks)
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        q_terms = tokenize(query)
        if not q_terms or not self._chunks:
            return []
        scores: list[tuple[Chunk, float]] = []
        for i, tf in enumerate(self._tf):
            dl = self._len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 1))
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                s += self._idf(term) * (f * (self.k1 + 1)) / (f + denom_norm)
            if s > 0:
                scores.append((self._chunks[i], s))
        scores.sort(key=lambda p: p[1], reverse=True)
        return scores[:k]

    def __len__(self) -> int:
        return len(self._chunks)
