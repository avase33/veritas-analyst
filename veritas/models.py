"""Core domain models (dataclasses)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Document:
    title: str
    text: str
    source: str = ""
    id: str = field(default_factory=lambda: new_id("doc"))
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    text: str                    # embedded/searched text (may include contextual header)
    body: str                    # original chunk body without the prepended header
    page: int
    section: str = ""
    id: str = field(default_factory=lambda: new_id("chk"))
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "doc_title": self.doc_title, "page": self.page,
                "section": self.section, "body": self.body}


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rank: int = 0
    sources: list[str] = field(default_factory=list)   # which retrievers surfaced it

    def to_dict(self) -> dict[str, Any]:
        return {"chunk": self.chunk.to_dict(), "score": round(self.score, 4),
                "rank": self.rank, "retrievers": self.sources}


@dataclass
class Citation:
    doc_title: str
    page: int
    section: str = ""
    quote: str = ""

    def label(self) -> str:
        loc = f"Page {self.page}" if self.page else "the document"
        if self.section:
            loc += f", {self.section}"
        return loc

    def to_dict(self) -> dict[str, Any]:
        return {"doc_title": self.doc_title, "page": self.page,
                "section": self.section, "quote": self.quote}


@dataclass
class AgentResult:
    agent: str
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Answer:
    query: str
    text: str
    grounded: bool
    citations: list[Citation] = field(default_factory=list)
    agents_used: list[str] = field(default_factory=list)
    retrieved: list[Retrieved] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "answer": self.text, "grounded": self.grounded,
                "citations": [c.to_dict() for c in self.citations],
                "agents_used": self.agents_used,
                "sources": [r.to_dict() for r in self.retrieved],
                "latency_ms": round(self.latency_ms, 2)}
