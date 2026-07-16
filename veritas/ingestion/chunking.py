"""Contextual, overlapping chunking.

Splits a document into overlapping windows that preserve context, tracking the
**page** and **section** each chunk came from (so answers can cite them). The key
"contextual chunking" differentiator: a raw window like *"Revenue was up 15%."*
is ambiguous, so we prepend the document title and the current section header to
the text that gets embedded and searched — while keeping the original body for
display and citation. This dramatically improves retrieval of otherwise-orphaned
facts.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..models import Chunk, Document

_WS = re.compile(r"\s+")
# a heading: markdown (#), or a short Title/UPPER line with no terminal period
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")
_BARE_HEADING = re.compile(r"^\s*([A-Z][A-Za-z0-9 ,&/()'-]{2,58})\s*$")
_PAGE_MARKER = re.compile(r"^\s*(?:={2,}\s*)?page\s+\d+\s*(?:={2,})?\s*$", re.I)


def _is_heading(line: str) -> str | None:
    m = _MD_HEADING.match(line)
    if m:
        return m.group(1).strip()
    s = line.strip()
    if not s or s.endswith((".", ":", ";", ",")):
        return None
    m = _BARE_HEADING.match(line)
    if m and 1 <= len(s.split()) <= 8 and not s[0].islower():
        # avoid treating ordinary sentences as headings
        words = s.split()
        if sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) - 1):
            return s
    return None


def _split_pages(text: str) -> list[str]:
    if "\x0c" in text:
        return text.split("\x0c")
    # also handle explicit '=== PAGE N ===' markers on documents built directly
    if any(_PAGE_MARKER.match(ln) for ln in text.splitlines()):
        pages: list[str] = []
        cur: list[str] = []
        started = False
        for ln in text.splitlines():
            if _PAGE_MARKER.match(ln):
                if started:
                    pages.append("\n".join(cur))
                    cur = []
                started = True
                continue
            cur.append(ln)
        pages.append("\n".join(cur))
        return pages if pages else [text]
    return [text]


def _words(text: str) -> list[str]:
    return _WS.sub(" ", text).strip().split(" ")


def chunk_document(doc: Document, settings: Settings | None = None) -> list[Chunk]:
    s = settings or Settings()
    size, overlap = max(20, s.chunk_tokens), max(0, min(s.chunk_overlap, s.chunk_tokens - 1))
    chunks: list[Chunk] = []

    for p_idx, page_text in enumerate(_split_pages(doc.text), start=1):
        section = ""
        # walk paragraphs to track the active section header, accumulate body words
        buffer: list[str] = []

        def flush_section(sec: str, words: list[str]) -> None:
            if not words:
                return
            step = size - overlap
            for start in range(0, len(words), step):
                window = words[start:start + size]
                if not window:
                    continue
                body = " ".join(window).strip()
                if len(body) < 3:
                    continue
                header = ""
                if s.contextual_headers:
                    header = doc.title + (f" — {sec}" if sec else "")
                text = (header + "\n" + body) if header else body
                chunks.append(Chunk(doc_id=doc.id, doc_title=doc.title, text=text,
                                    body=body, page=p_idx, section=sec))
                if start + size >= len(words):
                    break

        for line in page_text.splitlines():
            heading = _is_heading(line)
            if heading is not None:
                flush_section(section, buffer)
                buffer = []
                section = heading
                continue
            buffer.extend(w for w in _words(line) if w)
        flush_section(section, buffer)

    return chunks
