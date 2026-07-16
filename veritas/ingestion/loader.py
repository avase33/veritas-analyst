"""Document loading.

Loads .txt / .md natively and .pdf via the optional ``pypdf`` dependency. Page
boundaries are normalised to the form-feed character (``\\x0c``) so the chunker
can attach an accurate page number to every chunk — the basis for citations.
Text documents may mark pages explicitly with ``=== PAGE N ===`` lines or form
feeds; otherwise the whole document is treated as paginated by the chunker.
"""

from __future__ import annotations

import os
import re

from ..errors import IngestionError
from ..models import Document

PAGE_MARKER = re.compile(r"^\s*(?:={2,}\s*)?page\s+\d+\s*(?:={2,})?\s*$", re.I)


def _normalize_pages(text: str) -> str:
    """Convert explicit '=== PAGE N ===' lines into form-feed page breaks."""
    if "\x0c" in text:
        return text
    out_lines: list[str] = []
    first = True
    for line in text.splitlines():
        if PAGE_MARKER.match(line):
            if not first:
                out_lines.append("\x0c")
            first = False
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def load_text(text: str, title: str = "Untitled", source: str = "") -> Document:
    if not text or not text.strip():
        raise IngestionError("empty document")
    return Document(title=title, text=_normalize_pages(text), source=source)


def _load_pdf(path: str, title: str) -> Document:  # pragma: no cover - optional dep
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise IngestionError("PDF support requires the 'pdf' extra (pip install veritas-analyst[pdf])") from exc
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\x0c".join(pages)
    meta_title = title
    if reader.metadata and reader.metadata.title:
        meta_title = reader.metadata.title
    return Document(title=meta_title, text=text, source=path, metadata={"pages": len(pages)})


def load_document(path: str, title: str | None = None) -> Document:
    if not os.path.exists(path):
        raise IngestionError(f"file not found: {path}")
    title = title or os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path, title)
    with open(path, encoding="utf-8", errors="replace") as f:
        return load_text(f.read(), title=title, source=path)
