"""Document ingestion: loading + contextual chunking."""

from .loader import load_document, load_text
from .chunking import chunk_document

__all__ = ["load_document", "load_text", "chunk_document"]
