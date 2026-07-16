"""Exception hierarchy for Veritas."""

from __future__ import annotations


class VeritasError(Exception):
    """Base class for all Veritas errors."""


class ConfigError(VeritasError):
    """Invalid configuration."""


class IngestionError(VeritasError):
    """Failed to load or chunk a document."""


class RetrievalError(VeritasError):
    """Retrieval / vector-store failure."""


class AgentError(VeritasError):
    """An agent failed to produce a result."""
