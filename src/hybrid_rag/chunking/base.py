"""Base interface and factory for all chunking strategies.

Every concrete chunker inherits from :class:`BaseChunker` and implements
:meth:`chunk`.  The :func:`get_chunker` factory returns the right
implementation based on a strategy name string.
"""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from hybrid_rag.config import settings
from hybrid_rag.models import Chunk, Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ChunkerConfig:
    """Configuration parameters shared by all chunking strategies.

    Attributes
    ----------
    strategy:
        One of ``"fixed"``, ``"structural"``, or ``"semantic"``.
    chunk_size:
        Target chunk size in characters.
    chunk_overlap:
        Overlap in characters between consecutive fixed chunks.
    min_size:
        Minimum chunk size; chunks smaller than this are dropped.
    """

    strategy: str = dataclasses.field(default_factory=lambda: settings.chunk_strategy)
    chunk_size: int = dataclasses.field(default_factory=lambda: settings.chunk_size)
    chunk_overlap: int = dataclasses.field(default_factory=lambda: settings.chunk_overlap)
    min_size: int = dataclasses.field(default_factory=lambda: settings.chunk_min_size)


# ---------------------------------------------------------------------------
# Abstract base chunker
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies.

    Sub-classes must implement :meth:`chunk` and set :attr:`strategy`.

    Parameters
    ----------
    config:
        A :class:`ChunkerConfig` instance.  If *None*, a default config
        (reading from ``settings``) is used.
    """

    #: Sub-classes must override this with their strategy name.
    strategy: str = ""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* into a list of :class:`~hybrid_rag.models.Chunk` objects.

        Parameters
        ----------
        doc:
            A normalised ``Document`` to chunk.

        Returns
        -------
        list[Chunk]
            Ordered, non-empty list of chunks.  Implementations must
            filter out chunks smaller than ``config.min_size``.
        """

    def _make_chunk(
        self,
        doc: Document,
        text: str,
        position: int,
        heading_path: str = "",
        extra_meta: dict[str, Any] | None = None,
    ) -> Chunk:
        """Convenience factory that stamps ``strategy`` and ``doc_id``."""
        meta = dict(doc.metadata)
        meta.setdefault("source", doc.source_path)
        if extra_meta:
            meta.update(extra_meta)
        return Chunk(
            doc_id=doc.doc_id,
            text=text,
            heading_path=heading_path,
            strategy=self.strategy,
            position=position,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_chunker(
    strategy: str | None = None,
    config: ChunkerConfig | None = None,
    embedder: Callable[[list[str]], Any] | None = None,
) -> BaseChunker:
    """Return the chunker implementation for *strategy*.

    Imports are deferred to avoid circular dependencies between the
    concrete modules and this base module.

    Parameters
    ----------
    strategy:
        ``"fixed"``, ``"structural"``, or ``"semantic"``.  Defaults to
        ``settings.chunk_strategy``.
    config:
        Optional :class:`ChunkerConfig`.  When *None*, a default config
        is built; ``config.strategy`` is overridden with *strategy*.
    embedder:
        Optional callable for semantic embedding model.

    Returns
    -------
    BaseChunker
        Instantiated chunker ready to use.

    Raises
    ------
    ValueError
        For unknown strategy names.
    """
    if strategy is None:
        strategy = settings.chunk_strategy

    strategy = strategy.lower()

    if config is None:
        config = ChunkerConfig(strategy=strategy)
    else:
        config = dataclasses.replace(config, strategy=strategy)

    if strategy == "fixed":
        from hybrid_rag.chunking.fixed import FixedChunker  # noqa: PLC0415

        return FixedChunker(config)

    if strategy == "structural":
        from hybrid_rag.chunking.structural import StructuralChunker  # noqa: PLC0415

        return StructuralChunker(config)

    if strategy == "semantic":
        from hybrid_rag.chunking.semantic import SemanticChunker  # noqa: PLC0415

        return SemanticChunker(config, embedder=embedder)

    raise ValueError(
        f"Unknown chunking strategy '{strategy}'. "
        "Valid options: 'fixed', 'structural', 'semantic'."
    )
