"""Sparse (BM25) retrieval backed by a pre-built JSON BM25Index.

Loads JSON indexes produced by ``hybrid_rag.index.bm25_index`` lazily on first
query and caches the instance per chunking strategy.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hybrid_rag.config import settings
from hybrid_rag.index.bm25_index import BM25Index
from hybrid_rag.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Cache loaded BM25Index instances by strategy name
_index_cache: dict[str, BM25Index] = {}


def _get_index_path(strategy: str) -> Path:
    """Return the expected JSON index path for a given chunking strategy."""
    return Path(settings.bm25_index_dir) / f"bm25_{strategy}.json"


def _load_index(strategy: str) -> BM25Index:
    """Load (and cache) the BM25 index from disk for a given strategy.

    Raises:
        FileNotFoundError: If the index file has not been built yet.
        RuntimeError:      If the index file is corrupted or unreadable.
    """
    if strategy in _index_cache:
        return _index_cache[strategy]

    index_path = _get_index_path(strategy)
    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at '{index_path}'. "
            "Build it first by running the ingestion pipeline "
            f"(e.g. `python -m hybrid_rag.index.build --strategy {strategy}`)."
        )

    logger.info("Loading BM25 index from %s", index_path)
    try:
        idx = BM25Index()
        idx.load(index_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to deserialise BM25 index from '{index_path}': {exc}"
        ) from exc

    _index_cache[strategy] = idx
    logger.debug(
        "BM25 index loaded for strategy '%s' — %d documents in corpus",
        strategy,
        idx.count(),
    )
    return idx


def clear_cache() -> None:
    """Clear cached BM25Index instances (useful in test teardown)."""
    _index_cache.clear()


# ── Public API ────────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    top_k: int | None = None,
    strategy: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top-*k* chunks using BM25 keyword scoring.

    Args:
        query:    The raw query text (tokenised internally with identifier preservation).
        top_k:    How many results to return; defaults to ``settings.retrieval_top_k``.
        strategy: Chunking strategy to query; defaults to ``settings.chunk_strategy``.

    Returns:
        A list of :class:`~hybrid_rag.models.RetrievedChunk` objects ranked 1-based
        by descending BM25 score.  ``retrieval_type`` is ``"sparse"``.

    Raises:
        FileNotFoundError: If the BM25 index has not been built.
        RuntimeError:      If the index file is corrupted.
    """
    effective_top_k: int = top_k if top_k is not None else settings.retrieval_top_k
    effective_strategy: str = strategy or settings.chunk_strategy

    bm25 = _load_index(effective_strategy)
    chunks = bm25.query(query, top_k=effective_top_k)

    # Ensure source and source_path are consistently mapped
    for chunk in chunks:
        src = chunk.source or chunk.metadata.get("source") or chunk.metadata.get("source_path") or ""
        chunk.source = src
        chunk.metadata["source"] = src
        chunk.metadata["source_path"] = src

    logger.debug(
        "BM25 retrieval returned %d chunks (strategy=%s, top_k=%d)",
        len(chunks),
        effective_strategy,
        effective_top_k,
    )
    return chunks
