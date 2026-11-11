"""Dense (vector) retrieval using ChromaDB and SentenceTransformers embedder.

Integrates directly with VectorStore and EmbeddingService to enforce uniform collection
naming, BGE query prefixing, and score normalization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hybrid_rag.config import settings
from hybrid_rag.index.embedder import _BGE_QUERY_PREFIX, get_embedding_service
from hybrid_rag.index.vector_store import VectorStore
from hybrid_rag.models import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Lazy singletons ───────────────────────────────────────────────────────────

_embedder: Any | None = None


def _get_embedder() -> Any:
    """Return the cached SentenceTransformer embedder (loaded once)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info("Loading embedding model: %s", settings.embed_model)
        _embedder = SentenceTransformer(settings.embed_model)
    return _embedder


def _get_collection(strategy: str | None = None) -> Any:
    """Return the ChromaDB collection for the given or default strategy."""
    effective_strategy = strategy or settings.chunk_strategy
    store = VectorStore(strategy=effective_strategy)
    return store._get_collection()


# ── Public API ────────────────────────────────────────────────────────────────


def embed_query(query: str) -> list[float]:
    """Embed a query string with the configured model.

    Prepend BGE instruction prefix when using a BGE embedding model.

    Args:
        query: The raw query text.

    Returns:
        A list of floats representing the query vector.
    """
    try:
        service = get_embedding_service()
        vec = service.embed_query(query)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)
    except Exception:
        embedder = _get_embedder()
        prefixed = f"{_BGE_QUERY_PREFIX}{query}" if "bge" in settings.embed_model.lower() else query
        vector: list[float] = embedder.encode(prefixed, normalize_embeddings=True).tolist()
        return vector


def retrieve(
    query: str,
    top_k: int | None = None,
    strategy: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top-*k* chunks by cosine similarity to *query*.

    Args:
        query:    The raw query text.
        top_k:    How many results to return; defaults to ``settings.retrieval_top_k``.
        strategy: Chunking strategy to query; defaults to ``settings.chunk_strategy``.

    Returns:
        A list of :class:`~hybrid_rag.models.RetrievedChunk` objects, ranked 1-based
        by descending cosine similarity.  ``retrieval_type`` is ``"dense"``.

    Raises:
        RuntimeError: If the ChromaDB collection is empty.
    """
    effective_top_k: int = top_k if top_k is not None else settings.retrieval_top_k
    effective_strategy: str = strategy or settings.chunk_strategy

    store = VectorStore(strategy=effective_strategy)
    n_items = store.count()
    if n_items == 0:
        raise RuntimeError(
            "ChromaDB collection is empty — run the ingestion pipeline first "
            f"(e.g. `python -m hybrid_rag.index.build --strategy {effective_strategy}`)."
        )

    # Clamp top_k to the number of available items
    effective_top_k = min(effective_top_k, n_items)

    query_vector = embed_query(query)
    vec_arr = np.array(query_vector, dtype=np.float32)

    logger.debug(
        "Dense query | strategy=%s | top_k=%d | collection_size=%d",
        effective_strategy,
        effective_top_k,
        n_items,
    )

    chunks = store.query(vec_arr, top_k=effective_top_k)

    # Ensure source and source_path are consistently mapped
    for chunk in chunks:
        src = chunk.source or chunk.metadata.get("source") or chunk.metadata.get("source_path") or ""
        chunk.source = src
        chunk.metadata["source"] = src
        chunk.metadata["source_path"] = src

    logger.debug("Dense retrieval returned %d chunks", len(chunks))
    return chunks
