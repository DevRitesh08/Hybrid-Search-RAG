"""Unified retrieval pipeline.

Ties together dense, sparse (BM25), Reciprocal Rank Fusion, and cross-encoder
reranking into a single ``retrieve`` entry-point.  Callers choose a *mode* and
receive both the ranked chunks and a per-stage latency breakdown.

Supported modes
---------------
``"dense"``
    Pure vector search via ChromaDB.

``"bm25"``
    Pure BM25 keyword search.

``"hybrid"``
    Dense + BM25 merged with Reciprocal Rank Fusion.

``"hybrid_rerank"``
    Hybrid fusion followed by cross-encoder reranking.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from hybrid_rag.config import settings
from hybrid_rag.models import RetrievedChunk
from hybrid_rag.retrieval import dense, fusion, rerank, sparse

logger = logging.getLogger(__name__)

# Advertised mode registry (useful for validation in callers)
MODES: list[str] = ["dense", "bm25", "hybrid", "hybrid_rerank"]

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]

# ── Internal helpers ──────────────────────────────────────────────────────────


def _now_ms() -> float:
    """Return current time in milliseconds using the highest-resolution clock."""
    return time.perf_counter() * 1_000.0


# ── Public API ────────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    mode: str | None = None,
    top_k: int | None = None,
    strategy: str | None = None,
) -> tuple[list[RetrievedChunk], dict[str, float]]:
    """Run the full retrieval pipeline for *query* under the chosen *mode*.

    Args:
        query:    The raw user query string.
        mode:     One of ``"dense"``, ``"bm25"``, ``"hybrid"``, or
                  ``"hybrid_rerank"``.  Defaults to ``settings.retrieval_mode``.
        top_k:    Candidate pool size before optional reranking.  Defaults to
                  ``settings.retrieval_top_k``.
        strategy: Chunk strategy hint forwarded to the dense/sparse retrievers
                  (currently informational only).

    Returns:
        A 2-tuple ``(chunks, latency_dict)`` where:

        * ``chunks`` is a :class:`list` of :class:`~hybrid_rag.models.RetrievedChunk`
          sorted by descending relevance score.
        * ``latency_dict`` is a :class:`dict` with keys:

          - ``"retrieve_ms"``  — wall-clock time spent in retrieval (ms).
          - ``"rerank_ms"``    — wall-clock time spent in reranking (ms, 0 if skipped).

    Raises:
        ValueError:       If an unknown *mode* is supplied.
        FileNotFoundError: If BM25 index is requested but has not been built.
        RuntimeError:     If the ChromaDB collection is empty (dense/hybrid).

    Examples:
        >>> from hybrid_rag.retrieval.pipeline import retrieve, MODES
        >>> chunks, latency = retrieve("What is BERT?", mode="hybrid_rerank", top_k=20)
        >>> print(latency)
        {'retrieve_ms': 42.1, 'rerank_ms': 183.7}
    """
    effective_mode: str = mode if mode is not None else settings.retrieval_mode
    effective_top_k: int = top_k if top_k is not None else settings.retrieval_top_k

    if effective_mode not in MODES:
        raise ValueError(
            f"Unknown retrieval mode '{effective_mode}'.  "
            f"Valid modes are: {MODES}"
        )

    logger.info(
        "Pipeline.retrieve | mode=%s | top_k=%d | query=%r",
        effective_mode,
        effective_top_k,
        query[:80],
    )

    latency: dict[str, float] = {"retrieve_ms": 0.0, "rerank_ms": 0.0}
    chunks: list[RetrievedChunk] = []

    # ── Retrieval stage ───────────────────────────────────────────────────────
    t0 = _now_ms()

    if effective_mode == "dense":
        chunks = dense.retrieve(query, top_k=effective_top_k, strategy=strategy)

    elif effective_mode == "bm25":
        chunks = sparse.retrieve(query, top_k=effective_top_k, strategy=strategy)

    elif effective_mode == "hybrid":
        dense_chunks = dense.retrieve(query, top_k=effective_top_k, strategy=strategy)
        sparse_chunks = sparse.retrieve(query, top_k=effective_top_k, strategy=strategy)
        chunks = fusion.fuse(dense_chunks, sparse_chunks)[:effective_top_k]

    elif effective_mode == "hybrid_rerank":
        stage1_k = max(effective_top_k * 2, settings.retrieval_top_k)
        dense_chunks = dense.retrieve(query, top_k=stage1_k, strategy=strategy)
        sparse_chunks = sparse.retrieve(query, top_k=stage1_k, strategy=strategy)
        chunks = fusion.fuse(dense_chunks, sparse_chunks)

    latency["retrieve_ms"] = _now_ms() - t0

    logger.debug(
        "Retrieval stage done | mode=%s | chunks=%d | retrieve_ms=%.1f",
        effective_mode,
        len(chunks),
        latency["retrieve_ms"],
    )

    # ── Reranking stage (hybrid_rerank only) ──────────────────────────────────
    if effective_mode == "hybrid_rerank" and chunks:
        t1 = _now_ms()
        chunks = rerank.rerank(query, chunks, top_k=effective_top_k)
        latency["rerank_ms"] = _now_ms() - t1

        logger.debug(
            "Rerank stage done | chunks=%d | rerank_ms=%.1f",
            len(chunks),
            latency["rerank_ms"],
        )

    return chunks, latency


class HybridRetrievalPipeline:
    """Unified retrieval pipeline class wrapper."""

    @staticmethod
    def retrieve(
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, float]]:
        """Run the retrieval pipeline."""
        return retrieve(query, mode=mode, top_k=top_k, strategy=strategy)

