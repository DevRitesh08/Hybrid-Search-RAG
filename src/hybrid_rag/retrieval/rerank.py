"""Cross-encoder reranker using sentence-transformers' CrossEncoder.

The model is loaded lazily on first call and kept as a module-level singleton
to avoid repeated I/O across requests.

Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
(fast, effective MS-MARCO passage reranker)
"""

from __future__ import annotations

import logging
from typing import Any

from hybrid_rag.config import settings
from hybrid_rag.models import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Lazy singleton ────────────────────────────────────────────────────────────

_cross_encoder: Any | None = None
_load_attempted: bool = False


def _get_cross_encoder() -> Any | None:
    """Return the cached CrossEncoder (loaded once per process), or None if unavailable."""
    global _cross_encoder, _load_attempted
    if _cross_encoder is None and not _load_attempted:
        _load_attempted = True
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            logger.info("Loading cross-encoder reranker: %s", settings.rerank_model)
            _cross_encoder = CrossEncoder(settings.rerank_model)
            logger.debug("Cross-encoder loaded successfully")
        except Exception as exc:
            logger.warning(
                "Failed to load cross-encoder (%s): %s. Offline fallback mode active.",
                settings.rerank_model,
                exc,
            )
            _cross_encoder = None
    return _cross_encoder


def clear_cache() -> None:
    """Clear cached CrossEncoder singleton."""
    global _cross_encoder, _load_attempted
    _cross_encoder = None
    _load_attempted = False


def _build_fallback_chunks(
    chunks: list[RetrievedChunk],
    effective_top_k: int,
) -> list[RetrievedChunk]:
    """Construct fallback candidates with preserved upstream metadata and score bounds."""
    fallback_top = chunks[:effective_top_k]
    fallback_results: list[RetrievedChunk] = []
    for new_rank, chunk in enumerate(fallback_top, start=1):
        src_val = (
            chunk.source
            or chunk.metadata.get("source")
            or chunk.metadata.get("source_path")
            or ""
        )
        upstream_type = chunk.retrieval_type
        raw_score = float(chunk.score)

        if upstream_type == "hybrid":
            bounded_score = max(0.0, raw_score)
            score_scale = "rrf"
        elif upstream_type == "dense":
            bounded_score = min(1.0, max(0.0, raw_score))
            score_scale = "normalized"
        elif upstream_type == "sparse":
            bounded_score = max(0.0, raw_score)
            score_scale = "bm25"
        else:
            bounded_score = raw_score
            score_scale = "raw"

        meta = dict(chunk.metadata)
        meta["source"] = src_val
        meta["source_path"] = src_val
        meta["rerank_fallback"] = True
        meta["fallback"] = True
        meta["upstream_retrieval_type"] = upstream_type
        meta["original_retrieval_type"] = upstream_type
        meta["original_score"] = raw_score
        meta["score_scale"] = score_scale

        fallback_results.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                heading_path=chunk.heading_path,
                source=src_val,
                score=bounded_score,
                rank=new_rank,
                retrieval_type="reranked",
                metadata=meta,
            )
        )
    return fallback_results


# ── Public API ────────────────────────────────────────────────────────────────


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Score every (query, chunk) pair and return the top-*k* by cross-encoder score.

    In offline environments or if model loading fails, falls back gracefully to
    returning candidates in upstream order truncated to *top_k*, with
    ``retrieval_type="reranked"``.

    Args:
        query:  The raw query text.
        chunks: Candidate chunks from any upstream retriever.
        top_k:  How many chunks to return after reranking; defaults to
                ``settings.rerank_top_k``.

    Returns:
        A list of :class:`~hybrid_rag.models.RetrievedChunk` with:
        - ``retrieval_type`` set to ``"reranked"``
        - ``score`` set to the cross-encoder logit (or original score in fallback)
        - ``rank`` re-assigned starting from 1 (best = 1)

        Returns an empty list if *chunks* is empty.
    """
    if not chunks:
        logger.debug("Reranker: empty input — returning []")
        return []

    effective_top_k: int = top_k if top_k is not None else settings.rerank_top_k

    cross_encoder = None
    try:
        cross_encoder = _get_cross_encoder()
    except Exception as exc:
        logger.warning("Error retrieving cross-encoder: %s", exc)

    if cross_encoder is None:
        logger.warning(
            "Cross-encoder unavailable; applying offline passthrough ranking."
        )
        return _build_fallback_chunks(chunks, effective_top_k)

    try:
        # Build input pairs for a single batch call
        pairs: list[tuple[str, str]] = [(query, chunk.text) for chunk in chunks]

        logger.debug(
            "Reranking %d chunk(s) | top_k=%d | model=%s",
            len(pairs),
            effective_top_k,
            settings.rerank_model,
        )

        # predict() returns a numpy array or list of scores
        raw_scores_out = cross_encoder.predict(pairs)
        raw_scores: list[float] = (
            raw_scores_out.tolist()
            if hasattr(raw_scores_out, "tolist")
            else [float(s) for s in raw_scores_out]
        )

        # Pair scores with chunks and sort descending
        scored = sorted(
            zip(raw_scores, chunks, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )
        top = scored[:effective_top_k]

        reranked: list[RetrievedChunk] = []
        for new_rank, (score, chunk) in enumerate(top, start=1):
            src_val = (
                chunk.source
                or chunk.metadata.get("source")
                or chunk.metadata.get("source_path")
                or ""
            )
            meta = dict(chunk.metadata)
            meta["source"] = src_val
            meta["source_path"] = src_val
            reranked_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                heading_path=chunk.heading_path,
                source=src_val,
                score=float(score),
                rank=new_rank,
                retrieval_type="reranked",
                metadata=meta,
            )
            reranked.append(reranked_chunk)

        logger.debug("Reranker returned %d chunks", len(reranked))
        return reranked
    except Exception as exc:
        logger.warning(
            "Cross-encoder inference failed (%s); applying offline passthrough ranking.",
            exc,
        )
        return _build_fallback_chunks(chunks, effective_top_k)
