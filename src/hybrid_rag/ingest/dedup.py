"""Chunk deduplication for the Hybrid RAG ingest pipeline.

Removes near-duplicate chunks using cosine similarity of pre-computed
embeddings before they are indexed.  Near-duplicates increase index size,
hurt BM25 term frequencies, and confuse the reranker.
"""

from __future__ import annotations

import logging

import numpy as np

from hybrid_rag.config import settings
from hybrid_rag.models import Chunk

logger = logging.getLogger(__name__)

__all__ = ["_cosine_similarity", "cosine_similarity", "deduplicate"]


# ---------------------------------------------------------------------------
# Vector math helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Returns 0.0 if either vector has zero norm or non-finite values to avoid
    division-by-zero or NaN propagation.
    """
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# Backwards-compatibility alias for tests and internal callers
_cosine_similarity = cosine_similarity


def _max_similarity_to_set(
    candidate: np.ndarray,
    seen_embeddings: list[np.ndarray],
) -> tuple[float, int]:
    """Return the maximum cosine similarity between *candidate* and *seen_embeddings*.

    Parameters
    ----------
    candidate:
        1-D embedding of the candidate chunk.
    seen_embeddings:
        List of unit-norm embeddings already accepted into the kept set.

    Returns
    -------
    tuple[float, int]
        ``(max_similarity, index_of_closest)``; ``(-1.0, -1)`` if
        *seen_embeddings* is empty.
    """
    if not seen_embeddings:
        return -1.0, -1

    if not np.all(np.isfinite(candidate)):
        return 0.0, 0

    cand_norm = np.linalg.norm(candidate)
    if cand_norm == 0.0:
        return 0.0, 0

    cand_unit = candidate / cand_norm
    matrix = np.stack(seen_embeddings, axis=0)  # (N, D)
    sims = matrix @ cand_unit
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deduplicate(
    chunks: list[Chunk],
    embeddings: dict[str, np.ndarray] | np.ndarray | list[np.ndarray],
    threshold: float | None = None,
) -> tuple[list[Chunk], list[str]]:
    """Remove near-duplicate chunks using cosine similarity.

    Iterates *chunks* in order.  Each chunk is compared against the
    embeddings of all *kept* chunks so far.  If the maximum cosine
    similarity to any kept chunk exceeds *threshold*, the candidate is
    skipped (deduplicated); otherwise it is kept.

    Chunks whose ``chunk_id`` is not found in *embeddings* are always kept
    with a warning (we cannot compare without an embedding).

    Parameters
    ----------
    chunks:
        Ordered list of ``Chunk`` objects to deduplicate.
    embeddings:
        Pre-computed embedding vectors keyed by ``chunk_id``, or a 2-D array
        / list of 1-D arrays indexed by position.
    threshold:
        Cosine similarity above which a chunk is considered a duplicate.
        Defaults to ``settings.dedup_threshold`` (0.95).

    Returns
    -------
    tuple[list[Chunk], list[str]]
        ``(kept, skipped)`` where *skipped* contains the ``chunk_id`` of
        every deduplicated chunk.
    """
    if threshold is None:
        threshold = settings.dedup_threshold

    kept: list[Chunk] = []
    skipped: list[str] = []
    kept_embeddings: list[np.ndarray] = []

    for i, chunk in enumerate(chunks):
        cid = chunk.chunk_id

        if isinstance(embeddings, dict):
            if cid not in embeddings:
                logger.warning(
                    "chunk_id=%s has no embedding — keeping without similarity check.",
                    cid,
                )
                kept.append(chunk)
                continue
            emb = np.asarray(embeddings[cid], dtype=np.float32)
        elif isinstance(embeddings, (list, np.ndarray)):
            if i >= len(embeddings):
                logger.warning(
                    "Index %d out of bounds for embeddings array — keeping without check.",
                    i,
                )
                kept.append(chunk)
                continue
            emb = np.asarray(embeddings[i], dtype=np.float32)
        else:
            logger.warning(
                "Unsupported embeddings type %s — keeping chunk.",
                type(embeddings),
            )
            kept.append(chunk)
            continue

        # Ensure 1-D
        if emb.ndim != 1:
            emb = emb.flatten()

        # Guard against non-finite values (NaN / Inf) to avoid poisoning seen_embeddings
        if not np.all(np.isfinite(emb)):
            logger.warning(
                "chunk_id=%s has non-finite embedding — keeping chunk without updating kept_embeddings.",
                cid,
            )
            kept.append(chunk)
            continue

        max_sim, closest_idx = _max_similarity_to_set(emb, kept_embeddings)

        if max_sim > threshold:
            logger.debug(
                "Skipping chunk_id=%s (cosine_sim=%.4f > threshold=%.4f, closest_kept_idx=%d).",
                cid,
                max_sim,
                threshold,
                closest_idx,
            )
            skipped.append(cid)
        else:
            kept.append(chunk)
            norm = np.linalg.norm(emb)
            kept_embeddings.append(emb / norm if norm > 0.0 else emb)

    logger.info(
        "Deduplication complete: kept=%d, skipped=%d (threshold=%.4f).",
        len(kept),
        len(skipped),
        threshold,
    )
    return kept, skipped
