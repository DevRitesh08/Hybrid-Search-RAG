"""Reciprocal Rank Fusion (RRF) for combining dense and sparse result lists.

The RRF formula used here is:

    score(doc) = w_dense / (k + rank_dense(doc))
               + w_sparse / (k + rank_sparse(doc))

where ranks are 1-based and a document absent from one list is treated as
having rank = infinity (contributing a score of 0 from that side).

References
----------
Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
*Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.*
Proc. ACM SIGIR.
"""

from __future__ import annotations

import logging

from hybrid_rag.config import settings
from hybrid_rag.models import RetrievedChunk

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────


def fuse(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int | None = None,
    weight_dense: float | None = None,
    weight_sparse: float | None = None,
) -> list[RetrievedChunk]:
    """Merge two ranked lists via Reciprocal Rank Fusion.

    Args:
        dense_results:  Ranked output from the dense (vector) retriever.
        sparse_results: Ranked output from the sparse (BM25) retriever.
        k:              RRF smoothing constant (default: ``settings.rrf_k``).
                        Higher values reduce the influence of top-ranked documents.
        weight_dense:   Weight applied to the dense side (default: ``settings.rrf_weight_dense``).
        weight_sparse:  Weight applied to the sparse side (default: ``settings.rrf_weight_sparse``).

    Returns:
        A new list of :class:`~hybrid_rag.models.RetrievedChunk` objects sorted by
        descending RRF score.  ``retrieval_type`` is ``"hybrid"`` and ranks are
        re-assigned starting from 1.  When a chunk appears in both lists the text
        and metadata from the *dense* result are preserved (ChromaDB stores the
        original chunk text; BM25 uses the same text, but dense is considered
        richer/canonical).

    Notes:
        - Chunks absent from one list receive a score contribution of 0 from
          that side (rank = ∞ → 0 contribution).
        - If both lists are empty an empty list is returned immediately.
    """
    effective_k: int = k if k is not None else settings.rrf_k
    effective_w_dense: float = (
        weight_dense if weight_dense is not None else settings.rrf_weight_dense
    )
    effective_w_sparse: float = (
        weight_sparse if weight_sparse is not None else settings.rrf_weight_sparse
    )

    if not dense_results and not sparse_results:
        logger.debug("RRF fusion: both result lists are empty — returning []")
        return []

    # ── Build chunk-id → chunk lookup tables ─────────────────────────────────
    # Dense lookup: chunk_id → (rank, RetrievedChunk)
    dense_rank: dict[str, int] = {}
    dense_chunk: dict[str, RetrievedChunk] = {}
    for chunk in dense_results:
        dense_rank[chunk.chunk_id] = chunk.rank
        dense_chunk[chunk.chunk_id] = chunk

    # Sparse lookup: chunk_id → rank
    sparse_rank: dict[str, int] = {}
    sparse_chunk: dict[str, RetrievedChunk] = {}
    for chunk in sparse_results:
        sparse_rank[chunk.chunk_id] = chunk.rank
        sparse_chunk[chunk.chunk_id] = chunk

    # ── Collect all unique chunk IDs ──────────────────────────────────────────
    all_ids: set[str] = set(dense_rank.keys()) | set(sparse_rank.keys())

    logger.debug(
        "RRF fusion | dense=%d | sparse=%d | unique=%d | k=%d | w_d=%.2f | w_s=%.2f",
        len(dense_results),
        len(sparse_results),
        len(all_ids),
        effective_k,
        effective_w_dense,
        effective_w_sparse,
    )

    # ── Compute RRF scores ────────────────────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    for cid in all_ids:
        # Absent rank → contribution of 0 (rank = ∞)
        d_rank = dense_rank.get(cid)
        s_rank = sparse_rank.get(cid)

        score_d = effective_w_dense / (effective_k + d_rank) if d_rank is not None else 0.0
        score_s = effective_w_sparse / (effective_k + s_rank) if s_rank is not None else 0.0

        # Strictly non-negative
        rrf_scores[cid] = max(0.0, score_d + score_s)

    # ── Sort by 4-element compound tuple for deterministic stability across runs ─
    sorted_ids = sorted(
        all_ids,
        key=lambda cid: (
            -rrf_scores[cid],
            dense_rank.get(cid, 999999),
            sparse_rank.get(cid, 999999),
            cid,
        ),
    )

    # ── Reconstruct RetrievedChunk list ──────────────────────────────────────
    fused: list[RetrievedChunk] = []
    for new_rank, cid in enumerate(sorted_ids, start=1):
        # Prefer the dense chunk object (richer text from ChromaDB); fall back to sparse
        source = dense_chunk[cid] if cid in dense_chunk else sparse_chunk[cid]

        src_val = source.source or source.metadata.get("source") or source.metadata.get("source_path") or ""
        meta = dict(source.metadata)
        meta["source"] = src_val
        meta["source_path"] = src_val

        fused_chunk = RetrievedChunk(
            chunk_id=source.chunk_id,
            doc_id=source.doc_id,
            text=source.text,
            heading_path=source.heading_path,
            source=src_val,
            score=rrf_scores[cid],
            rank=new_rank,
            retrieval_type="hybrid",
            metadata=meta,
        )
        fused.append(fused_chunk)

    logger.debug("RRF fusion produced %d merged chunks", len(fused))
    return fused


# Alias for explicit API naming compatibility
reciprocal_rank_fusion = fuse

