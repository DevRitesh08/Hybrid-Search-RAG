"""Retrieval quality metrics for the Hybrid RAG evaluation suite.

All metrics are implemented from scratch (no sklearn dependency).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from hybrid_rag.models import EvalResult

logger = logging.getLogger(__name__)


# ── Individual metrics ────────────────────────────────────────────────────────


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """Return 1.0 if any relevant chunk appears in the top-k retrieved IDs.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved chunk IDs (most relevant first).
    relevant_ids:
        Ground-truth relevant chunk IDs for the query.
    k:
        Cut-off rank.

    Returns
    -------
    float
        1.0 if hit, 0.0 otherwise.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return 1.0 if top_k & relevant_set else 0.0


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """Compute recall at k: |relevant ∩ top-k| / |relevant|.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved chunk IDs (most relevant first).
    relevant_ids:
        Ground-truth relevant chunk IDs.
    k:
        Cut-off rank.

    Returns
    -------
    float
        Recall value in [0, 1]. Returns 0.0 if ``relevant_ids`` is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / len(relevant_set)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Return 1/rank of the first relevant item, or 0.0 if none found.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved chunk IDs (most relevant first).
    relevant_ids:
        Ground-truth relevant chunk IDs.

    Returns
    -------
    float
        Reciprocal rank in (0, 1], or 0.0 if no relevant item is retrieved.
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> float:
    """Compute normalized Discounted Cumulative Gain (nDCG) at k.

    Uses binary relevance: gain is 1 for a relevant item, 0 otherwise.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved chunk IDs (most relevant first).
    relevant_ids:
        Ground-truth relevant chunk IDs.
    k:
        Cut-off rank.

    Returns
    -------
    float
        nDCG value in [0, 1]. Returns 0.0 if ``relevant_ids`` is empty.
    """
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)

    def _dcg(ranked: list[str], cut: int) -> float:
        total = 0.0
        seen: set[str] = set()
        for i, chunk_id in enumerate(ranked[:cut], start=1):
            if chunk_id in relevant_set and chunk_id not in seen:
                seen.add(chunk_id)
                total += 1.0 / math.log2(i + 1)
        return total

    actual_dcg = _dcg(retrieved_ids, k)

    # Ideal DCG: place all relevant items at the top
    n_ideal = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

    if ideal_dcg == 0.0:
        return 0.0
    score = actual_dcg / ideal_dcg
    return max(0.0, min(1.0, score))


def mean_metric(values: list[float]) -> float:
    """Return the arithmetic mean of a list of floats.

    Parameters
    ----------
    values:
        Numeric values to average.

    Returns
    -------
    float
        Mean value, or 0.0 for an empty list.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


# ── Aggregated metrics ────────────────────────────────────────────────────────


@dataclass
class AggregatedMetrics:
    """Aggregated retrieval metrics for a single evaluation mode.

    Attributes
    ----------
    hit_rate:
        Mean hit rate @ 5 across all questions.
    recall:
        Mean recall @ 5 across all questions.
    mrr:
        Mean Reciprocal Rank across all questions.
    ndcg:
        Mean nDCG @ 10 across all questions.
    n_questions:
        Number of questions contributing to these metrics.
    mode:
        Pipeline mode identifier (e.g. "hybrid_rerank").
    """

    hit_rate: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    n_questions: int = 0
    mode: str = ""

    # Per-question scores kept for downstream analysis (bootstrap, etc.)
    hit_scores: list[float] = field(default_factory=list, repr=False)
    recall_scores: list[float] = field(default_factory=list, repr=False)
    mrr_scores: list[float] = field(default_factory=list, repr=False)
    ndcg_scores: list[float] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a plain-dict summary (no per-question score lists)."""
        return {
            "mode": self.mode,
            "n_questions": self.n_questions,
            "hit_rate@5": round(self.hit_rate, 4),
            "recall@5": round(self.recall, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@10": round(self.ndcg, 4),
        }


def aggregate(results: list[EvalResult]) -> AggregatedMetrics:
    """Aggregate per-question EvalResult objects into a single AggregatedMetrics.

    Parameters
    ----------
    results:
        List of per-question evaluation results (all must share the same mode).

    Returns
    -------
    AggregatedMetrics
        Populated with means and per-question score lists.
    """
    if not results:
        logger.warning("aggregate() called with empty results list")
        return AggregatedMetrics()

    mode = results[0].mode

    hit_scores = [float(r.hit_at_5) for r in results]
    rr_scores = [r.reciprocal_rank for r in results]
    ndcg_scores = [r.ndcg_at_10 for r in results]
    recall_scores = [r.recall_at_5 for r in results]

    return AggregatedMetrics(
        hit_rate=mean_metric(hit_scores),
        recall=mean_metric(recall_scores),
        mrr=mean_metric(rr_scores),
        ndcg=mean_metric(ndcg_scores),
        n_questions=len(results),
        mode=mode,
        hit_scores=hit_scores,
        recall_scores=recall_scores,
        mrr_scores=rr_scores,
        ndcg_scores=ndcg_scores,
    )
