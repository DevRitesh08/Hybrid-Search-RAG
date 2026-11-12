"""Confidence scoring — composite score based on retrieval, coverage, and completeness."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from hybrid_rag.config import settings
from hybrid_rag.generation.llm import get_llm
from hybrid_rag.generation.prompt import COMPLETENESS_PROMPT
from hybrid_rag.models import Citation, ConfidenceScore, RetrievedChunk

if TYPE_CHECKING:
    from hybrid_rag.generation.llm import LLMAdapter

_REFUSAL_TRIGGERS = (
    "don't have enough information",
    "do not have enough information",
    "do not have sufficient information",
    "not enough information in the provided context",
    "insufficient information",
    "context lacked sufficient information",
)


def is_refusal_text(answer: str) -> bool:
    """Return True if answer text explicitly acknowledges lack of context/information."""
    if not answer or not answer.strip():
        return True
    lower = answer.lower()
    return any(trigger in lower for trigger in _REFUSAL_TRIGGERS)


def compute_retrieval_score(chunks: list[RetrievedChunk]) -> float:
    """Compute normalized score from top retrieved chunks.

    Handles:
    - Normal cross-encoder reranked chunks: applies sigmoid to raw logits.
    - Reranker fallback chunks: detects fallback metadata and routes to upstream
      normalization (e.g. RRF scaling or [0, 1] clamping), preventing distorted
      sigmoid calibration (mapping 0.0 -> 0.5 or 0.016 -> 0.504).
    - Hybrid (RRF) chunks: scales relative to maximum theoretical RRF score.
    - Dense / BM25 chunks: clamps to [0.0, 1.0].
    """
    if not chunks:
        return 0.0

    top = chunks[0]

    # Handle reranked chunks
    if top.retrieval_type == "reranked":
        is_fallback = bool(
            top.metadata.get("rerank_fallback")
            or top.metadata.get("fallback")
        )
        upstream_type = (
            top.metadata.get("upstream_retrieval_type")
            or top.metadata.get("original_retrieval_type")
        )
        score_scale = top.metadata.get("score_scale")

        if is_fallback or score_scale in ("rrf", "normalized"):
            if upstream_type == "hybrid" or score_scale == "rrf":
                max_possible_rrf = (
                    settings.rrf_weight_dense + settings.rrf_weight_sparse
                ) / (settings.rrf_k + 1)
                return (
                    min(1.0, max(0.0, top.score / max_possible_rrf))
                    if max_possible_rrf > 0
                    else 0.5
                )
            return min(1.0, max(0.0, top.score))

        # Standard Cross-Encoder inference: raw logits (typically -10 to +10)
        try:
            return 1.0 / (1.0 + math.exp(-top.score))
        except OverflowError:
            return 1.0 if top.score > 0 else 0.0

    # RRF scores are typically in [0.005, 0.035] range for k=60
    if top.retrieval_type == "hybrid":
        max_possible_rrf = (settings.rrf_weight_dense + settings.rrf_weight_sparse) / (
            settings.rrf_k + 1
        )
        return min(1.0, top.score / max_possible_rrf) if max_possible_rrf > 0 else 0.5

    # Dense or BM25: fallback clamped between 0 and 1
    return min(1.0, max(0.0, top.score))


def compute_coverage_score(citations: list[Citation]) -> float:
    """Compute share of citations that were verified."""
    if not citations:
        return 0.0
    verified = sum(1 for c in citations if c.verified)
    return verified / len(citations)


def check_completeness(
    question: str,
    answer: str,
    llm: LLMAdapter | None = None,
) -> float:
    """Assess whether the answer addresses all parts of the question."""
    if not answer or not answer.strip() or is_refusal_text(answer):
        return 0.0

    if llm is None:
        llm = get_llm(model_name=settings.judge_model)

    prompt = COMPLETENESS_PROMPT.format(question=question, answer=answer)
    try:
        response = llm.generate(prompt=prompt).strip().upper()
        if "COMPLETE" in response and "INCOMPLETE" not in response and "PARTIAL" not in response:
            return 1.0
        elif "PARTIAL" in response:
            return 0.5
        else:
            return 0.0
    except Exception:
        return 0.5  # neutral fallback on failure


def compute_confidence(
    chunks: list[RetrievedChunk],
    citations: list[Citation],
    answer: str,
    question: str,
    llm: LLMAdapter | None = None,
) -> ConfidenceScore:
    """Compute composite confidence score from retrieval, coverage, and completeness."""
    retrieval = compute_retrieval_score(chunks)
    coverage = compute_coverage_score(citations)
    completeness = check_completeness(question, answer, llm=llm)

    composite = (
        settings.confidence_weight_retrieval * retrieval
        + settings.confidence_weight_coverage * coverage
        + settings.confidence_weight_completeness * completeness
    )
    composite = min(1.0, max(0.0, composite))

    return ConfidenceScore(
        retrieval=round(retrieval, 4),
        coverage=round(coverage, 4),
        completeness=round(completeness, 4),
        composite=round(composite, 4),
    )


def is_refusal_required(
    confidence: ConfidenceScore,
    answer: str,
    threshold: float | None = None,
) -> tuple[bool, str]:
    """Check whether a query response must trigger an explicit refusal.

    Returns:
        tuple of (refused: bool, refusal_reason: str).
    """
    cutoff = threshold if threshold is not None else settings.confidence_threshold

    if is_refusal_text(answer):
        return (
            True,
            "Context lacked sufficient information to answer the question.",
        )

    if confidence.composite < cutoff:
        return (
            True,
            f"Confidence score ({confidence.composite:.2f}) was below "
            f"the threshold ({cutoff:.2f}).",
        )

    return (False, "")
