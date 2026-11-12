"""End-to-end generation pipeline: answer -> verify -> confidence -> refusal."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from hybrid_rag.generation.answer import generate_answer
from hybrid_rag.generation.confidence import compute_confidence, is_refusal_required
from hybrid_rag.generation.prompt import CANONICAL_REFUSAL_MESSAGE
from hybrid_rag.generation.verify_citations import verify_citations
from hybrid_rag.models import LatencyBreakdown, RAGResponse, RetrievedChunk

if TYPE_CHECKING:
    from hybrid_rag.generation.llm import LLMAdapter


class RAGPipeline:
    """End-to-end RAG generation pipeline class for stateful or dependency-injected runs."""

    def __init__(
        self,
        llm: LLMAdapter | None = None,
        judge_llm: LLMAdapter | None = None,
    ) -> None:
        self.llm = llm
        self.judge_llm = judge_llm

    def run(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
        retrieve_latency_ms: int = 0,
        rerank_latency_ms: int = 0,
    ) -> RAGResponse:
        """Run generation, verification, confidence, and refusal logic."""
        return run_generation_pipeline(
            question=question,
            chunks=chunks,
            mode=mode,
            retrieve_latency_ms=retrieve_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            llm=self.llm,
            judge_llm=self.judge_llm,
        )


def run_generation_pipeline(
    question: str,
    chunks: list[RetrievedChunk],
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
    retrieve_latency_ms: int = 0,
    rerank_latency_ms: int = 0,
    llm: LLMAdapter | None = None,
    judge_llm: LLMAdapter | None = None,
) -> RAGResponse:
    """Run full generation, citation verification, confidence scoring, and explicit refusal.

    * When refused (due to insufficient context or low confidence):
      - Emits canonical refusal message.
      - Suppresses/clears hallucinated citations (sets citations = []).
    * Tracks granular latency breakdown across stages.

    Args:
        question: User query string.
        chunks: Retrieved context chunks.
        mode: Retrieval strategy mode tag.
        retrieve_latency_ms: Upstream retrieval latency in milliseconds.
        rerank_latency_ms: Upstream reranking latency in milliseconds.
        llm: Optional LLM adapter override for generation.
        judge_llm: Optional judge LLM adapter override for verification/confidence.

    Returns:
        Structured :class:`RAGResponse` containing answer, citations, confidence,
        retrieved chunks, refusal flags, and latency telemetry.
    """
    t_start = time.perf_counter()

    # 1. Generate answer
    t_gen_start = time.perf_counter()
    raw_answer, citations = generate_answer(question, chunks, llm=llm)
    gen_ms = int((time.perf_counter() - t_gen_start) * 1000)

    # 2. Verify citations against cited chunk texts and claims
    t_ver_start = time.perf_counter()
    verified_citations = verify_citations(
        raw_answer,
        citations,
        chunks,
        llm=judge_llm or llm,
    )
    ver_ms = int((time.perf_counter() - t_ver_start) * 1000)

    # 3. Compute composite confidence score
    confidence = compute_confidence(
        chunks=chunks,
        citations=verified_citations,
        answer=raw_answer,
        question=question,
        llm=judge_llm or llm,
    )

    # 4. Check explicit refusal conditions
    refused, refusal_reason = is_refusal_required(confidence, raw_answer)

    # In refusal mode: emit canonical refusal text and suppress hallucinated citations
    if refused:
        final_answer = CANONICAL_REFUSAL_MESSAGE
        final_citations = []
    else:
        final_answer = raw_answer
        final_citations = verified_citations

    total_ms = (
        retrieve_latency_ms
        + rerank_latency_ms
        + int((time.perf_counter() - t_start) * 1000)
    )

    latency = LatencyBreakdown(
        retrieve_ms=retrieve_latency_ms,
        rerank_ms=rerank_latency_ms,
        generate_ms=gen_ms,
        verify_ms=ver_ms,
        total_ms=total_ms,
    )

    return RAGResponse(
        question=question,
        answer=final_answer,
        refused=refused,
        refusal_reason=refusal_reason,
        citations=final_citations,
        confidence=confidence,
        retrieved=chunks,
        mode=mode,
        latency=latency,
    )
