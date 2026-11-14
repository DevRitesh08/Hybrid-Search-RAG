"""Unit tests for composite confidence scoring and refusal thresholding (Feature 15)."""

from __future__ import annotations

from hybrid_rag.config import settings
from hybrid_rag.generation.confidence import (
    check_completeness,
    compute_confidence,
    compute_coverage_score,
    compute_retrieval_score,
    is_refusal_required,
    is_refusal_text,
)
from hybrid_rag.generation.llm import LLMAdapter
from hybrid_rag.models import Citation, ConfidenceScore, RetrievedChunk


class MockCompletenessLLM(LLMAdapter):
    """Deterministic LLM judge for testing completeness evaluation."""

    def __init__(self, verdict: str = "COMPLETE") -> None:
        self.verdict = verdict

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        return f"{self.verdict}: Evaluated correctly"


def make_chunk(retrieval_type: str = "dense", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        text="Sample chunk content.",
        score=score,
        rank=1,
        retrieval_type=retrieval_type,
    )


def test_compute_retrieval_score_modes():
    """Verify normalization across reranked, hybrid, dense, and empty chunks."""
    assert compute_retrieval_score([]) == 0.0

    # Reranked: logit=0 -> sigmoid(0) = 0.5
    reranked_zero = make_chunk(retrieval_type="reranked", score=0.0)
    assert abs(compute_retrieval_score([reranked_zero]) - 0.5) < 1e-4

    # Reranked: large positive logit -> ~1.0
    reranked_pos = make_chunk(retrieval_type="reranked", score=10.0)
    assert compute_retrieval_score([reranked_pos]) > 0.99

    # Hybrid: score scaled relative to max possible RRF
    hybrid_chunk = make_chunk(retrieval_type="hybrid", score=0.015)
    score = compute_retrieval_score([hybrid_chunk])
    assert 0.0 <= score <= 1.0

    # Dense: score clamped [0.0, 1.0]
    dense_chunk = make_chunk(retrieval_type="dense", score=0.85)
    assert compute_retrieval_score([dense_chunk]) == 0.85


def test_compute_retrieval_score_reranker_fallback_rrf():
    """Verify that reranker fallback preserving RRF scores is properly normalized without sigmoid distortion."""
    max_rrf = (settings.rrf_weight_dense + settings.rrf_weight_sparse) / (settings.rrf_k + 1)

    # 1. Top possible RRF score should normalize to 1.0 (not ~0.504 via sigmoid)
    top_chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        text="Top match",
        score=max_rrf,
        rank=1,
        retrieval_type="reranked",
        metadata={"rerank_fallback": True, "upstream_retrieval_type": "hybrid"},
    )
    assert abs(compute_retrieval_score([top_chunk]) - 1.0) < 1e-4

    # 2. Zero RRF score should normalize to 0.0 (not 0.5 via sigmoid)
    zero_chunk = RetrievedChunk(
        chunk_id="c0",
        doc_id="d0",
        text="Zero match",
        score=0.0,
        rank=1,
        retrieval_type="reranked",
        metadata={"rerank_fallback": True, "upstream_retrieval_type": "hybrid"},
    )
    assert compute_retrieval_score([zero_chunk]) == 0.0

    # 3. Half of max RRF should normalize to 0.5
    half_chunk = RetrievedChunk(
        chunk_id="c_half",
        doc_id="d_half",
        text="Half match",
        score=max_rrf / 2.0,
        rank=1,
        retrieval_type="reranked",
        metadata={"rerank_fallback": True, "upstream_retrieval_type": "hybrid"},
    )
    assert abs(compute_retrieval_score([half_chunk]) - 0.5) < 1e-4


def test_compute_retrieval_score_reranker_fallback_dense():
    """Verify that reranker fallback preserving dense cosine scores is clamped [0, 1] without sigmoid."""
    chunk = RetrievedChunk(
        chunk_id="c_dense",
        doc_id="d_dense",
        text="Dense match",
        score=0.85,
        rank=1,
        retrieval_type="reranked",
        metadata={"rerank_fallback": True, "upstream_retrieval_type": "dense"},
    )
    assert compute_retrieval_score([chunk]) == 0.85


def test_compute_coverage_score():
    """Verify share of verified citations."""
    assert compute_coverage_score([]) == 0.0

    citations = [
        Citation(n=1, chunk_id="c1", verified=True),
        Citation(n=2, chunk_id="c2", verified=False),
        Citation(n=3, chunk_id="c3", verified=True),
        Citation(n=4, chunk_id="c4", verified=False),
    ]
    assert compute_coverage_score(citations) == 0.5

    all_verified = [Citation(n=1, chunk_id="c1", verified=True)]
    assert compute_coverage_score(all_verified) == 1.0


def test_is_refusal_text_detection():
    """Verify refusal text detection across variations."""
    assert is_refusal_text("I don't have enough information in the provided context to answer this question.") is True
    assert is_refusal_text("I do not have sufficient information to answer.") is True
    assert is_refusal_text("Context lacked sufficient information.") is True
    assert is_refusal_text("") is True
    assert is_refusal_text("AutoModel loads pretrained models directly.") is False


def test_check_completeness_judge_responses():
    """Verify completeness scoring for COMPLETE, PARTIAL, INCOMPLETE, and refusals."""
    # Refusal text immediately scores 0.0 without LLM call
    assert check_completeness("What is LoRA?", "I don't have enough information in the provided context.") == 0.0

    llm_comp = MockCompletenessLLM("COMPLETE")
    assert check_completeness("What is LoRA?", "LoRA is low rank adaptation.", llm=llm_comp) == 1.0

    llm_part = MockCompletenessLLM("PARTIAL")
    assert check_completeness("What is LoRA?", "LoRA is low rank adaptation.", llm=llm_part) == 0.5

    llm_incomp = MockCompletenessLLM("INCOMPLETE")
    assert check_completeness("What is LoRA?", "LoRA is low rank adaptation.", llm=llm_incomp) == 0.0


def test_compute_confidence_composite_and_properties():
    """Verify composite confidence calculation and compatibility property aliases."""
    chunks = [make_chunk(retrieval_type="dense", score=0.8)]
    citations = [Citation(n=1, chunk_id="c1", verified=True)]
    llm = MockCompletenessLLM("COMPLETE")

    conf = compute_confidence(
        chunks=chunks,
        citations=citations,
        answer="AutoModel is a helper class [1].",
        question="What is AutoModel?",
        llm=llm,
    )

    expected = (
        settings.confidence_weight_retrieval * 0.8
        + settings.confidence_weight_coverage * 1.0
        + settings.confidence_weight_completeness * 1.0
    )
    assert conf.composite == round(min(1.0, max(0.0, expected)), 4)
    assert 0.0 <= conf.composite <= 1.0

    # Verify compatibility property aliases
    assert conf.retrieval_score == conf.retrieval
    assert conf.citation_coverage == conf.coverage
    assert conf.completeness_score == conf.completeness
    assert conf.composite_score == conf.composite


def test_is_refusal_required():
    """Verify refusal triggers on low confidence or refusal text."""
    low_conf = ConfidenceScore(retrieval=0.2, coverage=0.0, completeness=0.0, composite=0.1)
    high_conf = ConfidenceScore(retrieval=0.9, coverage=1.0, completeness=1.0, composite=0.95)

    # 1. Low confidence -> refusal triggered
    refused, reason = is_refusal_required(low_conf, "Some generated answer.", threshold=0.35)
    assert refused is True
    assert "below the threshold" in reason

    # 2. Refusal phrase in answer -> refusal triggered even with high confidence
    refused2, reason2 = is_refusal_required(
        high_conf,
        "I don't have enough information in the provided context to answer this question.",
        threshold=0.35,
    )
    assert refused2 is True
    assert "lacked sufficient information" in reason2

    # 3. High confidence and substantive answer -> not refused
    refused3, reason3 = is_refusal_required(high_conf, "AutoModel is convenient.", threshold=0.35)
    assert refused3 is False
    assert reason3 == ""


def test_is_refusal_text_whitespace_and_tabs():
    """Verify that whitespace-only strings are treated as refusal text."""
    assert is_refusal_text("   ") is True
    assert is_refusal_text("\n\t\n  ") is True
    assert is_refusal_text("  \r\n  ") is True
    assert is_refusal_text("") is True
    assert is_refusal_text("Valid answer content.") is False


def test_is_refusal_required_whitespace_answer_high_retrieval():
    """Verify that whitespace answer with high retrieval score is refused rather than passed."""
    score = ConfidenceScore(retrieval=0.999, coverage=0.0, completeness=0.0, composite=0.3996)
    refused, reason = is_refusal_required(score, "    \n\t  ")
    assert refused is True
    assert "Context lacked sufficient information" in reason
