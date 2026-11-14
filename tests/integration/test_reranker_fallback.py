"""Integration test for reranker fallback metadata preservation and confidence score calibration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hybrid_rag.config import settings
from hybrid_rag.generation.confidence import (
    compute_confidence,
    compute_retrieval_score,
    is_refusal_required,
)
from hybrid_rag.generation.llm import LLMAdapter
from hybrid_rag.models import Citation, RetrievedChunk
from hybrid_rag.retrieval import pipeline, rerank


class MockLLM(LLMAdapter):
    """Deterministic LLM for integration testing."""

    def __init__(self, answer: str = "This is a verified test answer with factual detail.") -> None:
        self.answer = answer

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        if "address all parts" in prompt or "COMPLETE" in prompt or "Question:" in prompt:
            return "COMPLETE: The response fully answers the question."
        return self.answer


def make_test_chunk(
    chunk_id: str,
    score: float,
    rank: int,
    retrieval_type: str = "dense",
    source: str = "hf_docs/transformers.md",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=f"Detailed documentation text for {chunk_id} covering configuration and parameters.",
        heading_path="Transformers > Models > AutoModel",
        source=source,
        score=score,
        rank=rank,
        retrieval_type=retrieval_type,
        metadata={"source": source, "source_path": source},
    )


def test_pipeline_hybrid_rerank_offline_fallback_calibration():
    """Verify that in hybrid_rerank mode with offline cross-encoder, candidates preserve fallback metadata

    and compute_retrieval_score applies linear RRF scaling instead of sigmoid compression.
    """
    rerank.clear_cache()

    sample_dense = [make_test_chunk(f"d{i}", 0.95 - i * 0.05, i + 1, "dense") for i in range(10)]
    sample_sparse = [make_test_chunk(f"s{i}", 12.0 - i, i + 1, "sparse") for i in range(10)]

    # Make the top candidate rank 1 in both dense and sparse to achieve maximal RRF score
    sample_sparse[0] = make_test_chunk("d0", 12.0, 1, "sparse")

    with patch("hybrid_rag.retrieval.dense.retrieve", return_value=sample_dense), \
         patch("hybrid_rag.retrieval.sparse.retrieve", return_value=sample_sparse), \
         patch("hybrid_rag.retrieval.rerank._get_cross_encoder", return_value=None):

        chunks, lat = pipeline.retrieve("AutoModel configuration", mode="hybrid_rerank", top_k=5)

        # 1. Pipeline contract checks
        assert len(chunks) == 5
        assert chunks[0].chunk_id == "d0"
        assert chunks[0].rank == 1
        assert chunks[0].retrieval_type == "reranked"
        assert chunks[0].metadata["rerank_fallback"] is True
        assert chunks[0].metadata["fallback"] is True
        assert chunks[0].metadata["upstream_retrieval_type"] == "hybrid"
        assert chunks[0].metadata["score_scale"] == "rrf"

        # 2. Confidence retrieval scoring calibration
        retrieval_score = compute_retrieval_score(chunks)

        # In fallback mode, top chunk has maximal possible RRF score -> retrieval_score must be ~1.0
        # If sigmoid had been mistakenly applied, 1.0 / (1.0 + exp(-0.01639)) would yield ~0.5041.
        assert retrieval_score > 0.95
        assert retrieval_score <= 1.0
        # Sigmoid distortion check: assert it is significantly higher than distorted sigmoid floor
        assert retrieval_score > 0.60


def test_pipeline_hybrid_rerank_inference_error_fallback():
    """Verify that if cross-encoder inference throws an error, fallback activates and prevents false refusal."""
    rerank.clear_cache()

    sample_dense = [make_test_chunk(f"d{i}", 0.90 - i * 0.05, i + 1, "dense") for i in range(10)]
    sample_sparse = [make_test_chunk(f"d{i}", 10.0 - i, i + 1, "sparse") for i in range(10)]

    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.side_effect = RuntimeError("GPU out of memory or inference crashed")

    with patch("hybrid_rag.retrieval.dense.retrieve", return_value=sample_dense), \
         patch("hybrid_rag.retrieval.sparse.retrieve", return_value=sample_sparse), \
         patch("hybrid_rag.retrieval.rerank._get_cross_encoder", return_value=mock_cross_encoder):

        chunks, _ = pipeline.retrieve("test query", mode="hybrid_rerank", top_k=5)

        assert len(chunks) == 5
        assert chunks[0].metadata["rerank_fallback"] is True
        assert chunks[0].metadata["upstream_retrieval_type"] == "hybrid"

        retrieval_score = compute_retrieval_score(chunks)
        assert retrieval_score > 0.90


def test_fallback_confidence_composite_and_refusal_discrimination():
    """Verify end-to-end composite confidence discriminating good vs bad matches under reranker fallback."""
    max_rrf = (settings.rrf_weight_dense + settings.rrf_weight_sparse) / (settings.rrf_k + 1)
    mock_llm = MockLLM()

    # Case A: Good match on fallback
    good_chunk = RetrievedChunk(
        chunk_id="good_c",
        doc_id="good_d",
        text="Target modules can be set to ['q_proj', 'v_proj'].",
        score=max_rrf * 0.9,
        rank=1,
        retrieval_type="reranked",
        metadata={"rerank_fallback": True, "upstream_retrieval_type": "hybrid"},
    )
    good_citations = [Citation(n=1, chunk_id="good_c", verified=True)]

    conf_good = compute_confidence(
        question="How do I set target_modules?",
        answer="You can set target_modules to ['q_proj', 'v_proj'] [1].",
        chunks=[good_chunk],
        citations=good_citations,
        llm=mock_llm,
    )

    # Retrieval score: ~0.90; Coverage: 1.0; Completeness: 1.0 -> Composite confidence > 0.85
    assert conf_good.retrieval > 0.85
    assert conf_good.composite > 0.85
    refused, _ = is_refusal_required(conf_good, "You can set target_modules to ['q_proj', 'v_proj'] [1].")
    assert refused is False

    # Case B: Very poor match on fallback
    poor_chunk = RetrievedChunk(
        chunk_id="poor_c",
        doc_id="poor_d",
        text="Irrelevant snippet about something else.",
        score=max_rrf * 0.05,
        rank=1,
        retrieval_type="reranked",
        metadata={"rerank_fallback": True, "upstream_retrieval_type": "hybrid"},
    )
    # Poor retrieval with unverified citation
    poor_citations = [Citation(n=1, chunk_id="poor_c", verified=False)]

    conf_poor = compute_confidence(
        question="How do I set target_modules?",
        answer="I think it is something [1].",
        chunks=[poor_chunk],
        citations=poor_citations,
        llm=mock_llm,
    )

    # Retrieval score: ~0.05 (NOT ~0.50 floor from sigmoid!); Coverage: 0.0
    assert conf_poor.retrieval < 0.10
    # Composite score will be below refusal threshold (0.35)
    assert conf_poor.composite < settings.confidence_threshold
    refused_poor, reason = is_refusal_required(conf_poor, "I think it is something [1].")
    assert refused_poor is True
    assert "below the threshold" in reason
