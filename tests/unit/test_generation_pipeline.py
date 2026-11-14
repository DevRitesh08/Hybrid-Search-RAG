"""Unit tests for end-to-end generation pipeline and refusal citation suppression (Feature 16)."""

from __future__ import annotations

from hybrid_rag.config import settings
from hybrid_rag.generation.answer_pipeline import (
    run_generation_pipeline as answer_pipeline_run,
)
from hybrid_rag.generation.llm import LLMAdapter
from hybrid_rag.generation.pipeline import RAGPipeline, run_generation_pipeline
from hybrid_rag.generation.prompt import CANONICAL_REFUSAL_MESSAGE
from hybrid_rag.models import RAGResponse, RetrievedChunk


class MockPipelineLLM(LLMAdapter):
    """Deterministic mock LLM for end-to-end generation pipeline testing."""

    def __init__(self, response_text: str = "AutoModel is useful [1].") -> None:
        self.response_text = response_text

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        if "SUPPORT" in prompt.upper():
            return "SUPPORTED: Found in text"
        if "COMPLETE" in prompt.upper():
            return "COMPLETE: Fully addressed"
        return self.response_text


def make_chunks(n: int = 2) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c_{i}",
            doc_id=f"doc_{i}",
            text=f"AutoModel text content for chunk {i}.",
            source=f"docs/source_{i}.md",
            heading_path=f"Section {i}",
            score=0.9,
            rank=i,
            retrieval_type="reranked",
        )
        for i in range(1, n + 1)
    ]


def test_generation_pipeline_successful_run():
    """Verify standard answering pipeline with verified citations and high confidence."""
    chunks = make_chunks(2)
    mock_llm = MockPipelineLLM(response_text="AutoModel loads weights automatically [1].")

    resp = run_generation_pipeline(
        question="How does AutoModel work?",
        chunks=chunks,
        mode="hybrid_rerank",
        retrieve_latency_ms=15,
        rerank_latency_ms=25,
        llm=mock_llm,
    )

    assert isinstance(resp, RAGResponse)
    assert resp.refused is False
    assert resp.refusal_reason == ""
    assert "AutoModel loads weights" in resp.answer
    assert len(resp.citations) == 1
    assert resp.citations[0].n == 1
    assert resp.citations[0].verified is True
    assert resp.confidence.composite > settings.confidence_threshold
    assert resp.mode == "hybrid_rerank"

    # Latency tracking
    assert resp.latency.retrieve_ms == 15
    assert resp.latency.rerank_ms == 25
    assert resp.latency.total_ms >= 40


def test_generation_pipeline_refusal_on_empty_chunks():
    """Verify refusal triggers and returns canonical refusal message on empty context."""
    resp = run_generation_pipeline(
        question="What is NovaStar Bank interest?",
        chunks=[],
        mode="hybrid",
    )

    assert resp.refused is True
    assert resp.answer == CANONICAL_REFUSAL_MESSAGE
    assert resp.citations == []
    assert resp.refusal_reason != ""


def test_generation_pipeline_refusal_clears_hallucinated_citations():
    """Verify that when confidence is below threshold, hallucinated citations are suppressed."""
    chunks = [
        RetrievedChunk(
            chunk_id="c_unrelated",
            doc_id="d1",
            text="Unrelated text about cookies.",
            score=-9.0,  # very low retrieval score
            rank=1,
            retrieval_type="reranked",
        )
    ]

    # LLM hallucinates a citation [1] despite lacking knowledge
    class LowConfidenceLLM(LLMAdapter):
        def generate(self, prompt: str, system: str = "", **kwargs) -> str:
            if "SUPPORT" in prompt.upper():
                return "NOT_SUPPORTED: Chunk does not mention question"
            if "COMPLETE" in prompt.upper():
                return "INCOMPLETE: Missing information"
            return "Here is a hallucinated claim [1]."

    resp = run_generation_pipeline(
        question="What is the commercial loan interest rate?",
        chunks=chunks,
        llm=LowConfidenceLLM(),
    )

    assert resp.refused is True
    assert resp.answer == CANONICAL_REFUSAL_MESSAGE
    # Hallucinated citations must be completely cleared upon refusal!
    assert resp.citations == []
    assert "below the threshold" in resp.refusal_reason or "lacked sufficient" in resp.refusal_reason


def test_generation_pipeline_refusal_on_explicit_llm_statement():
    """Verify refusal triggers when LLM explicitly outputs lack of information."""
    chunks = make_chunks(1)
    mock_llm = MockPipelineLLM(
        response_text="I don't have enough information in the provided context to answer this question."
    )

    resp = run_generation_pipeline(
        question="What is an unknown topic?",
        chunks=chunks,
        llm=mock_llm,
    )

    assert resp.refused is True
    assert resp.answer == CANONICAL_REFUSAL_MESSAGE
    assert resp.citations == []
    assert "lacked sufficient information" in resp.refusal_reason


def test_rag_pipeline_class_injection():
    """Verify RAGPipeline class encapsulation and dependency injection."""
    mock_llm = MockPipelineLLM(response_text="Configured properly [1].")
    pipeline = RAGPipeline(llm=mock_llm, judge_llm=mock_llm)

    chunks = make_chunks(1)
    resp = pipeline.run("How to configure?", chunks=chunks)

    assert isinstance(resp, RAGResponse)
    assert resp.refused is False
    assert len(resp.citations) == 1


def test_rag_response_compatibility_aliases():
    """Verify RAGResponse query, is_refusal, and retrieved_chunks property aliases."""
    chunks = make_chunks(1)
    mock_llm = MockPipelineLLM(response_text="Valid answer [1].")
    resp = run_generation_pipeline("What is this?", chunks, llm=mock_llm)

    assert resp.query == resp.question
    assert resp.is_refusal == resp.refused
    assert resp.retrieved_chunks == resp.retrieved


def test_answer_pipeline_reexport_compatibility():
    """Verify backward-compatible re-export from answer_pipeline.py."""
    chunks = make_chunks(1)
    mock_llm = MockPipelineLLM(response_text="Backward compatibility answer [1].")

    resp = answer_pipeline_run("Query", chunks, llm=mock_llm)
    assert isinstance(resp, RAGResponse)
    assert resp.refused is False


def test_generation_pipeline_refusal_on_whitespace_only_answer():
    """Verify that pipeline emits canonical refusal when LLM produces whitespace-only text."""
    chunks = make_chunks(1)
    mock_llm = MockPipelineLLM(response_text="   \n\t  ")

    resp = run_generation_pipeline(
        question="What is the learning rate?",
        chunks=chunks,
        llm=mock_llm,
    )

    assert resp.refused is True
    assert resp.answer == CANONICAL_REFUSAL_MESSAGE
    assert resp.citations == []
