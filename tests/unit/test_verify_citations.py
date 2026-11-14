"""Unit tests for sentence-level citation verification, cache collision safety, and metrics (Feature 14)."""

from __future__ import annotations

from hybrid_rag.generation.citations import parse_citations
from hybrid_rag.generation.llm import LLMAdapter
from hybrid_rag.generation.verify_citations import (
    _split_into_claims,
    clear_verify_cache,
    compute_citation_metrics,
    negative_control_check,
    verify_citations,
    verify_single_citation,
)
from hybrid_rag.models import Citation, RetrievedChunk


class MockJudgeLLM(LLMAdapter):
    """Deterministic judge LLM for testing verification logic."""

    def __init__(self, supported: bool = True, explanation: str = "Supports claim directly") -> None:
        self.supported = supported
        self.explanation = explanation
        self.call_count = 0

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        self.call_count += 1
        verdict = "SUPPORTED" if self.supported else "NOT_SUPPORTED"
        return f"{verdict}: {self.explanation}"


def setup_function():
    clear_verify_cache()


def teardown_function():
    clear_verify_cache()


def make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        text=text,
        score=0.9,
        rank=1,
    )


def test_split_into_claims():
    """Verify sentence splitting and citation bracket stripping."""
    text = (
        "AutoModel loads weights from disk [1]. "
        "LoRA freezes base weights and injects trainable matrices [2, 3]! "
        "Short."
    )
    claims = _split_into_claims(text)
    assert len(claims) == 2
    assert claims[0] == "AutoModel loads weights from disk"
    assert claims[1] == "LoRA freezes base weights and injects trainable matrices"


def test_split_into_claims_empty():
    assert _split_into_claims("") == []


def test_verify_single_citation_supported_and_unsupported():
    """Verify single citation supported and unsupported verdicts."""
    supported_llm = MockJudgeLLM(supported=True, explanation="Claim appears in text")
    is_sup, reason = verify_single_citation("AutoModel loads weights", "AutoModel loads weights.", llm=supported_llm)
    assert is_sup is True
    assert "SUPPORTED" in reason

    clear_verify_cache()
    unsupported_llm = MockJudgeLLM(supported=False, explanation="No mention of topic")
    is_sup2, reason2 = verify_single_citation("Unrelated claim", "AutoModel loads weights.", llm=unsupported_llm)
    assert is_sup2 is False
    assert "NOT_SUPPORTED" in reason2


def test_verify_single_citation_cache_collision_resistance():
    """Verify that chunks with identical headers but different bodies produce distinct cache entries."""
    header = "# AutoModel Documentation\n\nOverview of the API."
    chunk_1 = f"{header}\n\nSection A details specific behavior 1."
    chunk_2 = f"{header}\n\nSection B details completely different behavior 2."

    mock_llm = MockJudgeLLM(supported=True)
    verify_single_citation("Claim A", chunk_1, llm=mock_llm)
    assert mock_llm.call_count == 1

    # Same claim against chunk_2 with identical header must NOT collide in cache
    verify_single_citation("Claim A", chunk_2, llm=mock_llm)
    assert mock_llm.call_count == 2


def test_verify_single_citation_caching_hit():
    """Verify repeated calls with identical claim and chunk use the cache."""
    mock_llm = MockJudgeLLM(supported=True)
    res1 = verify_single_citation("Consistent claim", "Chunk text content.", llm=mock_llm)
    assert mock_llm.call_count == 1

    res2 = verify_single_citation("Consistent claim", "Chunk text content.", llm=mock_llm)
    assert mock_llm.call_count == 1
    assert res1 == res2


def test_verify_citations_batch_with_sentence_claims():
    """Verify verify_citations associates citations with their specific sentence claims."""
    chunks = [
        make_chunk("c1", "AutoModel automatically loads model checkpoints."),
        make_chunk("c2", "LoRA adapts specific attention matrices."),
    ]
    citations = [
        Citation(n=1, chunk_id="c1", claim_text="AutoModel loads checkpoints", verified=False),
        Citation(n=2, chunk_id="c2", claim_text="LoRA adapts attention matrices", verified=False),
    ]
    answer = "AutoModel loads checkpoints [1]. LoRA adapts attention matrices [2]."

    mock_llm = MockJudgeLLM(supported=True)
    verified = verify_citations(answer, citations, chunks, llm=mock_llm)

    assert len(verified) == 2
    assert verified[0].verified is True
    assert verified[1].verified is True


def test_verify_citations_missing_chunk():
    """Verify citations pointing to nonexistent chunk IDs are flagged as unverified."""
    chunks = [make_chunk("c1", "Some content.")]
    citations = [Citation(n=9, chunk_id="missing_chunk_id", verified=False)]

    verified = verify_citations("Some answer [9].", citations, chunks)
    assert len(verified) == 1
    assert verified[0].verified is False
    assert "not found in retrieved results" in verified[0].verification_reason


def test_compute_citation_metrics():
    """Verify precision and recall computation for verified citations and claims."""
    citations = [
        Citation(n=1, chunk_id="c1", claim_text="AutoModel loads checkpoints", verified=True),
        Citation(n=2, chunk_id="c2", claim_text="LoRA adapts matrices", verified=False),
    ]
    claims = [
        "AutoModel loads checkpoints",
        "LoRA adapts matrices",
    ]

    metrics = compute_citation_metrics(citations, claims=claims)
    # 1 verified citation out of 2 total -> precision = 0.5
    assert metrics["precision"] == 0.5
    assert metrics["verified_citations"] == 1
    assert metrics["total_citations"] == 2

    # 1 claim backed by verified citation out of 2 total -> recall = 0.5
    assert metrics["recall"] == 0.5
    assert metrics["verified_claims"] == 1
    assert metrics["total_claims"] == 2


def test_compute_citation_metrics_empty():
    """Empty citations yields standard baseline metrics."""
    metrics = compute_citation_metrics([])
    assert metrics["precision"] == 1.0
    assert metrics["verified_citations"] == 0
    assert metrics["total_citations"] == 0


def test_negative_control_check():
    """Verify negative control check against unrelated chunks."""
    claims = ["Quantum computing requires dilution refrigerators."]
    chunks = [make_chunk("c1", "HuggingFace Transformers library provides pre-trained models.")]

    llm = MockJudgeLLM(supported=False, explanation="No mention of physics")
    support_rate = negative_control_check(claims, chunks, llm=llm)
    assert support_rate == 0.0


def test_compute_citation_metrics_multiple_claims_same_chunk_full_recall():
    """Verify that multiple claims citing the same chunk achieve 1.0 (100%) recall."""
    chunks = [
        make_chunk("c1", "AutoModel automatically loads model weights and instantiates configurations.")
    ]
    answer = "AutoModel loads weights from disk [1]. AutoModel also instantiates configuration classes [1]."

    citations = parse_citations(answer, chunks)
    for c in citations:
        c.verified = True

    metrics = compute_citation_metrics(citations, answer=answer)
    assert metrics["total_claims"] == 2
    assert metrics["verified_claims"] == 2
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_verify_citations_fallback_comma_and_range():
    """Verify that citations lacking claim_text originating from comma or range brackets locate sentence."""
    chunks = [
        make_chunk("c2", "Chunk 2 details."),
    ]
    answer = "First fact supported by multiple sources [1, 2]. Second independent fact [3]."
    citations = [Citation(n=2, chunk_id="c2", claim_text="", verified=False)]
    mock_llm = MockJudgeLLM(supported=True)
    verified = verify_citations(answer, citations, chunks, llm=mock_llm)
    assert len(verified) == 1
    assert "First fact supported by multiple sources" in verified[0].claim_text
    assert "Second independent fact" not in verified[0].claim_text
