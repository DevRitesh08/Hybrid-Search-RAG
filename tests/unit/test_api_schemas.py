"""Unit tests for FastAPI Pydantic schemas."""

from __future__ import annotations

from hybrid_rag.api.schemas import AskRequest, HealthResponse
from hybrid_rag.models import (
    Citation,
    ConfidenceScore,
    LatencyBreakdown,
    RAGResponse,
    RetrievedChunk,
)


def test_ask_request_valid():
    req = AskRequest(question="How does LoRA work?")
    assert req.question == "How does LoRA work?"
    assert req.mode is None
    assert req.top_k is None


def test_ask_request_with_options():
    req = AskRequest(question="What is AutoModel?", mode="hybrid", top_k=10)
    assert req.mode == "hybrid"
    assert req.top_k == 10


def test_rag_response_serialization():
    resp = RAGResponse(
        question="Test query?",
        answer="Test answer [1].",
        refused=False,
        citations=[Citation(n=1, chunk_id="c1", source="test.md", heading="H1", verified=True)],
        confidence=ConfidenceScore(retrieval=0.8, coverage=1.0, completeness=1.0, composite=0.9),
        retrieved=[RetrievedChunk(chunk_id="c1", doc_id="d1", text="Text", score=0.9, rank=1)],
        mode="hybrid_rerank",
        latency=LatencyBreakdown(total_ms=120),
    )
    json_data = resp.model_dump_json()
    assert "Test query?" in json_data
    assert "c1" in json_data
    assert resp.confidence.composite == 0.9


def test_health_response_schema():
    h = HealthResponse(status="healthy", version="0.1.0", chroma_ok=True, bm25_loaded=True)
    assert h.status == "healthy"
    assert h.chroma_ok is True
