"""Shared data models for the entire pipeline.

These are the canonical types passed between all modules.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ── Ingestion ─────────────────────────────────────────────────────────────────


class Document(BaseModel):
    """A normalized document loaded from disk."""

    doc_id: str = Field(description="Unique stable ID derived from source path")
    source_path: str = Field(description="Relative path within the corpus")
    title: str = Field(default="", description="Document title (from heading or filename)")
    text: str = Field(description="Full normalized text")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (front matter, section hierarchy, etc.)",
    )

    @property
    def id(self) -> str:
        return self.doc_id

    @classmethod
    def make_id(cls, source_path: str) -> str:
        """Stable document ID from source path."""
        return hashlib.sha1(source_path.encode()).hexdigest()[:16]


# ── Chunking ─────────────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """A single chunk produced by any chunking strategy."""

    chunk_id: str = Field(default="", description="Stable hash of doc_id + position + heading_path + text + strategy")
    doc_id: str = Field(description="Parent document ID")
    text: str = Field(description="Chunk text content")
    heading_path: str = Field(default="", description="Slash-separated or > separated heading hierarchy")
    strategy: str = Field(description="Chunking strategy used: fixed | structural | semantic")
    char_count: int = Field(default=0, description="Character count of this chunk")
    position: int = Field(description="Zero-based position within the document")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.chunk_id

    @property
    def document_id(self) -> str:
        return self.doc_id

    @property
    def heading_hierarchy(self) -> list[str]:
        if not self.heading_path:
            return []
        return [h.strip() for h in self.heading_path.split(">") if h.strip()]

    @model_validator(mode="before")
    @classmethod
    def compute_derived(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Auto-compute char_count and chunk_id if not provided."""
        text = data.get("text", "")
        data.setdefault("char_count", len(text))

        if "chunk_id" not in data or not data["chunk_id"]:
            pos = data.get("position", 0)
            hpath = data.get("heading_path", "")
            strat = data.get("strategy", "")
            doc_id = data.get("doc_id", "")
            raw = f"{doc_id}::{pos}::{hpath}::{text}::{strat}"
            data["chunk_id"] = hashlib.sha1(raw.encode()).hexdigest()[:20]
        return data


# ── Retrieval ─────────────────────────────────────────────────────────────────


class RetrievedChunk(BaseModel):
    """A chunk returned by any retrieval stage."""

    chunk_id: str
    doc_id: str
    text: str
    heading_path: str = ""
    source: str = ""
    score: float = Field(description="Retrieval or reranker score")
    rank: int = Field(description="1-based rank within the result list")
    retrieval_type: Literal["dense", "sparse", "hybrid", "reranked"] = "dense"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Generation ───────────────────────────────────────────────────────────────


class Citation(BaseModel):
    """A single citation extracted from an LLM answer."""

    n: int = Field(description="Citation number as it appears in the answer, e.g. [1]")
    chunk_id: str = Field(description="The cited chunk ID")
    source: str = Field(default="", description="Source document path")
    heading: str = Field(default="", description="Heading path of the cited chunk")
    claim_text: str = Field(
        default="",
        description="The specific claim sentence this citation supports",
    )
    verified: bool = Field(
        default=False,
        description="Whether the verifier confirmed this chunk supports the claim",
    )
    verification_reason: str = Field(default="", description="Verifier explanation")

    @property
    def is_verified(self) -> bool:
        """Alias for verified to match PROJECT.md interface contract."""
        return self.verified

    @is_verified.setter
    def is_verified(self, value: bool) -> None:
        self.verified = value

    @property
    def citation_index(self) -> int:
        """Alias for n to match PROJECT.md interface contract."""
        return self.n

    @citation_index.setter
    def citation_index(self, value: int) -> None:
        self.n = value


class ConfidenceScore(BaseModel):
    """Composite confidence breakdown."""

    retrieval: float = Field(ge=0.0, le=1.0, description="Retrieval quality score")
    coverage: float = Field(ge=0.0, le=1.0, description="Share of claims with verified citations")
    completeness: float = Field(ge=0.0, le=1.0, description="Judge completeness score")
    composite: float = Field(ge=0.0, le=1.0, description="Weighted composite score")

    @property
    def retrieval_score(self) -> float:
        return self.retrieval

    @property
    def citation_coverage(self) -> float:
        return self.coverage

    @property
    def completeness_score(self) -> float:
        return self.completeness

    @property
    def composite_score(self) -> float:
        return self.composite


class LatencyBreakdown(BaseModel):
    """Per-stage latency in milliseconds."""

    retrieve_ms: int = 0
    rerank_ms: int = 0
    generate_ms: int = 0
    verify_ms: int = 0
    total_ms: int = 0


class RAGResponse(BaseModel):
    """The canonical response schema for every query."""

    question: str
    answer: str
    refused: bool = False
    refusal_reason: str = Field(default="", description="Why the system refused to answer")
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceScore
    retrieved: list[RetrievedChunk] = Field(
        default_factory=list,
        description="All chunks considered for context",
    )
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def query(self) -> str:
        return self.question

    @property
    def is_refusal(self) -> bool:
        return self.refused

    @property
    def retrieved_chunks(self) -> list[RetrievedChunk]:
        return self.retrieved


# ── Evaluation ────────────────────────────────────────────────────────────────


class GoldenQuestion(BaseModel):
    """One entry from golden.jsonl."""

    id: str
    type: Literal["direct", "exact_term", "multi_hop", "unanswerable", "ambiguous"]
    question: str
    reference_answer: str = Field(default="", description="Empty for unanswerable")
    source_chunks_or_sections: list[str] = Field(default_factory=list)
    notes: str = ""


class EvalResult(BaseModel):
    """Per-question evaluation result."""

    question_id: str
    question: str
    mode: str
    hit_at_5: bool
    recall_at_5: float = 0.0
    reciprocal_rank: float
    ndcg_at_10: float
    judge_verdict: str = ""
    faithfulness: float = 0.0
    citation_accuracy: float = 0.0
    refused: bool = False
    latency_ms: int = 0
    error: str = ""
