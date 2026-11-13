"""Pydantic request/response schemas for the Hybrid RAG API.

All request and response models are defined here and re-exported for
use in route handlers and OpenAPI documentation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Re-export canonical response type so API clients import from one place
from hybrid_rag.models import RAGResponse  # noqa: F401


class AskRequest(BaseModel):
    """Request body for ``POST /v1/ask``."""

    question: str = Field(
        ...,
        description="The natural-language question to answer.",
    )
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] | None = Field(
        default=None,
        description=(
            "Retrieval mode override. When omitted the server uses the value "
            "from ``settings.retrieval_mode``."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of final chunks to return. Defaults to ``settings.rerank_top_k``.",
    )

    model_config = {"json_schema_extra": {"example": {"question": "What is the AutoModel class?"}}}


class IngestRequest(BaseModel):
    """Request body for ``POST /v1/ingest``."""

    strategy: str | None = Field(
        default=None,
        description=(
            "Chunking strategy override: ``fixed`` | ``structural`` | ``semantic``. "
            "When omitted the server uses ``settings.chunk_strategy``."
        ),
    )

    model_config = {"json_schema_extra": {"example": {"strategy": "structural"}}}


class IngestResponse(BaseModel):
    """Response body for ``POST /v1/ingest``."""

    chunks_added: int = Field(description="Number of new chunks written to the vector store.")
    chunks_skipped: int = Field(description="Number of duplicate chunks skipped by dedup.")
    docs_processed: int = Field(description="Number of source documents processed.")
    strategy: str = Field(description="Chunking strategy that was used.")
    duration_ms: int = Field(description="Total wall-clock time for the ingest run in ms.")


class DocumentInfo(BaseModel):
    """Metadata summary for a single ingested document returned by ``GET /v1/documents``."""

    doc_id: str = Field(description="Stable document identifier.")
    source: str = Field(description="Relative path of the source file within the corpus.")
    chunk_count: int = Field(description="Number of chunks stored for this document.")
    strategy: str = Field(description="Chunking strategy used when the document was ingested.")


class HealthResponse(BaseModel):
    """Response body for ``GET /healthz``."""

    status: str = Field(description="``ok`` or ``degraded``.")
    version: str = Field(description="Application version string.")
    chroma_ok: bool = Field(description="``True`` if ChromaDB is reachable.")
    bm25_loaded: bool = Field(description="``True`` if the BM25 index is loaded.")
