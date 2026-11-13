"""FastAPI dependencies: shared resources, middleware helpers, auth.

Resources (embedder, reranker, BM25 index, LLM) are created once during
the application lifespan and injected into route handlers via FastAPI's
Depends() mechanism.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hybrid_rag.index.embedder import EmbeddingService


# ── Application-state container ───────────────────────────────────────────────

class _AppState:
    """Mutable container attached to ``app.state`` during lifespan."""

    embedder: EmbeddingService | None = None
    reranker: Any | None = None   # sentence_transformers.CrossEncoder
    bm25_index: Any | None = None  # hybrid_rag.index.bm25.BM25Index (if exists)
    llm: Any | None = None         # hybrid_rag.generation.llm.LLMAdapter


_state = _AppState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load heavy resources once at startup; clean up on shutdown."""
    from hybrid_rag.config import settings
    from hybrid_rag.index.embedder import get_embedding_service

    logger.info("Starting up — loading embedder (%s)…", settings.embed_model)
    _state.embedder = get_embedding_service()

    # Reranker (cross-encoder) — optional; skip if sentence-transformers unavailable
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        logger.info("Loading reranker (%s)…", settings.rerank_model)
        _state.reranker = CrossEncoder(settings.rerank_model)
        logger.info("Reranker loaded.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranker could not be loaded: %s — rerank mode will degrade.", exc)

    # BM25 index — load from disk if available
    try:
        from hybrid_rag.index.bm25_index import BM25Index  # noqa: PLC0415

        bm25_path = settings.bm25_index_dir / f"bm25_{settings.chunk_strategy}.json"
        if bm25_path.exists():
            idx = BM25Index()
            idx.load(bm25_path)
            _state.bm25_index = idx
            logger.info("BM25 index loaded from %s.", bm25_path)
        else:
            logger.info("BM25 index %s not present at startup — will load lazily.", bm25_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 index not loaded: %s", exc)

    # LLM adapter
    try:
        from hybrid_rag.generation.llm import get_llm  # noqa: PLC0415

        _state.llm = get_llm()
        logger.info("LLM adapter initialised.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM adapter not loaded: %s", exc)

    app.state.hybrid_rag = _state  # noqa: WPS125 — deliberate
    yield

    logger.info("Shutting down…")
    if _state.bm25_index is not None and hasattr(_state.bm25_index, "close"):
        _state.bm25_index.close()


# ── FastAPI dependency functions ──────────────────────────────────────────────

def get_embedder(request: Request) -> EmbeddingService:
    """Return the shared EmbeddingService instance."""
    svc = request.app.state.hybrid_rag.embedder
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not available.",
        )
    return svc


def get_reranker(request: Request) -> Any:
    """Return the shared CrossEncoder reranker (may be None if not loaded)."""
    return request.app.state.hybrid_rag.reranker


def get_bm25_index(request: Request | None = None) -> Any:
    """Return the shared BM25Index (may be None if not loaded)."""
    if request is not None and hasattr(request, "app") and hasattr(request.app, "state") and hasattr(request.app.state, "hybrid_rag"):
        return request.app.state.hybrid_rag.bm25_index
    return _state.bm25_index


def get_vector_store(strategy: str = "structural") -> Any:
    """Return the VectorStore instance for the given chunking strategy."""
    from hybrid_rag.index.vector_store import VectorStore  # noqa: PLC0415

    return VectorStore(strategy=strategy)


def get_llm(request: Request) -> Any:
    """Return the shared LLMAdapter (raises 503 if not available)."""
    llm = request.app.state.hybrid_rag.llm
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM adapter not available.",
        )
    return llm


def get_retrieval_pipeline() -> Callable[..., Any]:
    """Return the retrieval pipeline callable.

    This is a factory dependency — the pipeline is built from the shared state
    at request time so the caller can pass a ``mode`` override.
    """
    def _build_pipeline(request: Request) -> Callable[..., Any]:
        try:
            from hybrid_rag.retrieval.pipeline import retrieve  # noqa: PLC0415
            return retrieve
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Retrieval pipeline not available: {exc}",
            ) from exc

    return _build_pipeline


# ── API-key guard ─────────────────────────────────────────────────────────────

_API_KEY = os.getenv("API_KEY", "")


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """Validate the ``X-API-Key`` header when ``API_KEY`` env var is set.

    If the env var is unset (empty string) the check is skipped, making
    development with no auth frictionless.
    """
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ── Request-ID middleware ─────────────────────────────────────────────────────

class RequestIDMiddleware:
    """ASGI middleware that attaches a UUID to every request.

    The ID is exposed as:
    - ``request.state.request_id`` inside route handlers.
    - ``X-Request-ID`` response header.
    """

    def __init__(self, app: Any) -> None:  # noqa: ANN401
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())

        # Stash on scope so Request.state picks it up
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id

        async def send_with_header(message: Any) -> None:  # noqa: ANN401
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-request-id"] = request_id.encode()
                message = {**message, "headers": list(headers.items())}
            await send(message)

        await self.app(scope, receive, send_with_header)
