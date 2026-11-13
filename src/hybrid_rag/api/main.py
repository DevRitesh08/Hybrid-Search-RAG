"""FastAPI application for Hybrid Search RAG."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from hybrid_rag import __version__
from hybrid_rag.api.deps import get_bm25_index, get_vector_store
from hybrid_rag.api.schemas import (
    AskRequest,
    DocumentInfo,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from hybrid_rag.config import settings
from hybrid_rag.generation.answer_pipeline import run_generation_pipeline
from hybrid_rag.models import RAGResponse
from hybrid_rag.retrieval.pipeline import retrieve

logger = logging.getLogger("hybrid_rag.api")
logging.basicConfig(level=getattr(logging, settings.log_level))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan event: warm up vector store and BM25 index."""
    logger.info("Initializing Hybrid RAG services...")
    # Pre-warm vector store and bm25
    try:
        get_vector_store()
        get_bm25_index()
        logger.info("Stores loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not pre-warm stores at startup: {e}")
    yield
    logger.info("Shutting down Hybrid RAG services.")


app = FastAPI(
    title="Hybrid RAG — HuggingFace Docs",
    version=__version__,
    lifespan=lifespan,
    description="Dense + BM25 + RRF + Cross-Encoder Reranker with Verified Citations",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next) -> Response:
    """Attach unique X-Request-ID header to every request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Service liveness and readiness probe."""
    chroma_ok = False
    bm25_ok = False

    try:
        store = get_vector_store()
        chroma_ok = store.count() >= 0
    except Exception:
        chroma_ok = False

    try:
        bm25 = get_bm25_index()
        bm25_ok = bm25.count() >= 0
    except Exception:
        bm25_ok = False

    return HealthResponse(
        status="healthy" if (chroma_ok or bm25_ok) else "degraded",
        version=__version__,
        chroma_ok=chroma_ok,
        bm25_loaded=bm25_ok,
    )


@app.post("/v1/ask", response_model=RAGResponse)
async def ask_question(request: AskRequest) -> RAGResponse:
    """Ask a question and receive a cited, verified response."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    loop = asyncio.get_event_loop()
    mode = request.mode or settings.retrieval_mode
    top_k = request.top_k or settings.rerank_top_k

    try:
        # Step 1: Retrieval (blocking CPU/network -> run in executor)
        chunks, latencies = await loop.run_in_executor(
            None,
            lambda: retrieve(
                query=request.question,
                mode=mode,
                top_k=top_k,
            ),
        )

        # Step 2: Generation + Verification + Confidence
        response = await loop.run_in_executor(
            None,
            lambda: run_generation_pipeline(
                question=request.question,
                chunks=chunks,
                mode=mode,
                retrieve_latency_ms=int(latencies.get("retrieve_ms", 0.0)),
                rerank_latency_ms=int(latencies.get("rerank_ms", 0.0)),
            ),
        )
        return response

    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/documents", response_model=list[DocumentInfo])
async def list_documents(strategy: str = "structural") -> list[DocumentInfo]:
    """List indexed documents."""
    try:
        store = get_vector_store(strategy)
        # ChromaDB does not have a direct distinct doc_id list, but we can query count
        count = store.count()
        return [
            DocumentInfo(
                doc_id="collection_total",
                source=f"{store.get_collection_name()}",
                chunk_count=count,
                strategy=strategy,
            )
        ]
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return []


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest_documents(req: IngestRequest) -> IngestResponse:
    """Trigger ingestion and indexing."""
    from hybrid_rag.index.build import build_indexes

    strategy = req.strategy or settings.chunk_strategy
    loop = asyncio.get_event_loop()
    try:
        stats = await loop.run_in_executor(None, lambda: build_indexes(strategy=strategy))
        return IngestResponse(
            chunks_added=stats.get("total_chunks", 0),
            chunks_skipped=stats.get("dedup_dropped", 0),
            docs_processed=stats.get("total_docs", 0),
            strategy=strategy,
            duration_ms=stats.get("duration_ms", 0),
        )
    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
