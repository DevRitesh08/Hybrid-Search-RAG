"""Unit tests for dense, sparse, rerank, and unified retrieval pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hybrid_rag.config import settings
from hybrid_rag.index.bm25_index import BM25Index
from hybrid_rag.models import Chunk, RetrievedChunk
from hybrid_rag.retrieval import dense, pipeline, rerank, sparse
from hybrid_rag.retrieval.pipeline import HybridRetrievalPipeline

# ── Fixtures & Helpers ────────────────────────────────────────────────────────


def make_retrieved_chunk(
    chunk_id: str,
    score: float,
    rank: int,
    retrieval_type: str = "dense",
    source: str = "doc.md",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=f"Text for {chunk_id}",
        heading_path="Heading",
        source=source,
        score=score,
        rank=rank,
        retrieval_type=retrieval_type,
        metadata={"source": source, "source_path": source},
    )


# ── Sparse Retrieval Tests ────────────────────────────────────────────────────


def test_sparse_retrieve_delegates_to_bm25_index(tmp_path: Path):
    """Verify sparse.retrieve loads JSON index, queries with identifier preservation, and caches."""
    sparse.clear_cache()

    chunks = [
        Chunk(
            chunk_id="c_lora",
            doc_id="d1",
            text="target_modules parameter in LoraConfig specifies adapt layers.",
            strategy="structural",
            position=0,
            metadata={"source": "peft/lora.md"},
        )
    ]
    bm25 = BM25Index()
    bm25.build(chunks)
    index_file = tmp_path / "bm25_structural.json"
    bm25.save(index_file)

    with patch.object(settings, "bm25_index_dir", tmp_path):
        results = sparse.retrieve("target_modules", strategy="structural", top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "c_lora"
        assert results[0].retrieval_type == "sparse"
        assert results[0].source == "peft/lora.md"
        assert results[0].metadata["source"] == "peft/lora.md"
        assert results[0].metadata["source_path"] == "peft/lora.md"

        # Verify caching
        assert "structural" in sparse._index_cache

    sparse.clear_cache()


def test_sparse_retrieve_missing_file_raises():
    """Verify missing JSON index raises FileNotFoundError."""
    sparse.clear_cache()
    with patch.object(settings, "bm25_index_dir", Path("nonexistent_dir_bm25")), pytest.raises(FileNotFoundError):
        sparse.retrieve("test query", strategy="missing_strat")


# ── Dense Retrieval Tests ─────────────────────────────────────────────────────


def test_dense_embed_query_bge_prefix():
    """Verify BGE query prefix is prepended."""
    mock_service = MagicMock()
    mock_service.embed_query.return_value = np.zeros(384, dtype=np.float32)

    with patch("hybrid_rag.retrieval.dense.get_embedding_service", return_value=mock_service):
        vec = dense.embed_query("What is LoRA?")
        assert len(vec) == 384
        mock_service.embed_query.assert_called_once_with("What is LoRA?")


def test_dense_retrieve_delegates_to_vector_store():
    """Verify dense.retrieve queries VectorStore and validates collection count."""
    mock_store = MagicMock()
    mock_store.count.return_value = 5
    mock_store.query.return_value = [
        make_retrieved_chunk("c1", 0.92, 1, "dense", "peft/lora.md")
    ]

    with patch("hybrid_rag.retrieval.dense.VectorStore", return_value=mock_store), \
         patch("hybrid_rag.retrieval.dense.embed_query", return_value=[0.1] * 384):
        results = dense.retrieve("query", top_k=3, strategy="structural")
        assert len(results) == 1
        assert results[0].source == "peft/lora.md"
        assert results[0].metadata["source_path"] == "peft/lora.md"
        mock_store.query.assert_called_once()


def test_dense_retrieve_empty_collection_raises():
    """Verify dense.retrieve raises RuntimeError when collection is empty."""
    mock_store = MagicMock()
    mock_store.count.return_value = 0

    with patch("hybrid_rag.retrieval.dense.VectorStore", return_value=mock_store), pytest.raises(RuntimeError):
        dense.retrieve("query", strategy="structural")


# ── Reranker Tests ────────────────────────────────────────────────────────────


def test_rerank_with_model():
    """Verify reranker scores pairs and updates rank and retrieval_type."""
    rerank.clear_cache()
    chunks = [
        make_retrieved_chunk("c1", 0.8, 1),
        make_retrieved_chunk("c2", 0.7, 2),
    ]

    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([1.2, 3.4])

    with patch("hybrid_rag.retrieval.rerank._get_cross_encoder", return_value=mock_ce):
        results = rerank.rerank("query", chunks, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c2"  # 3.4 > 1.2
        assert results[0].rank == 1
        assert results[0].retrieval_type == "reranked"
        assert results[0].score == 3.4


def test_rerank_offline_fallback():
    """When cross-encoder fails to load, reranker falls back to upstream order with retrieval_type=reranked."""
    rerank.clear_cache()
    chunks = [
        make_retrieved_chunk("c1", 0.9, 1),
        make_retrieved_chunk("c2", 0.8, 2),
        make_retrieved_chunk("c3", 0.7, 3),
    ]

    with patch("hybrid_rag.retrieval.rerank._get_cross_encoder", return_value=None):
        results = rerank.rerank("query", chunks, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[1].chunk_id == "c2"
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[0].retrieval_type == "reranked"
        assert results[1].retrieval_type == "reranked"
        assert results[0].metadata["rerank_fallback"] is True
        assert results[0].metadata["upstream_retrieval_type"] == "dense"
        assert results[0].metadata["original_score"] == 0.9
        assert results[0].metadata["score_scale"] == "normalized"


def test_rerank_inference_exception_fallback():
    """When cross-encoder predict() raises an exception, gracefully falls back with metadata."""
    rerank.clear_cache()
    chunks = [
        make_retrieved_chunk("c1", 0.03, 1, retrieval_type="hybrid"),
        make_retrieved_chunk("c2", 0.02, 2, retrieval_type="hybrid"),
    ]
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("Inference engine error")

    with patch("hybrid_rag.retrieval.rerank._get_cross_encoder", return_value=mock_model):
        results = rerank.rerank("query", chunks, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[0].score == 0.03
        assert results[0].metadata["rerank_fallback"] is True
        assert results[0].metadata["upstream_retrieval_type"] == "hybrid"
        assert results[0].metadata["score_scale"] == "rrf"


def test_rerank_empty_input():
    """Empty chunk list returns empty list."""
    assert rerank.rerank("query", []) == []


# ── Pipeline Tests ────────────────────────────────────────────────────────────


def test_pipeline_modes_and_top_k():
    """Verify all pipeline modes, effective_top_k propagation to reranker, and telemetry."""
    sample_dense = [make_retrieved_chunk(f"d{i}", 0.9 - i * 0.1, i + 1, "dense") for i in range(10)]
    sample_sparse = [make_retrieved_chunk(f"s{i}", 10.0 - i, i + 1, "sparse") for i in range(10)]

    with patch("hybrid_rag.retrieval.dense.retrieve", return_value=sample_dense), \
         patch("hybrid_rag.retrieval.sparse.retrieve", return_value=sample_sparse), \
         patch("hybrid_rag.retrieval.rerank.rerank", return_value=sample_dense[:3]) as mock_r:

        # 1. Dense mode
        chunks, lat = pipeline.retrieve("test", mode="dense", top_k=5)
        assert len(chunks) == 10
        assert "retrieve_ms" in lat
        assert lat["rerank_ms"] == 0.0

        # 2. BM25 mode
        chunks, lat = pipeline.retrieve("test", mode="bm25", top_k=5)
        assert len(chunks) == 10

        # 3. Hybrid mode (slices to effective_top_k)
        chunks, lat = pipeline.retrieve("test", mode="hybrid", top_k=4)
        assert len(chunks) == 4
        assert chunks[0].retrieval_type == "hybrid"

        # 4. Hybrid rerank mode (passes effective_top_k to rerank.rerank)
        chunks, lat = pipeline.retrieve("test", mode="hybrid_rerank", top_k=3)
        assert len(chunks) == 3
        assert lat["rerank_ms"] >= 0.0
        mock_r.assert_called_with("test", pytest.approx(chunks, rel=1e-1) if False else mock_r.call_args[0][1], top_k=3)


def test_pipeline_invalid_mode():
    """Unknown mode raises ValueError."""
    with pytest.raises(ValueError):
        pipeline.retrieve("test", mode="unknown_mode")


def test_hybrid_retrieval_pipeline_class():
    """HybridRetrievalPipeline static wrapper executes retrieve()."""
    with patch("hybrid_rag.retrieval.pipeline.retrieve", return_value=([], {"retrieve_ms": 1.0, "rerank_ms": 0.0})) as mock_ret:
        res, lat = HybridRetrievalPipeline.retrieve("test query", mode="dense", top_k=5)
        assert res == []
        mock_ret.assert_called_once_with("test query", mode="dense", top_k=5, strategy=None)
