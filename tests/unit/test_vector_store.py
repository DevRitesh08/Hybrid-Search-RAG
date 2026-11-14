"""Unit tests for ChromaDB VectorStore."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from hybrid_rag.config import settings
from hybrid_rag.index.vector_store import VectorStore, _NoOpEmbeddingFunction
from hybrid_rag.models import Chunk


def test_noop_embedding_function():
    """Verify _NoOpEmbeddingFunction prevents default ONNX download."""
    ef = _NoOpEmbeddingFunction()
    assert ef(["test doc"]) == []
    assert ef.name() == "default"


def test_vector_store_collection_naming():
    """Collection names must follow {chroma_collection_prefix}_{strategy}."""
    for strat in ["fixed", "structural", "semantic"]:
        store = VectorStore(strategy=strat)
        expected = f"{settings.chroma_collection_prefix}_{strat}"
        assert store.get_collection_name() == expected


def test_vector_store_chunk_metadata_alignment():
    """Metadata extraction must populate both source and source_path."""
    chunk1 = Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="Sample text 1",
        heading_path="Heading > Section",
        strategy="structural",
        position=0,
        metadata={"source": "transformers/model.md"},
    )
    meta1 = VectorStore._chunk_metadata(chunk1)
    assert meta1["source"] == "transformers/model.md"
    assert meta1["source_path"] == "transformers/model.md"
    assert meta1["heading_path"] == "Heading > Section"

    chunk2 = Chunk(
        chunk_id="c2",
        doc_id="d2",
        text="Sample text 2",
        strategy="fixed",
        position=1,
        metadata={"source_path": "peft/lora.md"},
    )
    meta2 = VectorStore._chunk_metadata(chunk2)
    assert meta2["source"] == "peft/lora.md"
    assert meta2["source_path"] == "peft/lora.md"


def test_vector_store_crud_and_score_normalization(tmp_path: Path):
    """Test full CRUD operations and verify cosine distance to similarity mapping."""
    with patch.object(settings, "chroma_path", tmp_path / "chroma_test"):
        store = VectorStore(strategy="structural")
        store.reset()
        assert store.count() == 0

        chunks = [
            Chunk(
                chunk_id="chk_1",
                doc_id="doc_1",
                text="HuggingFace AutoModel handles pretrained checkpoints.",
                heading_path="AutoModel",
                strategy="structural",
                position=0,
                metadata={"source": "hf/auto.md"},
            ),
            Chunk(
                chunk_id="chk_2",
                doc_id="doc_2",
                text="PEFT LoRA configures low-rank matrix adaptation.",
                heading_path="LoRA",
                strategy="structural",
                position=1,
                metadata={"source_path": "peft/lora.md"},
            ),
        ]

        # Use normalized float32 vectors
        vec1 = np.ones(384, dtype=np.float32) / np.sqrt(384)
        vec2 = np.zeros(384, dtype=np.float32)
        vec2[0] = 1.0

        # One as numpy array, one as list to test polymorphism
        embeddings = {
            "chk_1": vec1,
            "chk_2": vec2.tolist(),
        }

        added = store.add_chunks(chunks, embeddings)
        assert added == 2
        assert store.count() == 2

        # Query with vec1
        results = store.query(vec1, top_k=2)
        assert len(results) == 2
        # Identical vector should have distance close to 0 -> score close to 1.0
        assert results[0].chunk_id == "chk_1"
        assert 0.99 <= results[0].score <= 1.0
        assert all(0.0 <= r.score <= 1.0 for r in results)
        assert results[0].retrieval_type == "dense"
        assert results[0].source == "hf/auto.md"

        # Delete by doc_id
        deleted = store.delete_by_doc("doc_1")
        assert deleted == 1
        assert store.count() == 1

        # Reset
        store.reset()
        assert store.count() == 0
