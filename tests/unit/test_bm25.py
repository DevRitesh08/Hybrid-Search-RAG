"""Unit tests for BM25 sparse index."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hybrid_rag.index.bm25_index import BM25Index, tokenize
from hybrid_rag.models import Chunk


def test_bm25_tokenize_identifier_preservation():
    """Verify regex tokenizer preserves programming identifiers while stripping punctuation."""
    text = (
        "Configure target_modules=['q_proj', 'v_proj'] and lora_alpha=32. "
        "Use AutoModel.from_pretrained with torch_dtype=torch.bfloat16 and TaskType.CAUSAL_LM."
    )
    tokens = tokenize(text)
    assert "target_modules" in tokens
    assert "q_proj" in tokens
    assert "v_proj" in tokens
    assert "lora_alpha" in tokens
    assert "automodel" in tokens
    assert "from_pretrained" in tokens
    assert "torch_dtype" in tokens
    assert "bfloat16" in tokens
    assert "causal_lm" in tokens


def test_bm25_build_and_query():
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="The target_modules parameter in LoraConfig specifies which layers to adapt.",
            strategy="fixed",
            position=0,
            metadata={"source": "peft/lora.md"},
        ),
        Chunk(
            chunk_id="c2",
            doc_id="d2",
            text="AutoModel.from_pretrained loads the base weights into GPU memory.",
            strategy="fixed",
            position=1,
            metadata={"source_path": "transformers/auto.md"},
        ),
        Chunk(
            chunk_id="c3",
            doc_id="d3",
            text="General machine learning approaches often use gradient descent.",
            strategy="fixed",
            position=2,
            metadata={"source": "ml/general.md"},
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    assert index.count() == 3
    assert set(index.chunk_ids()) == {"c1", "c2", "c3"}

    # Exact term match: "target_modules" should strongly favor c1
    results = index.query("target_modules", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0.0
    assert results[0].source == "peft/lora.md"
    assert results[0].metadata["source"] == "peft/lora.md"
    assert results[0].metadata["source_path"] == "peft/lora.md"


def test_bm25_save_load_roundtrip():
    chunks = [
        Chunk(
            chunk_id="chk_alpha",
            doc_id="d_alpha",
            text="Setting lora_alpha to 32 scales the LoRA weights appropriately.",
            strategy="structural",
            position=0,
            metadata={"source": "peft/lora.md"},
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        saved_path = index.save(tmp_path)
        assert saved_path == tmp_path
        assert tmp_path.exists()

        loaded_index = BM25Index()
        loaded_index.load(tmp_path)

        assert loaded_index.count() == 1
        res = loaded_index.query("lora_alpha", top_k=1)
        assert len(res) == 1
        assert res[0].chunk_id == "chk_alpha"
        assert res[0].source == "peft/lora.md"
        assert res[0].metadata["source_path"] == "peft/lora.md"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_bm25_load_nonexistent_raises_error():
    """Attempting to load a nonexistent file should raise FileNotFoundError."""
    index = BM25Index()
    with pytest.raises(FileNotFoundError):
        index.load(Path("data/does_not_exist_bm25.json"))
