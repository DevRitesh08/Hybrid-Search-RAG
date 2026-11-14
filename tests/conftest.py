"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_rag.generation.llm import LLMAdapter
from hybrid_rag.models import Chunk, Document, RetrievedChunk


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provide realistic sample documents for testing."""
    return [
        Document(
            doc_id="doc_auto",
            source_path="transformers/auto.md",
            title="AutoModel Documentation",
            text="# AutoModel\n\nThe `AutoModel` class automatically loads weights for any pre-trained model checkpoint.",
            metadata={"source": "transformers/auto.md"},
        ),
        Document(
            doc_id="doc_lora",
            source_path="peft/lora.md",
            title="LoRA Configuration",
            text="# LoRA\n\nConfigure `target_modules=['q_proj', 'v_proj']` and `lora_alpha=16` in `LoraConfig`.",
            metadata={"source": "peft/lora.md"},
        ),
        Document(
            doc_id="doc_pipe",
            source_path="transformers/pipeline.md",
            title="Pipeline Quickstart",
            text="# Pipeline\n\nThe `pipeline('sentiment-analysis')` utility runs end-to-end inference.",
            metadata={"source": "transformers/pipeline.md"},
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Provide realistic sample chunks."""
    texts = [
        "The AutoModel class loads any supported checkpoint using from_pretrained.",
        "LoraConfig requires target_modules to define which linear layers to adapt.",
        "Pipeline sentiment-analysis runs out-of-the-box text classification inference.",
        "AutoTokenizer resolves vocabulary files corresponding to model checkpoints.",
        "LoRA rank r defaults to 8 and lora_alpha controls update scaling.",
    ]
    return [
        Chunk(
            chunk_id=f"chk_{i:03d}",
            doc_id=f"doc_{i}",
            text=text,
            heading_path="Overview > Basics",
            strategy="structural",
            position=i,
            metadata={"source": f"docs/file_{i}.md"},
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def mock_retrieved_chunks(sample_chunks: list[Chunk]) -> list[RetrievedChunk]:
    """Provide realistic retrieved chunks with scores and ranks."""
    return [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            text=c.text,
            heading_path=c.heading_path,
            source=c.metadata.get("source", ""),
            score=round(1.0 / (i + 1), 3),
            rank=i + 1,
            retrieval_type="dense",
            metadata=c.metadata,
        )
        for i, c in enumerate(sample_chunks)
    ]


class MockLLM(LLMAdapter):
    """Deterministic mock LLM for testing without network calls."""

    def __init__(self, canned_response: str = "This is supported by [1] and [2].") -> None:
        self.canned_response = canned_response

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        if "SUPPORT" in prompt.upper():
            return "SUPPORTED: Found directly in context"
        if "COMPLETE" in prompt.upper():
            return "COMPLETE: All parts addressed"
        return self.canned_response


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def mock_embedder() -> callable:
    """Deterministic random unit vectors."""
    rng = np.random.default_rng(1234)

    def _embed(texts: list[str]) -> np.ndarray:
        vecs = rng.normal(size=(len(texts), 384))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    return _embed
