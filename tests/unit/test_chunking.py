"""Unit tests for chunking strategies."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_rag.chunking.base import ChunkerConfig, get_chunker
from hybrid_rag.chunking.fixed import FixedChunker
from hybrid_rag.chunking.fixed import chunk_document as chunk_document_fixed
from hybrid_rag.chunking.semantic import SemanticChunker
from hybrid_rag.chunking.semantic import chunk_document as chunk_document_semantic
from hybrid_rag.chunking.structural import StructuralChunker
from hybrid_rag.chunking.structural import chunk_document as chunk_document_structural
from hybrid_rag.models import Document


def test_get_chunker_factory():
    """Factory should instantiate the correct chunker class."""
    assert isinstance(get_chunker("fixed"), FixedChunker)
    assert isinstance(get_chunker("structural"), StructuralChunker)
    assert isinstance(get_chunker("semantic"), SemanticChunker)
    assert isinstance(get_chunker("FIXED"), FixedChunker)
    assert isinstance(get_chunker("Structural"), StructuralChunker)

    with pytest.raises(ValueError):
        get_chunker("invalid_strategy")


def test_fixed_chunker_determinism(sample_documents: list[Document]):
    """Fixed chunker should return the exact same chunks across runs."""
    doc = sample_documents[0]
    config = ChunkerConfig(chunk_size=50, chunk_overlap=10, min_size=10)
    chunker = FixedChunker(config)

    run1 = chunker.chunk(doc)
    run2 = chunker.chunk(doc)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=True):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.position == c2.position


def test_structural_chunker_heading_extraction():
    """Structural chunker extracts heading paths and prepends them."""
    markdown = """# Top Title

## Section One
Content under section one.

### Subsection A
Detailed paragraph in subsection A.
"""
    doc = Document(
        doc_id="doc_struct",
        source_path="struct.md",
        title="Top Title",
        text=markdown,
    )

    chunker = StructuralChunker(ChunkerConfig(chunk_size=200, min_size=5))
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    # Ensure no empty chunks
    assert all(len(c.text.strip()) > 0 for c in chunks)
    # Check heading path was recorded
    heading_paths = [c.heading_path for c in chunks]
    assert any("Section One" in hp for hp in heading_paths)


def test_chunk_ids_are_unique_and_stable():
    """Chunk IDs should be unique within a doc and deterministic."""
    doc = Document(
        doc_id="doc_stability",
        source_path="stability.md",
        text="First sentence. Second sentence. Third sentence. Fourth sentence.",
    )
    chunker = FixedChunker(ChunkerConfig(chunk_size=30, chunk_overlap=5, min_size=5))

    chunks1 = chunker.chunk(doc)
    chunks2 = chunker.chunk(doc)

    ids1 = [c.chunk_id for c in chunks1]
    ids2 = [c.chunk_id for c in chunks2]

    assert len(ids1) == len(set(ids1)), "Chunk IDs must be unique"
    assert ids1 == ids2, "Chunk IDs must be deterministic"


def test_chunk_document_convenience_functions():
    """Module-level chunk_document should work across all strategy modules."""
    doc = Document(
        doc_id="doc_conv",
        source_path="conv.md",
        text="# Title\n\nThis is a test document with sufficient length for chunking.",
    )
    cfg = ChunkerConfig(chunk_size=50, min_size=5)

    f_chunks = chunk_document_fixed(doc, cfg)
    assert len(f_chunks) > 0
    assert f_chunks[0].strategy == "fixed"

    s_chunks = chunk_document_structural(doc, cfg)
    assert len(s_chunks) > 0
    assert s_chunks[0].strategy == "structural"

    sem_chunks = chunk_document_semantic(doc, cfg)
    assert len(sem_chunks) > 0
    assert sem_chunks[0].strategy == "semantic"


def test_structural_chunker_code_block_comment_immunity():
    """Comments starting with # inside code blocks must not be treated as headings."""
    markdown = """# API Guide

Here is the setup code:

```python
# Configure the pipeline
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
# Run classification
result = classifier("Hello world")
```

## Advanced Usage

Further instructions here.
"""
    doc = Document(
        doc_id="doc_code_comments",
        source_path="code.md",
        title="API Guide",
        text=markdown,
    )
    chunker = StructuralChunker(ChunkerConfig(chunk_size=500, min_size=5))
    chunks = chunker.chunk(doc)

    # Headings extracted should only be API Guide and API Guide > Advanced Usage
    for c in chunks:
        assert "# Configure the pipeline" not in c.heading_path
        assert "# Run classification" not in c.heading_path
        assert "Configure the pipeline" not in c.heading_path
        assert "Run classification" not in c.heading_path


def test_structural_chunker_oversized_sub_splitting():
    """Oversized sections (> 1.5 * chunk_size) must keep heading prefix on EVERY sub-split."""
    long_body = "The quick brown fox jumps over the lazy dog. " * 30  # ~1350 characters
    markdown = f"# Architecture\n\n## Component Breakdown\n\n{long_body}"
    doc = Document(
        doc_id="doc_oversized",
        source_path="arch.md",
        title="Architecture",
        text=markdown,
    )
    # chunk_size=200 -> threshold=300, so 1350 chars will produce multiple sub-chunks
    chunker = StructuralChunker(ChunkerConfig(chunk_size=200, chunk_overlap=20, min_size=10))
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    expected_prefix = "[Architecture > Component Breakdown]\n\n"
    for c in chunks:
        assert c.heading_path == "Architecture > Component Breakdown"
        assert c.text.startswith(expected_prefix), f"Chunk missing breadcrumb prefix: {c.text[:50]}"


def test_semantic_chunker_fallback_without_embedder():
    """SemanticChunker without embedder should pack sentences up to max_chars."""
    text = (
        "First sentence of the text. Second sentence has more information. "
        "Third sentence continues the theme. Fourth sentence concludes it."
    )
    doc = Document(
        doc_id="doc_sem_fallback",
        source_path="sem.md",
        text=text,
        metadata={"repo": "transformers", "author": "huggingface"},
    )
    chunker = SemanticChunker(ChunkerConfig(chunk_size=30, min_size=10))
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "semantic"
        assert c.metadata.get("repo") == "transformers"
        assert c.metadata.get("source") == "sem.md"


def test_semantic_chunker_with_mock_embedder():
    """SemanticChunker with embedder splits on cosine similarity drop below threshold."""
    sentences = [
        "Transformers provide general-purpose architectures.",
        "They are state of the art in natural language processing.",
        "Peft allows parameter efficient fine tuning.",
        "LoRA freezes pretrained weights and injects low rank matrices.",
    ]
    text = " ".join(sentences)
    doc = Document(
        doc_id="doc_sem_mock",
        source_path="sem_mock.md",
        text=text,
    )

    # Mock embedder: sentences 0 and 1 are identical vector [1, 0] (sim=1.0)
    # sentences 2 and 3 are identical vector [0, 1] (sim=1.0)
    # transition between 1 and 2 is orthogonal (sim=0.0 < 0.75)
    def mock_embedder(sents: list[str]) -> np.ndarray:
        vectors = []
        for s in sents:
            if "Transformers" in s or "natural language" in s:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors)

    chunker = SemanticChunker(
        config=ChunkerConfig(chunk_size=500, min_size=10),
        embedder=mock_embedder,
        similarity_threshold=0.75,
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) == 2
    assert "Transformers" in chunks[0].text
    assert "natural language" in chunks[0].text
    assert "Peft" in chunks[1].text
    assert "LoRA" in chunks[1].text


def test_semantic_chunker_code_block_preservation():
    """SemanticChunker must not fragment code blocks at method calls."""
    text = """Here is the training routine.

```python
model.eval()
inputs.to("cuda")
logger.info("Starting evaluation.")
```

Final summary sentence here.
"""
    doc = Document(
        doc_id="doc_code_preservation",
        source_path="code_pres.md",
        text=text,
    )
    chunker = SemanticChunker(ChunkerConfig(chunk_size=500, min_size=10))
    chunks = chunker.chunk(doc)

    full_output = "\n".join(c.text for c in chunks)
    assert 'model.eval()' in full_output
    assert 'inputs.to("cuda")' in full_output
    assert 'logger.info("Starting evaluation.")' in full_output


def test_chunkers_empty_and_whitespace_input():
    """All chunkers return empty list for empty or whitespace-only documents."""
    doc_empty = Document(doc_id="empty", source_path="empty.md", text="")
    doc_ws = Document(doc_id="ws", source_path="ws.md", text="   \n\n\t  ")

    for strat in ["fixed", "structural", "semantic"]:
        chunker = get_chunker(strat)
        assert chunker.chunk(doc_empty) == []
        assert chunker.chunk(doc_ws) == []


def test_chunkers_sub_min_size_handling():
    """All chunkers drop content smaller than min_size."""
    doc_tiny = Document(doc_id="tiny", source_path="tiny.md", text="Tiny.")
    cfg = ChunkerConfig(chunk_size=100, min_size=50)

    for strat in ["fixed", "structural", "semantic"]:
        chunker = get_chunker(strat, config=cfg)
        chunks = chunker.chunk(doc_tiny)
        assert chunks == []


def test_chunk_id_uniqueness_with_identical_paragraphs():
    """Identical text across different sections gets distinct chunk_ids."""
    text = """# Section Alpha

This exact identical sentence repeats.

# Section Beta

This exact identical sentence repeats.
"""
    doc = Document(doc_id="doc_repeat", source_path="repeat.md", text=text)
    chunker = StructuralChunker(ChunkerConfig(chunk_size=200, min_size=5))
    chunks = chunker.chunk(doc)

    assert len(chunks) == 2
    assert chunks[0].chunk_id != chunks[1].chunk_id
    assert chunks[0].position != chunks[1].position
    assert chunks[0].heading_path != chunks[1].heading_path


def test_semantic_chunker_code_block_double_newline_joining():
    """Fenced code blocks joined with prose must be separated by double newlines."""
    text = (
        "Here is the setup step.\n\n"
        "```python\n"
        "model = AutoModel.from_pretrained('bert-base-uncased')\n"
        "```\n\n"
        "Proceed with inference evaluation."
    )
    doc = Document(doc_id="sem_code_join", source_path="join.md", text=text)
    chunker = SemanticChunker(ChunkerConfig(chunk_size=1000, min_size=5))
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    chunk_text = chunks[0].text
    # Ensure code fence starts on its own line preceded by double newlines
    assert "setup step.\n\n```python" in chunk_text
    # Ensure code fence ends on its own line followed by double newlines
    assert "```\n\nProceed with inference" in chunk_text


def test_semantic_chunker_numbered_list_preservation():
    """Numbered list items must not have their digits split from item text."""
    text = (
        "1. Download model weights from the Hugging Face Hub.\n"
        "2. Instantiate the tokenizer with fast mode enabled.\n"
        "3. Run pipeline evaluation on GPU."
    )
    chunker = SemanticChunker(ChunkerConfig(chunk_size=1000, min_size=5))
    sentences = chunker._split_sentences(text)

    # Must produce 3 sentences, each retaining its enumeration digit
    assert len(sentences) == 3
    assert sentences[0] == "1. Download model weights from the Hugging Face Hub."
    assert sentences[1] == "2. Instantiate the tokenizer with fast mode enabled."
    assert sentences[2] == "3. Run pipeline evaluation on GPU."


def test_structural_chunker_four_backtick_fence_with_inner_comments():
    """Comments inside 4-backtick fence enclosing 3-backtick fence must not leak as headings."""
    text = (
        "# Markdown Documentation Guide\n\n"
        "Here is an example containing nested markdown:\n\n"
        "````markdown\n"
        "```python\n"
        "# Inner comment that looks like a heading\n"
        "x = 1\n"
        "```\n"
        "# Another comment inside 4-backticks\n"
        "````\n\n"
        "## Real Next Section\n\n"
        "This is genuine section content."
    )
    doc = Document(doc_id="struct_four_backticks", source_path="four.md", text=text)
    chunker = StructuralChunker(ChunkerConfig(chunk_size=1000, min_size=5))
    chunks = chunker.chunk(doc)

    # Real headings should ONLY be 'Markdown Documentation Guide' and 'Markdown Documentation Guide > Real Next Section'
    heading_paths = [c.heading_path for c in chunks]
    assert len(heading_paths) == 2
    assert heading_paths[0] == "Markdown Documentation Guide"
    assert heading_paths[1] == "Markdown Documentation Guide > Real Next Section"
    for c in chunks:
        assert "Inner comment" not in c.heading_path
        assert "Another comment" not in c.heading_path

