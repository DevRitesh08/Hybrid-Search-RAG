"""Unit tests for prompt formatting and context block construction (Feature 11)."""

from __future__ import annotations

from hybrid_rag.generation.prompt import (
    CANONICAL_REFUSAL_MESSAGE,
    CITATION_VERIFY_PROMPT,
    COMPLETENESS_PROMPT,
    SYSTEM_PROMPT,
    build_context_block,
    build_qa_prompt,
)
from hybrid_rag.models import RetrievedChunk


def make_chunk(
    chunk_id: str = "chk_1",
    doc_id: str = "doc_1",
    text: str = "Sample content text.",
    heading_path: str = "Section > Subsection",
    source: str = "docs/sample.md",
    score: float = 0.95,
    rank: int = 1,
    metadata: dict | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        heading_path=heading_path,
        source=source,
        score=score,
        rank=rank,
        metadata=metadata or {},
    )


def test_build_context_block_numbered_formatting():
    """Verify that chunks are formatted with 1-based indexing, source path, and heading hierarchy."""
    chunks = [
        make_chunk(chunk_id="c1", text="First chunk content", heading_path="Intro > Start", source="docs/intro.md"),
        make_chunk(chunk_id="c2", text="Second chunk content", heading_path="Advanced", source="docs/adv.md", rank=2),
    ]
    block = build_context_block(chunks)

    assert "[1] Source: docs/intro.md > Intro > Start" in block
    assert "First chunk content" in block
    assert "[2] Source: docs/adv.md > Advanced" in block
    assert "Second chunk content" in block


def test_build_context_block_empty():
    """Empty chunks list produces an empty string."""
    assert build_context_block([]) == ""


def test_build_context_block_without_heading():
    """Chunks with empty heading_path omit the trailing ' > ' separator."""
    chunk = make_chunk(heading_path="", source="docs/readme.md")
    block = build_context_block([chunk])

    assert "[1] Source: docs/readme.md\n\n" in block
    assert " > " not in block


def test_build_context_block_metadata_fallback():
    """Test fallback order: chunk.source -> metadata['source_path'] -> metadata['source'] -> doc_id."""
    # 1. source attribute set
    c1 = make_chunk(source="from_attr.md", metadata={"source_path": "from_meta_path.md", "source": "from_meta.md"})
    assert "Source: from_attr.md" in build_context_block([c1])

    # 2. source empty, metadata['source_path'] set
    c2 = make_chunk(source="", metadata={"source_path": "from_meta_path.md", "source": "from_meta.md"})
    assert "Source: from_meta_path.md" in build_context_block([c2])

    # 3. source and source_path empty, metadata['source'] set
    c3 = make_chunk(source="", metadata={"source": "from_meta.md"})
    assert "Source: from_meta.md" in build_context_block([c3])

    # 4. source and metadata empty, fallback to doc_id
    c4 = make_chunk(source="", doc_id="doc_fallback_999", metadata={})
    assert "Source: doc_fallback_999" in build_context_block([c4])


def test_build_qa_prompt_structure():
    """Verify question and context block insertion in user QA prompt."""
    context = "[1] Source: doc.md\n\nContent here"
    question = "How to configure AutoModel?"
    prompt = build_qa_prompt(question, context)

    assert "Context:\n\n[1] Source: doc.md\n\nContent here" in prompt
    assert "Question: How to configure AutoModel?" in prompt
    assert "Answer (cite using [N] notation):" in prompt


def test_system_prompt_rules():
    """Verify rules present in the system prompt."""
    assert "Answer ONLY using the numbered context blocks" in SYSTEM_PROMPT
    assert "Cite sources using [N] notation" in SYSTEM_PROMPT
    assert CANONICAL_REFUSAL_MESSAGE in SYSTEM_PROMPT
    assert "Do not speculate" in SYSTEM_PROMPT


def test_canonical_refusal_message():
    """Verify the exact refusal message text required by Requirement R3."""
    assert (
        CANONICAL_REFUSAL_MESSAGE
        == "I don't have enough information in the provided context to answer this question."
    )


def test_judge_prompts_templates():
    """Verify CITATION_VERIFY_PROMPT and COMPLETENESS_PROMPT contain required placeholders."""
    verify_rendered = CITATION_VERIFY_PROMPT.format(claim="Test claim", chunk_text="Chunk text")
    assert "Claim: Test claim" in verify_rendered
    assert "Text chunk:\nChunk text" in verify_rendered
    assert "SUPPORTED" in verify_rendered

    comp_rendered = COMPLETENESS_PROMPT.format(question="Q?", answer="A.")
    assert "Question: Q?" in comp_rendered
    assert "Answer: A." in comp_rendered
    assert "COMPLETE" in comp_rendered
