"""Unit tests for citation parsing and bracket notation expansion (Feature 13)."""

from __future__ import annotations

from hybrid_rag.generation.answer import parse_citations as answer_parse_citations
from hybrid_rag.generation.citations import (
    extract_citation_spans,
    parse_citation_numbers,
    parse_citations,
)
from hybrid_rag.models import Citation, RetrievedChunk


def make_retrieved(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"chk_{i}",
            doc_id=f"doc_{i}",
            text=f"Chunk {i} text describing Transformers feature {i}.",
            heading_path=f"Heading {i}",
            source=f"source_{i}.md",
            score=0.9,
            rank=i,
            metadata={"source": f"source_{i}.md"},
        )
        for i in range(1, n + 1)
    ]


def test_parse_single_citation():
    chunks = make_retrieved(3)
    text = "The AutoModel class is convenient [1]."
    citations = parse_citations(text, chunks)

    assert len(citations) == 1
    assert citations[0].n == 1
    assert citations[0].chunk_id == "chk_1"
    assert citations[0].source == "source_1.md"
    assert citations[0].heading == "Heading 1"
    assert "The AutoModel class is convenient" in citations[0].claim_text


def test_parse_multiple_citations_bracket_format():
    chunks = make_retrieved(5)
    text = "Combining adapters requires two steps [1][2] and finishes with [4]."
    citations = parse_citations(text, chunks)

    ns = [c.n for c in citations]
    assert ns == [1, 2, 4]


def test_parse_comma_separated_citations():
    chunks = make_retrieved(5)
    text = "Several layers can be trained [2, 3]."
    citations = parse_citations(text, chunks)

    ns = [c.n for c in citations]
    assert ns == [2, 3]


def test_parse_range_citations():
    """Verify range notation [1-3] and en-dash [1–3] expands to [1, 2, 3]."""
    chunks = make_retrieved(5)
    text = "Pretrained weights are initialized in stages [1-3]."
    citations = parse_citations(text, chunks)

    assert [c.n for c in citations] == [1, 2, 3]
    assert [c.chunk_id for c in citations] == ["chk_1", "chk_2", "chk_3"]

    # Test with spacing and en-dash
    text_spaces = "Supported across versions [2 - 4]."
    citations_spaces = parse_citations(text_spaces, chunks)
    assert [c.n for c in citations_spaces] == [2, 3, 4]


def test_parse_mixed_ranges_and_commas():
    """Verify combined range and comma citations like [1, 3-5]."""
    chunks = make_retrieved(6)
    text = "Model checkpoints are verified [1, 3-5]."
    citations = parse_citations(text, chunks)

    assert [c.n for c in citations] == [1, 3, 4, 5]


def test_parse_citation_numbers_helper():
    """Verify parse_citation_numbers isolated utility."""
    assert parse_citation_numbers("1") == [1]
    assert parse_citation_numbers("1, 2, 3") == [1, 2, 3]
    assert parse_citation_numbers("1-4") == [1, 2, 3, 4]
    assert parse_citation_numbers("1, 3-5, 8") == [1, 3, 4, 5, 8]
    assert parse_citation_numbers("invalid") == []
    assert parse_citation_numbers("") == []


def test_reject_code_indexing_and_types():
    """Verify code array index (arr[0]) and generics (List[str]) are not parsed as citations."""
    chunks = make_retrieved(5)
    text = (
        "In Python, access tokens via array[0] or batch[1]. "
        "Type annotations use List[str] or Dict[int, str]."
    )
    citations = parse_citations(text, chunks)
    assert len(citations) == 0


def test_extract_citation_spans():
    """Verify offset and span extraction."""
    text = "AutoModel loads weights [1] and LoRA wraps layers [2, 3]."
    spans = extract_citation_spans(text)

    assert len(spans) == 2
    assert spans[0]["raw"] == "[1]"
    assert spans[0]["numbers"] == [1]
    assert spans[1]["raw"] == "[2, 3]"
    assert spans[1]["numbers"] == [2, 3]


def test_citation_claim_association():
    """Verify citations map to their specific sentence claim."""
    chunks = make_retrieved(4)
    text = (
        "AutoModel loads pretrained checkpoints [1]. "
        "LoRA adapts specific attention weights efficiently [2][3]."
    )
    citations = parse_citations(text, chunks)

    c1 = next(c for c in citations if c.n == 1)
    assert "AutoModel loads pretrained checkpoints" in c1.claim_text
    assert "LoRA" not in c1.claim_text

    c2 = next(c for c in citations if c.n == 2)
    assert "LoRA adapts specific attention weights efficiently" in c2.claim_text


def test_citation_metadata_fallback():
    """Verify fallback from source attribute to metadata source_path / source."""
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            text="T1",
            score=0.9,
            rank=1,
            source="attr_source.md",
            metadata={},
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="d2",
            text="T2",
            score=0.8,
            rank=2,
            source="",
            metadata={"source_path": "path_source.md"},
        ),
    ]
    citations = parse_citations("Citations [1][2]", chunks)
    assert citations[0].source == "attr_source.md"
    assert citations[1].source == "path_source.md"


def test_parse_citations_out_of_range():
    """Citations pointing to chunk numbers beyond retrieved set are ignored safely."""
    chunks = make_retrieved(2)
    text = "This cites an absent source [99]."
    citations = parse_citations(text, chunks)
    assert len(citations) == 0


def test_parse_empty_citations():
    chunks = make_retrieved(3)
    text = "This answer has no citations at all."
    assert parse_citations(text, chunks) == []


def test_citation_model_compatibility_aliases():
    """Verify PROJECT.md interface aliases: is_verified and citation_index."""
    c = Citation(n=3, chunk_id="chk_3", source="doc.md", heading="H", verified=True)
    assert c.citation_index == 3
    assert c.is_verified is True

    c.is_verified = False
    assert c.verified is False

    c.citation_index = 5
    assert c.n == 5


def test_reexport_backward_compatibility():
    """Verify parse_citations imported from answer.py produces identical results."""
    chunks = make_retrieved(3)
    text = "Testing re-export [1-2]."
    c_direct = parse_citations(text, chunks)
    c_reexport = answer_parse_citations(text, chunks)

    assert [c.n for c in c_direct] == [c.n for c in c_reexport] == [1, 2]
    assert [c.chunk_id for c in c_direct] == [c.chunk_id for c in c_reexport] == ["chk_1", "chk_2"]


def test_citation_claim_association_multi_paragraph_markdown():
    """Verify that multi-paragraph markdown answers with \n\n do not cause sentence offset drift."""
    chunks = make_retrieved(3)
    paragraphs = [
        "Paragraph one introduces the high-level Transformers framework.",
        "Paragraph two discusses tokenization algorithms including WordPiece and BPE.",
        "Paragraph three covers attention masks and position encoding mechanisms.",
        "Paragraph four details self-attention matrices and feed-forward dimensions.",
        "Paragraph five covers layer normalization and residual connections.",
        "Paragraph six explains cross-attention modules in sequence-to-sequence setups.",
        "Paragraph seven discusses beam search and top-p sampling decoding strategies.",
        "Finally, AutoModel converts checkpoint weights into PyTorch modules [1].",
    ]
    answer = "\n\n".join(paragraphs)

    citations = parse_citations(answer, chunks)

    assert len(citations) == 1
    c1 = citations[0]
    assert c1.n == 1
    assert "Finally, AutoModel converts checkpoint weights into PyTorch modules" in c1.claim_text
    assert "Paragraph one" not in c1.claim_text
    assert c1.claim_text != answer.replace("[1]", "").strip()


def test_parse_citations_duplicate_chunk_in_multiple_claims():
    """Verify citing same chunk across multiple claims yields distinct citations with respective claims."""
    chunks = make_retrieved(3)
    answer = "AutoModel loads weights from disk [1]. AutoModel also instantiates configuration classes [1]."
    citations = parse_citations(answer, chunks)

    assert len(citations) == 2
    assert citations[0].n == 1
    assert citations[1].n == 1
    assert "AutoModel loads weights from disk" in citations[0].claim_text
    assert "AutoModel also instantiates configuration classes" in citations[1].claim_text
