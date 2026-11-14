"""Unit tests for Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from hybrid_rag.models import RetrievedChunk
from hybrid_rag.retrieval.fusion import fuse, reciprocal_rank_fusion


def make_chunk(
    chunk_id: str,
    score: float,
    rank: int,
    retrieval_type: str = "dense",
    source: str = "test.md",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=f"Text for {chunk_id}",
        heading_path="Test",
        source=source,
        score=score,
        rank=rank,
        retrieval_type=retrieval_type,
        metadata={"source": source, "source_path": source},
    )


def test_fusion_empty_inputs():
    """Empty inputs should return an empty list."""
    assert fuse([], []) == []


def test_fusion_single_side_dense():
    """When sparse is empty, dense rankings determine fused order."""
    dense = [make_chunk("c1", 0.9, 1), make_chunk("c2", 0.8, 2)]
    fused = fuse(dense, [])
    assert len(fused) == 2
    assert fused[0].chunk_id == "c1"
    assert fused[1].chunk_id == "c2"
    assert fused[0].rank == 1
    assert fused[0].retrieval_type == "hybrid"


def test_fusion_single_side_sparse():
    """When dense is empty, sparse rankings determine fused order."""
    sparse = [make_chunk("s1", 12.0, 1, "sparse"), make_chunk("s2", 6.0, 2, "sparse")]
    fused = fuse([], sparse)
    assert len(fused) == 2
    assert fused[0].chunk_id == "s1"
    assert fused[1].chunk_id == "s2"
    assert fused[0].rank == 1
    assert fused[0].retrieval_type == "hybrid"


def test_fusion_weight_dense_only():
    """weight_dense=1.0, weight_sparse=0.0 preserves pure dense order."""
    dense = [make_chunk("c1", 0.9, 1), make_chunk("c2", 0.8, 2)]
    sparse = [make_chunk("c2", 15.0, 1, "sparse"), make_chunk("c1", 5.0, 2, "sparse")]

    fused = fuse(dense, sparse, k=60, weight_dense=1.0, weight_sparse=0.0)
    assert fused[0].chunk_id == "c1"
    assert fused[1].chunk_id == "c2"


def test_fusion_weight_sparse_only():
    """weight_dense=0.0, weight_sparse=1.0 preserves pure sparse order."""
    dense = [make_chunk("c1", 0.9, 1), make_chunk("c2", 0.8, 2)]
    sparse = [make_chunk("c2", 15.0, 1, "sparse"), make_chunk("c1", 5.0, 2, "sparse")]

    fused = fuse(dense, sparse, k=60, weight_dense=0.0, weight_sparse=1.0)
    assert fused[0].chunk_id == "c2"
    assert fused[1].chunk_id == "c1"


def test_fusion_hand_computed_order():
    """Test known hand-calculated RRF scores.

    dense: c1 (rank 1), c2 (rank 2)
    sparse: c2 (rank 1), c3 (rank 2)
    k = 60, weights = 0.5, 0.5

    score(c1) = 0.5 / (60 + 1) + 0             = 0.5 / 61 ≈ 0.008197
    score(c2) = 0.5 / (60 + 2) + 0.5 / (60 + 1) = 0.5/62 + 0.5/61 ≈ 0.008065 + 0.008197 = 0.016262
    score(c3) = 0 + 0.5 / (60 + 2)             = 0.5 / 62 ≈ 0.008065

    Expected order: c2 > c1 > c3
    """
    dense = [make_chunk("c1", 0.9, 1), make_chunk("c2", 0.8, 2)]
    sparse = [make_chunk("c2", 10.0, 1, "sparse"), make_chunk("c3", 8.0, 2, "sparse")]

    fused = fuse(dense, sparse, k=60, weight_dense=0.5, weight_sparse=0.5)
    assert len(fused) == 3
    assert fused[0].chunk_id == "c2"
    assert fused[1].chunk_id == "c1"
    assert fused[2].chunk_id == "c3"

    assert fused[0].rank == 1
    assert fused[1].rank == 2
    assert fused[2].rank == 3
    assert all(c.score >= 0.0 for c in fused)
    assert all(c.retrieval_type == "hybrid" for c in fused)


def test_fusion_compound_sort_tie_breaking():
    """Verify 4-element compound sort tuple guarantees deterministic tie-breaking.

    When two candidates have identical combined RRF scores, tie-breaking falls back
    to dense rank, then sparse rank, then chunk ID string.
    """
    # Equal symmetric scores
    dense = [make_chunk("chunk_b", 0.9, 1), make_chunk("chunk_a", 0.8, 2)]
    sparse = [make_chunk("chunk_a", 10.0, 1, "sparse"), make_chunk("chunk_b", 9.0, 2, "sparse")]

    # c_b score: 0.5/61 + 0.5/62; c_a score: 0.5/62 + 0.5/61 -> identical!
    # Dense rank for chunk_b is 1, for chunk_a is 2 -> chunk_b wins on dense rank
    fused = fuse(dense, sparse, k=60, weight_dense=0.5, weight_sparse=0.5)
    assert fused[0].chunk_id == "chunk_b"
    assert fused[1].chunk_id == "chunk_a"


def test_fusion_alias_and_metadata_alignment():
    """reciprocal_rank_fusion is an alias to fuse, and source/source_path are preserved."""
    dense = [make_chunk("c1", 0.95, 1, source="docs/peft.md")]
    sparse = [make_chunk("c1", 11.0, 1, "sparse", source="docs/peft.md")]

    fused = reciprocal_rank_fusion(dense, sparse)
    assert len(fused) == 1
    assert fused[0].source == "docs/peft.md"
    assert fused[0].metadata["source"] == "docs/peft.md"
    assert fused[0].metadata["source_path"] == "docs/peft.md"
