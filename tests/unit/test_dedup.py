"""Unit tests for chunk deduplication."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_rag.ingest.dedup import (
    _cosine_similarity,
    cosine_similarity,
    deduplicate,
)
from hybrid_rag.models import Chunk


def test_cosine_similarity_basic():
    """Verify basic cosine similarity calculations and zero vector safety."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    v_zero = np.array([0.0, 0.0, 0.0])

    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0
    assert cosine_similarity(v1, v_zero) == 0.0
    assert cosine_similarity(v_zero, v_zero) == 0.0


def test_deduplicate_empty():
    kept, skipped = deduplicate([], {})
    assert kept == []
    assert skipped == []


def test_deduplicate_filters_duplicates(sample_chunks: list[Chunk]):
    """When chunks have identical embeddings, the duplicate should be skipped."""
    c1 = sample_chunks[0]
    c2 = sample_chunks[1]

    # Create dummy embeddings: c1 and c2 have cosine=1.0 (exact duplicate vectors)
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])  # Duplicate of vec1
    vec3 = np.array([0.0, 1.0, 0.0])  # Orthogonal to vec1

    c3 = sample_chunks[2]

    chunks = [c1, c2, c3]
    embeddings = {
        c1.chunk_id: vec1,
        c2.chunk_id: vec2,
        c3.chunk_id: vec3,
    }

    kept, skipped = deduplicate(chunks, embeddings, threshold=0.95)

    assert len(kept) == 2
    assert len(skipped) == 1
    assert c1 in kept
    assert c3 in kept
    assert c2.chunk_id in skipped


def test_deduplicate_threshold_boundary(sample_chunks: list[Chunk]):
    """Ensure strict inequality (> threshold) applies at threshold boundaries."""
    c_base = sample_chunks[0]
    c_below = sample_chunks[1]
    c_exact = sample_chunks[2]
    c_above = sample_chunks[3]

    base_vec = np.array([1.0, 0.0])
    # Angle theta such that cos(theta) = target
    theta_0949 = np.arccos(0.949)
    theta_0950 = np.arccos(0.950)
    theta_0951 = np.arccos(0.951)

    vec_below = np.array([np.cos(theta_0949), np.sin(theta_0949)])
    vec_exact = np.array([np.cos(theta_0950), np.sin(theta_0950)])
    vec_above = np.array([np.cos(theta_0951), np.sin(theta_0951)])

    # 1. Chunk with similarity 0.949 <= 0.95 must be KEPT
    kept, skipped = deduplicate(
        [c_base, c_below],
        {c_base.chunk_id: base_vec, c_below.chunk_id: vec_below},
        threshold=0.95,
    )
    assert len(kept) == 2
    assert len(skipped) == 0

    # 2. Chunk with similarity exactly 0.950 <= 0.95 must be KEPT
    kept, skipped = deduplicate(
        [c_base, c_exact],
        {c_base.chunk_id: base_vec, c_exact.chunk_id: vec_exact},
        threshold=0.95,
    )
    assert len(kept) == 2
    assert len(skipped) == 0

    # 3. Chunk with similarity 0.951 > 0.95 must be SKIPPED
    kept, skipped = deduplicate(
        [c_base, c_above],
        {c_base.chunk_id: base_vec, c_above.chunk_id: vec_above},
        threshold=0.95,
    )
    assert len(kept) == 1
    assert len(skipped) == 1
    assert c_above.chunk_id in skipped


def test_deduplicate_negative_similarities(sample_chunks: list[Chunk]):
    """Opposite/orthogonal vectors with negative cosine similarities are kept and argmax works."""
    c1 = sample_chunks[0]
    c2 = sample_chunks[1]
    c3 = sample_chunks[2]

    # All vectors point in negative directions
    embeddings = {
        c1.chunk_id: np.array([1.0, 0.0]),
        c2.chunk_id: np.array([-1.0, 0.0]),  # similarity -1.0
        c3.chunk_id: np.array([-0.5, -0.866]),  # similarity -0.5
    }

    kept, skipped = deduplicate([c1, c2, c3], embeddings, threshold=0.95)
    assert len(kept) == 3
    assert len(skipped) == 0


def test_deduplicate_zero_norm_vectors(sample_chunks: list[Chunk]):
    """Zero-norm vectors do not trigger division by zero errors and are kept."""
    c1 = sample_chunks[0]
    c2 = sample_chunks[1]

    embeddings = {
        c1.chunk_id: np.array([0.0, 0.0, 0.0]),
        c2.chunk_id: np.array([1.0, 0.0, 0.0]),
    }

    kept, skipped = deduplicate([c1, c2], embeddings, threshold=0.95)
    assert len(kept) == 2
    assert len(skipped) == 0


def test_deduplicate_missing_embedding_kept(sample_chunks: list[Chunk]):
    """Chunks missing from embeddings dictionary are kept with a warning."""
    c1 = sample_chunks[0]
    c2 = sample_chunks[1]

    embeddings = {c1.chunk_id: np.array([1.0, 0.0, 0.0])}

    kept, skipped = deduplicate([c1, c2], embeddings, threshold=0.95)
    assert len(kept) == 2
    assert c2 in kept


def test_deduplicate_array_and_list_embeddings(sample_chunks: list[Chunk]):
    """Deduplicate accepts numpy 2-D array or list of arrays."""
    c1 = sample_chunks[0]
    c2 = sample_chunks[1]
    c3 = sample_chunks[2]

    # Array embeddings: c1 and c2 identical, c3 orthogonal
    emb_array = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    kept, skipped = deduplicate([c1, c2, c3], emb_array, threshold=0.95)
    assert len(kept) == 2
    assert len(skipped) == 1
    assert c2.chunk_id in skipped


def test_cosine_similarity_alias_and_nan_guards():
    """Verify legacy _cosine_similarity alias and non-finite numerical guards."""
    assert _cosine_similarity is cosine_similarity

    nan_vec = np.array([np.nan, 1.0, 0.0])
    inf_vec = np.array([np.inf, 0.0, 0.0])
    valid_vec = np.array([1.0, 0.0, 0.0])

    assert _cosine_similarity(nan_vec, valid_vec) == 0.0
    assert _cosine_similarity(valid_vec, inf_vec) == 0.0
    assert _cosine_similarity(nan_vec, inf_vec) == 0.0


def test_deduplicate_non_finite_embeddings_do_not_poison_matrix(sample_chunks: list[Chunk]):
    """Chunks with NaN or Inf embeddings are kept safely and do not poison subsequent dedup."""
    c_valid1 = sample_chunks[0]
    c_nan = sample_chunks[1]
    c_valid2 = sample_chunks[2]  # Identical to c_valid1 -> should be skipped!

    vec_valid = np.array([1.0, 0.0, 0.0])
    vec_nan = np.full(3, np.nan)

    embeddings = {
        c_valid1.chunk_id: vec_valid,
        c_nan.chunk_id: vec_nan,
        c_valid2.chunk_id: vec_valid.copy(),
    }

    kept, skipped = deduplicate([c_valid1, c_nan, c_valid2], embeddings, threshold=0.95)

    assert len(kept) == 2
    assert len(skipped) == 1
    assert c_valid1 in kept
    assert c_nan in kept
    assert c_valid2.chunk_id in skipped

