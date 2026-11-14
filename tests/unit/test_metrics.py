"""Unit tests for retrieval evaluation metrics."""

from __future__ import annotations

import math

from hybrid_rag.eval.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_rate_at_k():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    
    # Hit in top-5
    assert hit_rate_at_k(retrieved, ["c3"], k=5) == 1.0
    # Miss outside top-3
    assert hit_rate_at_k(retrieved, ["c5"], k=3) == 0.0
    # Complete miss
    assert hit_rate_at_k(retrieved, ["c99"], k=5) == 0.0
    # Empty inputs
    assert hit_rate_at_k([], ["c1"], k=5) == 0.0
    assert hit_rate_at_k(retrieved, [], k=5) == 0.0


def test_recall_at_k():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = ["c1", "c3", "c9", "c10"]  # 4 total relevant

    # Top-5 has c1 and c3 -> 2 / 4 = 0.5
    assert recall_at_k(retrieved, relevant, k=5) == 0.5

    # Top-2 has c1 -> 1 / 4 = 0.25
    assert recall_at_k(retrieved, relevant, k=2) == 0.25

    # None relevant
    assert recall_at_k(retrieved, [], k=5) == 0.0


def test_reciprocal_rank():
    retrieved = ["c1", "c2", "c3", "c4"]

    # First relevant at rank 1 -> 1.0
    assert reciprocal_rank(retrieved, ["c1"]) == 1.0

    # First relevant at rank 2 -> 0.5
    assert reciprocal_rank(retrieved, ["c2"]) == 0.5

    # First relevant at rank 4 -> 0.25
    assert reciprocal_rank(retrieved, ["c4"]) == 0.25

    # None found -> 0.0
    assert reciprocal_rank(retrieved, ["c99"]) == 0.0


def test_ndcg_at_k_perfect_ranking():
    """All relevant items at the top should give nDCG = 1.0."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = ["c1", "c2"]

    assert ndcg_at_k(retrieved, relevant, k=5) == 1.0


def test_ndcg_at_k_hand_computed():
    """Hand-computed DCG / IDCG test.

    retrieved: [c2, c1], relevant: [c1]
    DCG@2: c2 is irrelevant (gain 0). c1 at pos 2 has gain 1 / log2(2 + 1) = 1 / log2(3) ≈ 0.63093
    IDCG@2: c1 at pos 1 has gain 1 / log2(1 + 1) = 1.0
    nDCG = 0.63093 / 1.0 ≈ 0.6309
    """
    retrieved = ["c2", "c1"]
    relevant = ["c1"]

    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert abs(ndcg_at_k(retrieved, relevant, k=2) - expected_ndcg) < 1e-4


def test_evaluate_question_retrieval_document_level(monkeypatch):
    """Failing test: evaluate_question_retrieval must rank at document level.
    
    When top retrieved chunks all belong to the relevant document,
    hit_at_5 must be True, MRR must be 1.0, and nDCG@10 must be 1.0.
    Under the buggy implementation (retrieved_ids + retrieved_sources),
    chunk IDs occupy ranks 1-5, pushing the document path to rank 6,
    causing hit_at_5 to fail and MRR to drop to 1/6.
    """
    from hybrid_rag.eval.run_eval import evaluate_question_retrieval
    from hybrid_rag.models import GoldenQuestion, RetrievedChunk

    # 5 chunks all from the target document
    fake_chunks = [
        RetrievedChunk(
            chunk_id=f"chunk_{i}",
            doc_id="peft/lora.md",
            text="some content",
            metadata={"source": "peft/lora.md"},
            score=0.9 - i * 0.1,
            rank=i + 1,
        )
        for i in range(5)
    ]

    question = GoldenQuestion(
        id="q_test",
        type="direct",
        question="What is LoRA?",
        reference_answer="Low-Rank Adaptation",
        source_chunks_or_sections=["peft/lora.md"],
    )

    monkeypatch.setattr(
        "hybrid_rag.eval.run_eval.retrieve",
        lambda q, mode, top_k: (fake_chunks, {}),
    )

    res = evaluate_question_retrieval(question, mode="dense", top_k=5)

    # Must be document-level rank 1
    assert res.hit_at_5 is True
    assert res.recall_at_5 == 1.0
    assert res.reciprocal_rank == 1.0
    assert res.ndcg_at_10 == 1.0

