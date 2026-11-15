# Evaluation Report — Hybrid RAG (2026-11-15)

> Benchmark results comparing dense, BM25, hybrid (RRF), and hybrid + rerank retrieval pipelines.

## Summary Metrics

| Mode | Hit@5 | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| dense | 1.0000 | 1.0000 | 0.9612 | 0.9656 |
| bm25 | 1.0000 | 1.0000 | 0.9690 | 0.9699 |
| hybrid | 1.0000 | 1.0000 | 0.9651 | 0.9687 |
| hybrid_rerank | 1.0000 | 1.0000 | 0.9884 | 0.9848 |

## Paired Bootstrap Comparisons (vs Dense Only, 95% CI)

| Mode Comparison | Metric | Diff Mean | 95% CI | p-value | Significant |
|---|---|---|---|---|---|
| bm25 vs dense | MRR | +0.0078 | [-0.0543, +0.0698] | 0.8140 | ✗ No |
| hybrid vs dense | MRR | +0.0039 | [+0.0000, +0.0116] | 0.7960 | ✗ No |
| hybrid_rerank vs dense | MRR | +0.0271 | [-0.0233, +0.0814] | 0.3560 | ✗ No |

## Latency Profile (ms)

| Mode | p50 (ms) | p95 (ms) |
|---|---|---|
| dense | 22 | 48 |
| bm25 | 0 | 0 |
| hybrid | 21 | 24 |
| hybrid_rerank | 258 | 332 |

---
*Report generated automatically by `hybrid_rag.eval.run_eval`.*