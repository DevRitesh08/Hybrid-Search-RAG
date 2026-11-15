# Evaluation Report — Hybrid RAG (2026-11-14)

> Benchmark results comparing dense, BM25, hybrid (RRF), and hybrid + rerank retrieval pipelines.

## Summary Metrics

| Mode | Hit@5 | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| dense | 0.9302 | 0.9302 | 0.1971 | 0.3960 |
| bm25 | 0.9535 | 0.9535 | 0.1973 | 0.3955 |
| hybrid | 0.9302 | 0.9302 | 0.1977 | 0.3965 |
| hybrid_rerank | 0.9767 | 0.9767 | 0.1992 | 0.3977 |

## Paired Bootstrap Comparisons (vs Dense Only, 95% CI)

| Mode Comparison | Metric | Diff Mean | 95% CI | p-value | Significant |
|---|---|---|---|---|---|
| bm25 vs dense | MRR | +0.0002 | [-0.0051, +0.0050] | 0.9100 | ✗ No |
| hybrid vs dense | MRR | +0.0006 | [+0.0000, +0.0017] | 0.7960 | ✗ No |
| hybrid_rerank vs dense | MRR | +0.0021 | [-0.0016, +0.0061] | 0.3500 | ✗ No |

## Latency Profile (ms)

| Mode | p50 (ms) | p95 (ms) |
|---|---|---|
| dense | 77 | 85 |
| bm25 | 0 | 0 |
| hybrid | 19 | 24 |
| hybrid_rerank | 308 | 354 |

---
*Report generated automatically by `hybrid_rag.eval.run_eval`.*