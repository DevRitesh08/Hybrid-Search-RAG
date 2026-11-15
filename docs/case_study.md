# Case Study: Production-Grade Hybrid Search RAG over Hugging Face Documentation

## Executive Summary

Standard RAG architectures often rely solely on dense vector retrieval. In technical domains like machine learning engineering—where user queries frequently feature exact code identifiers, class names, and hyperparameters (`target_modules`, `AutoModelForCausalLM`, `lora_alpha`, `get_peft_model`)—pure dense embeddings exhibit notable lexical blind spots.

This production system demonstrates a Hybrid Search RAG architecture over Hugging Face `transformers` and `peft` documentation, combining:
1. **Dense Retrieval**: `BAAI/bge-small-en-v1.5` embeddings in ChromaDB with cosine metric normalization.
2. **Sparse Lexical Retrieval**: Okapi BM25 (`rank_bm25`) with programming identifier preservation and deterministic tie-breaking.
3. **Reciprocal Rank Fusion (RRF)**: Parameterized rank aggregation ($k=60$) with tunable dense/sparse weights ($0.7 / 0.3$).
4. **Cross-Encoder Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scoring the top-20 fused candidate pool to the top-5 relevant chunks.
5. **Sentence-Level Citation Verification & Confidence Engine**: Post-generation claim attribution with explicit refusal logic for low-confidence or unanswerable queries.

---

## 1. Quantitative Benchmark Results

Evaluated over the 50-question golden evaluation dataset (43 answerable queries, 7 unanswerable negative controls) using paired bootstrap statistical testing ($n=1000$ iterations, 95% confidence intervals). Relevance is evaluated at document level with deduplication preserving first-seen rank.

### Retrieval Mode Comparison

| Mode | Hit@5 | Recall@5 | MRR | nDCG@10 | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|
| Dense Only (`bge-small-en-v1.5`) | 1.0000 | 1.0000 | 0.9612 | 0.9656 | 22 ms | 48 ms |
| BM25 Only (Okapi + exact tokens) | 1.0000 | 1.0000 | 0.9690 | 0.9699 | < 1 ms | < 1 ms |
| Hybrid (RRF) (0.7 / 0.3) | 1.0000 | 1.0000 | 0.9651 | 0.9687 | 21 ms | 24 ms |
| Hybrid + Cross-Encoder Rerank | 1.0000 | 1.0000 | 0.9884 | 0.9848 | 258 ms | 332 ms |

### Paired Bootstrap Hypothesis Testing (vs Dense Baseline, 95% CI)

| Comparison | Metric | Diff Mean | 95% Bootstrap CI | p-value | Significant |
|---|---|---|---|---|---|
| BM25 vs Dense | MRR | +0.0078 | [-0.0543, +0.0698] | 0.8140 | No |
| Hybrid vs Dense | MRR | +0.0039 | [+0.0000, +0.0116] | 0.7960 | No |
| Hybrid Rerank vs Dense | MRR | +0.0271 | [-0.0233, +0.0814] | 0.3560 | No |

*Statistical Assessment*: None of the observed metric differences reach statistical significance at the 95% confidence level. With 43 answerable questions, all 95% bootstrap confidence intervals for MRR improvement cross zero (or touch zero), with p-values well above 0.05. The numerical gains observed in MRR (+0.0271 for reranking) represent point estimates on this evaluation sample rather than statistically confirmed differences.

---

## 2. Key Architectural Insights & Ablations

### A. The BM25 Exact Identifier Advantage
In software documentation, specific tokens carry immense semantic weight that general-purpose bi-encoders can dilute in high-dimensional embedding space.
- Query: *"Which parameter in LoraConfig defines the specific linear layers to adapt?"*
  - **Dense Retrieval**: Retrieved general LoRA introduction and PEFT overview; `target_modules` ranked lower.
  - **BM25 Retrieval**: Ranked `peft/lora.md` #1 directly due to exact inverted-index matches for `LoraConfig` and `linear layers`.
  - **Trade-off**: BM25 achieved competitive retrieval (MRR 0.9690) with virtually zero retrieval latency (< 1 ms), confirming the value of sparse lexical indexing alongside dense embeddings.

### B. Structural Chunking Architecture
Documentation naturally contains hierarchical structure. Fixed-size windowing risks bisecting code blocks and function signatures across split boundaries. Structural chunking extracts markdown sections along heading boundaries and prepends the full breadcrumb path (e.g. `AutoModel > from_pretrained`) directly into each chunk header. This embeds document context locally, allowing downstream rerankers to recognize the functional context without needing full-document re-parsing.

### C. Cross-Encoder Joint Scoring & Latency Trade-Off
The bi-encoder embeds queries and documents independently ($O(N)$ dot products), whereas the cross-encoder computes full self-attention across $(query, document)$ tokens jointly.
- Cross-encoder reranking achieved an observed MRR of 0.9884 (vs 0.9612 for dense and 0.9651 for hybrid).
- However, latency increased from 21 ms (hybrid) to 258 ms p50 (and 332 ms p95) on CPU. For small technical corpora, this latency trade-off must be evaluated against the operational budget and requirements.

### D. Citation Discipline & Refusal Engine
To avoid presenting hallucinated answers on out-of-domain or adversarial queries:
- Answers with composite confidence $< 0.35$ trigger standard refusal: *"I don't have enough information in the provided context to answer this question."*
- Citations are strictly suppressed (`citations: []`) on all refused answers, eliminating misleading reference links.
- Negative controls in the evaluation set (e.g. out-of-domain terms) correctly trigger the refusal threshold.

---

## 3. Production Deployment Profile

- **FastAPI Backend**: Asynchronous endpoints (`/v1/ask`, `/v1/ingest`, `/healthz`) running under Uvicorn with threadpool offloading for CPU-intensive retrieval.
- **Streamlit Interactive UI**: Dual-column retrieval inspection, side-by-side hybrid vs dense toggling, and visual confidence meters.
- **Containerization**: Multi-stage `Dockerfile` and `docker-compose.yml` supporting isolated volume caching and automated health checks.
- **Continuous Integration**: GitHub Actions pipeline enforcing `ruff check`, `mypy`, and the unit test suite (`tests/unit/`).
