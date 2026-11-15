# Hybrid Search RAG — HuggingFace Docs

[![Release](https://img.shields.io/badge/release-v1.0-blue.svg)](https://github.com/DevRitesh08/Hybrid-Search-RAG/releases/tag/v1.0)
[![CI](https://github.com/DevRitesh08/Hybrid-Search-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/DevRitesh08/Hybrid-Search-RAG/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Dense + BM25 + RRF + Cross-Encoder Reranking** over HuggingFace Transformers & PEFT documentation — with verified citations, confidence scoring, and rigorous evaluation.

---

## Results

| Mode | Hit@5 | MRR | nDCG@10 | Latency p50 |
|---|---|---|---|---|
| Dense only | 1.0000 | 0.9612 | 0.9656 | 22 ms |
| BM25 only | 1.0000 | 0.9690 | 0.9699 | < 1 ms |
| Hybrid (RRF) | 1.0000 | 0.9651 | 0.9687 | 21 ms |
| Hybrid + Rerank | 1.0000 | 0.9884 | 0.9848 | 258 ms |

*95% CI from 1000-sample paired bootstrap on golden set (n=50, 43 answerable). Differences evaluated against dense baseline. Relevance is document-level: retrieved chunks are mapped to their source document paths and deduplicated preserving first-seen rank before metric evaluation. None of the pairwise differences vs dense reach statistical significance at 95% confidence (all 95% CIs include 0).*

---

## Architecture

```
Question
   │
   ▼
┌──────────────┐    ┌──────────────┐
│  Dense (BGE) │    │  BM25 (OKapi)│
│  ChromaDB    │    │  JSON index  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └────────┬──────────┘
                ▼
         RRF Fusion
         (20 candidates)
                │
                ▼
      Cross-Encoder Reranker
      (ms-marco-MiniLM-L-6)
                │
                ▼
         Top-5 Chunks
                │
                ▼
      Gemini Generation
      + Citation Verification
      + Confidence Scoring
                │
                ▼
      RAGResponse (JSON)
```

---

## Quickstart

### Docker (recommended)

```bash
git clone https://github.com/DevRitesh08/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

docker compose up -d
make docker-seed          # Downloads corpus, builds indexes, runs smoke test
# → API: http://localhost:8080/docs
# → Dashboard: http://localhost:8501
```

### Local

```bash
git clone https://github.com/DevRitesh08/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG
cp .env.example .env     # Add your GEMINI_API_KEY

make setup               # uv sync
make seed                # Download corpus → ingest → index

make api                 # FastAPI on :8080
make ui                  # Streamlit on :8501
```

---

## Evaluation

```bash
# Retrieval metrics only (no LLM cost)
make eval-retrieval

# Full eval: retrieval + generation + LLM judge
make eval-full
```

Reports land in `reports/` with timestamps. Confidence intervals computed via paired bootstrap (n=1000). Relevance is document-level: retrieved chunks are mapped to their unique document paths with duplicate ranks removed.

---

## Results Details

### Retrieval Variants (Golden Set, n=50)

| Mode | Hit@5 | Recall@5 | MRR | nDCG@10 | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|
| dense | 1.0000 | 1.0000 | 0.9612 | 0.9656 | 22 ms | 48 ms |
| bm25 | 1.0000 | 1.0000 | 0.9690 | 0.9699 | < 1 ms | < 1 ms |
| hybrid | 1.0000 | 1.0000 | 0.9651 | 0.9687 | 21 ms | 24 ms |
| hybrid_rerank | 1.0000 | 1.0000 | 0.9884 | 0.9848 | 258 ms | 332 ms |

### Paired Bootstrap Comparisons (vs Dense Only, 95% CI)

| Mode Comparison | Metric | Diff Mean | 95% CI | p-value | Significant |
|---|---|---|---|---|---|
| bm25 vs dense | MRR | +0.0078 | [-0.0543, +0.0698] | 0.8140 | No |
| hybrid vs dense | MRR | +0.0039 | [+0.0000, +0.0116] | 0.7960 | No |
| hybrid_rerank vs dense | MRR | +0.0271 | [-0.0233, +0.0814] | 0.3560 | No |

*Note: Differences between retrieval modes are not statistically significant. The 95% bootstrap confidence intervals for MRR improvements over the dense baseline include 0 in all cases (p > 0.35).*

---

## Design Decisions

### Why BM25 + Dense Hybrid?
Transformer docs are full of exact identifiers: `target_modules`, `lora_alpha`, `AutoModelForCausalLM`. Dense embeddings handle paraphrase well but can miss these exact terms. BM25 excels at exact match. RRF fusion gets both.

### Why Structural Chunking?
Markdown headings in HF docs map to logical concepts. Preserving heading path as a prefix ("Models > AutoModel > from_pretrained") gives the reranker context that pure text chunking loses.

### Why Cross-Encoder Reranking?
The bi-encoder (BGE) is fast but coarse. The cross-encoder scores (query, passage) jointly — much better signal. The latency cost is worth it for the top-5 slice.

### What Didn't Work
- Semantic chunking: similarity threshold tuning is fragile. On-CPU inference for sentence embeddings at chunk-build time was slow. Shipped as an option but `structural` is the default.
- Very high RRF dense weight (0.9/0.1): BM25 contribution was lost on semantic questions. 0.7/0.3 generalised better.

---

## Limitations

- Corpus: ~350 Markdown pages. Not every class is documented at the same depth.
- Golden set: 50 questions, one author. Subject to coverage bias.
- Judge: same model family for generation and judging (Gemini). Ideally different families.

---

## Repo Layout

```
src/hybrid_rag/
  config.py          — all tunables (pydantic-settings)
  models.py          — shared data types (Document, Chunk, RAGResponse, ...)
  ingest/            — loaders, normalizer, deduplicator
  chunking/          — fixed, structural, semantic chunkers
  index/             — embedder, ChromaDB wrapper, BM25 index, build orchestrator
  retrieval/         — dense, sparse, fusion (RRF), reranker, pipeline
  generation/        — LLM adapter, prompts, answer builder, citation verifier, confidence
  eval/              — golden loader, metrics, LLM judge, bootstrap CI, runner, reporter
  api/               — FastAPI app (ask, ingest, documents, healthz)
  cli.py             — command-line entry point
app/dashboard.py     — Streamlit UI
scripts/             — download_corpus.py, make_demo_bank.py
data/golden/         — golden.jsonl (50 questions)
reports/             — dated eval reports (committed)
tests/unit/          — fast offline tests (fusion, metrics, chunking, dedup, BM25, citations, API)
```

---

## Reproducing the Eval

```bash
# From a clean clone:
make setup
make seed          # ~5 min: downloads corpus, builds all three indexes

make eval          # ~20 min: retrieval + generation + judge
# → reports/retrieval_<date>.md
# → reports/full_<date>.md
```

All LLM calls are cached by input hash (`diskcache`). Re-running eval after the first pass is fast.

---

## Licence

MIT. Corpus: HuggingFace Transformers and PEFT docs are Apache 2.0 (see `data/raw/MANIFEST.json` for pinned commit). Raw docs are **not committed** — `scripts/download_corpus.py` fetches them at a pinned hash.
