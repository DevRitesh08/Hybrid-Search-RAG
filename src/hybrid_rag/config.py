"""Central configuration — all tunables live here.

Every parameter is overridable via environment variable or .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent  # repo root

# Ensure HF and Torch cache inside repo to bypass broken junctions
_hf_home = ROOT / ".cache" / "huggingface"
_hf_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf_home))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_hf_home))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))


class Settings(BaseSettings):
    """All settings, with environment-variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ─────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", description="Gemini API key")
    openai_api_key: str = Field(default="", description="OpenAI API key (optional ablation)")

    # ── Models ───────────────────────────────────────────────────────────────
    embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="SentenceTransformers embedding model",
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder reranker model",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model for generation",
    )
    judge_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model for LLM-as-judge evaluation",
    )

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_path: Path = Field(default=ROOT / ".chroma", description="ChromaDB persistence path")
    chroma_host: str = Field(default="localhost", description="ChromaDB host (http mode)")
    chroma_port: int = Field(default=8000, description="ChromaDB port (http mode)")
    chroma_mode: Literal["local", "http"] = Field(
        default="local",
        description="ChromaDB connection mode",
    )
    chroma_collection_prefix: str = Field(
        default="hybrid_rag",
        description="Prefix for ChromaDB collection names",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=20, description="Candidates to retrieve before reranking")
    rerank_top_k: int = Field(default=5, description="Final chunks after reranking")
    rrf_k: int = Field(default=60, description="RRF smoothing constant")
    rrf_weight_dense: float = Field(default=0.7, description="RRF weight for dense retrieval")
    rrf_weight_sparse: float = Field(default=0.3, description="RRF weight for BM25 retrieval")
    retrieval_mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = Field(
        default="hybrid_rerank",
        description="Default retrieval mode",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_strategy: Literal["fixed", "structural", "semantic"] = Field(
        default="structural",
        description="Default chunking strategy",
    )
    chunk_size: int = Field(default=500, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Chunk overlap in tokens")
    chunk_min_size: int = Field(default=50, description="Minimum chunk size; smaller are dropped")
    dedup_threshold: float = Field(
        default=0.95,
        description="Cosine similarity threshold for deduplication",
    )

    # ── Generation ────────────────────────────────────────────────────────────
    confidence_threshold: float = Field(
        default=0.35,
        description="Composite confidence below which the system refuses to answer",
    )
    max_context_chunks: int = Field(
        default=5,
        description="Maximum chunks included in the generation context",
    )
    llm_temperature: float = Field(default=0.1, description="LLM generation temperature")
    llm_max_tokens: int = Field(default=1024, description="LLM max output tokens")
    llm_retry_attempts: int = Field(default=3, description="Retry attempts for LLM calls")
    llm_retry_wait: float = Field(default=2.0, description="Base wait seconds between retries")

    # ── Confidence weights (must sum to ~1.0) ─────────────────────────────────
    confidence_weight_retrieval: float = Field(default=0.4, description="Weight: retrieval score")
    confidence_weight_coverage: float = Field(default=0.4, description="Weight: citation coverage")
    confidence_weight_completeness: float = Field(
        default=0.2,
        description="Weight: answer completeness",
    )

    # ── Caching ───────────────────────────────────────────────────────────────
    cache_dir: Path = Field(default=ROOT / ".cache" / "hybrid_rag", description="Disk-cache dir")
    embed_cache_enabled: bool = Field(default=True, description="Cache embeddings by chunk hash")
    llm_cache_enabled: bool = Field(default=True, description="Cache LLM calls by input hash")

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_raw_dir: Path = Field(default=ROOT / "data" / "raw")
    data_processed_dir: Path = Field(default=ROOT / "data" / "processed")
    golden_path: Path = Field(default=ROOT / "data" / "golden" / "golden.jsonl")
    bm25_index_dir: Path = Field(default=ROOT / ".chroma" / "bm25")
    reports_dir: Path = Field(default=ROOT / "reports")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    def model_post_init(self, __context: object) -> None:
        """Ensure all directories exist."""
        for attr in (
            "cache_dir",
            "data_raw_dir",
            "data_processed_dir",
            "bm25_index_dir",
            "reports_dir",
        ):
            path = getattr(self, attr)
            path.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()
