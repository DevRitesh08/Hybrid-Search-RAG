"""Document ingestion, normalization, and deduplication for Hybrid Search RAG."""

from hybrid_rag.ingest.dedup import (
    _cosine_similarity,
    cosine_similarity,
    deduplicate,
)
from hybrid_rag.ingest.fetch import (
    PEFT_COMMIT,
    TRANSFORMERS_COMMIT,
    create_fallback_corpus,
    download_github_tree,
    fetch_corpus,
)
from hybrid_rag.ingest.loaders import load_corpus, load_document
from hybrid_rag.ingest.normalize import batch_normalize, normalize

__all__ = [
    "PEFT_COMMIT",
    "TRANSFORMERS_COMMIT",
    "_cosine_similarity",
    "batch_normalize",
    "cosine_similarity",
    "create_fallback_corpus",
    "deduplicate",
    "download_github_tree",
    "fetch_corpus",
    "load_corpus",
    "load_document",
    "normalize",
]
