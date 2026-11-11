"""Embedding service for the Hybrid RAG pipeline.

Wraps SentenceTransformer (BAAI/bge-small-en-v1.5) with:
- Disk-based embedding cache (diskcache)
- Batch embedding with configurable batch size
- L2-normalised vectors (cosine ≡ dot product)
- Thread-safe singleton via double-checked lock
- BGE query prefix for query-time embedding
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from hybrid_rag.config import settings
from hybrid_rag.models import Chunk

if TYPE_CHECKING:
    import diskcache

logger = logging.getLogger(__name__)

# BGE instruction prefix for query embedding
_BGE_QUERY_PREFIX = "Represent this sentence: "

_DEFAULT_BATCH_SIZE = 64


class EmbeddingService:
    """Thread-safe singleton that wraps SentenceTransformer with disk caching.

    Attributes
    ----------
    model_name:
        HuggingFace model identifier (default from settings).
    batch_size:
        Number of texts embedded in a single forward pass.
    """

    _instance: EmbeddingService | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: object, **kwargs: object) -> EmbeddingService:  # noqa: D102
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialise (idempotent — only runs once due to singleton pattern)."""
        if getattr(self, "_initialised", False):
            return

        self.model_name: str = model_name or settings.embed_model
        self.batch_size: int = batch_size
        self._model: SentenceTransformer | None = None
        self._cache: diskcache.Cache | None = None
        self._model_lock = threading.Lock()
        self._initialised = True

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model (thread-safe)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    logger.info("Loading embedding model: %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
                    logger.info("Embedding model loaded.")
        return self._model

    def _get_cache(self) -> diskcache.Cache | None:
        """Lazy-load diskcache if caching is enabled."""
        if not settings.embed_cache_enabled:
            return None
        if self._cache is None:
            try:
                import diskcache  # noqa: PLC0415

                cache_path = settings.cache_dir / "embeddings"
                cache_path.mkdir(parents=True, exist_ok=True)
                self._cache = diskcache.Cache(str(cache_path))
                logger.debug("Embedding cache opened at %s", cache_path)
            except ImportError:
                logger.warning(
                    "diskcache not installed — embedding cache disabled. "
                    "Install with: pip install diskcache"
                )
        return self._cache

    @staticmethod
    def _text_hash(text: str) -> str:
        """Return a stable SHA-256 hex digest for a piece of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise a batch of row vectors in-place."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        return vectors / norms

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts and return L2-normalised vectors.

        Results are looked up / stored in the disk cache when caching is
        enabled.  Cache hits skip the model entirely.

        Parameters
        ----------
        texts:
            Arbitrary list of strings to embed.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), embedding_dim)``, dtype float32.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        cache = self._get_cache()
        model = self._get_model()

        dim = model.get_sentence_embedding_dimension() or 384
        result = np.zeros((len(texts), dim), dtype=np.float32)

        # Separate cache hits from misses
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if cache is not None:
                key = self._text_hash(text)
                cached = cache.get(key)
                if cached is not None:
                    result[i] = cached
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        if uncached_texts:
            logger.debug("Embedding %d texts (batch_size=%d)", len(uncached_texts), self.batch_size)
            raw = model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
            raw = self._normalise(raw.astype(np.float32))

            for local_i, global_i in enumerate(uncached_indices):
                result[global_i] = raw[local_i]
                if cache is not None:
                    key = self._text_hash(texts[global_i])
                    cache.set(key, raw[local_i])

        return result

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string with the BGE query instruction prefix.

        Parameters
        ----------
        text:
            Raw query text.

        Returns
        -------
        np.ndarray
            Shape ``(embedding_dim,)``, dtype float32.
        """
        prefixed = f"{_BGE_QUERY_PREFIX}{text}"
        vectors = self.embed([prefixed])
        return vectors[0]

    def embed_chunks(self, chunks: list[Chunk]) -> dict[str, np.ndarray]:
        """Embed a list of Chunk objects and return a mapping of chunk_id → vector.

        Parameters
        ----------
        chunks:
            Chunks to embed (uses ``chunk.text`` as the input).

        Returns
        -------
        dict[str, np.ndarray]
            Maps each ``chunk_id`` to its L2-normalised embedding vector.
        """
        if not chunks:
            return {}

        texts = [chunk.text for chunk in chunks]
        vectors = self.embed(texts)
        return {chunk.chunk_id: vectors[i] for i, chunk in enumerate(chunks)}


# Module-level convenience singleton
_service: EmbeddingService | None = None
_service_lock = threading.Lock()


def get_embedding_service(
    model_name: str | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> EmbeddingService:
    """Return the module-level EmbeddingService singleton.

    Parameters
    ----------
    model_name:
        Override the model name (only respected on first call).
    batch_size:
        Override the batch size (only respected on first call).

    Returns
    -------
    EmbeddingService
        The singleton instance.
    """
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = EmbeddingService(model_name=model_name, batch_size=batch_size)
    return _service
