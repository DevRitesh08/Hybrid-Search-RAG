"""ChromaDB vector-store wrapper for the Hybrid RAG pipeline.

One ChromaDB collection per chunking strategy, named:
    ``{settings.chroma_collection_prefix}_{strategy}``

Supports local persistent storage and HTTP client modes.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from hybrid_rag.config import settings
from hybrid_rag.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

# Chroma is imported lazily so the module can be imported without chromadb
# installed (e.g. in test environments that only need BM25).
try:
    import chromadb  # noqa: F401
    _CHROMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed — VectorStore will raise on use.")


class _NoOpEmbeddingFunction:
    """Explicit dummy embedding function to prevent ChromaDB default ONNX download."""

    def __call__(self, input: Any) -> list[Any]:
        return []

    @staticmethod
    def name() -> str:
        return "default"


class VectorStore:
    """Thin, lazy-initialised wrapper around a single ChromaDB collection.

    Parameters
    ----------
    strategy:
        Chunking strategy name (``fixed`` | ``structural`` | ``semantic``).
        Determines the collection name.
    """

    def __init__(self, strategy: str) -> None:
        self._strategy = strategy
        self._client: Any | None = None
        self._collection: Any | None = None
        self._lock = threading.RLock()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Return (or lazily create) the ChromaDB client."""
        if not _CHROMA_AVAILABLE:
            raise RuntimeError(
                "chromadb is not installed. Install with: pip install chromadb"
            )
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import chromadb  # noqa: PLC0415
                    import chromadb.config  # noqa: PLC0415

                    chroma_conf = chromadb.config.Settings(anonymized_telemetry=False)
                    if settings.chroma_mode == "http":
                        self._client = chromadb.HttpClient(
                            host=settings.chroma_host,
                            port=settings.chroma_port,
                            settings=chroma_conf,
                        )
                        logger.info(
                            "ChromaDB HTTP client connected to %s:%s",
                            settings.chroma_host,
                            settings.chroma_port,
                        )
                    else:
                        settings.chroma_path.mkdir(parents=True, exist_ok=True)
                        self._client = chromadb.PersistentClient(
                            path=str(settings.chroma_path),
                            settings=chroma_conf,
                        )
                        logger.info(
                            "ChromaDB PersistentClient at %s", settings.chroma_path
                        )
        return self._client

    def _get_collection(self) -> Any:
        """Return (or lazily create) the ChromaDB collection."""
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    client = self._get_client()
                    name = self.get_collection_name()
                    self._collection = client.get_or_create_collection(
                        name=name,
                        metadata={"hnsw:space": "cosine"},
                        embedding_function=_NoOpEmbeddingFunction(),
                    )
                    logger.info("ChromaDB collection ready: %s", name)
        return self._collection

    @staticmethod
    def _chunk_metadata(chunk: Chunk) -> dict[str, Any]:
        """Extract storable scalar metadata from a Chunk."""
        source_val = str(
            chunk.metadata.get("source")
            or chunk.metadata.get("source_path")
            or chunk.doc_id
        )
        return {
            "doc_id": chunk.doc_id,
            "source": source_val,
            "source_path": source_val,
            "heading_path": chunk.heading_path,
            "strategy": chunk.strategy,
            "position": chunk.position,
            "char_count": chunk.char_count,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def get_collection_name(self) -> str:
        """Return the ChromaDB collection name for this strategy.

        Returns
        -------
        str
            E.g. ``hybrid_rag_structural``.
        """
        return f"{settings.chroma_collection_prefix}_{self._strategy}"

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: dict[str, np.ndarray],
    ) -> int:
        """Upsert chunks with their embeddings into the collection.

        Parameters
        ----------
        chunks:
            List of Chunk objects to store.
        embeddings:
            Mapping of ``chunk_id`` → embedding vector (must cover all chunks).

        Returns
        -------
        int
            Number of chunks actually upserted.
        """
        if not chunks:
            return 0

        collection = self._get_collection()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        emb_list: list[list[float]] = []

        for chunk in chunks:
            vec = embeddings.get(chunk.chunk_id)
            if vec is None:
                logger.warning(
                    "No embedding found for chunk %s — skipping.", chunk.chunk_id
                )
                continue
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(self._chunk_metadata(chunk))
            emb_list.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))

        if not ids:
            return 0

        # Upsert in batches of 500 to stay within Chroma limits
        batch_size = 500
        total = 0
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=emb_list[start:end],
                metadatas=metadatas[start:end],
            )
            total += len(ids[start:end])

        logger.info("Upserted %d chunks into %s", total, self.get_collection_name())
        return total

    def query(self, embedding: np.ndarray, top_k: int = 10) -> list[RetrievedChunk]:
        """Dense nearest-neighbour retrieval.

        Parameters
        ----------
        embedding:
            Query vector (shape ``(dim,)``), must be L2-normalised.
        top_k:
            Number of results to return.

        Returns
        -------
        list[RetrievedChunk]
            Ranked results (highest similarity first).
        """
        collection = self._get_collection()
        n_results = min(top_k, self.count())
        if n_results == 0:
            return []

        response = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievedChunk] = []
        ids = response["ids"][0]
        docs = response["documents"][0]
        metas = response["metadatas"][0]
        distances = response["distances"][0]

        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, distances, strict=False), start=1
        ):
            # Chroma returns cosine *distance* (0=identical, 2=opposite).
            # Convert to similarity score in [0, 1].
            score = float(1.0 - dist / 2.0)
            source = meta.get("source") or meta.get("source_path") or ""
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id", ""),
                    text=doc,
                    heading_path=meta.get("heading_path", ""),
                    source=source,
                    score=score,
                    rank=rank,
                    retrieval_type="dense",
                    metadata=meta,
                )
            )
        return results

    def delete_by_doc(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document.

        Parameters
        ----------
        doc_id:
            The document ID whose chunks should be removed.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        collection = self._get_collection()
        # First, find all chunk IDs for this doc
        existing = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids_to_delete = existing["ids"]
        if not ids_to_delete:
            return 0
        collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d chunks for doc_id=%s from %s",
            len(ids_to_delete),
            doc_id,
            self.get_collection_name(),
        )
        return len(ids_to_delete)

    def count(self) -> int:
        """Return the total number of chunks in the collection.

        Returns
        -------
        int
        """
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Delete and recreate the ChromaDB collection for this strategy."""
        with self._lock:
            if _CHROMA_AVAILABLE:
                try:
                    client = self._get_client()
                    name = self.get_collection_name()
                    client.delete_collection(name=name)
                    logger.info("Reset collection: %s", name)
                except Exception as exc:
                    logger.debug("Collection reset skipped or not found: %s", exc)
                self._collection = None
