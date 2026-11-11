"""BM25 sparse index for the Hybrid RAG pipeline.

Uses ``rank_bm25.BM25Okapi`` with a custom tokenizer designed to preserve
technical identifiers (snake_case, camelCase, dotted names) while removing
common English stopwords.

The index is serialised as JSON (not pickle) for portability and safety.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from hybrid_rag.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:  # type: ignore[no-redef]
        """Pure-Python fallback for BM25Okapi when rank-bm25 is not yet installed."""
        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
            self.k1 = k1
            self.b = b
            self.epsilon = epsilon
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size else 0
            self.doc_len = [len(doc) for doc in corpus]
            self.doc_lens = self.doc_len  # alias for backwards compatibility
            self.average_idf = 0.0
            self.doc_freqs: list[dict[str, int]] = []
            self.nd: dict[str, int] = {}
            for doc in corpus:
                freq: dict[str, int] = {}
                for token in doc:
                    freq[token] = freq.get(token, 0) + 1
                self.doc_freqs.append(freq)
                for token in freq:
                    self.nd[token] = self.nd.get(token, 0) + 1
            self.idf: dict[str, float] = {}
            for token, doc_count in self.nd.items():
                self.idf[token] = math.log((self.corpus_size - doc_count + 0.5) / (doc_count + 0.5) + 1)

        def get_scores(self, query: list[str]) -> list[float]:
            scores = [0.0] * self.corpus_size
            for token in query:
                if token not in self.idf:
                    continue
                idf = self.idf[token]
                for i, doc_freq in enumerate(self.doc_freqs):
                    if token in doc_freq:
                        freq = doc_freq[token]
                        denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1)))
                        scores[i] += idf * (freq * (self.k1 + 1)) / denom
            return scores

# ── Tokenizer ─────────────────────────────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset(
    [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
    ]
)

# Split on whitespace and punctuation EXCEPT underscores.
# This keeps snake_case and dot.separated.names intact.
_SPLIT_PATTERN = re.compile(r"[\s\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\`\{\|\}\~]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing.

    Rules
    -----
    * Lowercase all text.
    * Split on whitespace and punctuation, **excluding** underscores so that
      identifiers like ``target_modules`` and ``AutoModel`` survive intact.
    * Remove tokens in the minimal stopword list.
    * Drop empty tokens.

    Parameters
    ----------
    text:
        Raw string to tokenize.

    Returns
    -------
    list[str]
        Ordered list of lowercase tokens.
    """
    lowered = text.lower()
    raw_tokens = _SPLIT_PATTERN.split(lowered)
    return [t for t in raw_tokens if t and t not in _STOPWORDS]


# ── BM25 Index ────────────────────────────────────────────────────────────────


class BM25Index:
    """In-memory BM25 sparse index with JSON persistence.

    Usage
    -----
    ::

        idx = BM25Index()
        idx.build(chunks)
        results = idx.query("what is LoRA?", top_k=5)
        idx.save(Path("bm25_structural.json"))

        # Later:
        idx2 = BM25Index()
        idx2.load(Path("bm25_structural.json"))
        results2 = idx2.query("LoRA fine-tuning", top_k=5)
    """

    def __init__(self) -> None:
        self._bm25: Any | None = None
        self._chunk_meta: list[dict[str, Any]] = []  # ordered, parallel to corpus
        self._corpus: list[list[str]] = []  # tokenized corpus

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of Chunk objects.

        Parameters
        ----------
        chunks:
            All chunks to be indexed.  Order is preserved.
        """
        logger.info("Building BM25 index over %d chunks …", len(chunks))
        self._corpus = [tokenize(chunk.text) for chunk in chunks]
        self._chunk_meta = []
        for chunk in chunks:
            source_val = str(
                chunk.metadata.get("source")
                or chunk.metadata.get("source_path")
                or chunk.doc_id
            )
            self._chunk_meta.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "heading_path": chunk.heading_path,
                    "source": source_val,
                    "source_path": source_val,
                    "strategy": chunk.strategy,
                    "position": chunk.position,
                    "char_count": chunk.char_count,
                }
            )
        self._bm25 = BM25Okapi(self._corpus)
        logger.info("BM25 index built (%d documents).", len(chunks))

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks by BM25 score.

        Parameters
        ----------
        text:
            Raw query string (tokenized internally).
        top_k:
            Number of results to return.

        Returns
        -------
        list[RetrievedChunk]
            Ranked from highest BM25 score to lowest.  Returns an empty list
            when the index has not been built.
        """
        if self._bm25 is None:
            logger.warning("BM25 index is empty — call build() or load() first.")
            return []

        query_tokens = tokenize(text)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        scores: list[float] = (
            raw_scores.tolist() if hasattr(raw_scores, "tolist") else [float(s) for s in raw_scores]
        )

        # Pair (score, index) and sort descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]

        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(top, start=1):
            meta = self._chunk_meta[idx]
            source = meta.get("source") or meta.get("source_path") or ""
            results.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    doc_id=meta["doc_id"],
                    text=meta["text"],
                    heading_path=meta.get("heading_path", ""),
                    source=source,
                    score=float(score),
                    rank=rank,
                    retrieval_type="sparse",
                    metadata={
                        "strategy": meta.get("strategy", ""),
                        "position": meta.get("position", 0),
                        "char_count": meta.get("char_count", 0),
                        "source": source,
                        "source_path": source,
                    },
                )
            )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> Path:
        """Serialise the index to a JSON file.

        The serialisation format stores the tokenized corpus and chunk metadata
        so the index can be fully reconstructed without the original Chunk
        objects.

        Parameters
        ----------
        path:
            Destination file path (will be created / overwritten). Defaults to
            settings.bm25_index_dir / f"bm25_{settings.chunk_strategy}.json".
        """
        if self._bm25 is None:
            raise RuntimeError("Cannot save an empty BM25 index — call build() first.")

        from hybrid_rag.config import settings

        target_path = (
            Path(path)
            if path is not None
            else settings.bm25_index_dir / f"bm25_{settings.chunk_strategy}.json"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        doc_len_attr = getattr(self._bm25, "doc_len", getattr(self._bm25, "doc_lens", []))
        # Extract BM25Okapi hyper-parameters so we can restore them exactly
        bm25_params = {
            "k1": float(self._bm25.k1),
            "b": float(self._bm25.b),
            "epsilon": float(getattr(self._bm25, "epsilon", 0.25)),
            "avgdl": float(self._bm25.avgdl),
            "corpus_size": int(self._bm25.corpus_size),
            "doc_freqs": [
                {term: int(count) for term, count in doc.items()}
                for doc in self._bm25.doc_freqs
            ],
            "idf": {term: float(val) for term, val in self._bm25.idf.items()},
            "doc_len": [int(dl) for dl in doc_len_attr],
            "average_idf": float(getattr(self._bm25, "average_idf", 0.0)),
        }

        payload = {
            "version": 1,
            "bm25_params": bm25_params,
            "corpus": self._corpus,
            "chunk_meta": self._chunk_meta,
        }

        with target_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=None)

        logger.info("BM25 index saved to %s (%d chunks).", target_path, len(self._chunk_meta))
        return target_path

    def load(self, path: Path | str | None = None) -> None:
        """Restore the index from a previously saved JSON file.

        Parameters
        ----------
        path:
            Path to the JSON file written by :meth:`save`. Defaults to
            settings.bm25_index_dir / f"bm25_{settings.chunk_strategy}.json".
        """
        from hybrid_rag.config import settings

        target_path = (
            Path(path)
            if path is not None
            else settings.bm25_index_dir / f"bm25_{settings.chunk_strategy}.json"
        )
        if not target_path.exists():
            raise FileNotFoundError(f"BM25 index file not found at '{target_path}'.")

        logger.info("Loading BM25 index from %s …", target_path)
        with target_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._corpus = payload["corpus"]
        self._chunk_meta = payload["chunk_meta"]
        params = payload["bm25_params"]

        # Reconstruct BM25Okapi from stored parameters without re-tokenizing
        bm25 = BM25Okapi.__new__(BM25Okapi)
        bm25.k1 = params["k1"]
        bm25.b = params["b"]
        bm25.epsilon = params.get("epsilon", 0.25)
        bm25.avgdl = params["avgdl"]
        bm25.corpus_size = params["corpus_size"]
        bm25.doc_freqs = [dict(d) for d in params["doc_freqs"]]
        bm25.idf = dict(params["idf"])
        bm25.doc_len = params["doc_len"]
        bm25.doc_lens = params["doc_len"]
        bm25.average_idf = params.get("average_idf", 0.0)
        self._bm25 = bm25

        logger.info(
            "BM25 index loaded: %d chunks, vocab size %d.",
            len(self._chunk_meta),
            len(self._bm25.idf),
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    def chunk_ids(self) -> list[str]:
        """Return the ordered list of chunk IDs in the index.

        Returns
        -------
        list[str]
            Ordered the same as the corpus passed to :meth:`build`.
        """
        return [meta["chunk_id"] for meta in self._chunk_meta]

    def count(self) -> int:
        """Return the number of documents in the index.

        Returns
        -------
        int
        """
        return len(self._chunk_meta)
