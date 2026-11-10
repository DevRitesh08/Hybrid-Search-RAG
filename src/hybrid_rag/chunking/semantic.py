"""Semantic similarity chunker.

Splits text on sentences and groups consecutive sentences whose embedding similarity
stays above a given threshold. Falls back to sentence length splitting when no embedder is given.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np

from hybrid_rag.chunking.base import BaseChunker, ChunkerConfig
from hybrid_rag.models import Chunk, Document


class SemanticChunker(BaseChunker):
    """Chunks documents based on semantic similarity between consecutive sentences."""

    strategy: str = "semantic"

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        embedder: Callable[[list[str]], Any] | None = None,
        similarity_threshold: float = 0.75,
    ) -> None:
        super().__init__(config)
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences while preserving code blocks."""
        fence_pattern = re.compile(
            r"^(`{3,}|~{3,})[^\n]*\n.*?\n\1[^\n]*$",
            re.MULTILINE | re.DOTALL,
        )

        segments: list[tuple[bool, str]] = []
        last_idx = 0
        for m in fence_pattern.finditer(text):
            if m.start() > last_idx:
                segments.append((False, text[last_idx:m.start()]))
            segments.append((True, m.group(0)))
            last_idx = m.end()
        if last_idx < len(text):
            segments.append((False, text[last_idx:]))

        results: list[str] = []
        sentence_re = re.compile(
            r"(?<!\d\.)(?<!\d\d\.)(?<!\d\d\d\.)(?<=[.!?])\s+|\n(?=\s*\d+\.\s+)"
        )
        for is_code, seg in segments:
            if is_code:
                s = seg.strip()
                if s:
                    results.append(s)
            else:
                for sent in sentence_re.split(seg):
                    s = sent.strip()
                    if s:
                        results.append(s)
        return results

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two 1D vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def _is_code_block(text: str) -> bool:
        """Check if text segment is a fenced code block."""
        s = text.strip()
        return (
            (s.startswith("```") and s.endswith("```") and len(s) >= 6)
            or (s.startswith("~~~") and s.endswith("~~~") and len(s) >= 6)
        )

    def _join_sentences(self, sentences: list[str]) -> str:
        """Join sentences into chunk text, separating code blocks with double newlines."""
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0]

        parts: list[str] = [sentences[0]]
        for i in range(1, len(sentences)):
            prev = sentences[i - 1]
            curr = sentences[i]
            sep = "\n\n" if (self._is_code_block(prev) or self._is_code_block(curr)) else " "
            parts.append(sep)
            parts.append(curr)
        return "".join(parts)

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split document into semantically cohesive chunks."""
        text = doc.text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # If embedder is available, compute similarities
        sentence_embeddings: list[np.ndarray] = []
        if self.embedder is not None and len(sentences) > 1:
            try:
                emb_array = self.embedder(sentences)
                sentence_embeddings = [np.asarray(emb_array[i]) for i in range(len(sentences))]
            except Exception:
                sentence_embeddings = []

        chunks: list[Chunk] = []
        current_sentences: list[str] = [sentences[0]]
        current_length = len(sentences[0])
        position = 0

        max_chars = self.config.chunk_size * 4  # rough token-to-char conversion

        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_len = len(sent)

            split_needed = False

            # Semantic split check if embeddings exist
            if sentence_embeddings and i < len(sentence_embeddings):
                sim = self._cosine_similarity(sentence_embeddings[i - 1], sentence_embeddings[i])
                if sim < self.similarity_threshold:
                    split_needed = True

            # Hard length limit check
            if current_length + sent_len > max_chars:
                split_needed = True

            if split_needed and current_length >= self.config.min_size:
                chunk_text = self._join_sentences(current_sentences)
                chunks.append(
                    self._make_chunk(
                        doc=doc,
                        text=chunk_text,
                        position=position,
                        heading_path="",
                    )
                )
                position += 1
                current_sentences = [sent]
                current_length = sent_len
            else:
                current_sentences.append(sent)
                sep_len = 2 if (self._is_code_block(current_sentences[-2]) or self._is_code_block(sent)) else 1
                current_length += sent_len + sep_len

        if current_sentences:
            chunk_text = self._join_sentences(current_sentences)
            if len(chunk_text) >= self.config.min_size:
                chunks.append(
                    self._make_chunk(
                        doc=doc,
                        text=chunk_text,
                        position=position,
                        heading_path="",
                    )
                )

        return chunks


def chunk_document(
    doc: Document,
    config: ChunkerConfig | None = None,
    embedder: Callable[[list[str]], Any] | None = None,
) -> list[Chunk]:
    """Split *doc* using SemanticChunker."""
    return SemanticChunker(config, embedder=embedder).chunk(doc)
