"""Fixed-size character-level chunker.

Uses ``langchain_text_splitters.RecursiveCharacterTextSplitter`` with a
separator hierarchy that respects paragraph and sentence boundaries before
falling back to pure character splits.
"""

from __future__ import annotations

import logging

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
        def __init__(
            self,
            separators: list[str] | None = None,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            **kwargs: object,
        ) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> list[str]:
            chunks: list[str] = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start = max(end - self.chunk_overlap, start + 1)
            return chunks

from hybrid_rag.chunking.base import BaseChunker, ChunkerConfig
from hybrid_rag.models import Chunk, Document

logger = logging.getLogger(__name__)

# Separator hierarchy — coarser boundaries first so natural prose units
# are preferred; character-level ``""`` is the last resort.
_SEPARATORS: list[str] = [
    "\n\n",   # paragraph boundary
    "\n",     # line boundary
    ". ",     # sentence boundary (English)
    "! ",
    "? ",
    "; ",
    ", ",
    " ",      # word boundary
    "",       # character-level fallback
]


class FixedChunker(BaseChunker):
    """Split documents into fixed-size character chunks with overlap.

    Parameters
    ----------
    config:
        A :class:`~hybrid_rag.chunking.base.ChunkerConfig` controlling
        ``chunk_size``, ``chunk_overlap``, and ``min_size``.
    """

    strategy: str = "fixed"

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        super().__init__(config)
        self._splitter = RecursiveCharacterTextSplitter(
            separators=_SEPARATORS,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,
        )

    # ------------------------------------------------------------------
    # BaseChunker interface
    # ------------------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* into fixed-size chunks.

        The splitter is seeded with the document text only; it is fully
        deterministic — identical input always produces identical output.

        Parameters
        ----------
        doc:
            Normalised ``Document`` to chunk.

        Returns
        -------
        list[Chunk]
            Non-empty chunks in document order, each stamped with
            ``strategy="fixed"`` and a sequential ``position``.
        """
        if not doc.text.strip():
            logger.warning("Document %s has empty text — returning no chunks.", doc.doc_id)
            return []

        raw_chunks: list[str] = self._splitter.split_text(doc.text)

        chunks: list[Chunk] = []
        position = 0
        for raw in raw_chunks:
            text = raw.strip()
            if len(text) < self.config.min_size:
                logger.debug(
                    "Dropping fixed chunk (len=%d < min_size=%d) from doc %s.",
                    len(text),
                    self.config.min_size,
                    doc.doc_id,
                )
                continue

            chunks.append(
                self._make_chunk(
                    doc=doc,
                    text=text,
                    position=position,
                )
            )
            position += 1

        logger.debug(
            "Fixed chunker produced %d chunks for doc %s.", len(chunks), doc.doc_id
        )
        return chunks


def chunk_document(doc: Document, config: ChunkerConfig | None = None) -> list[Chunk]:
    """Split *doc* into fixed-size chunks using FixedChunker."""
    return FixedChunker(config).chunk(doc)

