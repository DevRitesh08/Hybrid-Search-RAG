"""Heading-aware structural chunker.

Splits Markdown documents on heading boundaries, tracks the full heading
path, and optionally splits oversized sections with
``RecursiveCharacterTextSplitter``.
"""

from __future__ import annotations

import logging
import re

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

# Matches any ATX heading: ``# Title``, ``## Sub``, etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$", re.MULTILINE)

# Separators for the fallback RecursiveCharacterTextSplitter
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]


def _find_code_fence_spans(text: str) -> list[tuple[int, int]]:
    """Return [start, end) character spans of all fenced code blocks.

    Complies with CommonMark specifications:
    - Opening fence: sequence of N >= 3 backticks (`) or tildes (~).
    - Closing fence: sequence of M >= N matching fence characters with no info string.
    - Inner fences of shorter length (e.g. 3 backticks inside 4 backticks) do not
      prematurely close the outer block.
    """
    spans: list[tuple[int, int]] = []
    fence_pattern = re.compile(r"^(`{3,}|~{3,})([^\n]*)$", re.MULTILINE)
    in_fence_char: str | None = None
    in_fence_len: int = 0
    fence_start = -1

    for m in fence_pattern.finditer(text):
        fence_chars = m.group(1)
        rest_of_line = m.group(2).strip()
        char_type = fence_chars[0]
        char_len = len(fence_chars)

        if in_fence_char is None:
            in_fence_char = char_type
            in_fence_len = char_len
            fence_start = m.start()
        elif char_type == in_fence_char and char_len >= in_fence_len and not rest_of_line:
            line_end = text.find("\n", m.end())
            fence_end = len(text) if line_end == -1 else line_end + 1
            spans.append((fence_start, fence_end))
            in_fence_char = None
            in_fence_len = 0
            fence_start = -1

    if in_fence_char is not None and fence_start != -1:
        spans.append((fence_start, len(text)))

    return spans


def _build_heading_path(stack: list[tuple[int, str]]) -> str:
    """Format the heading stack as a `` > ``-separated path string."""
    return " > ".join(title for _, title in stack)


def _update_heading_stack(
    stack: list[tuple[int, str]],
    level: int,
    title: str,
) -> list[tuple[int, str]]:
    """Return a new stack with *title* at *level* appended correctly.

    Pops all entries at the same or deeper level before pushing.
    """
    new_stack = [(lvl, t) for lvl, t in stack if lvl < level]
    new_stack.append((level, title))
    return new_stack


def _split_section(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split a section text that is too long using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,
    )
    return splitter.split_text(text)


class StructuralChunker(BaseChunker):
    """Split documents on Markdown headings.

    Each top-level section becomes one chunk.  Sections that exceed
    ``chunk_size * 1.5`` characters are further split with
    :class:`langchain_text_splitters.RecursiveCharacterTextSplitter`.

    The full heading hierarchy is captured and stored in
    ``Chunk.heading_path``.  Each chunk's text is prepended with the
    heading path in the format::

        [H1 > H2 > H3]

        …chunk body…

    Parameters
    ----------
    config:
        :class:`~hybrid_rag.chunking.base.ChunkerConfig` instance.
    """

    strategy: str = "structural"

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        super().__init__(config)
        self._oversized_threshold = int(self.config.chunk_size * 1.5)

    # ------------------------------------------------------------------
    # BaseChunker interface
    # ------------------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* on heading boundaries.

        Parameters
        ----------
        doc:
            Normalised ``Document`` to chunk.

        Returns
        -------
        list[Chunk]
            Ordered, non-empty chunks with ``heading_path`` set.
        """
        if not doc.text.strip():
            logger.warning("Document %s is empty — no chunks produced.", doc.doc_id)
            return []

        sections = self._extract_sections(doc.text)
        chunks: list[Chunk] = []
        position = 0

        for heading_path, body in sections:
            body = body.strip()
            if not body:
                continue

            # Prepend heading path to the chunk text
            prefix = f"[{heading_path}]\n\n" if heading_path else ""

            # If oversized, split body and prepend prefix to every sub-chunk
            if len(body) > self._oversized_threshold:
                sub_parts = _split_section(
                    body,
                    self.config.chunk_size,
                    self.config.chunk_overlap,
                )
            else:
                sub_parts = [body]

            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                full_text = f"{prefix}{part}" if prefix else part
                if len(full_text) < self.config.min_size:
                    logger.debug(
                        "Dropping structural sub-chunk (len=%d) from doc %s.",
                        len(full_text),
                        doc.doc_id,
                    )
                    continue
                chunks.append(
                    self._make_chunk(
                        doc=doc,
                        text=full_text,
                        position=position,
                        heading_path=heading_path,
                    )
                )
                position += 1

        # Fallback: if heading parsing produced nothing, treat the entire
        # document as a single section.
        if not chunks:
            logger.debug(
                "No headings found in doc %s; treating as single section.", doc.doc_id
            )
            full_text = doc.text.strip()
            if len(full_text) >= self.config.min_size:
                chunks.append(
                    self._make_chunk(
                        doc=doc,
                        text=full_text,
                        position=0,
                        heading_path="",
                    )
                )

        logger.debug(
            "Structural chunker produced %d chunks for doc %s.", len(chunks), doc.doc_id
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_sections(self, text: str) -> list[tuple[str, str]]:
        """Parse the text into ``(heading_path, body)`` pairs.

        Returns a list in document order.  Content before the first
        heading is returned with an empty heading path.
        """
        sections: list[tuple[str, str]] = []
        heading_stack: list[tuple[int, str]] = []  # [(level, title), ...]

        fence_spans = _find_code_fence_spans(text)

        # Find all heading positions that are outside code fences
        all_matches = list(_HEADING_RE.finditer(text))
        matches = [
            m for m in all_matches
            if not any(start <= m.start() < end for start, end in fence_spans)
        ]

        if not matches:
            # No headings at all — entire document is one section
            return [("", text)]

        # Content before the first heading
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for i, match in enumerate(matches):
            level = len(match.group(1))  # number of `#` characters
            title = match.group(2).strip()

            heading_stack = _update_heading_stack(heading_stack, level, title)
            heading_path = _build_heading_path(heading_stack)

            # Body = text from end of this heading line to start of the next
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end]

            sections.append((heading_path, body))

        return sections


def chunk_document(doc: Document, config: ChunkerConfig | None = None) -> list[Chunk]:
    """Split *doc* on heading boundaries using StructuralChunker."""
    return StructuralChunker(config).chunk(doc)
