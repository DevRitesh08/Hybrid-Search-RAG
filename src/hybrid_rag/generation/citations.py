"""Bracket citation parser ([1], [1][2], [1, 2], [1-3]).

Implements robust parsing of inline bracket citations, range expansion,
code-syntax filtering, and mapping to retrieved context chunks.
"""

from __future__ import annotations

import re
from typing import Any

from hybrid_rag.models import Citation, RetrievedChunk

# Regex matching citation brackets: [1], [1, 2], [1-3], [1–3], [1, 2-4]
# Negative lookbehind (?<![A-Za-z0-9_]) prevents matching code indexing/types (e.g. arr[0], List[str])
_CITATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])\[(\s*\d+(?:\s*(?:,\s*|\s*[-–]\s*)\s*\d+)*\s*)\]"
)
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def extract_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Extract sentence spans from text preserving exact character offsets.

    Uses sentence-ending punctuation followed by whitespace or paragraph breaks
    as boundaries, preventing identifier splitting (e.g. 0.75, AutoModel.from_pretrained)
    and avoiding offset drift across multi-paragraph or varied-whitespace texts.

    Args:
        text: Input text.

    Returns:
        List of (start_offset, end_offset, sentence_text) tuples.
    """
    spans: list[tuple[int, int, str]] = []
    last_end = 0
    for match in _SENTENCE_BOUNDARY_PATTERN.finditer(text):
        start = last_end
        end = match.start()
        if end > start:
            spans.append((start, end, text[start:end]))
        last_end = match.end()
    if last_end < len(text):
        spans.append((last_end, len(text), text[last_end:]))
    return spans


def parse_citation_numbers(bracket_content: str) -> list[int]:
    """Parse inner bracket content into a sorted list of unique chunk indices.

    Supports:
    * Single number: ``"1"`` -> ``[1]``
    * Comma-separated: ``"1, 2, 3"`` -> ``[1, 2, 3]``
    * Range: ``"1-3"`` or ``"1–3"`` -> ``[1, 2, 3]``
    * Mixed: ``"1, 3-5, 8"`` -> ``[1, 3, 4, 5, 8]``

    Args:
        bracket_content: The string inside the citation brackets.

    Returns:
        Sorted list of integer citation indices.
    """
    numbers: set[int] = set()

    for part in bracket_content.split(","):
        part = part.strip()
        if not part:
            continue

        range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start <= end and (end - start) <= 100:
                numbers.update(range(start, end + 1))
            elif start > end:
                numbers.update([start, end])
        elif part.isdigit():
            numbers.add(int(part))

    return sorted(numbers)


def extract_citation_spans(text: str) -> list[dict[str, Any]]:
    """Extract all citation brackets with their character offsets and parsed indices.

    Args:
        text: Input text containing bracket citations.

    Returns:
        List of dicts with keys: ``raw``, ``start``, ``end``, ``numbers``.
    """
    spans: list[dict[str, Any]] = []
    for match in _CITATION_PATTERN.finditer(text):
        inner = match.group(1)
        numbers = parse_citation_numbers(inner)
        if numbers:
            spans.append(
                {
                    "raw": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "numbers": numbers,
                }
            )
    return spans


def parse_citations(text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Extract citation numbers [N] from text and map to retrieved chunks.

    Supports:
    * Single citations: ``[1]``
    * Grouped consecutive citations: ``[1][2]``
    * Comma-separated citations: ``[1, 2]``
    * Range citations: ``[1-3]`` -> ``[1, 2, 3]``
    * Rejects code indexing (e.g. ``arr[0]``) and generics (e.g. ``List[str]``).

    Args:
        text: LLM-generated answer containing citation brackets.
        chunks: List of retrieved chunks provided in context (1-based rank).

    Returns:
        List of :class:`Citation` objects mapped to valid chunks in numerical order.
    """
    if not text or not chunks:
        return []

    spans = extract_citation_spans(text)
    if not spans:
        return []

    chunk_map = {i + 1: chunk for i, chunk in enumerate(chunks)}

    # Map citations to their enclosing sentence claims using exact offset spans
    sentence_spans = extract_sentence_spans(text)

    # Track distinct (citation_number, claim_text) pairs preserving text order
    seen_pairs: set[tuple[int, str]] = set()
    citation_pairs: list[tuple[int, str]] = []

    for span in spans:
        start_pos = span["start"]
        containing_sentence = ""
        for s_start, s_end, s in sentence_spans:
            if s_start <= start_pos <= s_end:
                clean_s = _CITATION_PATTERN.sub("", s).strip()
                clean_s = re.sub(r"\s+([.!?])", r"\1", clean_s)
                containing_sentence = clean_s
                break

        if not containing_sentence:
            containing_sentence = _CITATION_PATTERN.sub("", text).strip()
            containing_sentence = re.sub(r"\s+([.!?])", r"\1", containing_sentence)

        for num in span["numbers"]:
            pair = (num, containing_sentence)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                citation_pairs.append(pair)

    citations: list[Citation] = []
    # Sort primarily by citation number n (stable sort preserves appearance order for ties)
    for n, claim_text in sorted(citation_pairs, key=lambda p: p[0]):
        if n in chunk_map:
            c = chunk_map[n]
            source = (
                c.source
                or c.metadata.get("source_path")
                or c.metadata.get("source")
                or ""
            )
            citations.append(
                Citation(
                    n=n,
                    chunk_id=c.chunk_id,
                    source=source,
                    heading=c.heading_path,
                    claim_text=claim_text,
                    verified=False,
                )
            )

    return citations
