"""Text normalisation for the Hybrid RAG ingest pipeline.

Cleans raw document text while carefully preserving code fences and
heading structure.  Exposes ``normalize(doc)`` for single-document use
and a ``__main__`` entry-point for batch processing.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from hybrid_rag.config import settings
from hybrid_rag.models import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# YAML front matter — strip if still present (loaders may have already done it)
_FRONT_MATTER_RE = re.compile(r"^\s*---\s*\n.*?\n---\s*\n", re.DOTALL)

# Markdown code fence sentinel (opening or closing ```) or ~~~
_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)

# Heading lines: # through ######
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)

# Boilerplate lines to remove (case-insensitive)
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*edit\s+this\s+page\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*was\s+this\s+(page\s+)?helpful\??\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*yes\s*[\|/]\s*no\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*on\s+this\s+page\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*table\s+of\s+contents\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*copy\s+link\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*jump\s+to\s+content\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*skip\s+to\s+(main\s+)?content\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*back\s+to\s+top\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\[edit\]\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*suggest\s+an\s+edit\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*view\s+source\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*permalink\s*$", re.IGNORECASE | re.MULTILINE),
]

# Collapse 3+ consecutive blank lines to exactly one blank line
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_code_blocks(text: str) -> list[tuple[str, bool]]:
    """Split *text* into alternating prose/code segments.

    Returns a list of ``(segment, is_code)`` pairs where ``is_code=True``
    means the segment is inside a fenced code block and must not be touched.
    """
    segments: list[tuple[str, bool]] = []
    in_code = False
    fence_char: str = ""
    fence_len: int = 0
    current: list[str] = []

    for line in text.splitlines(keepends=True):
        m = _CODE_FENCE_RE.match(line)
        if m:
            opener = m.group(1)
            if not in_code:
                # Opening fence
                segments.append(("".join(current), False))
                current = [line]
                in_code = True
                fence_char = opener[0]
                fence_len = len(opener)
            else:
                # Closing fence must match opening fence character and be at
                # least as long.
                if opener[0] == fence_char and len(opener) >= fence_len:
                    current.append(line)
                    segments.append(("".join(current), True))
                    current = []
                    in_code = False
                    fence_char = ""
                    fence_len = 0
                else:
                    current.append(line)
        else:
            current.append(line)

    if current:
        segments.append(("".join(current), in_code))  # unclosed fence → treat as code

    return segments


def _extract_headings(text: str) -> list[str]:
    """Return a flat list of heading texts in document order."""
    return [m.group(2).strip() for m in _HEADING_RE.finditer(text)]


def _clean_prose(prose: str) -> str:
    """Apply all normalisation steps to a prose (non-code) segment."""
    # Strip front matter (safety net — loaders may already have done this)
    prose = _FRONT_MATTER_RE.sub("", prose)

    # Remove boilerplate lines
    for pat in _BOILERPLATE_PATTERNS:
        prose = pat.sub("", prose)

    # Collapse multiple blank lines
    prose = _MULTI_BLANK_RE.sub("\n\n", prose)

    return prose


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize(doc: Document) -> Document:
    """Normalise *doc* and return a new ``Document`` with cleaned text.

    The original ``Document`` is not mutated.  The returned document has
    the same ``doc_id`` and ``source_path`` but an updated ``text`` and
    an enriched ``metadata`` dict containing a ``"headings"`` key.

    Parameters
    ----------
    doc:
        Input document as loaded by :mod:`hybrid_rag.ingest.loaders`.

    Returns
    -------
    Document
        Normalised document with ``metadata["headings"]`` populated.
    """
    text = doc.text

    # Split into prose/code segments first to prevent code comments from leaking into headings
    segments = _split_code_blocks(text)

    # Extract headings ONLY from prose segments
    headings: list[str] = []
    cleaned_parts: list[str] = []
    for segment, is_code in segments:
        if is_code:
            cleaned_parts.append(segment)
        else:
            headings.extend(_extract_headings(segment))
            cleaned_parts.append(_clean_prose(segment))

    cleaned_text = "".join(cleaned_parts)

    # Final whitespace normalisation at document level
    cleaned_text = _MULTI_BLANK_RE.sub("\n\n", cleaned_text).strip()

    updated_meta = dict(doc.metadata)
    updated_meta["headings"] = headings

    return Document(
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        title=doc.title,
        text=cleaned_text,
        metadata=updated_meta,
    )


# ---------------------------------------------------------------------------
# Batch normalization & CLI entry-point
# ---------------------------------------------------------------------------


def batch_normalize(
    raw_dir: Path,
    processed_dir: Path,
) -> int:
    """Normalise every document in *raw_dir* and write JSONL to *processed_dir*.

    Returns the count of successfully processed and written documents.
    """
    # Import here to avoid circular dependency at module level
    from hybrid_rag.ingest.loaders import load_corpus  # noqa: PLC0415

    documents = load_corpus(raw_dir)
    if not documents:
        logger.warning("No documents found in %s", raw_dir)
        return 0

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "documents.jsonl"

    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            norm = normalize(doc)
            fh.write(norm.model_dump_json() + "\n")
            written += 1

    logger.info("Wrote %d normalised documents to %s", written, output_path)
    return written


def main() -> None:
    """CLI entry-point for normalization."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    raw_dir = settings.data_raw_dir
    processed_dir = settings.data_processed_dir

    if len(sys.argv) >= 3:
        raw_dir = Path(sys.argv[1])
        processed_dir = Path(sys.argv[2])

    count = batch_normalize(raw_dir, processed_dir)
    print(f"Batch normalization complete: {count} documents processed.")


if __name__ == "__main__":
    main()

