"""Multi-format document loader for the Hybrid RAG ingest pipeline.

Supports Markdown (primary), plain text, and HTML files.
Recursively scans directories and returns ``Document`` objects.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from hybrid_rag.models import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional markdown-it-py for HTML stripping
# ---------------------------------------------------------------------------
try:
    import markdown_it  # noqa: F401

    _HAS_MARKDOWN_IT = True
except ImportError:  # pragma: no cover
    _HAS_MARKDOWN_IT = False
    logger.warning("markdown-it-py not installed; HTML stripping will use a regex fallback.")

# Supported extensions
_MARKDOWN_EXTS = {".md", ".mdx", ".markdown"}
_TEXT_EXTS = {".txt", ".rst"}
_HTML_EXTS = {".html", ".htm"}

_SUPPORTED_EXTS = _MARKDOWN_EXTS | _TEXT_EXTS | _HTML_EXTS

# YAML front matter pattern (--- … ---)
_FRONT_MATTER_RE = re.compile(r"^\s*---\s*\n.*?\n---\s*\n", re.DOTALL)

# Heading-extraction pattern
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    """Read a file, trying UTF-8 first, falling back to latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.debug("UTF-8 decode failed for %s; retrying with latin-1.", path)
        return path.read_text(encoding="latin-1")


def _strip_front_matter(text: str) -> tuple[str, dict[str, str]]:
    """Remove YAML front matter and return (clean_text, raw_front_matter_dict).

    The returned dict is intentionally minimal; full YAML parsing is deferred
    to the normalizer so the loader stays dependency-light.
    """
    meta: dict[str, str] = {}
    match = _FRONT_MATTER_RE.match(text)
    if match:
        fm_block = match.group(0)
        text = text[match.end():]
        # Naive key: value extraction (good enough for titles, dates, etc.)
        for line in fm_block.splitlines():
            if ":" in line and not line.strip().startswith("-"):
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val:
                    meta[key] = val
    return text, meta


def _extract_title(text: str, path: Path) -> str:
    """Return the first H1 heading text, or the filename stem as fallback."""
    match = _HEADING_RE.search(text)
    if match:
        return match.group(1).strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text via markdown-it-py render + tag stripping."""
    if _HAS_MARKDOWN_IT:
        # markdown-it renders Markdown → HTML; for HTML input we just strip tags.
        pass  # fall through to regex stripper
    # Simple but robust HTML tag stripper
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#\d+;", " ", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_document(path: Path, base_dir: Path | None = None) -> Document:
    """Load a single document from *path* and return a ``Document``.

    Parameters
    ----------
    path:
        Absolute or relative ``Path`` to the file.  Must be one of the
        supported types (``.md``, ``.mdx``, ``.markdown``, ``.txt``,
        ``.rst``, ``.html``, ``.htm``).
    base_dir:
        Optional root directory to compute a relative POSIX ``source_path``.

    Returns
    -------
    Document
        A fully-populated ``Document`` with ``doc_id``, ``source_path``,
        ``title``, ``text``, and ``metadata``.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If *path* does not exist.
    """
    file_path = Path(path)
    resolved = file_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = resolved.suffix.lower()
    if suffix not in _SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type '{suffix}' for {path}")

    raw = _read_text(resolved)
    metadata: dict[str, object] = {}

    if suffix in _HTML_EXTS:
        text = _html_to_text(raw)
        title = _extract_title(text, resolved)

    elif suffix in _MARKDOWN_EXTS:
        text, fm_meta = _strip_front_matter(raw)
        metadata.update(fm_meta)
        # Front-matter title overrides heading-based title
        title = fm_meta.get("title", "") or _extract_title(text, resolved)

    else:  # plain text / rst
        text = raw
        title = _extract_title(text, resolved)

    if base_dir is not None:
        try:
            source_path = resolved.relative_to(Path(base_dir).resolve()).as_posix()
        except ValueError:
            source_path = file_path.as_posix()
    else:
        source_path = file_path.as_posix() if not file_path.is_absolute() else resolved.as_posix()

    doc_id = Document.make_id(source_path)

    return Document(
        doc_id=doc_id,
        source_path=source_path,
        title=title,
        text=text,
        metadata=metadata,
    )


def load_corpus(directory: Path) -> list[Document]:
    """Recursively load all supported documents under *directory*.

    Files with unsupported extensions are silently skipped.  Errors for
    individual files are logged but do not abort the full scan.

    Parameters
    ----------
    directory:
        Root directory to walk.

    Returns
    -------
    list[Document]
        All successfully loaded documents, in filesystem traversal order.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {directory}")

    documents: list[Document] = []
    skipped = 0

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in _SUPPORTED_EXTS:
            continue
        try:
            doc = load_document(file_path, base_dir=directory)
            documents.append(doc)
            logger.debug("Loaded %s (%d chars)", file_path.name, len(doc.text))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", file_path, exc)
            skipped += 1

    logger.info(
        "Corpus loaded: %d documents from %s (%d skipped).",
        len(documents),
        directory,
        skipped,
    )
    return documents
