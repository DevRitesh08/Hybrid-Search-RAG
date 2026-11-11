"""Index build orchestrator for the Hybrid RAG pipeline.

Ties together:
 1. Document loading from ``data/processed/``
 2. Chunking (strategy-selected)
 3. Embedding (batch)
 4. Deduplication
 5. ChromaDB upsert
 6. BM25 build + save
 7. Corpus stats report

Usage (CLI)
-----------
::

    uv run python -m hybrid_rag.index.build --strategy structural
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from hybrid_rag.config import settings
from hybrid_rag.index.bm25_index import BM25Index
from hybrid_rag.index.embedder import get_embedding_service
from hybrid_rag.index.vector_store import VectorStore
from hybrid_rag.models import Chunk, Document

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_documents(processed_dir: Path) -> list[Document]:
    """Load all normalised documents from ``data/processed/``.

    Checks first for ``documents.jsonl`` (the standard output of normalizer),
    then falls back to scanning for ``.json`` files.

    Parameters
    ----------
    processed_dir:
        Directory to scan for document files.

    Returns
    -------
    list[Document]
    """
    docs: list[Document] = []
    jsonl_file = processed_dir / "documents.jsonl"
    if jsonl_file.exists():
        try:
            with jsonl_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        docs.append(Document.model_validate_json(line))
            logger.info("Loaded %d documents from %s", len(docs), jsonl_file)
            return docs
        except Exception as exc:
            logger.warning("Failed to load %s: %s", jsonl_file, exc)

    json_files = sorted(processed_dir.rglob("*.json"))
    for fp in json_files:
        try:
            with fp.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Accepts both a single document dict or a list of dicts
            if isinstance(data, list):
                for item in data:
                    docs.append(Document.model_validate(item))
            else:
                docs.append(Document.model_validate(data))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", fp, exc)

    if not docs:
        logger.warning("No documents found in %s", processed_dir)

    logger.info("Loaded %d documents from %s", len(docs), processed_dir)
    return docs


def _get_chunker(strategy: str) -> Any:
    """Return the appropriate chunker callable for a given strategy.

    The returned callable must accept a :class:`~hybrid_rag.models.Document`
    and return ``list[Chunk]``.

    Parameters
    ----------
    strategy:
        One of ``fixed``, ``structural``, ``semantic``.

    Returns
    -------
    Callable[[Document], list[Chunk]]

    Raises
    ------
    ImportError
        If the chunking module for the requested strategy is not yet
        implemented.
    ValueError
        If ``strategy`` is not recognised.
    """
    strat = strategy.lower()
    if strat == "fixed":
        try:
            from hybrid_rag.chunking.fixed import chunk_document

            return chunk_document
        except ImportError as exc:
            raise ImportError(
                "Fixed chunker not found at hybrid_rag.chunking.fixed"
            ) from exc

    if strat == "structural":
        try:
            from hybrid_rag.chunking.structural import chunk_document

            return chunk_document
        except ImportError as exc:
            raise ImportError(
                "Structural chunker not found at hybrid_rag.chunking.structural"
            ) from exc

    if strat == "semantic":
        try:
            from hybrid_rag.chunking.semantic import chunk_document

            return chunk_document
        except ImportError as exc:
            raise ImportError(
                "Semantic chunker not found at hybrid_rag.chunking.semantic"
            ) from exc

    raise ValueError(
        f"Unknown chunking strategy: '{strategy}'. "
        "Must be one of: fixed, structural, semantic."
    )


def _deduplicate(chunks: list[Chunk], embeddings: dict[str, Any]) -> tuple[list[Chunk], int]:
    """Deduplicate chunks using the dedup module if available.

    Falls back to returning all chunks unchanged when the dedup module has not
    yet been implemented.

    Parameters
    ----------
    chunks:
        All chunks before deduplication.
    embeddings:
        Mapping of ``chunk_id`` → embedding vector.

    Returns
    -------
    tuple[list[Chunk], int]
        ``(deduplicated_chunks, removed_count)``
    """
    try:
        from hybrid_rag.ingest.dedup import deduplicate

        kept, skipped = deduplicate(chunks, embeddings)
        return kept, len(skipped)
    except ImportError:
        logger.debug("hybrid_rag.ingest.dedup not available — skipping deduplication.")
        return chunks, 0


def _write_stats_report(
    report_path: Path,
    strategy: str,
    doc_count: int,
    chunk_count: int,
    dedup_removed: int,
    avg_char_count: float,
    elapsed_s: float,
) -> None:
    """Write a Markdown corpus-stats report.

    Parameters
    ----------
    report_path:
        Destination file path.
    strategy:
        Chunking strategy used.
    doc_count:
        Number of source documents.
    chunk_count:
        Final chunk count after deduplication.
    dedup_removed:
        Number of duplicate chunks removed.
    avg_char_count:
        Average character count per chunk.
    elapsed_s:
        Total wall-clock seconds for the build.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Corpus Statistics\n",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Strategy | `{strategy}` |",
        f"| Documents | {doc_count:,} |",
        f"| Chunks (after dedup) | {chunk_count:,} |",
        f"| Duplicates removed | {dedup_removed:,} |",
        f"| Avg chars / chunk | {avg_char_count:,.1f} |",
        f"| Build time | {elapsed_s:.1f}s |",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Stats report written to %s", report_path)


# ── Main build function ───────────────────────────────────────────────────────


def build_indexes(strategy: str) -> dict[str, Any]:
    """Build vector and BM25 indexes for the given chunking strategy.

    Steps
    -----
    1. Load normalised documents from ``data/processed/``.
    2. Chunk each document with the selected strategy.
    3. Embed all chunks in batch.
    4. Deduplicate via cosine similarity.
    5. Upsert embeddings into ChromaDB.
    6. Build and save BM25 index to JSON.
    7. Write ``reports/corpus_stats.md``.

    Parameters
    ----------
    strategy:
        One of ``fixed``, ``structural``, ``semantic``.

    Returns
    -------
    dict[str, Any]
        Stats dict with keys: ``strategy``, ``doc_count``, ``chunk_count``,
        ``dedup_removed``, ``avg_char_count``, ``elapsed_s``.
    """
    t0 = time.monotonic()

    # ── 1. Try to import rich.progress; fall back gracefully ─────────────────
    try:
        from rich.progress import (  # noqa: PLC0415
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        _rich_available = True
    except ImportError:
        _rich_available = False

    logger.info("=== Building indexes for strategy: %s ===", strategy)

    # ── 2. Load documents ─────────────────────────────────────────────────────
    docs = _load_documents(settings.data_processed_dir)
    if not docs:
        logger.warning("No documents found — aborting index build.")
        return {
            "strategy": strategy,
            "doc_count": 0,
            "chunk_count": 0,
            "dedup_removed": 0,
            "avg_char_count": 0.0,
            "elapsed_s": time.monotonic() - t0,
        }

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    chunker = _get_chunker(strategy)
    all_chunks: list[Chunk] = []

    if _rich_available:
        from rich.progress import (  # noqa: PLC0415
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Chunking documents …", total=len(docs))
            for doc in docs:
                chunks = chunker(doc)
                all_chunks.extend(chunks)
                progress.advance(task)
    else:
        for i, doc in enumerate(docs, 1):
            chunks = chunker(doc)
            all_chunks.extend(chunks)
            if i % 10 == 0 or i == len(docs):
                logger.info("  chunked %d/%d documents …", i, len(docs))

    logger.info("Total chunks before dedup: %d", len(all_chunks))

    # ── 4. Embed ──────────────────────────────────────────────────────────────
    embedder = get_embedding_service()

    if _rich_available:
        from rich.progress import (  # noqa: PLC0415
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Embedding chunks …", total=len(all_chunks))
            # Embed in batches to update the progress bar
            embeddings: dict[str, Any] = {}
            batch_size = 256
            for start in range(0, len(all_chunks), batch_size):
                batch = all_chunks[start : start + batch_size]
                batch_embs = embedder.embed_chunks(batch)
                embeddings.update(batch_embs)
                progress.advance(task, len(batch))
    else:
        logger.info("Embedding %d chunks …", len(all_chunks))
        embeddings = embedder.embed_chunks(all_chunks)

    # ── 5. Deduplicate ────────────────────────────────────────────────────────
    all_chunks, dedup_removed = _deduplicate(all_chunks, embeddings)
    logger.info(
        "After deduplication: %d chunks (%d removed)", len(all_chunks), dedup_removed
    )

    # ── 6. ChromaDB upsert ────────────────────────────────────────────────────
    vector_store = VectorStore(strategy=strategy)

    if _rich_available:
        from rich.progress import (  # noqa: PLC0415
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Upserting into ChromaDB …", total=None)
            added = vector_store.add_chunks(all_chunks, embeddings)
            progress.update(task, description=f"ChromaDB: {added} chunks upserted")
    else:
        added = vector_store.add_chunks(all_chunks, embeddings)

    logger.info("ChromaDB: %d chunks in collection '%s'", added, vector_store.get_collection_name())

    # ── 7. BM25 build + save ──────────────────────────────────────────────────
    bm25 = BM25Index()
    bm25.build(all_chunks)

    bm25_path = settings.bm25_index_dir / f"bm25_{strategy}.json"
    bm25.save(bm25_path)
    logger.info("BM25 index saved to %s", bm25_path)

    # ── 8. Stats ──────────────────────────────────────────────────────────────
    avg_char = (
        sum(c.char_count for c in all_chunks) / len(all_chunks) if all_chunks else 0.0
    )
    elapsed = time.monotonic() - t0

    report_path = settings.reports_dir / "corpus_stats.md"
    _write_stats_report(
        report_path=report_path,
        strategy=strategy,
        doc_count=len(docs),
        chunk_count=len(all_chunks),
        dedup_removed=dedup_removed,
        avg_char_count=avg_char,
        elapsed_s=elapsed,
    )

    stats: dict[str, Any] = {
        "strategy": strategy,
        "doc_count": len(docs),
        "chunk_count": len(all_chunks),
        "dedup_removed": dedup_removed,
        "avg_char_count": round(avg_char, 1),
        "elapsed_s": round(elapsed, 2),
    }
    logger.info("Build complete: %s", stats)
    return stats


# ── CLI entry point ───────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hybrid_rag.index.build",
        description="Build vector (ChromaDB) and sparse (BM25) indexes.",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "structural", "semantic"],
        default=settings.chunk_strategy,
        help="Chunking strategy to use (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    stats = build_indexes(strategy=args.strategy)
    print(json.dumps(stats, indent=2))
    sys.exit(0)
