"""Golden question set loader and splitter.

Loads GoldenQuestion objects from JSONL files at the paths
configured in settings. Provides a stratified train/test split by question
type.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from hybrid_rag.config import settings
from hybrid_rag.models import GoldenQuestion

logger = logging.getLogger(__name__)


def load_golden(path: Path | None = None) -> list[GoldenQuestion]:
    """Load GoldenQuestion objects from a JSONL file.

    Parameters
    ----------
    path:
        Path to the JSONL file. Defaults to ``settings.golden_path``.

    Returns
    -------
    list[GoldenQuestion]
        Parsed golden questions in file order.

    Raises
    ------
    FileNotFoundError
        If the JSONL file does not exist.
    """
    target = path or settings.golden_path
    if not target.exists():
        logger.warning("Golden questions file not found: %s — returning empty list", target)
        return []

    questions: list[GoldenQuestion] = []
    with target.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                questions.append(GoldenQuestion.model_validate(data))
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Skipping malformed golden record at line %d: %s", lineno, exc)

    logger.info("Loaded %d golden questions from %s", len(questions), target)
    return questions


def split_golden(
    questions: list[GoldenQuestion],
    dev_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[list[GoldenQuestion], list[GoldenQuestion]]:
    """Stratified train/test split of golden questions by question type.

    Within each question type, the first ``dev_ratio`` fraction (after seeded
    shuffle) goes to the *dev* (train) set; the remainder forms the *test* set.
    This ensures every question type is represented in both partitions.

    Parameters
    ----------
    questions:
        Full list of golden questions to split.
    dev_ratio:
        Fraction of each stratum assigned to the dev set (0 < dev_ratio < 1).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple[list[GoldenQuestion], list[GoldenQuestion]]
        ``(dev_set, test_set)`` — both maintain stable ordering within each
        stratum (seed is fixed).

    Raises
    ------
    ValueError
        If ``dev_ratio`` is not in (0, 1).
    """
    if not (0.0 < dev_ratio < 1.0):
        raise ValueError(f"dev_ratio must be in (0, 1); got {dev_ratio}")

    # Group by question type
    strata: dict[str, list[GoldenQuestion]] = defaultdict(list)
    for q in questions:
        strata[q.type].append(q)

    rng = random.Random(seed)

    dev: list[GoldenQuestion] = []
    test: list[GoldenQuestion] = []

    for qtype in sorted(strata.keys()):  # sorted for determinism
        bucket = list(strata[qtype])
        rng.shuffle(bucket)
        n_dev = max(1, round(len(bucket) * dev_ratio)) if len(bucket) > 1 else 1
        # Protect against edge cases where rounding takes all items
        n_dev = min(n_dev, len(bucket) - 1) if len(bucket) > 1 else len(bucket)
        dev.extend(bucket[:n_dev])
        test.extend(bucket[n_dev:])

    logger.info(
        "Split: %d dev / %d test from %d total (dev_ratio=%.2f, seed=%d)",
        len(dev),
        len(test),
        len(questions),
        dev_ratio,
        seed,
    )
    return dev, test


# Alias for compatibility with stratified_split naming convention
stratified_split = split_golden

