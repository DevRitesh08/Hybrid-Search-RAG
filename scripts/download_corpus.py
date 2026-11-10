"""Download pinned slice of Hugging Face Transformers + PEFT Markdown documentation."""

from __future__ import annotations

import logging

from hybrid_rag.config import settings
from hybrid_rag.ingest.fetch import fetch_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    manifest_path = settings.data_raw_dir / "MANIFEST.json"
    manifest = fetch_corpus(raw_dir=settings.data_raw_dir, manifest_path=manifest_path)
    t_count = manifest.get("transformers", {}).get("file_count", 0)
    p_count = manifest.get("peft", {}).get("file_count", 0)
    print(f"\n[OK] Download complete! Transformers: {t_count}, PEFT: {p_count}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
