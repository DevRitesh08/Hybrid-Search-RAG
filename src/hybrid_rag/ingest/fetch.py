"""Corpus fetcher and fallback documentation generator."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from hybrid_rag.config import settings

logger = logging.getLogger(__name__)

TRANSFORMERS_COMMIT = "8b17b2b5120f2c41864f1d4f291588647e30c498"
PEFT_COMMIT = "e2d4d98a063b46990e75a96ca735e581283626c0"


def download_github_tree(repo: str, commit: str, subpath: str, out_dir: Path) -> int:
    """Download Markdown files from a GitHub repository subfolder at a pinned commit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"
    headers = {"User-Agent": "HybridRAG-Docs-Downloader"}

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(api_url, headers=headers)
            if resp.status_code != 200:
                logger.warning(
                    "GitHub API returned %d for %s. Using fallback docs.",
                    resp.status_code,
                    repo,
                )
                return 0

            tree = resp.json().get("tree", [])
            md_files = [
                item
                for item in tree
                if item["type"] == "blob"
                and item["path"].startswith(subpath)
                and item["path"].endswith(".md")
            ]
            md_files = md_files[:200]
            downloaded = 0
            for item in tqdm(md_files, desc=f"Downloading {repo}"):
                raw_url = (
                    f"https://raw.githubusercontent.com/{repo}/{commit}/{item['path']}"
                )
                r = client.get(raw_url, headers=headers)
                if r.status_code == 200:
                    rel_path = item["path"][len(subpath) :].lstrip("/\\")
                    file_dest = out_dir / rel_path
                    file_dest.parent.mkdir(parents=True, exist_ok=True)
                    file_dest.write_text(r.text, encoding="utf-8")
                    downloaded += 1
            return downloaded
    except Exception as exc:
        logger.warning("Network download failed for %s: %s", repo, exc)
        return 0


def create_fallback_corpus(out_dir: Path) -> dict[str, int]:
    """Generate minimal offline fallback Markdown docs if network is unavailable."""
    sample_docs = {
        "transformers/quickstart.md": (
            "# Quickstart with Transformers\n\n"
            "The `pipeline` function is the easiest way to use a pre-trained model for inference.\n"
            "You can use `pipeline` for tasks like text classification, named entity recognition, and question answering.\n\n"
            "```python\n"
            "from transformers import pipeline\n"
            "classifier = pipeline('sentiment-analysis')\n"
            "classifier('We are very happy to show you the 🤗 Transformers library.')\n"
            "```\n\n"
            "The `AutoModel` and `AutoTokenizer` classes allow you to load any supported pre-trained model with `from_pretrained`.\n"
        ),
        "transformers/model_doc/auto.md": (
            "# AutoModel Class Reference\n\n"
            "The `AutoModel` class is a generic model class that will be instantiated as one of the base model classes of the library when created with the `AutoModel.from_pretrained()` class method.\n\n"
            "Parameters:\n"
            "- `pretrained_model_name_or_path`: Shortcut name or path to checkpoint directory.\n"
            "- `trust_remote_code`: Whether or not to allow custom code defined in the model repository.\n"
            "- `torch_dtype`: Override default torch dtype.\n"
        ),
        "peft/lora.md": (
            "# Low-Rank Adaptation (LoRA)\n\n"
            "LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture.\n\n"
            "Key parameters in `LoraConfig`:\n"
            "- `r`: The rank of the update matrices (int, default 8).\n"
            "- `target_modules`: The names of the modules to apply LoRA to, e.g. `['q_proj', 'v_proj']`.\n"
            "- `lora_alpha`: The alpha parameter for LoRA scaling (int, default 8).\n"
            "- `lora_dropout`: The dropout probability for LoRA layers (float, default 0.0).\n"
            "- `bias`: Bias type for LoRA ('none', 'all', or 'lora_only').\n"
        ),
        "peft/peft_model.md": (
            "# PeftModel\n\n"
            "The base class for PEFT models. Wraps a base transformer model and applies parameter-efficient adapters.\n\n"
            "```python\n"
            "from peft import get_peft_model, LoraConfig, TaskType\n"
            "peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=['q_proj', 'v_proj'])\n"
            "model = get_peft_model(base_model, peft_config)\n"
            "```\n"
        ),
    }

    counts = {"transformers": 0, "peft": 0}
    for rel_path, content in sample_docs.items():
        dest = out_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content.strip(), encoding="utf-8")
        repo_prefix = rel_path.split("/")[0]
        counts[repo_prefix] = counts.get(repo_prefix, 0) + 1
    return counts


def fetch_corpus(
    raw_dir: Path | None = None,
    manifest_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch the documentation corpus or generate fallback documents."""
    raw_dir = raw_dir or settings.data_raw_dir
    manifest_path = manifest_path or (raw_dir / "MANIFEST.json")

    if manifest_path.exists() and not force:
        logger.info("Corpus already present. Manifest: %s", manifest_path)
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    raw_dir.mkdir(parents=True, exist_ok=True)
    t_dir = raw_dir / "transformers"
    p_dir = raw_dir / "peft"

    t_count = download_github_tree(
        "huggingface/transformers",
        TRANSFORMERS_COMMIT,
        "docs/source/en",
        t_dir,
    )
    p_count = download_github_tree(
        "huggingface/peft",
        PEFT_COMMIT,
        "docs/source",
        p_dir,
    )

    if t_count == 0 or p_count == 0:
        fallback_counts = create_fallback_corpus(raw_dir)
        if t_count == 0:
            t_count = fallback_counts.get("transformers", 0)
        if p_count == 0:
            p_count = fallback_counts.get("peft", 0)

    manifest = {
        "transformers": {
            "repo": "huggingface/transformers",
            "commit": TRANSFORMERS_COMMIT,
            "file_count": t_count,
        },
        "peft": {
            "repo": "huggingface/peft",
            "commit": PEFT_COMMIT,
            "file_count": p_count,
        },
        "downloaded_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
