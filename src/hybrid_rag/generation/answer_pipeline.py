"""Backward-compatible re-export of canonical generation pipeline."""

from __future__ import annotations

from hybrid_rag.generation.pipeline import RAGPipeline, run_generation_pipeline

__all__ = ["RAGPipeline", "run_generation_pipeline"]
