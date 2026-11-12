"""Generation module — prompt engineering, LLM adapters, citations, verification, and confidence."""

from __future__ import annotations

from hybrid_rag.generation.answer import generate_answer
from hybrid_rag.generation.citations import (
    extract_citation_spans,
    parse_citation_numbers,
    parse_citations,
)
from hybrid_rag.generation.confidence import (
    check_completeness,
    compute_confidence,
    compute_coverage_score,
    compute_retrieval_score,
    is_refusal_required,
)
from hybrid_rag.generation.llm import (
    GeminiAdapter,
    LLMAdapter,
    get_llm,
    reset_llm_registry,
)
from hybrid_rag.generation.pipeline import RAGPipeline, run_generation_pipeline
from hybrid_rag.generation.prompt import (
    CANONICAL_REFUSAL_MESSAGE,
    CITATION_VERIFY_PROMPT,
    COMPLETENESS_PROMPT,
    SYSTEM_PROMPT,
    build_context_block,
    build_qa_prompt,
)
from hybrid_rag.generation.verify_citations import (
    clear_verify_cache,
    compute_citation_metrics,
    negative_control_check,
    verify_citations,
    verify_single_citation,
)

__all__ = [
    # Prompts
    "CANONICAL_REFUSAL_MESSAGE",
    "CITATION_VERIFY_PROMPT",
    "COMPLETENESS_PROMPT",
    "SYSTEM_PROMPT",
    "build_context_block",
    "build_qa_prompt",
    # LLM
    "GeminiAdapter",
    "LLMAdapter",
    "get_llm",
    "reset_llm_registry",
    # Citations
    "extract_citation_spans",
    "parse_citation_numbers",
    "parse_citations",
    # Answer & Verification
    "clear_verify_cache",
    "compute_citation_metrics",
    "generate_answer",
    "negative_control_check",
    "verify_citations",
    "verify_single_citation",
    # Confidence
    "check_completeness",
    "compute_confidence",
    "compute_coverage_score",
    "compute_retrieval_score",
    "is_refusal_required",
    # Pipeline
    "RAGPipeline",
    "run_generation_pipeline",
]
