"""Answer generation with context assembly and citation extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hybrid_rag.config import settings
from hybrid_rag.generation.citations import parse_citations
from hybrid_rag.generation.llm import get_llm
from hybrid_rag.generation.prompt import (
    CANONICAL_REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    build_context_block,
    build_qa_prompt,
)
from hybrid_rag.models import Citation, RetrievedChunk

if TYPE_CHECKING:
    from hybrid_rag.generation.llm import LLMAdapter

__all__ = ["generate_answer", "parse_citations"]


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    llm: LLMAdapter | None = None,
) -> tuple[str, list[Citation]]:
    """Generate an answer using retrieved chunks as numbered context."""
    if not chunks:
        return (
            CANONICAL_REFUSAL_MESSAGE,
            [],
        )

    if llm is None:
        llm = get_llm()

    # Limit to max configured context chunks
    context_chunks = chunks[: settings.max_context_chunks]
    context_text = build_context_block(context_chunks)
    prompt = build_qa_prompt(question, context_text)

    try:
        answer_text = llm.generate(prompt=prompt, system=SYSTEM_PROMPT)
    except Exception as e:
        answer_text = f"Error generating answer: {e}"

    citations = parse_citations(answer_text, context_chunks)
    return answer_text, citations
