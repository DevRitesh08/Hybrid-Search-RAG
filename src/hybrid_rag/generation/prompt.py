"""Prompt templates for the generation layer.

All LLM-facing string templates live here so they are easy to audit and edit
without touching business logic.
"""

from __future__ import annotations

from hybrid_rag.models import RetrievedChunk

# ── Canonical Refusal Message ─────────────────────────────────────────────────

CANONICAL_REFUSAL_MESSAGE = (
    "I don't have enough information in the provided context to answer this question."
)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise technical assistant for HuggingFace documentation.

Rules:
1. Answer ONLY using the numbered context blocks provided below.
2. Cite sources using [N] notation where N matches the context block number (e.g. [1], [2]).
3. If multiple blocks support a claim, cite all of them using grouped brackets [1][2], comma-separated brackets [1, 2], or ranges [1-3].
4. If the context is insufficient or unhelpful to answer the question, say exactly: "I don't have enough information in the provided context to answer this question."
5. Do not speculate, extrapolate, or use knowledge outside the provided context blocks.
6. Keep answers concise, factual, and strictly grounded in the context."""

# ── Context / QA builders ─────────────────────────────────────────────────────


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format a list of retrieved chunks as numbered context blocks.

    Each block follows the pattern::

        [1] Source: path/to/doc.md > Heading

        chunk text here

    Args:
        chunks: Ordered list of retrieved chunks (rank 1 = most relevant).

    Returns:
        A single string with all context blocks joined by newlines.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source_label = (
            chunk.source
            or chunk.metadata.get("source_path")
            or chunk.metadata.get("source")
            or chunk.doc_id
            or "unknown"
        )
        heading_part = f" > {chunk.heading_path}" if chunk.heading_path else ""
        header = f"[{i}] Source: {source_label}{heading_part}"
        parts.append(f"{header}\n\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def build_qa_prompt(question: str, context: str) -> str:
    """Compose the final user-turn prompt sent to the LLM.

    Args:
        question: The user's natural-language question.
        context:  The pre-formatted context block string from
                  :func:`build_context_block`.

    Returns:
        The full prompt string ready to pass to the LLM.
    """
    return (
        f"Context:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer (cite using [N] notation):"
    )


# ── Judge prompts ─────────────────────────────────────────────────────────────

CITATION_VERIFY_PROMPT = """Does the following text chunk SUPPORT the claim?

Claim: {claim}

Text chunk:
{chunk_text}

Answer with exactly one of:
- SUPPORTED: <brief reason>
- NOT_SUPPORTED: <brief reason>"""

COMPLETENESS_PROMPT = """Question: {question}

Answer: {answer}

Does this answer address all parts of the question? Answer:
- COMPLETE: <reason>
- PARTIAL: <what is missing>
- INCOMPLETE: <what is missing>"""
