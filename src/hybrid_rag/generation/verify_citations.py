"""Citation verification — verifies that cited chunks support the claims made."""

from __future__ import annotations

import hashlib
import random
import re
from typing import TYPE_CHECKING, Any

from hybrid_rag.config import settings
from hybrid_rag.generation.citations import (
    _CITATION_PATTERN,
    extract_citation_spans,
    extract_sentence_spans,
)
from hybrid_rag.generation.llm import get_llm
from hybrid_rag.generation.prompt import CITATION_VERIFY_PROMPT
from hybrid_rag.models import Citation, RetrievedChunk

if TYPE_CHECKING:
    from hybrid_rag.generation.llm import LLMAdapter

# In-memory verifier cache to avoid redundant LLM calls
_VERIFY_CACHE: dict[str, tuple[bool, str]] = {}


def clear_verify_cache() -> None:
    """Clear in-memory citation verification cache (used in testing)."""
    _VERIFY_CACHE.clear()


def _split_into_claims(text: str) -> list[str]:
    """Split answer into individual sentence-level claims.

    Strips citation bracket syntax before splitting to maintain sentence integrity.
    """
    if not text:
        return []
    # Remove citation brackets like [1], [1, 2], [1-3]
    cleaned = _CITATION_PATTERN.sub("", text)
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", cleaned)
    claims: list[str] = []
    for s in sentences:
        s_clean = s.strip()
        s_clean = re.sub(r"\s+([.!?])", r"\1", s_clean)
        s_clean = s_clean.rstrip(".!?").strip()
        if len(s_clean) > 8:
            claims.append(s_clean)
    return claims


def verify_single_citation(
    claim: str,
    chunk_text: str,
    llm: LLMAdapter | None = None,
) -> tuple[bool, str]:
    """Verify whether a text chunk supports a given claim.

    Uses a full-text SHA256 digest in the cache key to avoid collisions
    on shared documentation headers.
    """
    chunk_digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    cache_key = hashlib.sha256(f"{claim}::{chunk_digest}".encode()).hexdigest()

    if cache_key in _VERIFY_CACHE:
        return _VERIFY_CACHE[cache_key]

    if llm is None:
        llm = get_llm(model_name=settings.judge_model)

    prompt = CITATION_VERIFY_PROMPT.format(
        claim=claim,
        chunk_text=chunk_text[:1200],
    )

    try:
        response = llm.generate(prompt=prompt).strip()
        upper_resp = response.upper()
        if "SUPPORTED" in upper_resp and "NOT_SUPPORTED" not in upper_resp:
            res = (True, response)
        else:
            res = (False, response)
    except Exception as exc:
        res = (False, f"Verification failed: {exc}")

    _VERIFY_CACHE[cache_key] = res
    return res


def compute_citation_metrics(
    citations: list[Citation],
    claims: list[str] | None = None,
    answer: str = "",
) -> dict[str, Any]:
    """Compute precision and recall metrics for citations.

    * Precision: share of citations that were verified (verified_citations / total_citations).
    * Recall: share of distinct sentence claims backed by at least one verified citation.

    Args:
        citations: List of extracted citations.
        claims: Optional pre-split claim sentences.
        answer: Raw answer string used to extract claims if not provided.

    Returns:
        Dictionary containing precision, recall, and count statistics.
    """
    if claims is None:
        claims = _split_into_claims(answer) if answer else []

    total_citations = len(citations)
    verified_citations = sum(1 for c in citations if c.verified)
    precision = verified_citations / total_citations if total_citations > 0 else 1.0

    # Recall: fraction of claims backed by verified citations
    total_claims = len(claims)
    if total_claims == 0:
        recall = 1.0 if verified_citations > 0 or total_citations == 0 else 0.0
        verified_claims_count = 0
    else:
        verified_claims_set = {
            c.claim_text.strip()
            for c in citations
            if c.verified and c.claim_text and c.claim_text.strip()
        }
        # Count how many split claims match or are covered by verified citations
        matched = 0
        for claim in claims:
            c_clean = claim.strip()
            if any(c_clean in vc or vc in c_clean for vc in verified_claims_set):
                matched += 1
        verified_claims_count = matched
        recall = verified_claims_count / total_claims

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "verified_citations": verified_citations,
        "total_citations": total_citations,
        "verified_claims": verified_claims_count,
        "total_claims": total_claims,
    }


def verify_citations(
    answer: str,
    citations: list[Citation],
    chunks: list[RetrievedChunk],
    llm: LLMAdapter | None = None,
) -> list[Citation]:
    """Verify each citation in the list against its cited chunk and associated claim.

    Associates each citation with its sentence-level claim rather than the
    entire answer to avoid false rejections.
    """
    if not citations or not chunks:
        return citations

    chunk_map = {c.chunk_id: c for c in chunks}
    claims = _split_into_claims(answer)
    full_claim_text = " ".join(claims) if claims else answer

    # Pre-index sentence claims and their parsed citation numbers for bracket-aware fallback
    # Uses extract_sentence_spans to eliminate drift across multi-paragraph answers
    sentence_spans = extract_sentence_spans(answer)
    sentence_records: list[tuple[str, set[int]]] = []
    for _, _, s in sentence_spans:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_numbers: set[int] = set()
        for span in extract_citation_spans(s_clean):
            s_numbers.update(span["numbers"])
        clean_claim = _CITATION_PATTERN.sub("", s_clean).strip()
        clean_claim = re.sub(r"\s+([.!?])", r"\1", clean_claim)
        sentence_records.append((clean_claim, s_numbers))

    verified_citations: list[Citation] = []
    for citation in citations:
        chunk = chunk_map.get(citation.chunk_id)
        if chunk is None:
            citation.verified = False
            citation.verification_reason = "Cited chunk not found in retrieved results"
            verified_citations.append(citation)
            continue

        # Determine granular claim for this citation using bracket-aware fallback
        claim = citation.claim_text
        if not claim:
            for clean_claim, numbers in sentence_records:
                if citation.n in numbers:
                    claim = clean_claim
                    break
            if not claim:
                claim = full_claim_text
            citation.claim_text = claim

        is_supported, reason = verify_single_citation(
            claim=claim,
            chunk_text=chunk.text,
            llm=llm,
        )
        citation.verified = is_supported
        citation.verification_reason = reason
        verified_citations.append(citation)

    return verified_citations


def negative_control_check(
    claims: list[str],
    random_chunks: list[RetrievedChunk],
    llm: LLMAdapter | None = None,
) -> float:
    """Negative control: test claims against unrelated chunks.

    Support rate should be near zero.
    """
    if not claims or not random_chunks:
        return 0.0

    supported_count = 0
    total = min(len(claims), len(random_chunks), 10)

    for i in range(total):
        claim = claims[i]
        chunk = random.choice(random_chunks)
        supported, _ = verify_single_citation(claim, chunk.text, llm)
        if supported:
            supported_count += 1

    return supported_count / total if total > 0 else 0.0
