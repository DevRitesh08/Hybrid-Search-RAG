"""LLM-as-judge for answer correctness and faithfulness.

Uses Gemini (via google-genai) with structured prompts to:
- Evaluate whether a generated answer is correct relative to a reference.
- Measure the fraction of answer claims supported by retrieved context.

Responses are cached by SHA-256 hash of the inputs to avoid redundant API
calls during evaluation runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hybrid_rag.config import settings
from hybrid_rag.models import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Verdict type ──────────────────────────────────────────────────────────────
Verdict = str  # "correct" | "incorrect" | "partial"

# ── On-disk cache ─────────────────────────────────────────────────────────────
_CACHE_FILE: Path = settings.cache_dir / "judge_cache.json"
_judge_cache: dict[str, tuple[str, str]] = {}
_cache_loaded = False


def _load_cache() -> None:
    """Load the judge cache from disk into memory (lazy, once per process)."""
    global _judge_cache, _cache_loaded
    if _cache_loaded:
        return
    if _CACHE_FILE.exists():
        try:
            raw = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            _judge_cache = {k: tuple(v) for k, v in raw.items()}
            logger.debug("Judge cache loaded: %d entries from %s", len(_judge_cache), _CACHE_FILE)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not load judge cache (%s) — starting fresh.", exc)
            _judge_cache = {}
    _cache_loaded = True


def _save_cache() -> None:
    """Persist the in-memory cache to disk."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(
            json.dumps(_judge_cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Failed to save judge cache: %s", exc)


def _cache_key(question: str, answer: str, reference: str) -> str:
    """Return a stable SHA-256 key for a (question, answer, reference) triple."""
    raw = f"{question}\x00{answer}\x00{reference}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── LLM helper ────────────────────────────────────────────────────────────────

def _get_llm(model_name: str | None = None) -> Any:
    """Return a Gemini GenerativeModel client.

    Tries google-genai first (the ``google.genai`` namespace used by the
    Gemini SDK ≥ 0.5), falls back to google-generativeai if available.
    """
    name = model_name or settings.judge_model
    try:
        import google.genai as genai

        client = genai.Client(api_key=settings.gemini_api_key)
        return client, name
    except ImportError:
        pass

    try:
        import google.generativeai as genai_legacy

        genai_legacy.configure(api_key=settings.gemini_api_key)
        return genai_legacy.GenerativeModel(name), None
    except ImportError as exc:
        raise ImportError(
            "Neither 'google-genai' nor 'google-generativeai' is installed. "
            "Install with: pip install google-genai"
        ) from exc


def _call_llm(prompt: str, llm=None, model_name: str | None = None) -> str:
    """Send a prompt to Gemini and return the text response."""
    if llm is None:
        client, name = _get_llm(model_name)
    else:
        # Accept pre-constructed clients
        client, name = llm, model_name or settings.judge_model

    try:
        # google-genai ≥ 0.5 API
        import google.genai as genai
        import google.genai.types as genai_types

        if isinstance(client, genai.Client):
            response = client.models.generate_content(
                model=name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            return response.text or ""
    except (ImportError, AttributeError):
        pass

    # Fallback: legacy generativeai
    response = client.generate_content(prompt)
    return response.text or ""


# ── System prompts ────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are a strict answer-correctness judge for a question-answering system.

Compare the GENERATED ANSWER to the REFERENCE ANSWER for the given QUESTION.
Return a JSON object with exactly two keys:
  "verdict": one of "correct", "partial", or "incorrect"
  "reason":  a single sentence explaining your verdict

Scoring rubric:
- "correct"   — the generated answer conveys the same key facts as the reference,
                 with minor differences in wording.
- "partial"   — the generated answer contains some correct information but misses
                 important facts or introduces minor inaccuracies.
- "incorrect" — the generated answer contradicts the reference or is substantially wrong.

Respond with valid JSON only. No markdown fences, no extra text."""

_FAITHFULNESS_SYSTEM = """You are evaluating the faithfulness of an AI-generated answer to a set of retrieved context chunks.

Your task: estimate what fraction of the factual claims in the ANSWER are directly supported by the CONTEXT.
Return a JSON object with exactly two keys:
  "score":   a float from 0.0 (no claims supported) to 1.0 (all claims supported)
  "reason":  one sentence explaining your score

Respond with valid JSON only. No markdown fences, no extra text."""


# ── Public API ────────────────────────────────────────────────────────────────


def judge_answer(
    question: str,
    answer: str,
    reference: str,
    llm=None,
) -> tuple[str, str]:
    """Evaluate whether *answer* correctly addresses *question* given *reference*.

    Results are cached by SHA-256 hash of ``(question, answer, reference)`` so
    repeated calls within or across runs are free.

    Parameters
    ----------
    question:
        The original question posed to the RAG system.
    answer:
        The generated answer to evaluate.
    reference:
        The ground-truth reference answer.
    llm:
        Optional pre-constructed LLM client. If ``None``, a Gemini client is
        created from ``settings``.

    Returns
    -------
    tuple[str, str]
        ``(verdict, reason)`` where verdict is one of ``"correct"``,
        ``"partial"``, or ``"incorrect"``.
    """
    _load_cache()

    key = _cache_key(question, answer, reference)
    if key in _judge_cache:
        cached = _judge_cache[key]
        logger.debug("Judge cache hit for key %s…", key[:12])
        return cached[0], cached[1]

    prompt = (
        f"{_JUDGE_SYSTEM}\n\n"
        f"QUESTION: {question}\n\n"
        f"REFERENCE ANSWER: {reference}\n\n"
        f"GENERATED ANSWER: {answer}\n\n"
        "JSON response:"
    )

    try:
        raw = _call_llm(prompt, llm=llm)
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        raw_json = match.group(0) if match else raw
        data = json.loads(raw_json)
        verdict: str = str(data.get("verdict", "incorrect")).lower()
        reason: str = str(data.get("reason", ""))
        # Normalise unexpected values
        if verdict not in {"correct", "partial", "incorrect"}:
            verdict = "incorrect"
            reason = f"[normalised from '{verdict}'] {reason}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge LLM call failed: %s — defaulting to 'incorrect'", exc)
        verdict, reason = "incorrect", f"Judge error: {exc}"

    _judge_cache[key] = (verdict, reason)
    _save_cache()
    return verdict, reason


def compute_faithfulness(
    answer: str,
    chunks: list[RetrievedChunk],
    llm=None,
) -> float:
    """Estimate what fraction of answer claims are supported by retrieved chunks.

    Sends the answer and context to Gemini and asks for a 0–1 score.

    Parameters
    ----------
    answer:
        The generated answer text.
    chunks:
        Retrieved context chunks used to generate the answer.
    llm:
        Optional pre-constructed LLM client.

    Returns
    -------
    float
        Score in [0.0, 1.0]. Returns 0.0 on error.
    """
    if not answer or not chunks:
        return 0.0

    context_text = "\n\n".join(
        f"[Chunk {i + 1}] {chunk.text}" for i, chunk in enumerate(chunks[:10])
    )

    prompt = (
        f"{_FAITHFULNESS_SYSTEM}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "JSON response:"
    )

    try:
        raw = _call_llm(prompt, llm=llm)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Faithfulness LLM call failed: %s — returning 0.0", exc)
        return 0.0

