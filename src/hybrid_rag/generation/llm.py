"""LLM adapter layer for the generation module.

Provides an abstract :class:`LLMAdapter` base and a production-ready
:class:`GeminiAdapter` that wraps the ``google.genai`` SDK with:

* Exponential-backoff retries via ``tenacity``
* Optional disk-based response caching via ``diskcache``
* A module-level singleton registry so each provider is instantiated once.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from hybrid_rag.config import settings

logger = logging.getLogger(__name__)

# ── Singleton registry ────────────────────────────────────────────────────────
_instances: dict[str, LLMAdapter] = {}


def reset_llm_registry() -> None:
    """Clear cached LLM singleton instances (used in testing)."""
    _instances.clear()


def is_transient_error(exc: BaseException) -> bool:
    """Return True if exc represents a transient/recoverable API or network failure."""
    if isinstance(exc, (TypeError, ValueError, KeyError, AttributeError, IndexError, NotImplementedError)):
        return False

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()

    transient_indicators = (
        "resourceexhausted",
        "rate limit",
        "ratelimit",
        "quota",
        "429",
        "503",
        "502",
        "504",
        "unavailable",
        "deadlineexceeded",
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "temporary",
        "transient",
    )
    return any(ind in exc_type or ind in exc_msg for ind in transient_indicators)


# ── Abstract base ─────────────────────────────────────────────────────────────


class LLMAdapter(ABC):
    """Abstract interface for any LLM backend.

    All concrete adapters must implement :meth:`generate`.
    """

    @abstractmethod
    def generate(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        """Generate a text response for the given *prompt*.

        Args:
            prompt: The user-turn content to send to the model.
            system: Optional system instruction / persona text.
            **kwargs: Provider-specific overrides (e.g. ``temperature``).

        Returns:
            The model's text response as a plain string.
        """


# ── Gemini adapter ────────────────────────────────────────────────────────────


class GeminiAdapter(LLMAdapter):
    """Google Gemini adapter built on the ``google.genai`` SDK.

    Features:
    * Retries with exponential backoff (configurable via ``settings``).
    * Optional disk-cache keyed by MD5(prompt + system).

    Args:
        api_key:     Gemini API key.  Defaults to ``settings.gemini_api_key``.
        model:       Model identifier.  Defaults to ``settings.gemini_model``.
        temperature: Sampling temperature.  Defaults to ``settings.llm_temperature``.
        max_tokens:  Maximum output tokens.  Defaults to ``settings.llm_max_tokens``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key or settings.gemini_api_key
        self._model = model or settings.gemini_model
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self._extra_kwargs = kwargs

        self._client = None
        if self._api_key:
            try:
                import google.genai as genai  # local import keeps startup fast

                self._client = genai.Client(api_key=self._api_key)
            except Exception as exc:
                logger.warning("Could not initialise Google GenAI client: %s", exc)
                self._client = None

        # Disk cache (optional)
        self._cache = None
        if settings.llm_cache_enabled:
            try:
                import diskcache

                cache_path = settings.cache_dir / "llm"
                cache_path.mkdir(parents=True, exist_ok=True)
                self._cache = diskcache.Cache(str(cache_path))
                logger.debug("LLM disk-cache enabled at %s", cache_path)
            except Exception as exc:
                logger.warning("Could not initialize diskcache: %s", exc)
                self._cache = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return cache key incorporating model and generation parameters with length-prefixed framing."""
        m = model or settings.gemini_model
        t = settings.llm_temperature if temperature is None else temperature
        tok = settings.llm_max_tokens if max_tokens is None else max_tokens
        raw = f"{len(m)}:{m}::{t}::{tok}::{len(system)}:{system}::{len(prompt)}:{prompt}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _build_retry_decorator(self) -> Any:
        """Build a tenacity retry decorator using current settings with jitter."""
        try:
            from tenacity import (
                RetryError,  # noqa: F401
                retry,
                retry_if_exception,
                stop_after_attempt,
                wait_random_exponential,
            )

            r: Any = retry(
                stop=stop_after_attempt(settings.llm_retry_attempts),
                wait=wait_random_exponential(
                    multiplier=1,
                    min=settings.llm_retry_wait,
                    max=settings.llm_retry_wait * 8,
                ),
                retry=retry_if_exception(is_transient_error),
                reraise=False,
            )
            dummy = r(lambda: None)
            r.retry = getattr(dummy, "retry", None)
            return r
        except ImportError:
            return lambda fn: fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        """Call the Gemini model, with caching and retry.

        Args:
            prompt: User-turn content.
            system: System instruction.  Sent as the ``system_instruction``
                    parameter of the Gemini ``GenerateContent`` API.
            **kwargs: Optional overrides — ``temperature``, ``max_tokens``.

        Returns:
            Model response text.

        Raises:
            RuntimeError: If all retry attempts are exhausted or client is uninitialised.
        """
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        # 1. Cache lookup
        key = self._cache_key(
            prompt=prompt,
            system=system,
            model=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("LLM cache hit for key %s", key)
                return str(cached)

        if self._client is None:
            raise RuntimeError(
                f"Gemini client is not initialised (missing API key or SDK). "
                f"Target model: {self._model}"
            )

        # 2. Build generation config
        from tenacity import RetryError

        try:
            import google.genai.types as genai_types  # local import

            generation_config = genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system if system else None,
            )
        except Exception:
            generation_config = None

        # 3. Call with retry
        client = self._client
        assert client is not None

        @self._build_retry_decorator()
        def _call() -> str:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=generation_config,
            )
            return response.text or ""

        try:
            result = _call()
        except RetryError as exc:
            raise RuntimeError(
                f"Gemini generation failed after {settings.llm_retry_attempts} attempts: {exc}"
            ) from exc
        except Exception as exc:
            if is_transient_error(exc):
                raise RuntimeError(
                    f"Gemini generation failed after {settings.llm_retry_attempts} attempts: {exc}"
                ) from exc
            raise

        # 4. Store in cache
        if self._cache is not None:
            self._cache.set(key, result)

        return result


# ── Factory / singleton ───────────────────────────────────────────────────────


def get_llm(
    provider: str = "gemini",
    model_name: str | None = None,
    **kwargs: Any,
) -> LLMAdapter:
    """Return (or create) the singleton :class:`LLMAdapter` for *provider*.

    Supports optional *model_name* to instantiate model-specific adapters.

    Args:
        provider: Provider name string (case-insensitive, default "gemini").
        model_name: Optional model override (e.g. judge model or alternative checkpoint).
        **kwargs: Additional parameters passed to adapter constructor (e.g. temperature).

    Returns:
        The shared :class:`LLMAdapter` instance.

    Raises:
        ValueError: If *provider* is not recognised.
    """
    prov_key = provider.lower()
    instance_key = f"{prov_key}::{model_name}" if model_name else prov_key

    # Backward-compatible lookup: when default model is requested, check both
    # prov_key (e.g. "gemini") and legacy "gemini::" keys before instantiating a new adapter.
    if not model_name:
        if prov_key in _instances:
            return _instances[prov_key]
        legacy_key = f"{prov_key}::"
        if legacy_key in _instances:
            return _instances[legacy_key]
    elif instance_key in _instances:
        return _instances[instance_key]

    if prov_key == "gemini":
        _instances[instance_key] = GeminiAdapter(
            model=model_name,
            **kwargs,
        )
        logger.info(
            "Created GeminiAdapter instance (model=%s)",
            model_name or settings.gemini_model,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Supported: 'gemini'."
        )
    return _instances[instance_key]
