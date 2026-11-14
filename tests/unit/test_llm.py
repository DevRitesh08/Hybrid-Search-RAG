"""Unit tests for LLM adapter, retry decorator, caching, and factory (Feature 12)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.config import settings
from hybrid_rag.generation.llm import (
    GeminiAdapter,
    LLMAdapter,
    _instances,
    get_llm,
    is_transient_error,
    reset_llm_registry,
)


def setup_function():
    reset_llm_registry()


def teardown_function():
    reset_llm_registry()


def test_llm_adapter_cannot_instantiate_abstract():
    """Verify that abstract LLMAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        LLMAdapter()  # type: ignore[abstract]


def test_llm_adapter_concrete_implementation():
    """Verify that concrete implementations of LLMAdapter are valid."""
    class DummyAdapter(LLMAdapter):
        def generate(self, prompt: str, system: str = "", **kwargs) -> str:
            return f"Echo: {prompt}"

    adapter = DummyAdapter()
    assert adapter.generate("Hello world") == "Echo: Hello world"


def test_gemini_adapter_cache_key_differentiation():
    """Verify that cache key incorporates model, temperature, max_tokens, and prompt."""
    key_base = GeminiAdapter._cache_key(
        prompt="Tell me about LoRA",
        system="You are an expert",
        model="gemini-2.0-flash",
        temperature=0.1,
        max_tokens=512,
    )

    # 1. Same arguments produce same hash
    key_identical = GeminiAdapter._cache_key(
        prompt="Tell me about LoRA",
        system="You are an expert",
        model="gemini-2.0-flash",
        temperature=0.1,
        max_tokens=512,
    )
    assert key_base == key_identical

    # 2. Different model produces different hash
    key_diff_model = GeminiAdapter._cache_key(
        prompt="Tell me about LoRA",
        system="You are an expert",
        model="gemini-1.5-pro",
        temperature=0.1,
        max_tokens=512,
    )
    assert key_base != key_diff_model

    # 3. Different temperature produces different hash
    key_diff_temp = GeminiAdapter._cache_key(
        prompt="Tell me about LoRA",
        system="You are an expert",
        model="gemini-2.0-flash",
        temperature=0.7,
        max_tokens=512,
    )
    assert key_base != key_diff_temp

    # 4. Different max_tokens produces different hash
    key_diff_tokens = GeminiAdapter._cache_key(
        prompt="Tell me about LoRA",
        system="You are an expert",
        model="gemini-2.0-flash",
        temperature=0.1,
        max_tokens=1024,
    )
    assert key_base != key_diff_tokens

    # 5. Different prompt produces different hash
    key_diff_prompt = GeminiAdapter._cache_key(
        prompt="Tell me about AutoModel",
        system="You are an expert",
        model="gemini-2.0-flash",
        temperature=0.1,
        max_tokens=512,
    )
    assert key_base != key_diff_prompt


def test_is_transient_error_classification():
    """Verify selective transient error classifier."""
    # Transient errors -> True
    assert is_transient_error(ConnectionError("Connection reset by peer")) is True
    assert is_transient_error(TimeoutError("Request timed out")) is True
    assert is_transient_error(RuntimeError("429 ResourceExhausted: Quota exceeded")) is True
    assert is_transient_error(Exception("503 Service Unavailable")) is True
    assert is_transient_error(Exception("DeadlineExceeded while waiting")) is True

    # Non-transient developer/client errors -> False
    assert is_transient_error(TypeError("Unexpected keyword argument")) is False
    assert is_transient_error(ValueError("Invalid model configuration")) is False
    assert is_transient_error(KeyError("missing_key")) is False
    assert is_transient_error(AttributeError("'NoneType' object has no attribute")) is False


def test_get_llm_factory_and_singleton():
    """Verify get_llm returns singleton for same model and new instance for different model."""
    adapter_default = get_llm("gemini")
    adapter_default_2 = get_llm("gemini")
    assert adapter_default is adapter_default_2

    adapter_judge = get_llm("gemini", model_name="gemini-2.0-flash-judge")
    assert adapter_judge is not adapter_default
    assert adapter_judge._model == "gemini-2.0-flash-judge"

    # Subsequent call with same model_name retrieves existing singleton
    adapter_judge_2 = get_llm("gemini", model_name="gemini-2.0-flash-judge")
    assert adapter_judge is adapter_judge_2


def test_get_llm_unknown_provider_raises():
    """Verify get_llm raises ValueError for unsupported providers."""
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm("nonexistent_provider")


def test_gemini_adapter_uninitialized_client_raises():
    """Verify generate() raises clear RuntimeError when client is uninitialized."""
    adapter = GeminiAdapter(api_key="")
    adapter._client = None
    with pytest.raises(RuntimeError, match="Gemini client is not initialised"):
        adapter.generate("Hello")


def test_gemini_adapter_cache_hit_bypasses_client():
    """Verify that cached response is returned without calling client."""
    adapter = GeminiAdapter(api_key="test-key")
    adapter._cache = MagicMock()
    adapter._cache.get.return_value = "Cached response text"

    resp = adapter.generate("Cached prompt")
    assert resp == "Cached response text"
    adapter._cache.get.assert_called_once()


def test_get_llm_retrieves_registered_fixture_singleton():
    """Verify get_llm retrieves pre-set mock fixtures registered at _instances['gemini']."""
    reset_llm_registry()

    class FixtureMock(LLMAdapter):
        def generate(self, prompt: str, system: str = "", **kwargs) -> str:
            return "Fixture mock response"

    mock_instance = FixtureMock()
    _instances["gemini"] = mock_instance

    # Default call without arguments
    assert get_llm() is mock_instance
    # Call with provider
    assert get_llm("gemini") is mock_instance
    # Call with case variation
    assert get_llm("GEMINI") is mock_instance

    # Legacy key compatibility
    reset_llm_registry()
    _instances["gemini::"] = mock_instance
    assert get_llm("gemini") is mock_instance


def test_gemini_adapter_accepts_arbitrary_kwargs():
    """Verify GeminiAdapter and get_llm accept and preserve arbitrary kwargs without TypeError."""
    adapter = GeminiAdapter(
        api_key="key",
        model="gemini-2.0-flash",
        custom_timeout=45,
        custom_flag=True,
    )
    assert adapter._extra_kwargs == {"custom_timeout": 45, "custom_flag": True}

    reset_llm_registry()
    adapter2 = get_llm("gemini", custom_param="value")
    assert getattr(adapter2, "_extra_kwargs", {}).get("custom_param") == "value"


def test_cache_key_delimiter_boundary_injection_resistance():
    """Verify length-prefixed formatting prevents :: delimiter collisions between adjacent fields."""
    key1 = GeminiAdapter._cache_key(prompt="bar", system="foo::", model="m", temperature=0.1, max_tokens=100)
    key2 = GeminiAdapter._cache_key(prompt="::bar", system="foo", model="m", temperature=0.1, max_tokens=100)
    assert key1 != key2, "Length-prefixed cache keys must not collide on delimiter injection"


def test_gemini_adapter_tenacity_retry_recovery_on_transient_error():
    """Verify that GeminiAdapter retries transient errors and recovers if subsequent attempt succeeds."""
    adapter = GeminiAdapter(api_key="test-key")
    mock_client = MagicMock()

    mock_resp = MagicMock()
    mock_resp.text = "Success on attempt 3"
    mock_client.models.generate_content.side_effect = [
        ConnectionError("503 Service Unavailable"),
        RuntimeError("429 ResourceExhausted quota exceeded"),
        mock_resp,
    ]
    adapter._client = mock_client
    adapter._cache = None

    with patch.object(settings, "llm_retry_attempts", 3), \
         patch.object(settings, "llm_retry_wait", 0.001):
        result = adapter.generate("Test prompt")

    assert result == "Success on attempt 3"
    assert mock_client.models.generate_content.call_count == 3
