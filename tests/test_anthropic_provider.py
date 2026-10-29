"""Tests for the native Anthropic provider."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from providers.anthropic import MIN_THINKING_BUDGET, AnthropicModelProvider
from providers.registry import ModelProviderRegistry, set_default_registry
from providers.shared import ProviderType

# Every provider key blanked; individual tests enable what they need.
BASE_ENV = {
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "XAI_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "DIAL_API_KEY": "",
    "CUSTOM_API_URL": "",
    "AZURE_OPENAI_API_KEY": "",
    "AZURE_OPENAI_ENDPOINT": "",
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_API_URL": "",
}


def _reset_restrictions():
    import utils.model_restrictions

    utils.model_restrictions._restriction_service = None


class TestRequestShape:
    """The Messages API request the provider assembles."""

    def _provider(self):
        return AnthropicModelProvider(api_key="sk-ant-test")

    def test_system_max_tokens_and_thinking_budget(self):
        provider = self._provider()
        caps = provider.get_capabilities("claude-haiku-4-5")
        params = provider._build_request("hello", "claude-haiku-4-5", caps, "be brief", 0.3, None, "medium", None)
        assert params["system"] == "be brief"
        assert params["max_tokens"] == 64000  # required by the Messages API
        assert params["thinking"] == {"type": "enabled", "budget_tokens": int(32000 * 0.33)}
        assert "output_config" not in params
        # Extended thinking fixes temperature: the caller's value is dropped.
        assert "temperature" not in params
        assert params["messages"][0]["content"][0] == {"type": "text", "text": "hello"}

    def test_minimal_mode_clamps_to_api_floor(self):
        provider = self._provider()
        caps = provider.get_capabilities("claude-haiku-4-5")
        params = provider._build_request("x", "claude-haiku-4-5", caps, None, 0.3, None, "minimal", None)
        # 0.5% of 32000 = 160, below the API's floor -> clamped up.
        assert params["thinking"]["budget_tokens"] == MIN_THINKING_BUDGET

    def test_budget_must_fit_inside_max_tokens(self):
        provider = self._provider()
        caps = provider.get_capabilities("claude-haiku-4-5")
        # A tiny output window leaves no room for a valid budget: thinking is
        # dropped entirely rather than sending an API-invalid combination.
        params = provider._build_request("x", "claude-haiku-4-5", caps, None, 0.3, 2000, "max", None)
        assert "thinking" not in params
        assert params["temperature"] == 0.3

    @pytest.mark.parametrize(
        "model",
        [
            "claude-fable-5-1",
            "claude-fable-5",
            "claude-opus-5-5",
            "claude-opus-5",
            "claude-sonnet-5-5",
            "claude-sonnet-5",
            "claude-opus-4-8",
            "claude-haiku-5-5",
        ],
    )
    def test_adaptive_models_send_effort_and_no_sampling(self, model):
        # These models reject budget_tokens and sampling parameters with a 400.
        provider = self._provider()
        caps = provider.get_capabilities(model)
        params = provider._build_request("hello", model, caps, None, 0.3, None, "high", None)
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}
        assert params["max_tokens"] == 128000
        for key in ("temperature", "top_p", "top_k"):
            assert key not in params

    @pytest.mark.parametrize(
        "mode,effort",
        [("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("max", "max")],
    )
    def test_thinking_modes_map_onto_effort(self, mode, effort):
        provider = self._provider()
        caps = provider.get_capabilities("claude-opus-5-5")
        params = provider._build_request("x", "claude-opus-5-5", caps, None, 1.0, None, mode, None)
        assert params["output_config"] == {"effort": effort}

    def test_adaptive_model_without_thinking_mode_sends_neither(self):
        provider = self._provider()
        caps = provider.get_capabilities("claude-opus-5-5")
        params = provider._build_request("x", "claude-opus-5-5", caps, None, 0.3, None, None, None)
        assert "thinking" not in params
        assert "output_config" not in params
        assert "temperature" not in params

    def test_adaptive_model_ignores_tight_output_window(self):
        # No budget to fit, so a small max_tokens still gets thinking.
        provider = self._provider()
        caps = provider.get_capabilities("claude-sonnet-5-5")
        params = provider._build_request("x", "claude-sonnet-5-5", caps, None, 0.3, 2000, "max", None)
        assert params["max_tokens"] == 2000
        assert params["thinking"] == {"type": "adaptive"}

    def test_temperature_validation_skipped_for_samplingless_models(self):
        provider = self._provider()
        provider.validate_parameters("opus", 0.3)  # never sent, so never rejected
        with pytest.raises(ValueError):
            provider.validate_parameters("claude-haiku-4-5", 5.0)

    def test_non_thinking_model_keeps_temperature(self):
        provider = self._provider()
        caps = SimpleNamespace(
            supports_images=False,
            supports_extended_thinking=False,
            supports_adaptive_thinking=False,
            supports_temperature=True,
            max_output_tokens=4096,
        )
        params = provider._build_request("x", "claude-nothink", caps, None, 0.42, None, "high", None)
        assert "thinking" not in params
        assert params["temperature"] == 0.42


class TestGeneration:
    """generate_content / streaming against a mocked SDK client."""

    def _response_stub(self):
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", text="hmm"), SimpleNamespace(type="text", text="hello world")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=34),
            stop_reason="end_turn",
        )

    def _streaming_client(self, text_deltas=()):
        stream = MagicMock()
        stream.text_stream = iter(text_deltas)
        stream.get_final_message.return_value = self._response_stub()
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = stream
        stream_cm.__exit__.return_value = False
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = stream_cm
        return mock_client

    def test_generate_content_extracts_text_and_usage(self):
        provider = AnthropicModelProvider(api_key="sk-ant-test")
        mock_client = self._streaming_client()
        provider._client = mock_client

        response = provider.generate_content("say hello", "claude-sonnet-5", thinking_mode="low")

        assert response.content == "hello world"  # thinking blocks never reach output
        assert response.usage == {"input_tokens": 12, "output_tokens": 34, "total_tokens": 46}
        assert response.provider == ProviderType.ANTHROPIC
        assert response.model_name == "claude-sonnet-5"
        assert response.metadata["stop_reason"] == "end_turn"
        sent = mock_client.messages.stream.call_args.kwargs
        assert sent["model"] == "claude-sonnet-5"

    def test_generate_content_streams_to_avoid_sdk_long_request_guard(self):
        # The SDK raises before sending a non-streaming request whose
        # max_tokens exceeds ~21K, which every catalogued model's default does.
        provider = AnthropicModelProvider(api_key="sk-ant-test")
        mock_client = self._streaming_client()
        provider._client = mock_client

        provider.generate_content("hi", "claude-opus-5-5", temperature=0.3)

        mock_client.messages.create.assert_not_called()
        sent = mock_client.messages.stream.call_args.kwargs
        assert sent["max_tokens"] == 128000
        assert sent["thinking"] == {"type": "adaptive"}
        assert sent["output_config"] == {"effort": "medium"}
        assert "temperature" not in sent

    @pytest.mark.parametrize(
        "alias,model",
        [
            ("fable", "claude-fable-5-1"),
            ("fable5", "claude-fable-5"),
            ("opus", "claude-opus-5-5"),
            ("opus5", "claude-opus-5"),
            ("opus4", "claude-opus-4-8"),
            ("sonnet", "claude-sonnet-5-5"),
            ("claude", "claude-sonnet-5-5"),
            ("sonnet5", "claude-sonnet-5"),
            ("sonnet4", "claude-sonnet-4-6"),
            ("haiku", "claude-haiku-5-5"),
            ("haiku4", "claude-haiku-4-5"),
        ],
    )
    def test_alias_resolves_before_request(self, alias, model):
        provider = AnthropicModelProvider(api_key="sk-ant-test")
        mock_client = self._streaming_client()
        provider._client = mock_client

        provider.generate_content("hi", alias)
        assert mock_client.messages.stream.call_args.kwargs["model"] == model

    def test_streaming_yields_text_then_final_usage(self):
        provider = AnthropicModelProvider(api_key="sk-ant-test")
        provider._client = self._streaming_client(["hel", "lo"])

        chunks = list(provider.generate_content_stream("say hello", "claude-haiku-4-5"))

        assert [c.text for c in chunks] == ["hel", "lo", ""]
        assert chunks[-1].is_final is True
        assert chunks[-1].usage["total_tokens"] == 46
        assert all(not c.is_final for c in chunks[:-1])


class TestBaseUrlOverride:
    def test_env_override_used(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_URL": "https://gateway.example.com"}, clear=False):
            provider = AnthropicModelProvider(api_key="sk-ant-test")
            assert provider._base_url == "https://gateway.example.com"

    def test_placeholder_counts_as_unset(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_URL": "your_anthropic_api_url_here"}, clear=False):
            provider = AnthropicModelProvider(api_key="sk-ant-test")
            assert provider._base_url is None


class TestRegistrationAndPrecedence:
    """Configure-level behavior: key activation, precedence, restrictions."""

    def _configure(self, env):
        from providers.configure import configure_providers

        registry = ModelProviderRegistry(config={})
        set_default_registry(registry)
        configure_providers(registry)
        return registry

    def test_priority_order_places_native_before_openrouter(self):
        """Lock ANTHROPIC's slot in the priority order: native wins alias collisions."""
        order = list(ModelProviderRegistry.PROVIDER_PRIORITY_ORDER)
        assert ProviderType.ANTHROPIC in order
        assert order.index(ProviderType.ANTHROPIC) < order.index(ProviderType.OPENROUTER)

    def test_anthropic_key_registers_and_exposes_catalog(self):
        _reset_restrictions()
        with patch.dict(os.environ, {**BASE_ENV, "ANTHROPIC_API_KEY": "sk-ant-real"}, clear=True):
            registry = self._configure(os.environ)
            models = registry.get_available_models(respect_restrictions=True)
            assert "claude-fable-5" in models
            assert "claude-sonnet-4-6" in models
            provider = registry.get_provider_for_model("opus")
            assert provider.get_provider_type() == ProviderType.ANTHROPIC

    def test_placeholder_does_not_register_and_error_names_key(self):
        _reset_restrictions()
        with patch.dict(os.environ, {**BASE_ENV, "ANTHROPIC_API_KEY": "your_anthropic_api_key_here"}, clear=True):
            from providers.configure import configure_providers

            registry = ModelProviderRegistry(config={})
            set_default_registry(registry)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                configure_providers(registry)

    def test_dual_key_alias_precedence(self):
        _reset_restrictions()
        with patch.dict(
            os.environ, {**BASE_ENV, "ANTHROPIC_API_KEY": "sk-ant-real", "OPENROUTER_API_KEY": "sk-or-real"}, clear=True
        ):
            registry = self._configure(os.environ)
            provider = registry.get_provider_for_model("opus")
            assert provider.get_provider_type() == ProviderType.ANTHROPIC

    def test_openrouter_only_resolution_unchanged(self):
        _reset_restrictions()
        with patch.dict(os.environ, {**BASE_ENV, "OPENROUTER_API_KEY": "sk-or-real"}, clear=True):
            registry = self._configure(os.environ)
            provider = registry.get_provider_for_model("opus")
            assert provider.get_provider_type() == ProviderType.OPENROUTER

    def test_allowed_models_restriction_narrows_set(self):
        _reset_restrictions()
        try:
            with patch.dict(
                os.environ,
                {**BASE_ENV, "ANTHROPIC_API_KEY": "sk-ant-real", "ANTHROPIC_ALLOWED_MODELS": "claude-haiku-4-5"},
                clear=True,
            ):
                registry = self._configure(os.environ)
                models = registry.get_available_models(respect_restrictions=True)
                anthropic_models = [m for m in models if m.startswith("claude-")]
                assert anthropic_models == ["claude-haiku-4-5"]
        finally:
            _reset_restrictions()
