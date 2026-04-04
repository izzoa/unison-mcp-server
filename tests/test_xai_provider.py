"""Tests for X.AI provider implementation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from providers.shared import ProviderType
from providers.xai import XAIModelProvider
from tests.model_test_helpers import (  # noqa: F401 - get_any_model removed
    get_all_aliases,
    get_all_model_names,
    get_flagship_model,
    get_flash_model,
    get_model_with_thinking,
)


class TestXAIProvider:
    """Test X.AI provider functionality."""

    def setup_method(self):
        """Set up clean state before each test."""
        # Clear restriction service cache before each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        """Clean up after each test to avoid singleton issues."""
        # Clear restriction service cache after each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    @patch.dict(os.environ, {"XAI_API_KEY": "test-key"})
    def test_initialization(self):
        """Test provider initialization."""
        provider = XAIModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.XAI
        assert provider.base_url == "https://api.x.ai/v1"

    def test_initialization_with_custom_url(self):
        """Test provider initialization with custom base URL."""
        provider = XAIModelProvider("test-key", base_url="https://custom.x.ai/v1")
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.x.ai/v1"

    def test_model_validation(self):
        """Test that all registered models and their aliases validate."""
        provider = XAIModelProvider("test-key")

        # All canonical model names should be valid
        for model_name in get_all_model_names(ProviderType.XAI):
            assert provider.validate_model_name(model_name) is True, f"Canonical model {model_name!r} should validate"

        # All aliases should be valid
        for model_name, aliases in get_all_aliases(ProviderType.XAI).items():
            for alias in aliases:
                assert (
                    provider.validate_model_name(alias) is True
                ), f"Alias {alias!r} for {model_name!r} should validate"

        # Non-XAI models should be invalid
        assert provider.validate_model_name("invalid-model") is False
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("gemini-pro") is False

    def test_resolve_model_name(self):
        """Test that every alias resolves to a canonical model name."""
        provider = XAIModelProvider("test-key")
        all_canonical = set(get_all_model_names(ProviderType.XAI))

        # Every alias should resolve to a valid canonical model name
        for _model_name, aliases in get_all_aliases(ProviderType.XAI).items():
            for alias in aliases:
                resolved = provider._resolve_model_name(alias)
                assert (
                    resolved in all_canonical
                ), f"Alias {alias!r} resolved to {resolved!r} which is not a canonical model"

        # Every canonical name should resolve to itself
        for model_name in all_canonical:
            assert provider._resolve_model_name(model_name) == model_name

    def test_get_capabilities_flagship(self):
        """Test getting model capabilities for the flagship XAI model."""
        provider = XAIModelProvider("test-key")

        flagship = get_flagship_model(ProviderType.XAI)
        assert flagship is not None, "XAI should have at least one model"

        capabilities = provider.get_capabilities(flagship)
        assert capabilities.model_name == flagship
        assert capabilities.friendly_name.startswith("X.AI")
        assert capabilities.context_window >= 100_000
        assert capabilities.provider == ProviderType.XAI
        assert capabilities.supports_system_prompts is True
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_images is True

        # Temperature constraint should have sensible defaults
        assert capabilities.temperature_constraint.min_temp >= 0.0
        assert capabilities.temperature_constraint.max_temp >= 1.0
        assert 0.0 <= capabilities.temperature_constraint.default_temp <= 1.0

    def test_get_capabilities_fast_model(self):
        """Test getting model capabilities for the fast-tier XAI model."""
        provider = XAIModelProvider("test-key")

        fast = get_flash_model(ProviderType.XAI)
        assert fast is not None, "XAI should have a fast-tier model"

        capabilities = provider.get_capabilities(fast)
        assert capabilities.model_name == fast
        assert capabilities.friendly_name.startswith("X.AI")
        assert capabilities.context_window >= 100_000
        assert capabilities.provider == ProviderType.XAI
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_images is True

    def test_get_capabilities_with_alias(self):
        """Test getting model capabilities via an alias resolves properly."""
        provider = XAIModelProvider("test-key")
        all_canonical = set(get_all_model_names(ProviderType.XAI))

        # Pick the first model that has aliases
        for model_name, aliases in get_all_aliases(ProviderType.XAI).items():
            if aliases:
                # Use an alias that differs from the canonical name
                non_canonical_aliases = [a for a in aliases if a != model_name]
                if non_canonical_aliases:
                    alias = non_canonical_aliases[0]
                    capabilities = provider.get_capabilities(alias)
                    assert (
                        capabilities.model_name in all_canonical
                    ), f"Capabilities for alias {alias!r} should have a canonical model_name"
                    assert provider.validate_model_name(alias) is True
                    break

    def test_unsupported_model_capabilities(self):
        """Test error handling for unsupported models."""
        provider = XAIModelProvider("test-key")

        with pytest.raises(ValueError, match="Unsupported model 'invalid-model' for provider xai"):
            provider.get_capabilities("invalid-model")

    def test_extended_thinking_flags(self):
        """X.AI models with thinking support should expose it correctly."""
        provider = XAIModelProvider("test-key")

        thinking_model = get_model_with_thinking(ProviderType.XAI)
        if thinking_model is not None:
            assert provider.get_capabilities(thinking_model).supports_extended_thinking is True

        # Every alias of a thinking-capable model should also report thinking support
        for model_name, aliases in get_all_aliases(ProviderType.XAI).items():
            caps = provider.get_capabilities(model_name)
            if caps.supports_extended_thinking:
                for alias in aliases:
                    assert (
                        provider.get_capabilities(alias).supports_extended_thinking is True
                    ), f"Alias {alias!r} of thinking model {model_name!r} should support thinking"

    def test_provider_type(self):
        """Test provider type identification."""
        provider = XAIModelProvider("test-key")
        assert provider.get_provider_type() == ProviderType.XAI

    def test_model_restrictions_allow_single(self):
        """Test that model restrictions correctly allow/block models."""
        all_models = get_all_model_names(ProviderType.XAI)
        if len(all_models) < 2:
            pytest.skip("Need at least 2 models to test restrictions")

        allowed_model = all_models[0]
        blocked_model = all_models[1]

        with patch.dict(os.environ, {"XAI_ALLOWED_MODELS": allowed_model}):
            import utils.model_restrictions

            utils.model_restrictions._restriction_service = None

            restricted_provider = XAIModelProvider("test-key")

            # The allowed model should validate
            assert restricted_provider.validate_model_name(allowed_model) is True

            # The blocked model should not validate
            assert restricted_provider.validate_model_name(blocked_model) is False

    def test_model_restrictions_allow_aliases(self):
        """Test that restrictions on a model also allow its aliases."""
        all_models = get_all_model_names(ProviderType.XAI)
        all_aliases = get_all_aliases(ProviderType.XAI)

        # Find a model that has aliases differing from its canonical name
        target_model = None
        for m in all_models:
            aliases = all_aliases.get(m, [])
            non_canonical = [a for a in aliases if a == m]
            if non_canonical:
                target_model = m
                break

        if target_model is None:
            pytest.skip("No model with usable aliases found")

        with patch.dict(os.environ, {"XAI_ALLOWED_MODELS": target_model}):
            import utils.model_restrictions

            utils.model_restrictions._restriction_service = None

            restricted_provider = XAIModelProvider("test-key")
            assert restricted_provider.validate_model_name(target_model) is True

    @patch.dict(os.environ, {"XAI_ALLOWED_MODELS": ""})
    def test_empty_restrictions_allows_all(self):
        """Test that empty restrictions allow all models."""
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        provider = XAIModelProvider("test-key")

        # Every canonical model and alias should be allowed
        for model_name in get_all_model_names(ProviderType.XAI):
            assert (
                provider.validate_model_name(model_name) is True
            ), f"Model {model_name!r} should be allowed with empty restrictions"
        for _model_name, aliases in get_all_aliases(ProviderType.XAI).items():
            for alias in aliases:
                assert (
                    provider.validate_model_name(alias) is True
                ), f"Alias {alias!r} should be allowed with empty restrictions"

    def test_friendly_name(self):
        """Test friendly name constant and per-model friendly names."""
        provider = XAIModelProvider("test-key")
        assert provider.FRIENDLY_NAME == "X.AI"

        # Every model's friendly_name should start with the provider prefix
        for model_name in get_all_model_names(ProviderType.XAI):
            capabilities = provider.get_capabilities(model_name)
            assert capabilities.friendly_name.startswith(
                "X.AI"
            ), f"Model {model_name!r} friendly_name should start with 'X.AI'"

    def test_supported_models_structure(self):
        """Test that MODEL_CAPABILITIES has the correct structure."""
        provider = XAIModelProvider("test-key")
        from providers.shared import ModelCapabilities

        all_models = get_all_model_names(ProviderType.XAI)
        assert len(all_models) >= 1, "XAI should have at least one registered model"

        for model_name in all_models:
            config = provider.MODEL_CAPABILITIES[model_name]
            assert isinstance(config, ModelCapabilities), f"Model {model_name!r} should be a ModelCapabilities instance"
            assert hasattr(config, "context_window")
            assert hasattr(config, "supports_extended_thinking")
            assert hasattr(config, "aliases")
            assert config.context_window >= 100_000, f"Model {model_name!r} context_window should be at least 100K"

            # Every model should have at least one alias
            assert len(config.aliases) >= 1, f"Model {model_name!r} should have at least one alias"

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_resolves_alias_before_api_call(self, mock_openai_class):
        """Test that generate_content resolves aliases before making API calls.

        This is the CRITICAL test that ensures aliases get resolved to canonical
        model names before being sent to the X.AI API.
        """
        provider = XAIModelProvider("test-key")
        all_aliases = get_all_aliases(ProviderType.XAI)

        # Find an alias that differs from its canonical name
        test_alias = None
        expected_canonical = None
        for model_name, aliases in all_aliases.items():
            non_canonical = [a for a in aliases if a != model_name]
            if non_canonical:
                test_alias = non_canonical[0]
                expected_canonical = model_name
                break

        if test_alias is None:
            pytest.skip("No non-canonical alias found for testing")

        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = expected_canonical  # API returns the resolved model name
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        # Re-init provider so it picks up the mock client
        provider = XAIModelProvider("test-key")

        # Call generate_content with the alias
        result = provider.generate_content(prompt="Test prompt", model_name=test_alias, temperature=0.7)

        # Verify the API was called with the RESOLVED canonical name
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # CRITICAL ASSERTION: The API should receive the canonical name, not the alias
        assert (
            call_kwargs["model"] == expected_canonical
        ), f"Expected canonical {expected_canonical!r} but API received {call_kwargs['model']!r}"

        # Verify other parameters
        assert call_kwargs["temperature"] == 0.7
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Test prompt"

        # Verify response
        assert result.content == "Test response"
        assert provider.validate_model_name(result.model_name)

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_all_aliases_resolve(self, mock_openai_class):
        """Test that all aliases resolve to canonical names in generate_content."""
        from unittest.mock import MagicMock

        provider = XAIModelProvider("test-key")
        all_canonical = set(get_all_model_names(ProviderType.XAI))

        # Set up mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        provider = XAIModelProvider("test-key")

        for model_name, aliases in get_all_aliases(ProviderType.XAI).items():
            for alias in aliases:
                mock_response.model = model_name
                provider.generate_content(prompt="Test", model_name=alias, temperature=0.7)
                call_kwargs = mock_client.chat.completions.create.call_args[1]
                assert call_kwargs["model"] in all_canonical, (
                    f"Alias {alias!r} should resolve to a canonical model, " f"got {call_kwargs['model']!r}"
                )
