"""Tests for OpenAI provider implementation."""

import os
from unittest.mock import MagicMock, patch

from providers.openai import OpenAIModelProvider
from providers.shared import ProviderType
from tests.model_test_helpers import get_all_model_names, get_model_with_thinking


class TestOpenAIProvider:
    """Test OpenAI provider functionality."""

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

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_initialization(self):
        """Test provider initialization."""
        provider = OpenAIModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.OPENAI
        assert provider.base_url == "https://api.openai.com/v1"

    def test_initialization_with_custom_url(self):
        """Test provider initialization with custom base URL."""
        provider = OpenAIModelProvider("test-key", base_url="https://custom.openai.com/v1")
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.openai.com/v1"

    def test_model_validation(self):
        """Test model name validation."""
        provider = OpenAIModelProvider("test-key")

        # All canonical model names from the registry should validate
        for model_name in get_all_model_names(ProviderType.OPENAI):
            assert provider.validate_model_name(model_name) is True, f"Canonical model '{model_name}' should be valid"

        # Well-known aliases should also validate
        well_known_aliases = [
            "mini",
            "o3mini",
            "o4mini",
            "gpt5",
            "gpt5-mini",
            "gpt5mini",
            "gpt5.2",
            "gpt5.1",
            "gpt5.1-codex",
            "codex-mini",
        ]
        for alias in well_known_aliases:
            assert provider.validate_model_name(alias) is True, f"Alias '{alias}' should be valid"

        # Test invalid model
        assert provider.validate_model_name("invalid-model") is False
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("gemini-pro") is False

    def test_resolve_model_name(self):
        """Test model name resolution."""
        provider = OpenAIModelProvider("test-key")

        # Test that aliases resolve to valid models
        aliases = [
            "mini",
            "o3mini",
            "o4mini",
            "gpt5",
            "gpt5-mini",
            "gpt5mini",
            "gpt5.2",
            "gpt5.1",
            "gpt5.1-codex",
            "codex-mini",
        ]
        for alias in aliases:
            resolved = provider._resolve_model_name(alias)
            assert provider.validate_model_name(
                resolved
            ), f"Alias '{alias}' resolved to '{resolved}' which is not valid"

        # Test full name passthrough -- canonical names should resolve to themselves
        for model_name in get_all_model_names(ProviderType.OPENAI):
            resolved = provider._resolve_model_name(model_name)
            assert resolved == model_name, f"Canonical name '{model_name}' should pass through, got '{resolved}'"

    def test_get_capabilities_o3(self):
        """Test getting model capabilities for O3."""
        provider = OpenAIModelProvider("test-key")

        capabilities = provider.get_capabilities("o3")
        assert provider.validate_model_name(capabilities.model_name)
        assert capabilities.friendly_name  # non-empty
        assert capabilities.context_window > 0
        assert capabilities.provider == ProviderType.OPENAI
        assert not capabilities.supports_extended_thinking
        assert capabilities.supports_system_prompts is True
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True

        # Test temperature constraint (O3 has fixed temperature)
        assert capabilities.temperature_constraint.value == 1.0

    def test_get_capabilities_with_alias(self):
        """Test getting model capabilities with alias resolves correctly."""
        provider = OpenAIModelProvider("test-key")

        capabilities = provider.get_capabilities("mini")
        assert provider.validate_model_name(capabilities.model_name)
        assert capabilities.friendly_name  # non-empty
        assert capabilities.context_window > 0
        assert capabilities.provider == ProviderType.OPENAI

    def test_get_capabilities_gpt5(self):
        """Test getting model capabilities for GPT-5."""
        provider = OpenAIModelProvider("test-key")

        capabilities = provider.get_capabilities("gpt-5")
        assert provider.validate_model_name(capabilities.model_name)
        assert capabilities.friendly_name  # non-empty
        assert capabilities.context_window > 0
        assert capabilities.max_output_tokens > 0
        assert capabilities.provider == ProviderType.OPENAI
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_system_prompts is True
        assert capabilities.supports_streaming is False
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_temperature is True

    def test_get_capabilities_gpt5_mini(self):
        """Test getting model capabilities for GPT-5-mini."""
        provider = OpenAIModelProvider("test-key")

        capabilities = provider.get_capabilities("gpt-5-mini")
        assert provider.validate_model_name(capabilities.model_name)
        assert capabilities.friendly_name  # non-empty
        assert capabilities.context_window > 0
        assert capabilities.max_output_tokens > 0
        assert capabilities.provider == ProviderType.OPENAI
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_system_prompts is True
        assert capabilities.supports_streaming is False
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_temperature is True

    def test_get_capabilities_gpt52(self):
        """Test GPT-5.2 capabilities reflect new metadata."""
        provider = OpenAIModelProvider("test-key")

        capabilities = provider.get_capabilities("gpt-5.2")
        assert provider.validate_model_name(capabilities.model_name)
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_json_mode is True
        assert capabilities.allow_code_generation is True

    def test_get_capabilities_gpt51_codex(self):
        """Test GPT-5.1 Codex is responses-only and non-streaming."""
        provider = OpenAIModelProvider("test-key")

        capabilities = provider.get_capabilities("gpt-5.1-codex")
        assert provider.validate_model_name(capabilities.model_name)
        assert capabilities.supports_streaming is False
        assert capabilities.use_openai_response_api is True
        assert capabilities.allow_code_generation is True

    def test_get_capabilities_gpt51_codex_mini(self):
        """Test GPT-5.1 Codex mini exposes streaming and code generation."""
        provider = OpenAIModelProvider("test-key")

        capabilities = provider.get_capabilities("gpt-5.1-codex-mini")
        assert provider.validate_model_name(capabilities.model_name)
        assert capabilities.supports_streaming is True
        assert capabilities.allow_code_generation is True

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_resolves_alias_before_api_call(self, mock_openai_class):
        """Test that generate_content resolves aliases before making API calls.

        This is the CRITICAL test that was missing - verifying that aliases
        like 'gpt4.1' get resolved to the canonical name before being sent to OpenAI API.
        """
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4.1-2025-04-14"  # API returns the resolved model name
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIModelProvider("test-key")

        # Determine what "gpt4.1" resolves to dynamically
        resolved_name = provider._resolve_model_name("gpt4.1")

        # Call generate_content with alias 'gpt4.1'
        result = provider.generate_content(
            prompt="Test prompt",
            model_name="gpt4.1",
            temperature=1.0,
        )

        # Verify the API was called with the RESOLVED model name
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # CRITICAL ASSERTION: The API should receive the resolved name, not "gpt4.1"
        assert (
            call_kwargs["model"] == resolved_name
        ), f"Expected '{resolved_name}' but API received '{call_kwargs['model']}'"

        # Verify other parameters
        assert call_kwargs["temperature"] == 1.0
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Test prompt"

        # Verify response
        assert result.content == "Test response"
        assert result.model_name == resolved_name  # Should be the resolved name

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_other_aliases(self, mock_openai_class):
        """Test other alias resolutions in generate_content."""
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

        provider = OpenAIModelProvider("test-key")

        # Test o3mini -> resolved name
        resolved_o3mini = provider._resolve_model_name("o3mini")
        mock_response.model = resolved_o3mini
        provider.generate_content(prompt="Test", model_name="o3mini", temperature=1.0)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == resolved_o3mini

        # Test o4mini -> resolved name
        resolved_o4mini = provider._resolve_model_name("o4mini")
        mock_response.model = resolved_o4mini
        provider.generate_content(prompt="Test", model_name="o4mini", temperature=1.0)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == resolved_o4mini

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_no_alias_passthrough(self, mock_openai_class):
        """Test that full model names pass through unchanged."""
        # Set up mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "o3-mini"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIModelProvider("test-key")

        # Test full model name passes through unchanged (use o3-mini since o3-pro has special handling)
        provider.generate_content(prompt="Test", model_name="o3-mini", temperature=1.0)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "o3-mini"  # Should be unchanged

    def test_extended_thinking_capabilities(self):
        """Thinking-mode support should be reflected via ModelCapabilities."""
        provider = OpenAIModelProvider("test-key")

        # Dynamically find models that support extended thinking
        thinking_model = get_model_with_thinking(ProviderType.OPENAI)
        assert thinking_model is not None, "At least one OpenAI model should support thinking"
        assert provider.get_capabilities(thinking_model).supports_extended_thinking is True

        # Well-known aliases that should support thinking (GPT-5 family)
        thinking_aliases = ["gpt-5", "gpt-5-mini"]
        for alias in thinking_aliases:
            if provider.validate_model_name(alias):
                assert provider.get_capabilities(alias).supports_extended_thinking is True

        # Well-known aliases that should NOT support thinking (O-series)
        non_thinking_aliases = ["o3", "o3-mini", "o4-mini"]
        for alias in non_thinking_aliases:
            if provider.validate_model_name(alias):
                assert provider.get_capabilities(alias).supports_extended_thinking is False

        # Invalid models should not validate, treat as unsupported
        assert not provider.validate_model_name("invalid-model")

    @patch("providers.openai_compatible.OpenAI")
    def test_o3_pro_routes_to_responses_endpoint(self, mock_openai_class):
        """Test that o3-pro model routes to the /v1/responses endpoint (mock test)."""
        # Set up mock for OpenAI client responses endpoint
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        # New o3-pro format: direct output_text field
        mock_response.output_text = "4"
        mock_response.model = "o3-pro"
        mock_response.id = "test-id"
        mock_response.created_at = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.responses.create.return_value = mock_response

        provider = OpenAIModelProvider("test-key")

        # Generate content with o3-pro
        result = provider.generate_content(prompt="What is 2 + 2?", model_name="o3-pro", temperature=1.0)

        # Verify responses.create was called
        mock_client.responses.create.assert_called_once()
        call_args = mock_client.responses.create.call_args[1]
        assert call_args["model"] == "o3-pro"
        assert call_args["input"][0]["role"] == "user"
        assert "What is 2 + 2?" in call_args["input"][0]["content"][0]["text"]

        # Verify the response
        assert result.content == "4"
        assert result.model_name == "o3-pro"
        assert result.metadata["endpoint"] == "responses"

    @patch("providers.openai_compatible.OpenAI")
    def test_non_o3_pro_uses_chat_completions(self, mock_openai_class):
        """Test that non-o3-pro models use the standard chat completions endpoint."""
        # Set up mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "o3-mini"
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIModelProvider("test-key")

        # Generate content with o3-mini (not o3-pro)
        result = provider.generate_content(prompt="Test prompt", model_name="o3-mini", temperature=1.0)

        # Verify chat.completions.create was called
        mock_client.chat.completions.create.assert_called_once()

        # Verify the response
        assert result.content == "Test response"
        assert provider.validate_model_name(result.model_name)
