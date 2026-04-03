"""
Test suite for intelligent auto mode fallback logic

Tests the new dynamic model selection based on available API keys
"""

import os
from unittest.mock import Mock, patch

import pytest

from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType


class TestIntelligentFallback:
    """Test intelligent model fallback logic"""

    def setup_method(self):
        """Setup for each test - clear registry and reset providers"""
        # Store original providers for restoration
        registry = ModelProviderRegistry()
        self._original_providers = registry._providers.copy()
        self._original_initialized = registry._initialized_providers.copy()

        # Clear registry completely
        ModelProviderRegistry._instance = None

    def teardown_method(self):
        """Cleanup after each test - restore original providers"""
        # Restore original registry state
        registry = ModelProviderRegistry()
        registry._providers.clear()
        registry._initialized_providers.clear()
        registry._providers.update(self._original_providers)
        registry._initialized_providers.update(self._original_initialized)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key", "GEMINI_API_KEY": ""}, clear=False)
    def test_prefers_openai_when_available(self):
        """Test that a valid OpenAI model is preferred when OpenAI API key is available"""
        # Register only OpenAI provider for this test
        from providers.openai import OpenAIModelProvider

        ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)

        fallback_model = ModelProviderRegistry.get_preferred_fallback_model()
        # Should return a valid OpenAI model (highest intelligence_score for BALANCED)
        provider = ModelProviderRegistry.get_provider(ProviderType.OPENAI)
        assert provider.validate_model_name(fallback_model)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "GEMINI_API_KEY": "test-gemini-key"}, clear=False)
    def test_prefers_gemini_when_openai_unavailable(self):
        """Test that a valid Gemini model is used when only Gemini API key is available"""
        # Register only Gemini provider for this test
        from providers.gemini import GeminiModelProvider

        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

        fallback_model = ModelProviderRegistry.get_preferred_fallback_model()
        # Should return a valid Gemini model
        provider = ModelProviderRegistry.get_provider(ProviderType.GOOGLE)
        assert provider.validate_model_name(fallback_model)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key", "GEMINI_API_KEY": "test-gemini-key"}, clear=False)
    def test_returns_valid_model_when_both_available(self):
        """Test that a valid model is returned when both API keys are available"""
        # Register both OpenAI and Gemini providers
        from providers.gemini import GeminiModelProvider
        from providers.openai import OpenAIModelProvider

        ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

        fallback_model = ModelProviderRegistry.get_preferred_fallback_model()
        # Gemini has priority in PROVIDER_PRIORITY_ORDER, so should return a Gemini model
        provider = ModelProviderRegistry.get_provider(ProviderType.GOOGLE)
        assert provider.validate_model_name(fallback_model)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False)
    def test_fallback_when_no_keys_available(self):
        """Test fallback raises ValueError when no API keys are available"""
        # Register providers but with no API keys available
        from providers.gemini import GeminiModelProvider
        from providers.openai import OpenAIModelProvider

        ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

        with pytest.raises(ValueError, match="No models available"):
            ModelProviderRegistry.get_preferred_fallback_model()

    def test_available_providers_with_keys(self):
        """Test the get_available_providers_with_keys method"""
        from providers.gemini import GeminiModelProvider
        from providers.openai import OpenAIModelProvider

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key", "GEMINI_API_KEY": ""}, clear=False):
            # Clear and register providers
            ModelProviderRegistry._instance = None
            ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
            ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

            available = ModelProviderRegistry.get_available_providers_with_keys()
            assert ProviderType.OPENAI in available
            assert ProviderType.GOOGLE not in available

        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "GEMINI_API_KEY": "test-key"}, clear=False):
            # Clear and register providers
            ModelProviderRegistry._instance = None
            ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
            ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

            available = ModelProviderRegistry.get_available_providers_with_keys()
            assert ProviderType.GOOGLE in available
            assert ProviderType.OPENAI not in available

    def test_auto_mode_conversation_memory_integration(self):
        """Test that conversation memory uses intelligent fallback in auto mode"""
        from utils.conversation_memory import ThreadContext, build_conversation_history

        # Mock auto mode - patch the config module where these values are defined
        with (
            patch("config.IS_AUTO_MODE", True),
            patch("config.DEFAULT_MODEL", "auto"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key", "GEMINI_API_KEY": ""}, clear=False),
        ):
            # Register only OpenAI provider for this test
            from providers.openai import OpenAIModelProvider

            ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)

            # Create a context with at least one turn so it doesn't exit early
            from utils.conversation_memory import ConversationTurn

            context = ThreadContext(
                thread_id="test-123",
                created_at="2023-01-01T00:00:00Z",
                last_updated_at="2023-01-01T00:00:00Z",
                tool_name="chat",
                turns=[ConversationTurn(role="user", content="Test message", timestamp="2023-01-01T00:00:30Z")],
                initial_context={},
            )

            # This should use o4-mini for token calculations since OpenAI is available
            with patch("utils.model_context.ModelContext") as mock_context_class:
                mock_context_instance = Mock()
                mock_context_class.return_value = mock_context_instance
                mock_context_instance.calculate_token_allocation.return_value = Mock(
                    file_tokens=10000, history_tokens=5000
                )
                # Mock estimate_tokens to return integers for proper summing
                mock_context_instance.estimate_tokens.return_value = 100

                history, tokens = build_conversation_history(context, model_context=None)

                # Verify that ModelContext was called with a valid OpenAI model (the intelligent fallback)
                mock_context_class.assert_called_once()
                called_model = mock_context_class.call_args[0][0]
                from providers.openai import OpenAIModelProvider

                openai_provider = OpenAIModelProvider(api_key="dummy")
                assert openai_provider.validate_model_name(called_model)

    def test_auto_mode_with_gemini_only(self):
        """Test auto mode behavior when only Gemini API key is available"""
        from utils.conversation_memory import ThreadContext, build_conversation_history

        with (
            patch("config.IS_AUTO_MODE", True),
            patch("config.DEFAULT_MODEL", "auto"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "", "GEMINI_API_KEY": "test-key"}, clear=False),
        ):
            # Register only Gemini provider for this test
            from providers.gemini import GeminiModelProvider

            ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

            from utils.conversation_memory import ConversationTurn

            context = ThreadContext(
                thread_id="test-456",
                created_at="2023-01-01T00:00:00Z",
                last_updated_at="2023-01-01T00:00:00Z",
                tool_name="analyze",
                turns=[ConversationTurn(role="assistant", content="Test response", timestamp="2023-01-01T00:00:30Z")],
                initial_context={},
            )

            with patch("utils.model_context.ModelContext") as mock_context_class:
                mock_context_instance = Mock()
                mock_context_class.return_value = mock_context_instance
                mock_context_instance.calculate_token_allocation.return_value = Mock(
                    file_tokens=10000, history_tokens=5000
                )
                # Mock estimate_tokens to return integers for proper summing
                mock_context_instance.estimate_tokens.return_value = 100

                history, tokens = build_conversation_history(context, model_context=None)

                # Should use a valid Gemini model when only Gemini is available
                mock_context_class.assert_called_once()
                called_model = mock_context_class.call_args[0][0]
                from providers.gemini import GeminiModelProvider

                gemini_provider = GeminiModelProvider(api_key="dummy")
                assert gemini_provider.validate_model_name(called_model)

    def test_non_auto_mode_unchanged(self):
        """Test that non-auto mode behavior is unchanged"""
        from providers.gemini import GeminiModelProvider
        from utils.conversation_memory import ThreadContext, build_conversation_history

        gemini_pro = GeminiModelProvider(api_key="test-key")._resolve_model_name("pro")

        with patch("config.IS_AUTO_MODE", False), patch("config.DEFAULT_MODEL", gemini_pro):
            from utils.conversation_memory import ConversationTurn

            context = ThreadContext(
                thread_id="test-789",
                created_at="2023-01-01T00:00:00Z",
                last_updated_at="2023-01-01T00:00:00Z",
                tool_name="thinkdeep",
                turns=[
                    ConversationTurn(role="user", content="Test in non-auto mode", timestamp="2023-01-01T00:00:30Z")
                ],
                initial_context={},
            )

            with patch("utils.model_context.ModelContext") as mock_context_class:
                mock_context_instance = Mock()
                mock_context_class.return_value = mock_context_instance
                mock_context_instance.calculate_token_allocation.return_value = Mock(
                    file_tokens=10000, history_tokens=5000
                )
                # Mock estimate_tokens to return integers for proper summing
                mock_context_instance.estimate_tokens.return_value = 100

                history, tokens = build_conversation_history(context, model_context=None)

                # Should use the configured DEFAULT_MODEL, not the intelligent fallback
                mock_context_class.assert_called_once_with(gemini_pro)


if __name__ == "__main__":
    pytest.main([__file__])
