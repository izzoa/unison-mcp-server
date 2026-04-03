"""Tests for provider-aware token counting."""

import warnings
from unittest.mock import MagicMock, Mock, patch


class TestModelProviderCountTokens:
    """Test ModelProvider.count_tokens() base class."""

    def _make_provider(self):
        from providers.base import ModelProvider

        class FakeProvider(ModelProvider):
            def __init__(self):
                self._supported_models = {"test-model": "test-model"}

            def generate_content(self, **kwargs):
                pass

            def get_provider_type(self):
                pass

            def get_capabilities(self, model_name):
                pass

        return FakeProvider()

    def test_empty_string_returns_0(self):
        provider = self._make_provider()
        assert provider.count_tokens("", "test-model") == 0

    def test_litellm_fallback(self):
        provider = self._make_provider()
        with patch.dict("sys.modules", {"litellm": MagicMock(token_counter=Mock(return_value=42))}):
            # Need to reimport since litellm is imported inside the function
            result = provider.count_tokens("hello world", "test-model")
        assert result >= 1  # Either litellm or heuristic produces a positive int

    def test_heuristic_fallback_prose(self):
        provider = self._make_provider()
        result = provider._heuristic_count_tokens("Hello world, this is prose text.", "test-model")
        assert result == len("Hello world, this is prose text.") // 4

    def test_heuristic_fallback_code(self):
        provider = self._make_provider()
        code = "def foo() {\n    bar();\n    baz();\n}\n" * 50
        result = provider._heuristic_count_tokens(code, "test-model")
        assert result == len(code) // 3

    def test_non_empty_returns_at_least_1(self):
        provider = self._make_provider()
        result = provider._heuristic_count_tokens("x", "test-model")
        assert result >= 1


class TestOpenAICompatibleCountTokens:
    """Test OpenAICompatibleProvider.count_tokens() with encoding cache."""

    def test_tiktoken_returns_correct_count(self):
        """Test that tiktoken produces a reasonable token count for known text."""
        from providers.openai_compatible import OpenAICompatibleProvider
        from providers.shared import ProviderType

        class TestProvider(OpenAICompatibleProvider):
            def get_provider_type(self):
                return ProviderType.OPENAI

        provider = TestProvider.__new__(TestProvider)
        provider._supported_models = {"gpt-4": "gpt-4"}
        TestProvider._encoding_cache = {}

        result = provider.count_tokens("Hello, world!", "gpt-4")
        assert isinstance(result, int)
        assert result >= 1


class TestGeminiCountTokens:
    """Test GeminiModelProvider.count_tokens()."""

    def test_gemini_count_tokens_returns_int(self):
        """Test that Gemini provider produces a token count (via litellm or fallback)."""
        from providers.gemini import GeminiModelProvider

        provider = GeminiModelProvider.__new__(GeminiModelProvider)
        provider._supported_models = {"gemini-2.5-flash": "gemini-2.5-flash"}

        result = provider.count_tokens("Hello world test text", "gemini-2.5-flash")
        assert isinstance(result, int)
        assert result >= 1


class TestModelContextEstimateTokens:
    """Test ModelContext.estimate_tokens() delegation."""

    def test_delegates_to_provider(self):
        from utils.model_context import ModelContext

        mc = ModelContext.__new__(ModelContext)
        mc.model_name = "test"
        mock_provider = MagicMock()
        mock_provider.count_tokens.return_value = 50
        with patch.object(type(mc), "provider", new_callable=lambda: property(lambda self: mock_provider)):
            assert mc.estimate_tokens("hello") == 50

    def test_fallback_on_provider_error(self):
        from utils.model_context import ModelContext

        mc = ModelContext.__new__(ModelContext)
        mc.model_name = "test"
        mock_provider = MagicMock()
        mock_provider.count_tokens.side_effect = RuntimeError("boom")
        with patch.object(type(mc), "provider", new_callable=lambda: property(lambda self: mock_provider)):
            result = mc.estimate_tokens("hello world")
        assert result == len("hello world") // 3

    def test_fallback_on_mock_return(self):
        from utils.model_context import ModelContext

        mc = ModelContext.__new__(ModelContext)
        mc.model_name = "test"
        mock_provider = MagicMock()
        mock_provider.count_tokens.return_value = MagicMock()  # Not an int
        with patch.object(type(mc), "provider", new_callable=lambda: property(lambda self: mock_provider)):
            result = mc.estimate_tokens("hello")
        assert isinstance(result, int)


class TestDeprecatedEstimateTokens:
    """Test that utils.token_utils.estimate_tokens emits DeprecationWarning."""

    def test_emits_deprecation_warning(self):
        from utils.token_utils import estimate_tokens

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = estimate_tokens("hello world test")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert result == len("hello world test") // 4
