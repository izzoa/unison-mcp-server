"""Test the MODEL_CAPABILITIES aliases structure across all providers."""

from providers.dial import DIALModelProvider
from providers.gemini import GeminiModelProvider
from providers.openai import OpenAIModelProvider
from providers.shared import ProviderType
from providers.xai import XAIModelProvider
from tests.model_test_helpers import get_all_aliases, is_valid_model


class TestSupportedModelsAliases:
    """Test that all providers have correctly structured MODEL_CAPABILITIES with aliases."""

    def test_gemini_provider_aliases(self):
        """Test Gemini provider's alias structure."""
        provider = GeminiModelProvider("test-key")

        # Check that all models have ModelCapabilities with aliases
        for model_name, config in provider.MODEL_CAPABILITIES.items():
            assert hasattr(config, "aliases"), f"{model_name} must have aliases attribute"
            assert isinstance(config.aliases, list), f"{model_name} aliases must be a list"

        # Test that well-known aliases resolve to valid models
        well_known_aliases = ["flash", "gemini", "pro", "flashlite", "flash-lite"]
        for alias in well_known_aliases:
            assert is_valid_model(ProviderType.GOOGLE, alias), (
                f"Alias '{alias}' should resolve to a valid Gemini model"
            )

        # Test alias resolution returns valid models
        alias_inputs = ["gemini", "pro", "flash", "flashlite", "flash3"]
        for alias in alias_inputs:
            resolved = provider._resolve_model_name(alias)
            assert provider.validate_model_name(resolved), (
                f"Alias '{alias}' resolved to '{resolved}' which is not a valid model"
            )

        # Test case insensitive resolution returns valid models
        for alias in ["Flash", "PRO", "GEMINI"]:
            resolved = provider._resolve_model_name(alias)
            assert provider.validate_model_name(resolved), (
                f"Case-insensitive alias '{alias}' resolved to '{resolved}' which is not valid"
            )

        # Verify that "pro" and "gemini" resolve to the same flagship model
        assert provider._resolve_model_name("pro") == provider._resolve_model_name("gemini")

    def test_openai_provider_aliases(self):
        """Test OpenAI provider's alias structure."""
        provider = OpenAIModelProvider("test-key")

        # Check that all models have ModelCapabilities with aliases
        for model_name, config in provider.MODEL_CAPABILITIES.items():
            assert hasattr(config, "aliases"), f"{model_name} must have aliases attribute"
            assert isinstance(config.aliases, list), f"{model_name} aliases must be a list"

        # Test that well-known aliases resolve to valid models
        well_known_aliases = [
            "mini", "o4mini", "o3mini", "o3pro", "gpt4.1",
            "gpt5.2", "gpt5.1-codex", "codex-mini",
        ]
        for alias in well_known_aliases:
            assert is_valid_model(ProviderType.OPENAI, alias), (
                f"Alias '{alias}' should resolve to a valid OpenAI model"
            )

        # Test alias resolution returns valid models
        alias_inputs = [
            "mini", "o3mini", "o3pro", "o4mini", "gpt4.1",
            "gpt5.2", "gpt5.1", "gpt5.1-codex", "codex-mini",
        ]
        for alias in alias_inputs:
            resolved = provider._resolve_model_name(alias)
            assert provider.validate_model_name(resolved), (
                f"Alias '{alias}' resolved to '{resolved}' which is not a valid model"
            )

        # Test case insensitive resolution
        for alias in ["Mini", "O3MINI", "Gpt5.1"]:
            resolved = provider._resolve_model_name(alias)
            assert provider.validate_model_name(resolved), (
                f"Case-insensitive alias '{alias}' resolved to '{resolved}' which is not valid"
            )

    def test_xai_provider_aliases(self):
        """Test XAI provider's alias structure."""
        provider = XAIModelProvider("test-key")

        # Check that all models have ModelCapabilities with aliases
        for model_name, config in provider.MODEL_CAPABILITIES.items():
            assert hasattr(config, "aliases"), f"{model_name} must have aliases attribute"
            assert isinstance(config.aliases, list), f"{model_name} aliases must be a list"

        # Test that well-known aliases resolve to valid models
        well_known_aliases = ["grok", "grok4"]
        for alias in well_known_aliases:
            assert is_valid_model(ProviderType.XAI, alias), (
                f"Alias '{alias}' should resolve to a valid XAI model"
            )

        # Test alias resolution returns valid models
        alias_inputs = ["grok", "grok4", "grok-4.1-fast-reasoning", "grok-4.1-fast-reasoning-latest"]
        for alias in alias_inputs:
            resolved = provider._resolve_model_name(alias)
            assert provider.validate_model_name(resolved), (
                f"Alias '{alias}' resolved to '{resolved}' which is not a valid model"
            )

        # Test case insensitive resolution
        for alias in ["Grok", "GROK-4.1-FAST-REASONING"]:
            resolved = provider._resolve_model_name(alias)
            assert provider.validate_model_name(resolved), (
                f"Case-insensitive alias '{alias}' resolved to '{resolved}' which is not valid"
            )

    def test_dial_provider_aliases(self):
        """Test DIAL provider's alias structure."""
        provider = DIALModelProvider("test-key")

        # Check that all models have ModelCapabilities with aliases
        for model_name, config in provider.MODEL_CAPABILITIES.items():
            assert hasattr(config, "aliases"), f"{model_name} must have aliases attribute"
            assert isinstance(config.aliases, list), f"{model_name} aliases must be a list"

        # Test that well-known aliases resolve to valid models
        well_known_aliases = ["o3", "o4-mini", "sonnet-4.1", "opus-4.1", "gemini-2.5-pro"]
        for alias in well_known_aliases:
            assert provider.validate_model_name(alias), (
                f"Alias '{alias}' should resolve to a valid DIAL model"
            )

        # Test alias resolution returns valid models
        alias_inputs = ["o3", "o4-mini", "sonnet-4.1", "opus-4.1"]
        for alias in alias_inputs:
            resolved = provider._resolve_model_name(alias)
            assert provider.validate_model_name(resolved), (
                f"Alias '{alias}' resolved to '{resolved}' which is not a valid model"
            )

        # Test case insensitive resolution
        for alias in ["O3", "SONNET-4.1"]:
            resolved = provider._resolve_model_name(alias)
            assert provider.validate_model_name(resolved), (
                f"Case-insensitive alias '{alias}' resolved to '{resolved}' which is not valid"
            )

    def test_list_models_includes_aliases(self):
        """Test that list_models returns both base models and aliases."""
        # Test Gemini
        gemini_provider = GeminiModelProvider("test-key")
        gemini_models = gemini_provider.list_models(respect_restrictions=False)
        # Verify at least some base models AND aliases are present
        gemini_aliases = get_all_aliases(ProviderType.GOOGLE)
        for model_name in gemini_aliases:
            assert model_name in gemini_models, f"Base model '{model_name}' should be in list_models"
            for alias in gemini_aliases[model_name]:
                assert alias in gemini_models, f"Alias '{alias}' should be in list_models"

        # Test OpenAI
        openai_provider = OpenAIModelProvider("test-key")
        openai_models = openai_provider.list_models(respect_restrictions=False)
        openai_aliases = get_all_aliases(ProviderType.OPENAI)
        for model_name in openai_aliases:
            assert model_name in openai_models, f"Base model '{model_name}' should be in list_models"
            for alias in openai_aliases[model_name]:
                assert alias in openai_models, f"Alias '{alias}' should be in list_models"

        # Test XAI
        xai_provider = XAIModelProvider("test-key")
        xai_models = xai_provider.list_models(respect_restrictions=False)
        xai_aliases = get_all_aliases(ProviderType.XAI)
        for model_name in xai_aliases:
            assert model_name in xai_models, f"Base model '{model_name}' should be in list_models"
            for alias in xai_aliases[model_name]:
                assert alias in xai_models, f"Alias '{alias}' should be in list_models"

        # Test DIAL
        dial_provider = DIALModelProvider("test-key")
        dial_models = dial_provider.list_models(respect_restrictions=False)
        # Just verify that the model list is non-empty and contains entries
        assert len(dial_models) > 0, "DIAL provider should have models"

    def test_list_models_all_known_variant_includes_aliases(self):
        """Unified list_models should support lowercase, alias-inclusive listings."""
        # Test Gemini
        gemini_provider = GeminiModelProvider("test-key")
        gemini_all = gemini_provider.list_models(
            respect_restrictions=False,
            include_aliases=True,
            lowercase=True,
            unique=True,
        )
        # All should be lowercase
        assert all(model == model.lower() for model in gemini_all)
        # Should include base models and aliases
        assert len(gemini_all) > 0

        # Test OpenAI
        openai_provider = OpenAIModelProvider("test-key")
        openai_all = openai_provider.list_models(
            respect_restrictions=False,
            include_aliases=True,
            lowercase=True,
            unique=True,
        )
        # All should be lowercase
        assert all(model == model.lower() for model in openai_all)
        assert len(openai_all) > 0

    def test_no_string_shorthand_in_supported_models(self):
        """Test that no provider has string-based shorthands anymore."""
        providers = [
            GeminiModelProvider("test-key"),
            OpenAIModelProvider("test-key"),
            XAIModelProvider("test-key"),
            DIALModelProvider("test-key"),
        ]

        for provider in providers:
            for model_name, config in provider.MODEL_CAPABILITIES.items():
                # All values must be ModelCapabilities objects, not strings or dicts
                from providers.shared import ModelCapabilities

                assert isinstance(config, ModelCapabilities), (
                    f"{provider.__class__.__name__}.MODEL_CAPABILITIES['{model_name}'] "
                    f"must be a ModelCapabilities object, not {type(config).__name__}"
                )

    def test_resolve_returns_original_if_not_found(self):
        """Test that _resolve_model_name returns original name if alias not found."""
        providers = [
            GeminiModelProvider("test-key"),
            OpenAIModelProvider("test-key"),
            XAIModelProvider("test-key"),
            DIALModelProvider("test-key"),
        ]

        for provider in providers:
            # Test with unknown model name
            assert provider._resolve_model_name("unknown-model") == "unknown-model"
            assert provider._resolve_model_name("gpt-4") == "gpt-4"
            assert provider._resolve_model_name("claude-3") == "claude-3"
