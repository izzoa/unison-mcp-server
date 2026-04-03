"""Tests for the LiteLLM adapter module — prefix mapping, field mapping, inference, and merge."""

import json
import tempfile

import pytest

from providers.litellm_adapter import set_discovery_enabled
from providers.shared import ProviderType
from providers.shared.temperature import FixedTemperatureConstraint, RangeTemperatureConstraint


@pytest.fixture(autouse=True)
def enable_litellm_discovery():
    """Re-enable LiteLLM discovery for this test module (disabled globally in conftest)."""
    set_discovery_enabled(True)
    yield
    set_discovery_enabled(False)


class TestLiteLLMAdapterPrefixMapping:
    """Test provider prefix → ProviderType mapping."""

    def test_strip_openai_prefix(self):
        from providers.litellm_adapter import _strip_prefix

        assert _strip_prefix("openai/gpt-5") == "gpt-5"

    def test_strip_gemini_prefix(self):
        from providers.litellm_adapter import _strip_prefix

        assert _strip_prefix("gemini/gemini-3-pro-preview") == "gemini-3-pro-preview"

    def test_strip_xai_prefix(self):
        from providers.litellm_adapter import _strip_prefix

        assert _strip_prefix("xai/grok-4") == "grok-4"

    def test_strip_vertex_ai_prefix(self):
        from providers.litellm_adapter import _strip_prefix

        assert _strip_prefix("vertex_ai/gemini-2.5-pro") == "gemini-2.5-pro"

    def test_strip_ollama_prefix(self):
        from providers.litellm_adapter import _strip_prefix

        assert _strip_prefix("ollama/llama3.2") == "llama3.2"

    def test_no_prefix_passthrough(self):
        from providers.litellm_adapter import _strip_prefix

        assert _strip_prefix("some-bare-model") == "some-bare-model"


class TestLiteLLMFieldMapping:
    """Test LiteLLM → ModelCapabilities field mapping."""

    def test_map_fields_basic(self):
        from providers.litellm_adapter import _map_fields

        entry = {
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "supports_vision": True,
            "supports_function_calling": True,
            "supports_response_schema": True,
            "supports_reasoning": False,
            "supports_system_messages": True,
            "mode": "chat",
        }
        result = _map_fields("gpt-4o", entry)

        assert result["context_window"] == 128000
        assert result["max_output_tokens"] == 16384
        assert result["supports_images"] is True
        assert result["supports_function_calling"] is True
        assert result["supports_json_mode"] is True
        assert result["supports_extended_thinking"] is False

    def test_map_fields_missing_values_default_to_zero_or_false(self):
        from providers.litellm_adapter import _map_fields

        result = _map_fields("minimal-model", {})

        assert result["context_window"] == 0
        assert result["max_output_tokens"] == 0
        assert result["supports_images"] is False
        assert result["supports_function_calling"] is False


class TestInferIntelligenceScore:
    """Test intelligence score inference from model name."""

    def test_pro_model(self):
        from providers.litellm_adapter import infer_intelligence_score

        assert infer_intelligence_score("gemini-3-pro-preview", {}) == 16

    def test_ultra_model(self):
        from providers.litellm_adapter import infer_intelligence_score

        assert infer_intelligence_score("gemini-ultra", {}) == 17

    def test_flash_model(self):
        from providers.litellm_adapter import infer_intelligence_score

        assert infer_intelligence_score("gemini-2.5-flash", {}) == 12

    def test_mini_model(self):
        from providers.litellm_adapter import infer_intelligence_score

        assert infer_intelligence_score("gpt-4.1-mini", {}) == 10

    def test_nano_model(self):
        from providers.litellm_adapter import infer_intelligence_score

        assert infer_intelligence_score("gpt-4.1-nano", {}) == 7

    def test_unknown_defaults_to_12(self):
        from providers.litellm_adapter import infer_intelligence_score

        assert infer_intelligence_score("some-unknown-model", {}) == 12

    def test_large_context_boosts_score(self):
        from providers.litellm_adapter import infer_intelligence_score

        # nano would normally be 7 but 1M context boosts it to at least 14
        assert infer_intelligence_score("nano-thing", {"context_window": 1_000_000}) >= 14

    def test_medium_context_boosts_score(self):
        from providers.litellm_adapter import infer_intelligence_score

        assert infer_intelligence_score("nano-thing", {"context_window": 200_000}) >= 13


class TestInferAliases:
    """Test alias generation."""

    def test_strips_first_hyphen(self):
        from providers.litellm_adapter import infer_aliases

        aliases = infer_aliases("gpt-5.2-mini")
        assert "gpt5.2-mini" in aliases

    def test_no_alias_when_no_leading_hyphen(self):
        from providers.litellm_adapter import infer_aliases

        aliases = infer_aliases("llama3.2")
        assert aliases == []


class TestInferThinkingSupport:
    """Test extended thinking inference."""

    def test_reasoning_model(self):
        from providers.litellm_adapter import infer_thinking_support

        supports, tokens = infer_thinking_support("o3-mini", {"_litellm_supports_reasoning": True})
        assert supports is True
        assert tokens == 32768

    def test_non_reasoning_model(self):
        from providers.litellm_adapter import infer_thinking_support

        supports, tokens = infer_thinking_support("gpt-4o", {"_litellm_supports_reasoning": False})
        assert supports is False
        assert tokens == 0


class TestInferTemperatureConstraint:
    """Test temperature constraint inference."""

    def test_reasoning_model_fixed(self):
        from providers.litellm_adapter import infer_temperature_constraint

        supports, constraint = infer_temperature_constraint("o3-mini", {})
        assert supports is False
        assert isinstance(constraint, FixedTemperatureConstraint)

    def test_o4_model_fixed(self):
        from providers.litellm_adapter import infer_temperature_constraint

        supports, constraint = infer_temperature_constraint("o4-mini", {})
        assert supports is False

    def test_standard_model_range(self):
        from providers.litellm_adapter import infer_temperature_constraint

        supports, constraint = infer_temperature_constraint("gpt-5", {})
        assert supports is True
        assert isinstance(constraint, RangeTemperatureConstraint)

    def test_reasoning_suffix(self):
        from providers.litellm_adapter import infer_temperature_constraint

        supports, constraint = infer_temperature_constraint("grok-4-fast-reasoning", {})
        assert supports is False


class TestInferDefaults:
    """Test full defaults inference."""

    def test_defaults_structure(self):
        from providers.litellm_adapter import infer_defaults

        defaults = infer_defaults(
            "gpt-5",
            {"context_window": 400000, "supports_images": True},
            ProviderType.OPENAI,
        )

        assert defaults["intelligence_score"] == 13  # no family pattern, but 400K context boosts from 12 to 13
        assert defaults["friendly_name"] == "OpenAI (gpt-5)"
        assert defaults["allow_code_generation"] is False
        assert defaults["max_image_size_mb"] == 20.0

    def test_defaults_no_vision(self):
        from providers.litellm_adapter import infer_defaults

        defaults = infer_defaults("model", {"supports_images": False}, ProviderType.GOOGLE)
        assert defaults["max_image_size_mb"] == 0.0


class TestGracefulFallback:
    """Test behavior when litellm is not available."""

    def test_is_available_reflects_import(self):
        from providers.litellm_adapter import is_available

        # litellm is installed in test env, so should be True
        assert is_available() is True

    def test_get_models_empty_for_unknown_provider(self):
        from providers.litellm_adapter import get_models_for_provider

        result = get_models_for_provider(ProviderType.DIAL)
        assert result == {}


class TestRegistryMerge:
    """Test the merge logic in CustomModelRegistryBase."""

    def test_json_only_model_preserved(self):
        """Models in JSON but not LiteLLM should be kept as-is."""
        config_data = {
            "models": [
                {
                    "model_name": "custom-internal-model",
                    "aliases": ["internal"],
                    "intelligence_score": 15,
                    "description": "Internal model",
                    "context_window": 8192,
                    "max_output_tokens": 4096,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            from providers.registries.openai import OpenAIModelRegistry

            registry = OpenAIModelRegistry(config_path=temp_path)
            # The custom model should still be there
            caps = registry.resolve("custom-internal-model")
            assert caps is not None
            assert caps.intelligence_score == 15
        finally:
            import os

            os.unlink(temp_path)

    def test_litellm_models_auto_discovered(self):
        """LiteLLM-only models should appear with auto_discovered=True."""
        config_data = {"models": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            from providers.registries.openai import OpenAIModelRegistry

            registry = OpenAIModelRegistry(config_path=temp_path)
            models = registry.list_models()
            # With empty JSON, all models come from LiteLLM
            if models:
                caps = registry.model_map[models[0]]
                assert caps.auto_discovered is True
        finally:
            import os

            os.unlink(temp_path)

    def test_curated_models_not_overwritten_by_litellm(self):
        """When model exists in both, JSON values should be preserved (curated wins)."""
        # gpt-4o exists in litellm. Create a JSON entry with specific values.
        config_data = {
            "models": [
                {
                    "model_name": "gpt-4o",
                    "aliases": ["4o"],
                    "intelligence_score": 15,
                    "description": "Test override",
                    "context_window": 999,
                    "max_output_tokens": 1,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            from providers.registries.openai import OpenAIModelRegistry

            registry = OpenAIModelRegistry(config_path=temp_path)
            caps = registry.resolve("gpt-4o")
            assert caps is not None
            # JSON values should be preserved — curated models are authoritative
            assert caps.context_window == 999
            assert caps.max_output_tokens == 1
            assert caps.intelligence_score == 15
            assert caps.description == "Test override"
            assert "4o" in caps.aliases
        finally:
            import os

            os.unlink(temp_path)

    def test_alias_conflict_resolution(self):
        """Auto-discovered model aliases should not conflict with curated aliases."""
        config_data = {
            "models": [
                {
                    "model_name": "gpt-4o",
                    "aliases": ["4o", "gpt4o"],
                    "intelligence_score": 15,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            from providers.registries.openai import OpenAIModelRegistry

            registry = OpenAIModelRegistry(config_path=temp_path)
            # Verify no duplicate alias errors were raised during load
            assert registry.resolve("gpt-4o") is not None
            assert registry.resolve("4o") is not None
        finally:
            import os

            os.unlink(temp_path)
