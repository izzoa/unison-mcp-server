"""Tests for the ``supports_batch`` capability flag and its catalog data."""

import json
import re
from pathlib import Path
from unittest.mock import patch

from providers.registries.openrouter import OpenRouterModelRegistry
from providers.shared import ModelCapabilities, ProviderType

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "conf" / "openrouter_models.json"
BATCH_LISTINGS_PATH = Path(__file__).parent / "fixtures" / "openrouter_batch_listings.json"

# Curated entries with a '<slug>:batch' listing in OpenRouter's public catalog (2026-11-21).
EXPECTED_BATCH_MODELS = {
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.1",
    "openai/o3",
    "openai/o3-mini",
    "openai/o4-mini",
    "openai/gpt-5",
    "openai/gpt-5.2-pro",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5.2",
}


def _caps(**overrides):
    return ModelCapabilities(provider=ProviderType.OPENROUTER, model_name="x/y", friendly_name="X", **overrides)


class TestSupportsBatchField:
    def test_defaults_to_false_when_unspecified(self):
        assert _caps().supports_batch is False

    def test_can_be_enabled_explicitly(self):
        assert _caps(supports_batch=True).supports_batch is True


class TestOpenRouterCatalogBatchFlag:
    def test_known_batch_priced_model_reports_true(self):
        registry = OpenRouterModelRegistry()
        assert registry.resolve("anthropic/claude-opus-4.5").supports_batch is True
        assert registry.resolve("openai/gpt-5.2").supports_batch is True

    def test_unlisted_model_reports_false(self):
        registry = OpenRouterModelRegistry()
        assert registry.resolve("google/gemini-2.5-pro").supports_batch is False
        assert registry.resolve("x-ai/grok-4").supports_batch is False

    def test_alias_resolves_to_batch_capability(self):
        registry = OpenRouterModelRegistry()
        assert registry.resolve("opus").supports_batch is True

    def test_exactly_the_expected_entries_are_batch_capable(self):
        registry = OpenRouterModelRegistry()
        batch_models = {
            name for name in registry.list_models() if (registry.resolve(name) or _caps()).supports_batch is True
        }
        # Only compare curated entries (LiteLLM auto-discovery is never batch capable)
        curated = {m["model_name"] for m in json.loads(CATALOG_PATH.read_text())["models"]}
        assert batch_models & curated == EXPECTED_BATCH_MODELS

    def test_flag_matches_catalog_snapshot_of_batch_listings(self):
        """supports_batch is derived from '<slug>:batch' listings, not hardcoded."""
        snapshot = json.loads(BATCH_LISTINGS_PATH.read_text())["batch_listings"]
        listed = set(snapshot)
        entries = json.loads(CATALOG_PATH.read_text())["models"]
        for entry in entries:
            expected = f"{entry['model_name']}:batch" in listed
            assert entry["supports_batch"] is expected, entry["model_name"]

    def test_field_is_documented_in_readme(self):
        readme = json.loads(CATALOG_PATH.read_text())["_README"]["field_descriptions"]
        assert "supports_batch" in readme
        assert re.search(r":batch", readme["supports_batch"])

    def test_litellm_auto_discovered_models_default_false(self):
        from providers.registries.base import CAPABILITY_FIELD_NAMES

        assert "supports_batch" in CAPABILITY_FIELD_NAMES
        litellm_models = {
            "acme/shiny-model": {
                "context_window": 32000,
                "max_output_tokens": 4096,
                "supports_images": False,
                "supports_function_calling": True,
                "supports_json_mode": True,
            }
        }
        registry = OpenRouterModelRegistry()
        with (
            patch("providers.litellm_adapter.is_available", return_value=True),
            patch("providers.litellm_adapter.get_models_for_provider", return_value=litellm_models),
        ):
            merged = registry._merge_with_litellm([])
        by_name = {c.model_name: c for c in merged}
        assert by_name["acme/shiny-model"].auto_discovered is True
        assert by_name["acme/shiny-model"].supports_batch is False
