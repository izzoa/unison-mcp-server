"""Adapter for LiteLLM's bundled model metadata.

This module queries ``litellm.model_cost`` at import time and exposes
helpers that the registry base class uses during ``reload()`` to discover
models and populate baseline :class:`ModelCapabilities` fields.

If the ``litellm`` package is not installed the module degrades gracefully:
every public function returns empty results and a warning is logged once.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .shared import ProviderType
from .shared.temperature import FixedTemperatureConstraint, RangeTemperatureConstraint, TemperatureConstraint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of litellm — graceful fallback when not installed
# ---------------------------------------------------------------------------
_model_cost: dict[str, dict[str, Any]] = {}
_litellm_available = False

try:
    from litellm import get_model_info as _litellm_get_model_info
    from litellm import model_cost as _raw_model_cost

    _model_cost = _raw_model_cost
    _litellm_available = True
except Exception:  # pragma: no cover
    logger.warning("litellm is not installed — model discovery will use JSON files only")

    def _litellm_get_model_info(model: str, **kwargs: Any) -> dict[str, Any]:  # type: ignore[misc]
        return {}


# ---------------------------------------------------------------------------
# Provider prefix → ProviderType mapping
# ---------------------------------------------------------------------------
_PREFIX_TO_PROVIDER: dict[str, ProviderType] = {
    "openai/": ProviderType.OPENAI,
    "gemini/": ProviderType.GOOGLE,
    "vertex_ai/": ProviderType.GOOGLE,
    "azure/": ProviderType.AZURE,
    "xai/": ProviderType.XAI,
    "openrouter/": ProviderType.OPENROUTER,
    "ollama/": ProviderType.CUSTOM,
}

# Reverse mapping: ProviderType → list of litellm_provider strings to match
_PROVIDER_TO_LITELLM: dict[ProviderType, set[str]] = {
    ProviderType.OPENAI: {"openai"},
    ProviderType.GOOGLE: {"gemini", "vertex_ai"},
    ProviderType.AZURE: {"azure"},
    ProviderType.XAI: {"xai"},
    ProviderType.OPENROUTER: {"openrouter"},
    ProviderType.CUSTOM: {"ollama"},
}

# Fields that LiteLLM is authoritative for (the "base" fields)
LITELLM_AUTHORITATIVE_FIELDS = frozenset(
    {
        "context_window",
        "max_output_tokens",
        "supports_images",
        "supports_function_calling",
        "supports_json_mode",
    }
)

# Fields that JSON overrides are authoritative for (Unison-specific)
JSON_OVERRIDE_FIELDS = frozenset(
    {
        "intelligence_score",
        "aliases",
        "friendly_name",
        "description",
        "max_thinking_tokens",
        "temperature_constraint",
        "supports_temperature",
        "allow_code_generation",
        "use_openai_response_api",
        "default_reasoning_effort",
        "max_image_size_mb",
    }
)


_discovery_enabled = True


def set_discovery_enabled(enabled: bool) -> None:
    """Toggle LiteLLM model discovery globally (useful for tests)."""
    global _discovery_enabled
    _discovery_enabled = enabled


def is_available() -> bool:
    """Return True if litellm is installed, model_cost is populated, and discovery is enabled."""
    return _litellm_available and bool(_model_cost) and _discovery_enabled


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------
def get_models_for_provider(provider_type: ProviderType) -> dict[str, dict[str, Any]]:
    """Return a mapping of flat model names → LiteLLM metadata dicts for a provider.

    Only chat-mode models are included. Keys are stripped of their provider
    prefix (e.g. ``openai/gpt-5`` becomes ``gpt-5``).
    """
    if not is_available():
        return {}

    litellm_providers = _PROVIDER_TO_LITELLM.get(provider_type)
    if not litellm_providers:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for full_key, entry in _model_cost.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("mode") != "chat":
            continue
        if entry.get("litellm_provider") not in litellm_providers:
            continue

        flat_name = _strip_prefix(full_key)
        if not flat_name:
            continue

        result[flat_name] = _map_fields(flat_name, entry)

    return result


def _strip_prefix(litellm_key: str) -> str:
    """Remove the provider prefix from a LiteLLM model key."""
    for prefix in _PREFIX_TO_PROVIDER:
        if litellm_key.startswith(prefix):
            return litellm_key[len(prefix) :]
    # Some entries have no prefix (e.g. bare model names under openai)
    return litellm_key


def _map_fields(flat_name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a LiteLLM model_cost entry to a dict matching ModelCapabilities fields."""
    max_input = entry.get("max_input_tokens") or 0
    max_output = entry.get("max_output_tokens") or 0

    supports_vision = bool(entry.get("supports_vision", False))
    supports_function_calling = bool(entry.get("supports_function_calling", False))
    supports_json = bool(entry.get("supports_response_schema", False))
    supports_reasoning = bool(entry.get("supports_reasoning", False))
    supports_system = bool(entry.get("supports_system_messages", True))

    return {
        "context_window": max_input,
        "max_output_tokens": max_output,
        "supports_images": supports_vision,
        "supports_function_calling": supports_function_calling,
        "supports_json_mode": supports_json,
        "supports_extended_thinking": supports_reasoning,
        "supports_system_prompts": supports_system,
        "supports_streaming": True,
        # Pass raw reasoning flag for inference helpers
        "_litellm_supports_reasoning": supports_reasoning,
        "_litellm_raw": entry,
    }


# ---------------------------------------------------------------------------
# Inference helpers for auto-discovered models
# ---------------------------------------------------------------------------

# Pattern → intelligence score mapping (checked in order, first match wins)
_FAMILY_SCORE_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"(ultra|opus)", re.I), 17),
    (re.compile(r"pro", re.I), 16),
    (re.compile(r"(flash|fast)", re.I), 12),
    (re.compile(r"(mini|lite|small)", re.I), 10),
    (re.compile(r"nano", re.I), 7),
]


def infer_intelligence_score(model_name: str, model_info: dict[str, Any]) -> int:
    """Guess an intelligence score (1-20) from model name and metadata."""
    score = 12  # default middle ground

    for pattern, value in _FAMILY_SCORE_PATTERNS:
        if pattern.search(model_name):
            score = value
            break

    # Context window boost
    ctx = model_info.get("context_window", 0)
    if ctx >= 1_000_000:
        score = max(score, 14)
    elif ctx >= 200_000:
        score = max(score, 13)

    return min(20, max(1, score))


def infer_aliases(model_name: str) -> list[str]:
    """Generate simple aliases by stripping hyphens and dots."""
    aliases: list[str] = []

    # e.g. "gpt-5.2-mini" → "gpt5.2-mini"
    no_hyphens_first = re.sub(r"^([a-zA-Z]+)-", r"\1", model_name)
    if no_hyphens_first != model_name:
        aliases.append(no_hyphens_first)

    return aliases


def infer_thinking_support(model_name: str, model_info: dict[str, Any]) -> tuple[bool, int]:
    """Return (supports_extended_thinking, max_thinking_tokens)."""
    if model_info.get("_litellm_supports_reasoning") or model_info.get("supports_extended_thinking"):
        return True, 32768
    return False, 0


def infer_temperature_constraint(model_name: str, model_info: dict[str, Any]) -> tuple[bool, TemperatureConstraint]:
    """Return (supports_temperature, constraint)."""
    name_lower = model_name.lower()

    # Check for known reasoning-only patterns
    reasoning_patterns = ("o1", "o3", "o4", "-reasoning")
    for pat in reasoning_patterns:
        if pat in name_lower:
            return False, FixedTemperatureConstraint(1.0)

    return True, RangeTemperatureConstraint(0.0, 2.0, 0.3)


def infer_defaults(
    model_name: str,
    model_info: dict[str, Any],
    provider_type: ProviderType,
) -> dict[str, Any]:
    """Build a complete defaults dict for an auto-discovered model.

    The returned dict contains all Unison-specific fields with inferred or
    safe-default values, ready to be merged with LiteLLM's base fields.
    """
    score = infer_intelligence_score(model_name, model_info)
    aliases = infer_aliases(model_name)
    thinks, think_tokens = infer_thinking_support(model_name, model_info)
    supports_temp, temp_constraint = infer_temperature_constraint(model_name, model_info)

    supports_images = model_info.get("supports_images", False)

    provider_labels = {
        ProviderType.OPENAI: "OpenAI",
        ProviderType.GOOGLE: "Gemini",
        ProviderType.AZURE: "Azure",
        ProviderType.XAI: "X.AI",
        ProviderType.OPENROUTER: "OpenRouter",
        ProviderType.CUSTOM: "Custom",
        ProviderType.DIAL: "DIAL",
    }
    label = provider_labels.get(provider_type, "Unknown")

    return {
        "intelligence_score": score,
        "aliases": aliases,
        "friendly_name": f"{label} ({model_name})",
        "description": "Auto-discovered via LiteLLM",
        "max_thinking_tokens": think_tokens,
        "supports_extended_thinking": thinks,
        "supports_temperature": supports_temp,
        "temperature_constraint": temp_constraint,
        "allow_code_generation": False,
        "use_openai_response_api": False,
        "default_reasoning_effort": None,
        "max_image_size_mb": 20.0 if supports_images else 0.0,
    }
