"""Dynamic model query helpers for tests.

Instead of hardcoding model names like ``"gemini-2.5-flash"`` in assertions,
tests should use these helpers to query the live registry.  When the model
catalog changes (via LiteLLM refresh or manual edits), tests automatically
adapt without breaking.

Usage::

    from tests.model_test_helpers import get_flagship_model, get_flash_model

    def test_auto_mode_picks_flash():
        model = auto_mode_select(ToolModelCategory.FAST_RESPONSE)
        assert model == get_flash_model(ProviderType.GOOGLE)
"""

from __future__ import annotations

from providers.shared import ModelCapabilities, ProviderType

_FAST_TIER_PATTERNS = ("flash", "mini", "lite", "fast", "nano")


def _get_caps(provider_type: ProviderType) -> dict[str, ModelCapabilities]:
    """Return MODEL_CAPABILITIES for a provider (uses the cached registry)."""
    from providers.gemini import GeminiModelProvider
    from providers.openai import OpenAIModelProvider
    from providers.xai import XAIModelProvider

    provider_map = {
        ProviderType.GOOGLE: GeminiModelProvider,
        ProviderType.OPENAI: OpenAIModelProvider,
        ProviderType.XAI: XAIModelProvider,
    }

    cls = provider_map.get(provider_type)
    if cls is None:
        return {}
    cls._ensure_registry()
    return dict(cls.MODEL_CAPABILITIES)


def get_any_model(provider_type: ProviderType) -> str | None:
    """Return any valid model name from the provider, or None."""
    caps = _get_caps(provider_type)
    return next(iter(caps), None)


def get_flagship_model(provider_type: ProviderType) -> str | None:
    """Return the highest-intelligence-scored model for the provider."""
    caps = _get_caps(provider_type)
    if not caps:
        return None
    return max(caps, key=lambda m: caps[m].intelligence_score)


def get_flash_model(provider_type: ProviderType) -> str | None:
    """Return a fast-tier model (flash/mini/lite/fast) for the provider."""
    caps = _get_caps(provider_type)
    fast = [m for m in caps if any(p in m.lower() for p in _FAST_TIER_PATTERNS)]
    if not fast:
        return None
    return max(fast, key=lambda m: caps[m].intelligence_score)


def get_model_with_thinking(provider_type: ProviderType) -> str | None:
    """Return a model with ``supports_extended_thinking``, or None."""
    caps = _get_caps(provider_type)
    thinking = [m for m in caps if caps[m].supports_extended_thinking]
    if not thinking:
        return None
    return max(thinking, key=lambda m: caps[m].intelligence_score)


def get_all_model_names(provider_type: ProviderType) -> list[str]:
    """Return all model names for the provider."""
    return list(_get_caps(provider_type).keys())


def get_all_aliases(provider_type: ProviderType) -> dict[str, list[str]]:
    """Return {model_name: [aliases]} for the provider."""
    caps = _get_caps(provider_type)
    return {m: c.aliases for m, c in caps.items() if c.aliases}


def is_valid_model(provider_type: ProviderType, model_name: str) -> bool:
    """Check if a model name exists in the provider's registry."""
    caps = _get_caps(provider_type)
    if model_name in caps:
        return True
    # Check aliases
    for c in caps.values():
        if model_name.lower() in [a.lower() for a in c.aliases]:
            return True
    return False
