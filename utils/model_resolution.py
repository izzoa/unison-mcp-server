"""
Model resolution utilities for the MCP server.

Contains parse_model_option() for parsing model:option format strings,
and resolve_fallback_model() for unified fallback model resolution.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def parse_model_option(model_string: str) -> tuple[str, Optional[str]]:
    """
    Parse model:option format into model name and option.

    Handles different formats:
    - OpenRouter models: preserve :free, :beta, :preview suffixes as part of model name
    - Ollama/Custom models: split on : to extract tags like :latest
    - Consensus stance: extract options like :for, :against

    Args:
        model_string: String that may contain "model:option" format

    Returns:
        tuple: (model_name, option) where option may be None
    """
    if ":" in model_string and not model_string.startswith("http"):  # Avoid parsing URLs
        # Check if this looks like an OpenRouter model (contains /)
        if "/" in model_string and model_string.count(":") == 1:
            # Could be openai/gpt-4:something - check what comes after colon
            parts = model_string.split(":", 1)
            suffix = parts[1].strip().lower()

            # Known OpenRouter suffixes to preserve
            if suffix in ["free", "beta", "preview"]:
                return model_string.strip(), None

        # For other patterns (Ollama tags, consensus stances), split normally
        parts = model_string.split(":", 1)
        model_name = parts[0].strip()
        model_option = parts[1].strip() if len(parts) > 1 else None
        return model_name, model_option
    return model_string.strip(), None


def resolve_fallback_model(tool: Any, error_context: str) -> str:
    """
    Resolve a fallback model when the primary model is unavailable.

    Tries in order:
    1. Tool's preferred model for its category via get_preferred_fallback_model()
    2. First available model from the registry
    3. Raises ValueError if no models are available

    Args:
        tool: Tool instance (may be None) to determine model category preference
        error_context: Description of why fallback is needed (for error messages and logging)

    Returns:
        Model name string for the fallback model

    Raises:
        ValueError: If no fallback model can be found
    """
    from providers.registry import get_default_registry

    registry = get_default_registry()
    fallback_model = None
    if tool is not None:
        try:
            fallback_model = registry.get_preferred_fallback_model(tool.get_model_category())
        except Exception as fallback_exc:
            logger.debug(f"Unable to resolve fallback model for {tool.name}: {fallback_exc}")

    if fallback_model is None:
        available_models = registry.get_available_model_names()
        if available_models:
            fallback_model = available_models[0]

    if fallback_model is None:
        raise ValueError(f"No fallback model available: {error_context}")

    return fallback_model
