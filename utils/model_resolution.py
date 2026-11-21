"""
Model resolution utilities for the MCP server.

Contains trim_model_identifier() (identifier preservation), resolution_metadata() (reporting
alias resolution in response metadata), and resolve_fallback_model() for unified fallback
model resolution.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def trim_model_identifier(model_string: str) -> str:
    """
    Return a caller-supplied model identifier exactly as given, minus surrounding whitespace.

    A model identifier belongs to the provider that serves it, so nothing inside it is
    interpreted here. A colon-delimited segment (``:free``, ``:batch``, ``:online``,
    ``:exacto``, an Ollama ``:3b`` tag, or a variant a provider has not shipped yet) is part of
    the identifier and reaches the provider intact. Interpreting segments against a server-side
    list silently resolved a different (often pricier, or less capable) model than the caller
    named; an identifier the provider does not serve now fails visibly as an unknown model.

    Server-defined per-model options are never encoded into the identifier: consensus stance is
    read from its own ``stance`` field.

    Args:
        model_string: The model identifier as supplied by the caller.

    Returns:
        The identifier with leading and trailing whitespace removed.
    """
    return model_string.strip()


def resolution_metadata(requested: str | None, resolved: str | None) -> dict[str, str]:
    """
    Describe a difference between the model a caller named and the model that ran.

    Alias resolution legitimately maps one identifier to another (``flash`` to
    ``gemini-2.5-flash``). When it does, the response says so instead of hiding it, which keeps a
    resolution to a differently priced or differently capable model from ever being silent.

    Args:
        requested: The identifier the caller supplied.
        resolved: The identifier the provider actually used.

    Returns:
        ``{"model_requested": ..., "model_resolved": ...}`` when the two differ (ignoring case
        and surrounding whitespace), otherwise an empty dict.
    """
    if not isinstance(requested, str) or not isinstance(resolved, str) or not requested or not resolved:
        return {}
    requested = requested.strip()
    resolved = resolved.strip()
    if requested.casefold() == resolved.casefold():
        return {}
    return {"model_requested": requested, "model_resolved": resolved}


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
