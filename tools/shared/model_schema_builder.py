"""
Model schema building functionality for Unison MCP tools.

This module provides the ModelSchemaBuilder class which encapsulates all logic
related to building JSON schema definitions for the model field in tool schemas.
It handles model enumeration, availability checks, capability ranking, and
auto-mode detection.

Extracted from BaseTool to improve separation of concerns and testability.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tools.shared.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ModelSchemaBuilder:
    """
    Builds the JSON schema for the model field in MCP tool definitions.

    This class encapsulates model enumeration, ranking, formatting, and
    schema generation logic that was previously embedded in BaseTool.

    Args:
        tool_name: The unique name identifier for the tool.
        tool: Optional reference to the BaseTool instance for accessing
              get_model_category(), registry caches, and other tool methods.
    """

    def __init__(self, tool_name: str, tool: Optional["BaseTool"] = None):
        self.tool_name = tool_name
        self.tool = tool

    def is_effective_auto_mode(self) -> bool:
        """
        Check if we're in effective auto mode for schema generation.

        This determines whether the model parameter should be required in the tool schema.
        Used at initialization time when schemas are generated.

        Returns:
            bool: True if model parameter should be required in the schema
        """
        from config import DEFAULT_MODEL
        from providers.registry import ModelProviderRegistry

        # Case 1: Explicit auto mode
        if DEFAULT_MODEL.lower() == "auto":
            return True

        # Case 2: Model not available (fallback to auto mode)
        if DEFAULT_MODEL.lower() != "auto":
            provider = ModelProviderRegistry.get_provider_for_model(DEFAULT_MODEL)
            if not provider:
                return True

        return False

    def _should_require_model_selection(self, model_name: str) -> bool:
        """
        Check if we should require the CLI to select a model at runtime.

        This is called during request execution to determine if we need
        to return an error asking the CLI to provide a model parameter.

        Args:
            model_name: The model name from the request or DEFAULT_MODEL

        Returns:
            bool: True if we should require model selection
        """
        # Case 1: Model is explicitly "auto"
        if model_name.lower() == "auto":
            return True

        # Case 2: Requested model is not available
        from providers.registry import ModelProviderRegistry

        provider = ModelProviderRegistry.get_provider_for_model(model_name)
        if not provider:
            logger.warning(f"Model '{model_name}' is not available with current API keys. Requiring model selection.")
            return True

        return False

    def _get_available_models(self) -> list[str]:
        """
        Get list of models available from enabled providers.

        Only returns models from providers that have valid API keys configured.
        This fixes the namespace collision bug where models from disabled providers
        were shown to the CLI, causing routing conflicts.

        Returns:
            List of model names from enabled providers only
        """
        from providers.registry import ModelProviderRegistry
        from utils.env import get_env

        # Get models from enabled providers only (those with valid API keys)
        all_models = ModelProviderRegistry.get_available_model_names()

        # Add OpenRouter models if OpenRouter is configured
        openrouter_key = get_env("OPENROUTER_API_KEY")
        if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
            try:
                registry = self.tool._get_openrouter_registry()
                # Add all aliases from the registry (includes OpenRouter cloud models)
                for alias in registry.list_aliases():
                    if alias not in all_models:
                        all_models.append(alias)
            except Exception as e:
                import logging

                logging.debug(f"Failed to add OpenRouter models to enum: {e}")

        # Add custom models if custom API is configured
        custom_url = get_env("CUSTOM_API_URL")
        if custom_url:
            try:
                registry = self.tool._get_custom_registry()
                for alias in registry.list_aliases():
                    if alias not in all_models:
                        all_models.append(alias)
            except Exception as e:
                import logging

                logging.debug(f"Failed to add custom models to enum: {e}")

        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for model in all_models:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)

        return unique_models

    def _format_available_models_list(self) -> str:
        """Return a human-friendly list of available models or guidance when none found."""

        summaries, total, has_restrictions = self._get_ranked_model_summaries()
        if not summaries:
            return (
                "No models detected. Configure provider credentials or set DEFAULT_MODEL to a valid option. "
                "If the user requested a specific model, respond with this notice instead of substituting another model."
            )
        display = "; ".join(summaries)
        remainder = total - len(summaries)
        if remainder > 0:
            display = f"{display}; +{remainder} more (use the `listmodels` tool for the full roster)"
        return display

    @staticmethod
    def _format_context_window(tokens: int) -> Optional[str]:
        """Convert a raw context window into a short display string."""

        if not tokens or tokens <= 0:
            return None

        if tokens >= 1_000_000:
            if tokens % 1_000_000 == 0:
                return f"{tokens // 1_000_000}M ctx"
            return f"{tokens / 1_000_000:.1f}M ctx"

        if tokens >= 1_000:
            if tokens % 1_000 == 0:
                return f"{tokens // 1_000}K ctx"
            return f"{tokens / 1_000:.1f}K ctx"

        return f"{tokens} ctx"

    def _collect_ranked_capabilities(self) -> list[tuple[int, str, Any]]:
        """Gather available model capabilities sorted by capability rank."""

        from providers.registry import ModelProviderRegistry

        ranked: list[tuple[int, str, Any]] = []
        available = ModelProviderRegistry.get_available_models(respect_restrictions=True)

        for model_name, provider_type in available.items():
            provider = ModelProviderRegistry.get_provider(provider_type)
            if not provider:
                continue

            try:
                capabilities = provider.get_capabilities(model_name)
            except ValueError:
                continue

            rank = capabilities.get_effective_capability_rank()
            ranked.append((rank, model_name, capabilities))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    @staticmethod
    def _normalize_model_identifier(name: str) -> str:
        """Normalize model names for deduplication across providers."""

        normalized = name.lower()
        if ":" in normalized:
            normalized = normalized.split(":", 1)[0]
        if "/" in normalized:
            normalized = normalized.split("/", 1)[-1]
        return normalized

    def _get_ranked_model_summaries(self, limit: int = 5) -> tuple[list[str], int, bool]:
        """Return formatted, ranked model summaries and restriction status."""

        ranked = self._collect_ranked_capabilities()

        # Build allowlist map (provider -> lowercase names) when restrictions are active
        allowed_map: dict[Any, set[str]] = {}
        try:
            from utils.model_restrictions import get_restriction_service

            restriction_service = get_restriction_service()
            if restriction_service:
                from providers.shared import ProviderType

                for provider_type in ProviderType:
                    allowed = restriction_service.get_allowed_models(provider_type)
                    if allowed:
                        allowed_map[provider_type] = {name.lower() for name in allowed if name}
        except Exception:
            allowed_map = {}

        filtered: list[tuple[int, str, Any]] = []
        seen_normalized: set[str] = set()

        for rank, model_name, capabilities in ranked:
            canonical_name = getattr(capabilities, "model_name", model_name)
            canonical_lower = canonical_name.lower()
            alias_lower = model_name.lower()
            provider_type = getattr(capabilities, "provider", None)

            if allowed_map:
                if provider_type not in allowed_map:
                    continue
                allowed_set = allowed_map[provider_type]
                if canonical_lower not in allowed_set and alias_lower not in allowed_set:
                    continue

            normalized = self._normalize_model_identifier(canonical_name)
            if normalized in seen_normalized:
                continue

            seen_normalized.add(normalized)
            filtered.append((rank, canonical_name, capabilities))

        summaries: list[str] = []
        for rank, canonical_name, capabilities in filtered[:limit]:
            details: list[str] = []

            context_str = self._format_context_window(capabilities.context_window)
            if context_str:
                details.append(context_str)

            if capabilities.supports_extended_thinking:
                details.append("thinking")

            if capabilities.allow_code_generation:
                details.append("code-gen")

            base = f"{canonical_name} (score {rank}"
            if details:
                base = f"{base}, {', '.join(details)}"
            summaries.append(f"{base})")

        return summaries, len(filtered), bool(allowed_map)

    def _get_restriction_note(self) -> Optional[str]:
        """Return a string describing active per-provider allowlists, if any."""
        from utils.env import get_env

        env_labels = {
            "OPENAI_ALLOWED_MODELS": "OpenAI",
            "GOOGLE_ALLOWED_MODELS": "Google",
            "XAI_ALLOWED_MODELS": "X.AI",
            "OPENROUTER_ALLOWED_MODELS": "OpenRouter",
            "DIAL_ALLOWED_MODELS": "DIAL",
        }

        notes: list[str] = []
        for env_var, label in env_labels.items():
            raw = get_env(env_var)
            if not raw:
                continue

            models = sorted({token.strip() for token in raw.split(",") if token.strip()})
            if not models:
                continue

            notes.append(f"{label}: {', '.join(models)}")

        if not notes:
            return None

        return "Policy allows only \u2192 " + "; ".join(notes)

    def _build_model_unavailable_message(self, model_name: str) -> str:
        """Compose a consistent error message for unavailable model scenarios."""
        from providers import ModelProviderRegistry

        tool_category = self.tool.get_model_category()
        suggested_model = ModelProviderRegistry.get_preferred_fallback_model(tool_category)
        available_models_text = self._format_available_models_list()

        return (
            f"Model '{model_name}' is not available with current API keys. "
            f"Available models: {available_models_text}. "
            f"Suggested model for {self.tool.get_name()}: '{suggested_model}' "
            f"(category: {tool_category.value}). If the user explicitly requested a model, you MUST use that exact name or report this error back\u2014do not substitute another model."
        )

    def _build_auto_mode_required_message(self) -> str:
        """Compose the auto-mode prompt when an explicit model selection is required."""
        from providers import ModelProviderRegistry

        tool_category = self.tool.get_model_category()
        suggested_model = ModelProviderRegistry.get_preferred_fallback_model(tool_category)
        available_models_text = self._format_available_models_list()

        return (
            "Model parameter is required in auto mode. "
            f"Available models: {available_models_text}. "
            f"Suggested model for {self.tool.get_name()}: '{suggested_model}' "
            f"(category: {tool_category.value}). When the user names a model, relay that exact name\u2014never swap in another option."
        )

    def get_model_field_schema(self) -> dict[str, Any]:
        """
        Generate the model field schema based on auto mode configuration.

        When auto mode is enabled, the model parameter becomes required
        and includes detailed descriptions of each model's capabilities.

        Returns:
            Dict containing the model field JSON schema
        """

        from config import DEFAULT_MODEL

        # Use the centralized effective auto mode check
        if self.is_effective_auto_mode():
            description = (
                "Currently in auto model selection mode. CRITICAL: When the user names a model, you MUST use that exact name unless the server rejects it. "
                "If no model is provided, you may use the `listmodels` tool to review options and select an appropriate match."
            )
            summaries, total, restricted = self._get_ranked_model_summaries()
            remainder = max(0, total - len(summaries))
            if summaries:
                top_line = "; ".join(summaries)
                if remainder > 0:
                    label = "Allowed models" if restricted else "Top models"
                    top_line = f"{label}: {top_line}; +{remainder} more via `listmodels`."
                else:
                    label = "Allowed models" if restricted else "Top models"
                    top_line = f"{label}: {top_line}."
                description = f"{description} {top_line}"

            restriction_note = self._get_restriction_note()
            if restriction_note and (remainder > 0 or not summaries):
                description = f"{description} {restriction_note}."
            return {
                "type": "string",
                "description": description,
            }

        description = (
            f"The default model is '{DEFAULT_MODEL}'. Override only when the user explicitly requests a different model, and use that exact name. "
            "If the requested model fails validation, surface the server error instead of substituting another model. When unsure, use the `listmodels` tool for details."
        )
        summaries, total, restricted = self._get_ranked_model_summaries()
        remainder = max(0, total - len(summaries))
        if summaries:
            top_line = "; ".join(summaries)
            if remainder > 0:
                label = "Allowed models" if restricted else "Preferred alternatives"
                top_line = f"{label}: {top_line}; +{remainder} more via `listmodels`."
            else:
                label = "Allowed models" if restricted else "Preferred alternatives"
                top_line = f"{label}: {top_line}."
            description = f"{description} {top_line}"

        restriction_note = self._get_restriction_note()
        if restriction_note and (remainder > 0 or not summaries):
            description = f"{description} {restriction_note}."

        return {
            "type": "string",
            "description": description,
        }
