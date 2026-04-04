"""Model provider registry for managing available providers."""

import logging
from typing import TYPE_CHECKING, Optional

from utils.env import get_env

from .base import ModelProvider
from .shared import ProviderType

if TYPE_CHECKING:
    from tools.models import ToolModelCategory


# ---------------------------------------------------------------------------
# Module-level default registry accessor
# ---------------------------------------------------------------------------

_default_registry: Optional["ModelProviderRegistry"] = None


def get_default_registry() -> "ModelProviderRegistry":
    """Return the server-created default registry instance.

    Raises:
        RuntimeError: If ``set_default_registry()`` has not been called yet.
    """
    if _default_registry is None:
        raise RuntimeError("Registry not initialized — call set_default_registry() during server startup")
    return _default_registry


def set_default_registry(registry: "ModelProviderRegistry") -> None:
    """Set the module-level default registry (called once at server startup)."""
    global _default_registry
    _default_registry = registry


# ---------------------------------------------------------------------------
# ModelProviderRegistry — regular class (no singleton)
# ---------------------------------------------------------------------------


class ModelProviderRegistry:
    """Central catalogue of provider implementations used by the MCP server.

    Role
        Holds the mapping between :class:`ProviderType` values and concrete
        :class:`ModelProvider` subclasses/factories.  At runtime the registry
        is responsible for instantiating providers, caching them for reuse, and
        mediating lookup of providers and model names in provider priority
        order.

    Core responsibilities
        * Resolve API keys and other runtime configuration for each provider
        * Lazily create provider instances so unused backends incur no cost
        * Expose convenience methods for enumerating available models and
          locating which provider can service a requested model name or alias
        * Honour the project-wide provider priority policy so namespaces (or
          alias collisions) are resolved deterministically.
    """

    # Provider priority order for model selection
    # Native APIs first, then custom endpoints, then catch-all providers
    PROVIDER_PRIORITY_ORDER = [
        ProviderType.GOOGLE,  # Direct Gemini access
        ProviderType.OPENAI,  # Direct OpenAI access
        ProviderType.AZURE,  # Azure-hosted OpenAI deployments
        ProviderType.XAI,  # Direct X.AI GROK access
        ProviderType.DIAL,  # DIAL unified API access
        ProviderType.CUSTOM,  # Local/self-hosted models
        ProviderType.OPENROUTER,  # Catch-all for cloud models
    ]

    def __init__(self, config: dict[str, str]) -> None:
        """Create a new, independent registry instance.

        Args:
            config: Mapping of environment-variable names to values
                    (e.g. ``{"GEMINI_API_KEY": "real-key"}``).
        """
        self._providers: dict[ProviderType, type] = {}
        self._initialized_providers: dict[ProviderType, ModelProvider] = {}
        self._config: dict[str, str] = config
        logging.debug("REGISTRY: Created new registry instance %s", id(self))

    def register_provider(self, provider_type: ProviderType, provider_class: type[ModelProvider]) -> None:
        """Register a new provider class.

        Args:
            provider_type: Type of the provider (e.g., ProviderType.GOOGLE)
            provider_class: Class that implements ModelProvider interface
        """
        self._providers[provider_type] = provider_class
        # Invalidate any cached instance so subsequent lookups use the new registration
        self._initialized_providers.pop(provider_type, None)

    def get_provider(self, provider_type: ProviderType, force_new: bool = False) -> Optional[ModelProvider]:
        """Get an initialized provider instance.

        Args:
            provider_type: Type of provider to get
            force_new: Force creation of new instance instead of using cached

        Returns:
            Initialized ModelProvider instance or None if not available
        """
        # Return cached instance if available and not forcing new
        if not force_new and provider_type in self._initialized_providers:
            return self._initialized_providers[provider_type]

        # Check if provider class is registered
        if provider_type not in self._providers:
            return None

        # Get API key from environment
        api_key = self._get_api_key_for_provider(provider_type)

        # Get provider class or factory function
        provider_class = self._providers[provider_type]

        # For custom providers, handle special initialization requirements
        if provider_type == ProviderType.CUSTOM:
            # Check if it's a factory function (callable but not a class)
            if callable(provider_class) and not isinstance(provider_class, type):
                # Factory function - call it with api_key parameter
                provider = provider_class(api_key=api_key)
            else:
                # Regular class - need to handle URL requirement
                custom_url = self._config.get("CUSTOM_API_URL") or get_env("CUSTOM_API_URL", "") or ""
                if not custom_url:
                    if api_key:  # Key is set but URL is missing
                        logging.warning("CUSTOM_API_KEY set but CUSTOM_API_URL missing – skipping Custom provider")
                    return None
                # Use empty string as API key for custom providers that don't need auth (e.g., Ollama)
                # This allows the provider to be created even without CUSTOM_API_KEY being set
                api_key = api_key or ""
                # Initialize custom provider with both API key and base URL
                provider = provider_class(api_key=api_key, base_url=custom_url)
        elif provider_type == ProviderType.GOOGLE:
            # For Gemini, check if custom base URL is configured
            if not api_key:
                return None
            gemini_base_url = self._config.get("GEMINI_BASE_URL") or get_env("GEMINI_BASE_URL")
            provider_kwargs = {"api_key": api_key}
            if gemini_base_url:
                provider_kwargs["base_url"] = gemini_base_url
                logging.info(f"Initialized Gemini provider with custom endpoint: {gemini_base_url}")
            provider = provider_class(**provider_kwargs)
        elif provider_type == ProviderType.AZURE:
            if not api_key:
                return None

            azure_endpoint = self._config.get("AZURE_OPENAI_ENDPOINT") or get_env("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                logging.warning("AZURE_OPENAI_ENDPOINT missing – skipping Azure OpenAI provider")
                return None

            azure_version = self._config.get("AZURE_OPENAI_API_VERSION") or get_env("AZURE_OPENAI_API_VERSION")
            provider = provider_class(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_version,
            )
        else:
            if not api_key:
                return None
            # Initialize non-custom provider with just API key
            provider = provider_class(api_key=api_key)

        # Cache the instance
        self._initialized_providers[provider_type] = provider

        return provider

    def get_provider_for_model(self, model_name: str) -> Optional[ModelProvider]:
        """Get provider instance for a specific model name.

        Provider priority order:
        1. Native APIs (GOOGLE, OPENAI) - Most direct and efficient
        2. CUSTOM - For local/private models with specific endpoints
        3. OPENROUTER - Catch-all for cloud models via unified API

        Args:
            model_name: Name of the model (e.g., "gemini-2.5-flash", "gpt5")

        Returns:
            ModelProvider instance that supports this model
        """
        logging.debug(f"get_provider_for_model called with model_name='{model_name}'")

        logging.debug(f"Registry instance: {self}")
        logging.debug(f"Available providers in registry: {list(self._providers.keys())}")

        for provider_type in self.PROVIDER_PRIORITY_ORDER:
            if provider_type in self._providers:
                logging.debug(f"Found {provider_type} in registry")
                # Get or create provider instance
                provider = self.get_provider(provider_type)
                if provider and provider.validate_model_name(model_name):
                    logging.debug(f"{provider_type} validates model {model_name}")
                    return provider
                else:
                    logging.debug(f"{provider_type} does not validate model {model_name}")
            else:
                logging.debug(f"{provider_type} not found in registry")

        logging.debug(f"No provider found for model {model_name}")
        return None

    def get_available_providers(self) -> list[ProviderType]:
        """Get list of registered provider types."""
        return list(self._providers.keys())

    def get_available_models(self, respect_restrictions: bool = True) -> dict[str, ProviderType]:
        """Get mapping of all available models to their providers.

        Args:
            respect_restrictions: If True, filter out models not allowed by restrictions

        Returns:
            Dict mapping model names to provider types
        """
        # Import here to avoid circular imports
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service() if respect_restrictions else None
        models: dict[str, ProviderType] = {}

        for provider_type in self._providers:
            provider = self.get_provider(provider_type)
            if not provider:
                continue

            try:
                available = provider.list_models(respect_restrictions=respect_restrictions)
            except NotImplementedError:
                logging.warning("Provider %s does not implement list_models", provider_type)
                continue

            if restriction_service and restriction_service.has_restrictions(provider_type):
                restricted_display = self._collect_restricted_display_names(
                    provider,
                    provider_type,
                    available,
                    restriction_service,
                )
                if restricted_display:
                    for model_name in restricted_display:
                        models[model_name] = provider_type
                    continue

            for model_name in available:
                # =====================================================================================
                # CRITICAL: Prevent double restriction filtering (Fixed Issue #98)
                # =====================================================================================
                # Previously, both the provider AND registry applied restrictions, causing
                # double-filtering that resulted in "no models available" errors.
                #
                # Logic: If respect_restrictions=True, provider already filtered models,
                # so registry should NOT filter them again.
                # TEST COVERAGE: tests/test_provider_routing_bugs.py::TestOpenRouterAliasRestrictions
                # =====================================================================================
                if (
                    restriction_service
                    and not respect_restrictions  # Only filter if provider didn't already filter
                    and not restriction_service.is_allowed(provider_type, model_name)
                ):
                    logging.debug("Model %s filtered by restrictions", model_name)
                    continue
                models[model_name] = provider_type

        return models

    @staticmethod
    def _collect_restricted_display_names(
        provider: ModelProvider,
        provider_type: ProviderType,
        available: list[str],
        restriction_service,
    ) -> list[str] | None:
        """Derive the human-facing model list when restrictions are active."""

        allowed_models = restriction_service.get_allowed_models(provider_type)
        if not allowed_models:
            return None

        allowed_details: list[tuple[str, int]] = []

        for model_name in sorted(allowed_models):
            try:
                capabilities = provider.get_capabilities(model_name)
            except (AttributeError, ValueError):
                continue

            try:
                rank = capabilities.get_effective_capability_rank()
                rank_value = float(rank)
            except (AttributeError, TypeError, ValueError):
                rank_value = 0.0

            allowed_details.append((model_name, rank_value))

        if allowed_details:
            allowed_details.sort(key=lambda item: (-item[1], item[0]))
            return [name for name, _ in allowed_details]

        # Fallback: intersect the allowlist with the provider-advertised names.
        available_lookup = {name.lower(): name for name in available}
        display_names: list[str] = []
        for model_name in sorted(allowed_models):
            lowered = model_name.lower()
            if lowered in available_lookup:
                display_names.append(available_lookup[lowered])

        return display_names

    def get_available_model_names(self, provider_type: Optional[ProviderType] = None) -> list[str]:
        """Get list of available model names, optionally filtered by provider.

        This respects model restrictions automatically.

        Args:
            provider_type: Optional provider to filter by

        Returns:
            List of available model names
        """
        available_models = self.get_available_models(respect_restrictions=True)

        if provider_type:
            # Filter by specific provider
            return [name for name, ptype in available_models.items() if ptype == provider_type]
        else:
            # Return all available models
            return list(available_models.keys())

    def _get_api_key_for_provider(self, provider_type: ProviderType) -> Optional[str]:
        """Get API key for a provider from config dict or environment variables.

        Args:
            provider_type: Provider type to get API key for

        Returns:
            API key string or None if not found
        """
        key_mapping = {
            ProviderType.GOOGLE: "GEMINI_API_KEY",
            ProviderType.OPENAI: "OPENAI_API_KEY",
            ProviderType.AZURE: "AZURE_OPENAI_API_KEY",
            ProviderType.XAI: "XAI_API_KEY",
            ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
            ProviderType.CUSTOM: "CUSTOM_API_KEY",  # Can be empty for providers that don't need auth
            ProviderType.DIAL: "DIAL_API_KEY",
        }

        env_var = key_mapping.get(provider_type)
        if not env_var:
            return None

        return self._config.get(env_var) or get_env(env_var)

    def _get_allowed_models_for_provider(self, provider: ModelProvider, provider_type: ProviderType) -> list[str]:
        """Get a list of allowed canonical model names for a given provider.

        Args:
            provider: The provider instance to get models for
            provider_type: The provider type for restriction checking

        Returns:
            List of model names that are both supported and allowed
        """
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()

        allowed_models = []

        # Get the provider's supported models
        try:
            # Use list_models to get all supported models (handles both regular and custom providers)
            supported_models = provider.list_models(respect_restrictions=False)
        except (NotImplementedError, AttributeError):
            # Fallback to provider-declared capability maps if list_models not implemented
            model_map = getattr(provider, "MODEL_CAPABILITIES", None)
            supported_models = list(model_map.keys()) if isinstance(model_map, dict) else []

        # Filter by restrictions
        for model_name in supported_models:
            if restriction_service.is_allowed(provider_type, model_name):
                allowed_models.append(model_name)

        return allowed_models

    def get_preferred_fallback_model(self, tool_category: Optional["ToolModelCategory"] = None) -> str:
        """Get the preferred fallback model based on provider priority and tool category.

        This method orchestrates model selection by:
        1. Getting allowed models for each provider (respecting restrictions)
        2. Asking providers for their preference from the allowed list
        3. Falling back to first available model if no preference given

        Args:
            tool_category: Optional category to influence model selection

        Returns:
            Model name string for fallback use
        """
        from tools.models import ToolModelCategory

        effective_category = tool_category or ToolModelCategory.BALANCED
        first_available_model = None

        # Ask each provider for their preference in priority order
        for provider_type in self.PROVIDER_PRIORITY_ORDER:
            provider = self.get_provider(provider_type)
            if provider:
                # 1. Registry filters the models first
                allowed_models = self._get_allowed_models_for_provider(provider, provider_type)

                if not allowed_models:
                    continue

                # 2. Keep track of the first available model as fallback
                if not first_available_model:
                    first_available_model = sorted(allowed_models)[0]

                # 3. Ask provider to pick from allowed list
                preferred_model = provider.get_preferred_model(effective_category, allowed_models)

                if preferred_model:
                    logging.debug(
                        f"Provider {provider_type.value} selected '{preferred_model}' for category '{effective_category.value}'"
                    )
                    return preferred_model

        # If no provider returned a preference, use first available model
        if first_available_model:
            logging.debug(f"No provider preference, using first available: {first_available_model}")
            return first_available_model

        # Ultimate fallback if no providers have models
        raise ValueError("No models available from any configured provider")

    def get_available_providers_with_keys(self) -> list[ProviderType]:
        """Get list of provider types that have valid API keys.

        Returns:
            List of ProviderType values for providers with valid API keys
        """
        available = []
        for provider_type in self._providers:
            if self.get_provider(provider_type) is not None:
                available.append(provider_type)
        return available

    def clear_cache(self) -> None:
        """Clear cached provider instances."""
        self._initialized_providers.clear()

    def unregister_provider(self, provider_type: ProviderType) -> None:
        """Unregister a provider (mainly for testing)."""
        self._providers.pop(provider_type, None)
        self._initialized_providers.pop(provider_type, None)

    def get_all_health_status(self) -> list[dict]:
        """Collect health status from all initialised provider instances."""
        statuses = []
        for provider_type, provider in self._initialized_providers.items():
            try:
                statuses.append(provider.get_health_status())
            except Exception:
                logging.debug("Failed to get health status for %s", provider_type, exc_info=True)
        return statuses
