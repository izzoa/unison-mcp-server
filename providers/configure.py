"""
Provider configuration for the Unison MCP Server.

Validates API keys and registers providers using a single-pass data-driven
approach. Handles special cases (Azure, Custom, OpenRouter) and provider
cleanup, restriction validation, and auto-mode checks.
"""

import atexit
import logging
from dataclasses import dataclass
from typing import Any

from utils.env import get_env

logger = logging.getLogger(__name__)


@dataclass
class ProviderSpec:
    """Declarative specification for a standard provider."""

    env_key: str
    display_name: str
    provider_type: Any  # ProviderType enum value
    provider_class: Any  # Provider class or factory
    is_native: bool = True


def configure_providers():
    """
    Configure and validate AI providers based on available API keys.

    Uses a single-pass data-driven approach for standard providers,
    with special handling for Azure, Custom, and OpenRouter.

    Raises:
        ValueError: If no valid API keys are found or conflicting configurations detected
    """
    from providers import ModelProviderRegistry
    from providers.azure_openai import AzureOpenAIProvider
    from providers.custom import CustomProvider
    from providers.dial import DIALModelProvider
    from providers.gemini import GeminiModelProvider
    from providers.openai import OpenAIModelProvider
    from providers.openrouter import OpenRouterProvider
    from providers.shared import ProviderType
    from providers.xai import XAIModelProvider
    from utils.model_restrictions import get_restriction_service

    # Log environment variable status for debugging
    logger.debug("Checking environment variables for API keys...")
    api_keys_to_check = ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "CUSTOM_API_URL"]
    for key in api_keys_to_check:
        value = get_env(key)
        logger.debug(f"  {key}: {'[PRESENT]' if value else '[MISSING]'}")

    # Standard providers: single-pass validation + registration
    STANDARD_PROVIDERS = [
        ProviderSpec("GEMINI_API_KEY", "Gemini", ProviderType.GOOGLE, GeminiModelProvider),
        ProviderSpec("OPENAI_API_KEY", "OpenAI", ProviderType.OPENAI, OpenAIModelProvider),
        ProviderSpec("XAI_API_KEY", "X.AI (GROK)", ProviderType.XAI, XAIModelProvider),
        ProviderSpec("DIAL_API_KEY", "DIAL", ProviderType.DIAL, DIALModelProvider),
    ]

    valid_providers = []
    registered_providers = []
    has_native_apis = False
    has_openrouter = False
    has_custom = False

    # Single-pass: validate key and register in one step
    for spec in STANDARD_PROVIDERS:
        key = get_env(spec.env_key)
        placeholder = f"your_{spec.env_key.lower()}_here"
        if key and key != placeholder:
            valid_providers.append(spec.display_name)
            has_native_apis = True
            ModelProviderRegistry.register_provider(spec.provider_type, spec.provider_class)
            registered_providers.append(spec.provider_type.value)
            logger.info(f"{spec.display_name} API key found")
            logger.debug(f"Registered provider: {spec.provider_type.value}")
        else:
            if not key:
                logger.debug(f"{spec.display_name} API key not found in environment")
            else:
                logger.debug(f"{spec.display_name} API key is placeholder value")

    # Special case: Azure OpenAI (requires model registry check)
    azure_key = get_env("AZURE_OPENAI_API_KEY")
    azure_endpoint = get_env("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_key != "your_azure_openai_key_here" and azure_endpoint:
        try:
            from providers.registries.azure import AzureModelRegistry

            azure_registry = AzureModelRegistry()
            if azure_registry.list_models():
                valid_providers.append("Azure OpenAI")
                has_native_apis = True
                ModelProviderRegistry.register_provider(ProviderType.AZURE, AzureOpenAIProvider)
                registered_providers.append(ProviderType.AZURE.value)
                logger.info("Azure OpenAI configuration detected")
                logger.debug(f"Registered provider: {ProviderType.AZURE.value}")
            else:
                logger.warning(
                    "Azure OpenAI models configuration is empty. Populate conf/azure_models.json or set AZURE_MODELS_CONFIG_PATH."
                )
        except Exception as exc:
            logger.warning(f"Failed to load Azure OpenAI models: {exc}")

    # Special case: Custom provider (Ollama, vLLM, etc.)
    custom_url = get_env("CUSTOM_API_URL")
    if custom_url:
        # IMPORTANT: Always read CUSTOM_API_KEY even if empty
        # - Some providers (vLLM, LM Studio, enterprise APIs) require authentication
        # - Others (Ollama) work without authentication (empty key)
        # - DO NOT remove this variable - it's needed for provider factory function
        custom_key = get_env("CUSTOM_API_KEY", "") or ""  # Default to empty (Ollama doesn't need auth)
        custom_model = get_env("CUSTOM_MODEL_NAME", "llama3.2") or "llama3.2"
        valid_providers.append(f"Custom API ({custom_url})")
        has_custom = True
        logger.info(f"Custom API endpoint found: {custom_url} with model {custom_model}")
        if custom_key:
            logger.debug("Custom API key provided for authentication")
        else:
            logger.debug("No custom API key provided (using unauthenticated access)")

        # Factory function that creates CustomProvider with proper parameters
        def custom_provider_factory(api_key=None):
            base_url = get_env("CUSTOM_API_URL", "") or ""
            return CustomProvider(api_key=api_key or "", base_url=base_url)

        ModelProviderRegistry.register_provider(ProviderType.CUSTOM, custom_provider_factory)
        registered_providers.append(ProviderType.CUSTOM.value)
        logger.debug(f"Registered provider: {ProviderType.CUSTOM.value}")

    # Special case: OpenRouter last (catch-all for everything else)
    openrouter_key = get_env("OPENROUTER_API_KEY")
    logger.debug(f"OpenRouter key check: key={'[PRESENT]' if openrouter_key else '[MISSING]'}")
    if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
        valid_providers.append("OpenRouter")
        has_openrouter = True
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        registered_providers.append(ProviderType.OPENROUTER.value)
        logger.info("OpenRouter API key found - Multiple models available via OpenRouter")
        logger.debug(f"Registered provider: {ProviderType.OPENROUTER.value}")
    else:
        if not openrouter_key:
            logger.debug("OpenRouter API key not found in environment")
        else:
            logger.debug("OpenRouter API key is placeholder value")

    # Log all registered providers
    if registered_providers:
        logger.info(f"Registered providers: {', '.join(registered_providers)}")

    # Require at least one valid provider
    if not valid_providers:
        raise ValueError(
            "At least one API configuration is required. Please set either:\n"
            "- GEMINI_API_KEY for Gemini models\n"
            "- OPENAI_API_KEY for OpenAI models\n"
            "- XAI_API_KEY for X.AI GROK models\n"
            "- DIAL_API_KEY for DIAL models\n"
            "- OPENROUTER_API_KEY for OpenRouter (multiple models)\n"
            "- CUSTOM_API_URL for local models (Ollama, vLLM, etc.)"
        )

    logger.info(f"Available providers: {', '.join(valid_providers)}")

    # Log provider priority
    priority_info = []
    if has_native_apis:
        priority_info.append("Native APIs (Gemini, OpenAI)")
    if has_custom:
        priority_info.append("Custom endpoints")
    if has_openrouter:
        priority_info.append("OpenRouter (catch-all)")

    if len(priority_info) > 1:
        logger.info(f"Provider priority: {' → '.join(priority_info)}")

    # Register cleanup function for providers
    def cleanup_providers():
        """Clean up all registered providers on shutdown."""
        try:
            registry = ModelProviderRegistry()
            if hasattr(registry, "_initialized_providers"):
                for provider in list(registry._initialized_providers.values()):
                    try:
                        if provider and hasattr(provider, "close"):
                            provider.close()
                    except Exception:
                        pass
        except Exception:
            pass

    atexit.register(cleanup_providers)

    # Check and log model restrictions
    restriction_service = get_restriction_service()
    restrictions = restriction_service.get_restriction_summary()

    if restrictions:
        logger.info("Model restrictions configured:")
        for provider_name, allowed_models in restrictions.items():
            if isinstance(allowed_models, list):
                logger.info(f"  {provider_name}: {', '.join(allowed_models)}")
            else:
                logger.info(f"  {provider_name}: {allowed_models}")

        # Validate restrictions against known models
        provider_instances = {}
        provider_types_to_validate = [ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.XAI, ProviderType.DIAL]
        for provider_type in provider_types_to_validate:
            provider = ModelProviderRegistry.get_provider(provider_type)
            if provider:
                provider_instances[provider_type] = provider

        if provider_instances:
            restriction_service.validate_against_known_models(provider_instances)
    else:
        logger.info("No model restrictions configured - all models allowed")

    # Check if auto mode has any models available after restrictions
    from config import IS_AUTO_MODE

    if IS_AUTO_MODE:
        available_models = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        if not available_models:
            logger.error(
                "Auto mode is enabled but no models are available after applying restrictions. "
                "Please check your OPENAI_ALLOWED_MODELS and GOOGLE_ALLOWED_MODELS settings."
            )
            raise ValueError(
                "No models available for auto mode due to restrictions. "
                "Please adjust your allowed model settings or disable auto mode."
            )
