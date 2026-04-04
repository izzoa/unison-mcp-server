"""Model provider abstractions for supporting multiple AI providers."""

from .azure_openai import AzureOpenAIProvider
from .base import ModelProvider, StreamChunk
from .gemini import GeminiModelProvider
from .openai import OpenAIModelProvider
from .openai_compatible import OpenAICompatibleProvider
from .openrouter import OpenRouterProvider
from .registry import ModelProviderRegistry, get_default_registry, set_default_registry
from .shared import ModelCapabilities, ModelResponse

__all__ = [
    "ModelProvider",
    "StreamChunk",
    "ModelResponse",
    "ModelCapabilities",
    "ModelProviderRegistry",
    "get_default_registry",
    "set_default_registry",
    "AzureOpenAIProvider",
    "GeminiModelProvider",
    "OpenAIModelProvider",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
]
