"""Anthropic (Claude) model provider implementation."""

import base64
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from anthropic import Anthropic

from utils.env import get_env
from utils.image_utils import validate_image

from .base import ModelProvider, StreamChunk
from .registries.anthropic import AnthropicModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)

# The Messages API rejects extended-thinking budgets below this floor.
MIN_THINKING_BUDGET = 1024


class AnthropicModelProvider(RegistryBackedProviderMixin, ModelProvider):
    """First-party Anthropic integration built on the official SDK (Messages API).

    Advertises thinking-mode budgets for extended-thinking models, supports an
    ``ANTHROPIC_API_URL`` base override for enterprise gateways, and performs
    image pre-processing before forwarding a request.
    """

    FRIENDLY_NAME = "Anthropic"

    REGISTRY_CLASS = AnthropicModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # Thinking mode configurations - percentages of the model's
    # max_thinking_tokens (same table the Gemini provider uses).
    THINKING_BUDGETS = {
        "minimal": 0.005,
        "low": 0.08,
        "medium": 0.33,
        "high": 0.67,
        "max": 1.0,
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Anthropic provider with API key and optional base URL."""
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._client = None
        base_url = kwargs.get("base_url") or get_env("ANTHROPIC_API_URL")
        if base_url == "your_anthropic_api_url_here":
            # Template placeholder counts as unset, like CUSTOM_API_URL's.
            base_url = None
        self._base_url = base_url
        self._invalidate_capability_cache()

    # ------------------------------------------------------------------
    # Client access
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Lazy initialization of the Anthropic client."""
        if self._client is None:
            client_kwargs = {"api_key": self.api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
                logger.debug("Initializing Anthropic client with base_url=%s", self._base_url)
            self._client = Anthropic(**client_kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------

    def _build_request(
        self,
        prompt: str,
        resolved_model_name: str,
        capabilities,
        system_prompt: str | None,
        temperature: float,
        max_output_tokens: int | None,
        thinking_mode: str,
        images: list[str] | None,
    ) -> dict:
        """Assemble Messages API request parameters.

        The Messages API requires ``max_tokens`` on every request, and fixes
        sampling temperature when extended thinking is enabled — a
        caller-supplied temperature is dropped in that case rather than
        surfacing an API rejection.
        """
        blocks: list[dict] = [{"type": "text", "text": prompt}]

        if images and capabilities.supports_images:
            for image_path in images:
                try:
                    image_block = self._process_image(image_path)
                    if image_block:
                        blocks.append(image_block)
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {e}")
                    continue
        elif images and not capabilities.supports_images:
            logger.warning(f"Model {resolved_model_name} does not support images, ignoring {len(images)} image(s)")

        params: dict = {
            "model": resolved_model_name,
            "messages": [{"role": "user", "content": blocks}],
            "max_tokens": max_output_tokens or capabilities.max_output_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            params["system"] = system_prompt

        if capabilities.supports_extended_thinking and thinking_mode in self.THINKING_BUDGETS:
            model_config = self.get_all_model_capabilities().get(resolved_model_name)
            if model_config and model_config.max_thinking_tokens > 0:
                budget = int(model_config.max_thinking_tokens * self.THINKING_BUDGETS[thinking_mode])
                budget = max(budget, MIN_THINKING_BUDGET)
                # The budget must fit inside max_tokens with output headroom.
                if budget >= params["max_tokens"]:
                    budget = params["max_tokens"] - MIN_THINKING_BUDGET
                if budget >= MIN_THINKING_BUDGET:
                    params["thinking"] = {"type": "enabled", "budget_tokens": budget}
                    if temperature != 1.0:
                        logger.debug(
                            "Dropping temperature=%s for %s: the Messages API fixes temperature "
                            "when extended thinking is enabled",
                            temperature,
                            resolved_model_name,
                        )
                    params.pop("temperature")

        return params

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int | None = None,
        thinking_mode: str = "medium",
        images: list[str] | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using a Claude model via the Messages API."""
        self.validate_parameters(model_name, temperature)
        capabilities = self.get_capabilities(model_name)
        resolved_model_name = self._resolve_model_name(model_name)

        params = self._build_request(
            prompt,
            resolved_model_name,
            capabilities,
            system_prompt,
            temperature,
            max_output_tokens,
            thinking_mode,
            images,
        )
        thinking_enabled = "thinking" in params

        attempt_counter = {"value": 0}

        def _attempt() -> ModelResponse:
            attempt_counter["value"] += 1
            response = self.client.messages.create(**params)
            content = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
            return ModelResponse(
                content=content,
                usage=self._extract_usage(response),
                model_name=resolved_model_name,
                friendly_name="Anthropic",
                provider=ProviderType.ANTHROPIC,
                metadata={
                    "thinking_mode": thinking_mode if thinking_enabled else None,
                    "stop_reason": getattr(response, "stop_reason", None),
                },
            )

        try:
            return self._run_with_retries(
                operation=_attempt,
                max_attempts=4,
                delays=[1, 3, 5, 8],
                log_prefix=f"Anthropic API ({resolved_model_name})",
            )
        except Exception as exc:
            attempts = max(attempt_counter["value"], 1)
            error_msg = (
                f"Anthropic API error for model {resolved_model_name} after {attempts} attempt"
                f"{'s' if attempts > 1 else ''}: {exc}"
            )
            raise RuntimeError(error_msg) from exc

    def generate_content_stream(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int | None = None,
        thinking_mode: str = "medium",
        images: list[str] | None = None,
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """Stream content via the SDK's streaming helper.

        ``text_stream`` yields only text deltas — thinking deltas never reach
        tool output, matching how the Gemini provider handles thinking content.
        The final chunk carries usage metadata from the completed message.
        """
        self.validate_parameters(model_name, temperature)
        capabilities = self.get_capabilities(model_name)
        resolved_model_name = self._resolve_model_name(model_name)

        params = self._build_request(
            prompt,
            resolved_model_name,
            capabilities,
            system_prompt,
            temperature,
            max_output_tokens,
            thinking_mode,
            images,
        )

        # Circuit breaker: the streaming path bypasses _run_with_retries, so
        # gate and record health explicitly (mirrors the Gemini provider).
        if not self._circuit_breaker.allow_request():
            from utils.circuit_breaker import ProviderUnavailable

            raise ProviderUnavailable(
                provider_name=self._circuit_breaker._provider_name,
                circuit_state=self._circuit_breaker.state.value,
            )

        try:
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    yield StreamChunk(text=text, is_final=False)
                final_message = stream.get_final_message()

            self._circuit_breaker.record_success()
            yield StreamChunk(text="", is_final=True, usage=self._extract_usage(final_message))

        except Exception as exc:
            if self._is_provider_unhealthy_error(exc):
                self._circuit_breaker.record_failure()
            error_msg = f"Anthropic streaming error for model {resolved_model_name}: {exc}"
            raise RuntimeError(error_msg) from exc

    # ------------------------------------------------------------------
    # Provider surface
    # ------------------------------------------------------------------

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.ANTHROPIC

    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from a Messages API response."""
        usage: dict[str, int] = {}
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            input_tokens = getattr(usage_obj, "input_tokens", None)
            output_tokens = getattr(usage_obj, "output_tokens", None)
            if input_tokens is not None:
                usage["input_tokens"] = input_tokens
            if output_tokens is not None:
                usage["output_tokens"] = output_tokens
            if input_tokens is not None and output_tokens is not None:
                usage["total_tokens"] = input_tokens + output_tokens
        return usage

    def _process_image(self, image_path: str) -> dict | None:
        """Process an image into a Messages API image block."""
        try:
            image_bytes, mime_type = validate_image(image_path)

            if image_path.startswith("data:"):
                _, data = image_path.split(",", 1)
            else:
                data = base64.b64encode(image_bytes).decode()

            return {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": data}}

        except ValueError as e:
            logger.warning(str(e))
            return None
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> str | None:
        """Select the best Claude model for *category* using capability metadata."""
        return self.select_preferred_model(category, allowed_models)


# Load registry data at import time for registry consumers
AnthropicModelProvider._ensure_registry()
