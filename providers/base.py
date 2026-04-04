"""Base interfaces and common behaviour for model providers."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from utils.circuit_breaker import CircuitBreaker, CircuitState, ProviderUnavailable

from .shared import ModelCapabilities, ModelResponse, ProviderType


@dataclass
class StreamChunk:
    """A single chunk yielded by ``generate_content_stream()``.

    Attributes:
        text: The incremental content from the model for this chunk.
        is_final: ``True`` when this is the last chunk in the stream.
            The final chunk may have empty ``text`` if all content was
            sent in prior chunks.
        usage: Token usage metadata (input_tokens, output_tokens,
            total_tokens).  Present only on the final chunk when the
            provider reports usage information; ``None`` otherwise.
    """

    text: str
    is_final: bool = False
    usage: Optional[dict] = field(default=None)


try:
    import httpx

    _HttpxTimeoutException = httpx.TimeoutException
    _HttpxConnectError = httpx.ConnectError
    _HttpxHTTPStatusError = httpx.HTTPStatusError
except ImportError:
    _HttpxTimeoutException = None
    _HttpxConnectError = None
    _HttpxHTTPStatusError = None

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """Abstract base class for all model backends in the MCP server.

    Role
        Defines the interface every provider must implement so the registry,
        restriction service, and tools have a uniform surface for listing
        models, resolving aliases, and executing requests.

    Responsibilities
        * expose static capability metadata for each supported model via
          :class:`ModelCapabilities`
        * accept user prompts, forward them to the underlying SDK, and wrap
          responses in :class:`ModelResponse`
        * report tokenizer counts for budgeting and validation logic
        * advertise provider identity (``ProviderType``) so restriction
          policies can map environment configuration onto providers
        * validate whether a model name or alias is recognised by the provider

    Shared helpers like temperature validation, alias resolution, and
    restriction-aware ``list_models`` live here so concrete subclasses only
    need to supply their catalogue and wire up SDK-specific behaviour.
    """

    # All concrete providers must define their supported models
    MODEL_CAPABILITIES: dict[str, Any] = {}

    _RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504, 529}
    _NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}

    def __init__(self, api_key: str, **kwargs):
        """Initialize the provider with API key and optional configuration."""
        self.api_key = api_key
        self.config = kwargs
        self._sorted_capabilities_cache: Optional[list[tuple[str, ModelCapabilities]]] = None
        self._circuit_breaker = self._create_circuit_breaker()

    def _create_circuit_breaker(self) -> CircuitBreaker:
        """Create a circuit breaker with env-var-configurable thresholds."""
        import os

        def _env_int(key: str, default: int) -> int:
            raw = os.environ.get(key)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                logger.warning(
                    "Invalid value '%s' for %s, using default %d",
                    raw,
                    key,
                    default,
                )
                return default

        def _env_float(key: str, default: float) -> float:
            raw = os.environ.get(key)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                logger.warning(
                    "Invalid value '%s' for %s, using default %s",
                    raw,
                    key,
                    default,
                )
                return default

        return CircuitBreaker(
            failure_threshold=_env_int("CIRCUIT_FAILURE_THRESHOLD", 5),
            reset_timeout_seconds=_env_float("CIRCUIT_RESET_TIMEOUT_SECONDS", 60.0),
            half_open_max_calls=_env_int("CIRCUIT_HALF_OPEN_MAX_CALLS", 1),
            provider_name=self.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Provider identity & capability surface
    # ------------------------------------------------------------------
    @abstractmethod
    def get_provider_type(self) -> ProviderType:
        """Return the concrete provider identity."""

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Resolve capability metadata for a model name.

        This centralises the alias resolution → lookup → restriction check
        pipeline so providers only override the pieces they genuinely need to
        customise. Subclasses usually only override ``_lookup_capabilities`` to
        integrate a registry or dynamic source, or ``_finalise_capabilities`` to
        tweak the returned object.

        Args:
            model_name: Canonical model name or its alias
        """

        resolved_model_name = self._resolve_model_name(model_name)
        capabilities = self._lookup_capabilities(resolved_model_name, model_name)

        if capabilities is None:
            self._raise_unsupported_model(model_name)

        self._ensure_model_allowed(capabilities, resolved_model_name, model_name)
        return self._finalise_capabilities(capabilities, resolved_model_name, model_name)

    def get_all_model_capabilities(self) -> dict[str, ModelCapabilities]:
        """Return statically declared capabilities when available."""

        model_map = getattr(self, "MODEL_CAPABILITIES", None)
        if isinstance(model_map, dict) and model_map:
            return {k: v for k, v in model_map.items() if isinstance(v, ModelCapabilities)}
        return {}

    def get_capabilities_by_rank(self) -> list[tuple[str, ModelCapabilities]]:
        """Return model capabilities sorted by effective capability rank."""

        if self._sorted_capabilities_cache is not None:
            return list(self._sorted_capabilities_cache)

        model_configs = self.get_all_model_capabilities()
        if not model_configs:
            self._sorted_capabilities_cache = []
            return []

        items = list(model_configs.items())
        items.sort(key=lambda item: (-item[1].get_effective_capability_rank(), item[0]))
        self._sorted_capabilities_cache = items
        return list(items)

    def _invalidate_capability_cache(self) -> None:
        """Clear cached sorted capability data (call after dynamic updates)."""

        self._sorted_capabilities_cache = None

    def list_models(
        self,
        *,
        respect_restrictions: bool = True,
        include_aliases: bool = True,
        lowercase: bool = False,
        unique: bool = False,
    ) -> list[str]:
        """Return formatted model names supported by this provider."""

        model_configs = self.get_all_model_capabilities()
        if not model_configs:
            return []

        restriction_service = None
        if respect_restrictions:
            from utils.model_restrictions import get_restriction_service

            restriction_service = get_restriction_service()

        if restriction_service:
            allowed_configs = {}
            for model_name, config in model_configs.items():
                if restriction_service.is_allowed(self.get_provider_type(), model_name):
                    allowed_configs[model_name] = config
            model_configs = allowed_configs

        if not model_configs:
            return []

        return ModelCapabilities.collect_model_names(
            model_configs,
            include_aliases=include_aliases,
            lowercase=lowercase,
            unique=unique,
        )

    # ------------------------------------------------------------------
    # Capability-based model selection
    # ------------------------------------------------------------------

    # Name patterns that indicate fast-tier models
    _FAST_TIER_PATTERNS = ("flash", "mini", "lite", "fast", "nano")

    def select_preferred_model(
        self,
        category: "ToolModelCategory",
        allowed_models: list[str],
    ) -> Optional[str]:
        """Select the best model from *allowed_models* for a tool category.

        Uses ``intelligence_score`` and capability flags from
        :class:`ModelCapabilities` instead of hardcoded preference lists.

        Subclasses can override ``get_preferred_model`` if truly custom
        logic is needed, but the default implementation delegates here.
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if len(allowed_models) == 1:
            return allowed_models[0]

        caps_map = self.get_all_model_capabilities()

        def _score(model: str) -> int:
            cap = caps_map.get(model)
            return cap.intelligence_score if cap else 0

        def _has_thinking(model: str) -> bool:
            cap = caps_map.get(model)
            return bool(cap and cap.supports_extended_thinking)

        def _is_fast_tier(model: str) -> bool:
            name = model.lower()
            return any(p in name for p in self._FAST_TIER_PATTERNS)

        if category == ToolModelCategory.EXTENDED_REASONING:
            thinking = [m for m in allowed_models if _has_thinking(m)]
            pool = thinking if thinking else allowed_models
            return max(pool, key=lambda m: (_score(m), m))

        if category == ToolModelCategory.FAST_RESPONSE:
            fast = [m for m in allowed_models if _is_fast_tier(m)]
            if fast:
                return max(fast, key=lambda m: (_score(m), m))
            # No fast-tier → pick the lowest-scored (cheapest) model
            return min(allowed_models, key=lambda m: (_score(m), m))

        # BALANCED or default — highest intelligence_score
        return max(allowed_models, key=lambda m: (_score(m), m))

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using the model.

        This is the core method that all providers must implement to generate responses
        from their models. Providers should handle model-specific capabilities and
        constraints appropriately.

        Args:
            prompt: The main user prompt/query to send to the model
            model_name: Canonical model name or its alias that the provider supports
            system_prompt: Optional system instructions to prepend to the prompt for
                          establishing context, behavior, or role
            temperature: Controls randomness in generation (0.0=deterministic, 1.0=creative),
                        default 0.3. Some models may not support temperature control
            max_output_tokens: Optional maximum number of tokens to generate in the response.
                              If not specified, uses the model's default limit
            **kwargs: Additional provider-specific parameters that vary by implementation
                     (e.g., thinking_mode for Gemini, top_p for OpenAI, images for vision models)

        Returns:
            ModelResponse: Standardized response object containing:
                - content: The generated text response
                - usage: Token usage statistics (input/output/total)
                - model_name: The model that was actually used
                - friendly_name: Human-readable provider/model identifier
                - provider: The ProviderType enum value
                - metadata: Provider-specific metadata (finish_reason, safety info, etc.)

        Raises:
            ValueError: If the model is not supported, parameters are invalid,
                       or the model is restricted by policy
            RuntimeError: If the API call fails after retries
        """

    def generate_content_stream(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """Yield response chunks incrementally as the model generates output.

        This method has the same parameter signature as
        :meth:`generate_content`.  It returns a generator that yields one or
        more :class:`StreamChunk` objects.  The **last** chunk always has
        ``is_final=True``.

        The default implementation calls :meth:`generate_content`
        synchronously and yields the full response as a single
        ``StreamChunk``.  Concrete providers **MAY** override this method
        with a native streaming implementation that uses the provider
        SDK's streaming API for reduced time-to-first-token.

        Streaming contract:
            * Every chunk has a ``text`` field with incremental content.
            * Intermediate chunks have ``is_final=False`` and
              ``usage=None``.
            * The last chunk has ``is_final=True`` and ``usage`` set to
              token usage metadata when available from the provider.
            * Concatenating all chunk ``text`` values produces the same
              string as the ``content`` field of the :class:`ModelResponse`
              that :meth:`generate_content` would return for the same
              input.
            * If the provider errors mid-stream, the generator raises the
              exception; any chunks already yielded remain valid.

        Args:
            prompt: The main user prompt/query to send to the model.
            model_name: Canonical model name or alias.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature (0.0–1.0), default 0.3.
            max_output_tokens: Optional cap on generated tokens.
            **kwargs: Additional provider-specific parameters.

        Yields:
            StreamChunk: Incremental response chunks.
        """
        response = self.generate_content(
            prompt,
            model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        yield StreamChunk(
            text=response.content,
            is_final=True,
            usage=response.usage,
        )

    async def async_generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Async wrapper for content generation.

        Default implementation delegates to the synchronous ``generate_content()``
        via ``asyncio.to_thread()`` so the event loop is not blocked.  Concrete
        providers MAY override this with a native async implementation when their
        SDK provides an async client.

        Thread-safety note: providers used with the async path (e.g. concurrent
        consensus) MUST have thread-safe SDK clients because multiple calls may
        execute concurrently in the default thread-pool executor.
        """
        return await asyncio.to_thread(
            self.generate_content,
            prompt,
            model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens using the best available method for this provider.

        Fallback chain:
        1. Subclass override (e.g., tiktoken for OpenAI, litellm for Gemini)
        2. litellm.token_counter() (covers most known models)
        3. Content-aware character heuristic (code ~3 chars/token, prose ~4 chars/token)
        """
        resolved_model = self._resolve_model_name(model_name)

        if not text:
            return 0

        # Tier 2: try litellm.token_counter as a universal fallback
        try:
            import litellm

            count = litellm.token_counter(model=resolved_model, text=text)
            logger.debug("Counted %s tokens for model %s via litellm", count, resolved_model)
            return count
        except Exception:
            pass

        # Tier 3: content-aware character heuristic
        return self._heuristic_count_tokens(text, resolved_model)

    def _heuristic_count_tokens(self, text: str, resolved_model: str) -> int:
        """Content-aware character heuristic for token estimation.

        Detects code density via simple indicator scan and adjusts the ratio:
        - Code-heavy text: ~3 chars/token
        - Prose: ~4 chars/token
        """
        # Simple code indicator scan on a sample
        sample = text[:2000]
        code_indicators = sum(1 for ch in sample if ch in "{}();[]")
        indentation_lines = sum(1 for line in sample.split("\n") if line.startswith("    ") or line.startswith("\t"))
        total_lines = max(1, sample.count("\n") + 1)

        code_density = (code_indicators / max(1, len(sample))) + (indentation_lines / total_lines)

        if code_density > 0.05:
            estimated = max(1, len(text) // 3)
        else:
            estimated = max(1, len(text) // 4)

        logger.debug(
            "Estimating %s tokens for model %s via character heuristic (code_density=%.3f)",
            estimated,
            resolved_model,
            code_density,
        )
        return estimated

    def close(self) -> None:
        """Clean up any resources held by the provider."""

        return

    # ------------------------------------------------------------------
    # Circuit breaker health
    # ------------------------------------------------------------------
    def get_health_status(self) -> dict:
        """Return diagnostic health info including circuit breaker state."""
        status = self._circuit_breaker.health_status()
        status["provider_type"] = self.get_provider_type().value
        status["healthy"] = self._circuit_breaker.state is CircuitState.CLOSED
        return status

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------
    def _extract_status_code(self, error: Exception) -> Optional[int]:
        """Extract an HTTP status code from an exception, if available."""
        for attr in ("status_code", "code"):
            val = getattr(error, attr, None)
            if isinstance(val, int):
                return val
        resp = getattr(error, "response", None)
        if resp is not None:
            val = getattr(resp, "status_code", None)
            if isinstance(val, int):
                return val
        return None

    def _is_error_retryable(self, error: Exception) -> bool:
        """Three-tier error classification: class hierarchy -> status code -> string fallback."""

        error_str = str(error).lower()

        # Pre-check: 429/rate-limit is never retried by this layer
        if "429" in error_str or "rate limit" in error_str:
            return False

        # Tier 1: Exception class hierarchy (most reliable)
        if _HttpxTimeoutException is not None and isinstance(error, _HttpxTimeoutException):
            return True
        if _HttpxConnectError is not None and isinstance(error, _HttpxConnectError):
            return True
        if _HttpxHTTPStatusError is not None and isinstance(error, _HttpxHTTPStatusError):
            code = self._extract_status_code(error)
            if code is not None:
                if code in self._RETRYABLE_STATUS_CODES:
                    return True
                if code in self._NON_RETRYABLE_STATUS_CODES:
                    return False

        # Tier 2: Numeric status code from error attributes
        code = self._extract_status_code(error)
        if code is not None:
            if code in self._RETRYABLE_STATUS_CODES:
                return True
            if code in self._NON_RETRYABLE_STATUS_CODES:
                return False

        # Tier 3: String pattern fallback (least reliable, kept for compatibility)
        retryable_indicators = [
            "timeout",
            "connection",
            "temporary",
            "unavailable",
            "retry",
            "reset",
            "refused",
            "broken pipe",
            "tls",
            "handshake",
            "network",
            "500",
            "502",
            "503",
            "504",
        ]
        return any(indicator in error_str for indicator in retryable_indicators)

    def _run_with_retries(
        self,
        operation: Callable[[], Any],
        *,
        max_attempts: int,
        delays: Optional[list[float]] = None,
        log_prefix: str = "",
    ):
        """Execute ``operation`` with circuit breaker and retry semantics.

        The circuit breaker is checked before entering the retry loop.
        On success (even after retries), the breaker records success.
        When all retries are exhausted, the breaker records a failure.

        Args:
            operation: Callable returning the provider result.
            max_attempts: Maximum number of attempts (>=1).
            delays: Optional list of sleep durations between attempts.
            log_prefix: Optional identifier for log clarity.

        Returns:
            Whatever ``operation`` returns.

        Raises:
            ProviderUnavailable: If the circuit breaker is open.
            The last exception when all retries fail or the error is not retryable.
        """

        # Circuit breaker check — fail fast if the provider is known-down
        if not self._circuit_breaker.allow_request():
            raise ProviderUnavailable(
                provider_name=self._circuit_breaker._provider_name,
                circuit_state=self._circuit_breaker.state.value,
            )

        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        attempts = max_attempts
        delays = delays or []
        last_exc: Optional[Exception] = None

        for attempt_index in range(attempts):
            try:
                result = operation()
                self._circuit_breaker.record_success()
                return result
            except Exception as exc:  # noqa: BLE001 - bubble exact provider errors
                last_exc = exc
                attempt_number = attempt_index + 1

                # Decide whether to retry based on subclass hook
                retryable = self._is_error_retryable(exc)
                if not retryable or attempt_number >= attempts:
                    self._circuit_breaker.record_failure()
                    raise

                delay_idx = min(attempt_index, len(delays) - 1) if delays else -1
                delay = delays[delay_idx] if delay_idx >= 0 else 0.0

                if delay > 0:
                    logger.warning(
                        "%s retryable error (attempt %s/%s): %s. Retrying in %ss...",
                        log_prefix or self.__class__.__name__,
                        attempt_number,
                        attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "%s retryable error (attempt %s/%s): %s. Retrying...",
                        log_prefix or self.__class__.__name__,
                        attempt_number,
                        attempts,
                        exc,
                    )

        # Should never reach here because loop either returns or raises
        self._circuit_breaker.record_failure()
        raise last_exc if last_exc else RuntimeError("Retry loop exited without result")

    async def _run_with_retries_async(
        self,
        operation: Callable[[], Any],
        *,
        max_attempts: int,
        delays: Optional[list[float]] = None,
        log_prefix: str = "",
    ):
        """Async counterpart of ``_run_with_retries()``.

        Accepts an **async** callable and uses ``asyncio.sleep()`` for
        inter-attempt delays so the event loop is not blocked.  Intended for
        providers that override ``async_generate_content()`` with a native
        async implementation — the default thread-wrapped path uses the sync
        ``_run_with_retries()`` inside the thread where blocking is fine.

        Thread-safety note: this helper shares the same circuit-breaker
        instance as the sync variant.  Providers must ensure their async
        client is safe for concurrent access.
        """
        if not self._circuit_breaker.allow_request():
            raise ProviderUnavailable(
                provider_name=self._circuit_breaker._provider_name,
                circuit_state=self._circuit_breaker.state.value,
            )

        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        delays = delays or []
        last_exc: Optional[Exception] = None

        for attempt_index in range(max_attempts):
            try:
                result = await operation()
                self._circuit_breaker.record_success()
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                attempt_number = attempt_index + 1

                retryable = self._is_error_retryable(exc)
                if not retryable or attempt_number >= max_attempts:
                    self._circuit_breaker.record_failure()
                    raise

                delay_idx = min(attempt_index, len(delays) - 1) if delays else -1
                delay = delays[delay_idx] if delay_idx >= 0 else 0.0

                if delay > 0:
                    logger.warning(
                        "%s retryable error (attempt %s/%s): %s. Retrying in %ss...",
                        log_prefix or self.__class__.__name__,
                        attempt_number,
                        max_attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "%s retryable error (attempt %s/%s): %s. Retrying...",
                        log_prefix or self.__class__.__name__,
                        attempt_number,
                        max_attempts,
                        exc,
                    )

        self._circuit_breaker.record_failure()
        raise last_exc if last_exc else RuntimeError("Retry loop exited without result")

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------
    def validate_model_name(self, model_name: str) -> bool:
        """
        Return ``True`` when the model resolves to an allowed capability.

        Args:
            model_name: Canonical model name or its alias
        """

        try:
            self.get_capabilities(model_name)
        except ValueError:
            return False
        return True

    def validate_parameters(self, model_name: str, temperature: float, **kwargs) -> None:
        """
        Validate model parameters against capabilities.

        Args:
            model_name: Canonical model name or its alias
        """

        capabilities = self.get_capabilities(model_name)

        if not capabilities.temperature_constraint.validate(temperature):
            constraint_desc = capabilities.temperature_constraint.get_description()
            raise ValueError(f"Temperature {temperature} is invalid for model {model_name}. {constraint_desc}")

    # ------------------------------------------------------------------
    # Preference / registry hooks
    # ------------------------------------------------------------------
    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Get the preferred model from this provider for a given category."""

        return None

    def get_model_registry(self) -> Optional[dict[str, Any]]:
        """Return the model registry backing this provider, if any."""

        return None

    # ------------------------------------------------------------------
    # Capability lookup pipeline
    # ------------------------------------------------------------------
    def _lookup_capabilities(
        self,
        canonical_name: str,
        requested_name: Optional[str] = None,
    ) -> Optional[ModelCapabilities]:
        """Return ``ModelCapabilities`` for the canonical model name."""

        return self.get_all_model_capabilities().get(canonical_name)

    def _ensure_model_allowed(
        self,
        capabilities: ModelCapabilities,
        canonical_name: str,
        requested_name: str,
    ) -> None:
        """Raise ``ValueError`` if the model violates restriction policy."""

        try:
            from utils.model_restrictions import get_restriction_service
        except Exception:  # pragma: no cover - only triggered if service import breaks
            return

        restriction_service = get_restriction_service()
        if not restriction_service:
            return

        if restriction_service.is_allowed(self.get_provider_type(), canonical_name, requested_name):
            return

        raise ValueError(
            f"{self.get_provider_type().value} model '{canonical_name}' is not allowed by restriction policy."
        )

    def _finalise_capabilities(
        self,
        capabilities: ModelCapabilities,
        canonical_name: str,
        requested_name: str,
    ) -> ModelCapabilities:
        """Allow subclasses to adjust capability metadata before returning."""

        return capabilities

    def _raise_unsupported_model(self, model_name: str) -> None:
        """Raise the canonical unsupported-model error."""

        raise ValueError(f"Unsupported model '{model_name}' for provider {self.get_provider_type().value}.")

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model shorthand to full name.

        This implementation uses the hook methods to support different
        model configuration sources.

        Args:
            model_name: Canonical model name or its alias

        Returns:
            Resolved model name
        """
        # Get model configurations from the hook method
        model_configs = self.get_all_model_capabilities()

        # First check if it's already a base model name (case-sensitive exact match)
        if model_name in model_configs:
            return model_name

        # Check case-insensitively for both base models and aliases
        model_name_lower = model_name.lower()

        # Check base model names case-insensitively
        for base_model in model_configs:
            if base_model.lower() == model_name_lower:
                return base_model

        # Check aliases from the model configurations
        alias_map = ModelCapabilities.collect_aliases(model_configs)
        for base_model, aliases in alias_map.items():
            if any(alias.lower() == model_name_lower for alias in aliases):
                return base_model

        # If not found, return as-is
        return model_name
