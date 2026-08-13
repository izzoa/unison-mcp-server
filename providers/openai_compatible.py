"""Base class for OpenAI-compatible API providers."""

import copy
import ipaddress
import logging
import threading
from collections.abc import Generator
from urllib.parse import urlparse

from openai import OpenAI

from utils.env import get_env, suppress_env_vars
from utils.image_utils import validate_image

from .base import ModelProvider, StreamChunk
from .shared import ModelCapabilities, ModelResponse, ProviderType

try:
    from openai import APIConnectionError as _OpenAIConnectionError
    from openai import APIStatusError as _OpenAIStatusError
    from openai import APITimeoutError as _OpenAITimeoutError
except ImportError:
    _OpenAIStatusError = None
    _OpenAITimeoutError = None
    _OpenAIConnectionError = None


class EmptyContentError(RuntimeError):
    """A 200 OK response that carries no assistant content.

    Raised rather than returned so the condition reaches the retry classifier
    as a typed error. Carries ``finish_reason`` so retryability can be decided
    from a field instead of from the message text — see
    :meth:`OpenAICompatibleProvider._is_error_retryable`, which matches on this
    class before any string inspection happens.
    """

    def __init__(
        self,
        message: str,
        *,
        finish_reason: Optional[str] = None,
        reasoning_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens


class OpenAICompatibleProvider(ModelProvider):
    """Shared implementation for OpenAI API lookalikes.

    The class owns HTTP client configuration (timeouts, proxy hardening,
    custom headers) and normalises the OpenAI SDK responses into
    :class:`~providers.shared.ModelResponse`.  Concrete subclasses only need to
    provide capability metadata and any provider-specific request tweaks.
    """

    DEFAULT_HEADERS = {}
    FRIENDLY_NAME = "OpenAI Compatible"

    def __init__(self, api_key: str, base_url: str = None, **kwargs):
        """Initialize the provider with API key and optional base URL.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            **kwargs: Additional configuration options including timeout
        """
        self._allowed_alias_cache: dict[str, str] = {}
        super().__init__(api_key, **kwargs)
        self._client = None
        # Serialize lazy client construction: the property mutates global
        # os.environ (proxy var suppression) and does a check-then-act on
        # self._client, both of which race under concurrent dispatch (consensus).
        # Distinct from any subclass lock (e.g. DIAL's _client_lock guarding its
        # deployment-client cache) so acquiring this inside that one cannot
        # deadlock a non-reentrant lock.
        self._client_init_lock = threading.Lock()
        self.base_url = base_url
        self.organization = kwargs.get("organization")
        self.allowed_models = self._parse_allowed_models()

        # Configure timeouts - especially important for custom/local endpoints
        self.timeout_config = self._configure_timeouts(**kwargs)

        # Validate base URL for security
        if self.base_url:
            self._validate_base_url()

        # Warn if using external URL without authentication
        if self.base_url and not self._is_localhost_url() and not api_key:
            logging.warning(
                f"Using external URL '{self.base_url}' without API key. "
                "This may be insecure. Consider setting an API key for authentication."
            )

    def _ensure_model_allowed(
        self,
        capabilities: ModelCapabilities,
        canonical_name: str,
        requested_name: str,
    ) -> None:
        """Respect provider-specific allowlists before default restriction checks."""

        super()._ensure_model_allowed(capabilities, canonical_name, requested_name)

        if self.allowed_models is not None:
            requested = requested_name.lower()
            canonical = canonical_name.lower()

            if requested not in self.allowed_models and canonical not in self.allowed_models:
                allowed = False
                for allowed_entry in list(self.allowed_models):
                    normalized_resolved = self._allowed_alias_cache.get(allowed_entry)
                    if normalized_resolved is None:
                        try:
                            resolved_name = self._resolve_model_name(allowed_entry)
                        except Exception:
                            continue

                        if not resolved_name:
                            continue

                        normalized_resolved = resolved_name.lower()
                        self._allowed_alias_cache[allowed_entry] = normalized_resolved

                    if normalized_resolved == canonical:
                        # Canonical match discovered via alias resolution – mark as allowed and
                        # memoise the canonical entry for future lookups.
                        allowed = True
                        self._allowed_alias_cache[canonical] = canonical
                        self.allowed_models.add(canonical)
                        break

                if not allowed:
                    raise ValueError(
                        f"Model '{requested_name}' is not allowed by restriction policy. Allowed models: {sorted(self.allowed_models)}"
                    )

    def _parse_allowed_models(self) -> set[str] | None:
        """Parse allowed models from environment variable.

        Returns:
            Set of allowed model names (lowercase) or None if not configured
        """
        # Get provider-specific allowed models
        provider_type = self.get_provider_type().value.upper()
        env_var = f"{provider_type}_ALLOWED_MODELS"
        models_str = get_env(env_var, "") or ""

        if models_str:
            # Parse and normalize to lowercase for case-insensitive comparison
            models = {m.strip().lower() for m in models_str.split(",") if m.strip()}
            if models:
                logging.info(f"Configured allowed models for {self.FRIENDLY_NAME}: {sorted(models)}")
                self._allowed_alias_cache = {}
                return models

        # Log info if no allow-list configured for proxy providers
        if self.get_provider_type() not in [ProviderType.GOOGLE, ProviderType.OPENAI]:
            logging.info(
                f"Model allow-list not configured for {self.FRIENDLY_NAME} - all models permitted. "
                f"To restrict access, set {env_var} with comma-separated model names."
            )

        return None

    def _configure_timeouts(self, **kwargs):
        """Configure timeout settings based on provider type and custom settings.

        Custom URLs and local models often need longer timeouts due to:
        - Network latency on local networks
        - Extended thinking models taking longer to respond
        - Local inference being slower than cloud APIs

        Returns:
            httpx.Timeout object with appropriate timeout settings
        """
        import httpx

        # Default timeouts - more generous for custom/local endpoints
        default_connect = 30.0  # 30 seconds for connection (vs OpenAI's 5s)
        default_read = 600.0  # 10 minutes for reading (same as OpenAI default)
        default_write = 600.0  # 10 minutes for writing
        default_pool = 600.0  # 10 minutes for pool

        # For custom/local URLs, use even longer timeouts
        if self.base_url and self._is_localhost_url():
            default_connect = 60.0  # 1 minute for local connections
            default_read = 1800.0  # 30 minutes for local models (extended thinking)
            default_write = 1800.0  # 30 minutes for local models
            default_pool = 1800.0  # 30 minutes for local models
            logging.info(f"Using extended timeouts for local endpoint: {self.base_url}")
        elif self.base_url:
            default_connect = 45.0  # 45 seconds for custom remote endpoints
            default_read = 900.0  # 15 minutes for custom remote endpoints
            default_write = 900.0  # 15 minutes for custom remote endpoints
            default_pool = 900.0  # 15 minutes for custom remote endpoints
            logging.info(f"Using extended timeouts for custom endpoint: {self.base_url}")

        # Allow override via kwargs or environment variables in future, for now...
        connect_timeout = kwargs.get("connect_timeout")
        if connect_timeout is None:
            connect_timeout_raw = get_env("CUSTOM_CONNECT_TIMEOUT")
            connect_timeout = float(connect_timeout_raw) if connect_timeout_raw is not None else float(default_connect)

        read_timeout = kwargs.get("read_timeout")
        if read_timeout is None:
            read_timeout_raw = get_env("CUSTOM_READ_TIMEOUT")
            read_timeout = float(read_timeout_raw) if read_timeout_raw is not None else float(default_read)

        write_timeout = kwargs.get("write_timeout")
        if write_timeout is None:
            write_timeout_raw = get_env("CUSTOM_WRITE_TIMEOUT")
            write_timeout = float(write_timeout_raw) if write_timeout_raw is not None else float(default_write)

        pool_timeout = kwargs.get("pool_timeout")
        if pool_timeout is None:
            pool_timeout_raw = get_env("CUSTOM_POOL_TIMEOUT")
            pool_timeout = float(pool_timeout_raw) if pool_timeout_raw is not None else float(default_pool)

        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=write_timeout, pool=pool_timeout)

        logging.debug(
            f"Configured timeouts - Connect: {connect_timeout}s, Read: {read_timeout}s, "
            f"Write: {write_timeout}s, Pool: {pool_timeout}s"
        )

        return timeout

    def _is_localhost_url(self) -> bool:
        """Check if the base URL points to localhost or local network.

        Returns:
            True if URL is localhost or local network, False otherwise
        """
        if not self.base_url:
            return False

        try:
            parsed = urlparse(self.base_url)
            hostname = parsed.hostname

            # Check for common localhost patterns
            if hostname in ["localhost", "127.0.0.1", "::1"]:
                return True

            # Check for private network ranges (local network)
            if hostname:
                try:
                    ip = ipaddress.ip_address(hostname)
                    return ip.is_private or ip.is_loopback
                except ValueError:
                    # Not an IP address, might be a hostname
                    pass

            return False
        except Exception:
            return False

    def _validate_base_url(self) -> None:
        """Validate base URL for security (SSRF protection).

        Raises:
            ValueError: If URL is invalid or potentially unsafe
        """
        if not self.base_url:
            return

        try:
            parsed = urlparse(self.base_url)

            # Check URL scheme - only allow http/https
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")

            # Check hostname exists
            if not parsed.hostname:
                raise ValueError("URL must include a hostname")

            # Check port is valid (if specified)
            port = parsed.port
            if port is not None and (port < 1 or port > 65535):
                raise ValueError(f"Invalid port number: {port}. Must be between 1 and 65535.")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid base URL '{self.base_url}': {str(e)}")

    @property
    def client(self):
        """Lazy initialization of OpenAI client with security checks and timeout configuration."""
        if self._client is not None:
            return self._client
        with self._client_init_lock:
            # Double-checked: another thread may have built it while we waited.
            if self._client is not None:
                return self._client
            import httpx

            proxy_env_vars = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]

            with suppress_env_vars(*proxy_env_vars):
                try:
                    # Create a custom httpx client that explicitly avoids proxy parameters
                    timeout_config = (
                        self.timeout_config
                        if hasattr(self, "timeout_config") and self.timeout_config
                        else httpx.Timeout(30.0)
                    )

                    # Create httpx client with minimal config to avoid proxy conflicts
                    # Note: proxies parameter was removed in httpx 0.28.0
                    # Check for test transport injection
                    if hasattr(self, "_test_transport"):
                        # Use custom transport for testing (HTTP recording/replay)
                        http_client = httpx.Client(
                            transport=self._test_transport,
                            timeout=timeout_config,
                            follow_redirects=True,
                        )
                    else:
                        # Normal production client
                        http_client = httpx.Client(
                            timeout=timeout_config,
                            follow_redirects=True,
                        )

                    # Keep client initialization minimal to avoid proxy parameter conflicts
                    client_kwargs = {
                        "api_key": self.api_key,
                        "http_client": http_client,
                    }

                    if self.base_url:
                        client_kwargs["base_url"] = self.base_url

                    if self.organization:
                        client_kwargs["organization"] = self.organization

                    # Add default headers if any
                    if self.DEFAULT_HEADERS:
                        client_kwargs["default_headers"] = self.DEFAULT_HEADERS.copy()

                    logging.debug(
                        "OpenAI client initialized with custom httpx client and timeout: %s",
                        timeout_config,
                    )

                    # Create OpenAI client with custom httpx client
                    self._client = OpenAI(**client_kwargs)

                except Exception as e:
                    # If all else fails, try absolute minimal client without custom httpx
                    logging.warning(
                        "Failed to create client with custom httpx, falling back to minimal config: %s",
                        e,
                    )
                    try:
                        minimal_kwargs = {"api_key": self.api_key}
                        if self.base_url:
                            minimal_kwargs["base_url"] = self.base_url
                        self._client = OpenAI(**minimal_kwargs)
                    except Exception as fallback_error:
                        logging.error("Even minimal OpenAI client creation failed: %s", fallback_error)
                        raise

        return self._client

    def _sanitize_for_logging(self, params: dict) -> dict:
        """Sanitize sensitive data from parameters before logging.

        Args:
            params: Dictionary of API parameters

        Returns:
            dict: Sanitized copy of parameters safe for logging
        """
        sanitized = copy.deepcopy(params)

        # Sanitize messages content
        if "input" in sanitized:
            for msg in sanitized.get("input", []):
                if isinstance(msg, dict) and "content" in msg:
                    for content_item in msg.get("content", []):
                        if isinstance(content_item, dict) and "text" in content_item:
                            # Truncate long text and add ellipsis
                            text = content_item["text"]
                            if len(text) > 100:
                                content_item["text"] = text[:100] + "... [truncated]"

        # Remove any API keys that might be in headers/auth
        sanitized.pop("api_key", None)
        sanitized.pop("authorization", None)

        return sanitized

    def _safe_extract_output_text(self, response) -> str:
        """Safely extract output_text from o3-pro response with validation.

        Args:
            response: Response object from OpenAI SDK

        Returns:
            str: The output text content

        Raises:
            ValueError: If output_text is missing, None, or not a string
        """
        logging.debug("Response object type: %s", type(response).__name__)

        if not hasattr(response, "output_text"):
            raise ValueError(f"o3-pro response missing output_text field. Response type: {type(response).__name__}")

        content = response.output_text

        if content is None:
            raise ValueError("o3-pro returned None for output_text")

        if not isinstance(content, str):
            raise ValueError(f"o3-pro output_text is not a string. Got type: {type(content).__name__}")

        # Log only the length, never the content: the model may echo back secrets
        # from reviewed code, and LOG_LEVEL can be DEBUG. An unescaped f-string of
        # model output would also permit log-line forgery.
        logging.debug("Extracted output_text: %d chars", len(content))

        return content

    @staticmethod
    def _to_responses_content(content, role: str) -> list:
        """Convert a chat-format message ``content`` into Responses API content.

        Chat-completions content is either a plain string or a list of parts
        (``{"type": "text", ...}`` / ``{"type": "image_url", ...}``). The Responses
        API uses ``input_text`` / ``output_text`` and ``input_image`` instead.
        Previously a list was assigned verbatim to an ``input_text`` ``text``
        field, producing a malformed payload (400) and never mapping images, so
        vision was unusable on every responses-API model.
        """
        text_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, str):
            return [{"type": text_type, "text": content}]

        parts: list = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    parts.append({"type": text_type, "text": part.get("text", "")})
                elif ptype == "image_url":
                    image_url = part.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if url:
                        parts.append({"type": "input_image", "image_url": url})
                elif ptype in ("input_text", "output_text", "input_image"):
                    parts.append(part)  # already in Responses format

        if not parts:
            parts.append({"type": text_type, "text": content if isinstance(content, str) else ""})
        return parts

    def _generate_with_responses_endpoint(
        self,
        model_name: str,
        messages: list,
        temperature: float,
        max_output_tokens: int | None = None,
        capabilities: ModelCapabilities | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using the /v1/responses endpoint for reasoning models."""
        # Convert messages to the correct format for responses endpoint
        input_messages: list = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                # For o3-pro, system messages should be handled carefully to avoid policy violations
                # Instead of prefixing with "System:", we'll include the system content naturally
                input_messages.append({"role": "user", "content": self._to_responses_content(content, "user")})
            elif role == "user":
                input_messages.append({"role": "user", "content": self._to_responses_content(content, "user")})
            elif role == "assistant":
                input_messages.append(
                    {"role": "assistant", "content": self._to_responses_content(content, "assistant")}
                )

        # Prepare completion parameters for responses endpoint
        # Based on OpenAI documentation, use nested reasoning object for responses endpoint
        effort = "medium"
        if capabilities and capabilities.default_reasoning_effort:
            effort = capabilities.default_reasoning_effort

        completion_params = {
            "model": model_name,
            "input": input_messages,
            "reasoning": {"effort": effort},
        }

        # Only include store parameter for providers that support it.
        # OpenRouter's /responses endpoint rejects store:true via Zod validation (Issue #348).
        # This is an endpoint-level limitation, not model-specific, so we omit for all
        # OpenRouter /responses calls. If OpenRouter later supports store, revisit this logic.
        if self.get_provider_type() != ProviderType.OPENROUTER:
            completion_params["store"] = True
        else:
            logging.debug(f"Omitting 'store' parameter for OpenRouter provider (model: {model_name})")

        # Add max tokens if specified. The Responses API parameter is
        # ``max_output_tokens`` (``max_completion_tokens`` is chat-completions
        # only and raises TypeError on responses.create).
        if max_output_tokens:
            completion_params["max_output_tokens"] = max_output_tokens

        # For responses endpoint, we only add parameters that are explicitly supported
        # Remove unsupported chat completion parameters that may cause API errors

        # Retry logic with progressive delays
        max_retries = 4
        retry_delays = [1, 3, 5, 8]
        attempt_counter = {"value": 0}

        def _attempt() -> ModelResponse:
            attempt_counter["value"] += 1

            # Log only structural metadata, not message text. The previous INFO log
            # dumped message prefixes (system + user prompt + history) even after
            # sanitizing api_key, which persisted prompt content to disk on every
            # attempt.
            logging.debug(
                "Responses API request: model=%s, messages=%d, reasoning_effort=%s",
                completion_params.get("model"),
                len(completion_params.get("input", [])),
                (completion_params.get("reasoning") or {}).get("effort"),
            )

            response = self.client.responses.create(**completion_params)

            content = self._safe_extract_output_text(response)

            # _extract_usage handles both chat-completions and Responses-API
            # field names, so a single call covers all shapes.
            usage = self._extract_usage(response) or None

            return ModelResponse(
                content=content,
                usage=usage,
                model_name=model_name,
                friendly_name=self.FRIENDLY_NAME,
                provider=self.get_provider_type(),
                metadata={
                    "model": getattr(response, "model", model_name),
                    "id": getattr(response, "id", ""),
                    "created": getattr(response, "created_at", 0),
                    "endpoint": "responses",
                },
            )

        try:
            return self._run_with_retries(
                operation=_attempt,
                max_attempts=max_retries,
                delays=retry_delays,
                log_prefix="responses endpoint",
            )
        except Exception as exc:
            attempts = max(attempt_counter["value"], 1)
            error_msg = f"responses endpoint error after {attempts} attempt{'s' if attempts > 1 else ''}: {exc}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from exc

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int | None = None,
        images: list[str] | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using the OpenAI-compatible API.

        Args:
            prompt: User prompt to send to the model
            model_name: Canonical model name or its alias
            system_prompt: Optional system prompt for model behavior
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens to generate
            images: Optional list of image paths or data URLs to include with the prompt (for vision models)
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse with generated content and metadata
        """
        # Validate model name against allow-list
        if not self.validate_model_name(model_name):
            raise ValueError(f"Model '{model_name}' not in allowed models list. Allowed models: {self.allowed_models}")

        capabilities: ModelCapabilities | None
        try:
            capabilities = self.get_capabilities(model_name)
        except Exception as exc:
            logging.debug(f"Falling back to generic capabilities for {model_name}: {exc}")
            capabilities = None

        # Get effective temperature for this model from capabilities when available
        if capabilities:
            effective_temperature = capabilities.get_effective_temperature(temperature)
            if effective_temperature is not None and effective_temperature != temperature:
                logging.debug(
                    f"Adjusting temperature from {temperature} to {effective_temperature} for model {model_name}"
                )
        else:
            effective_temperature = temperature

        # Only validate if temperature is not None (meaning the model supports it)
        if effective_temperature is not None:
            # Validate parameters with the effective temperature
            self.validate_parameters(model_name, effective_temperature)

        # Resolve to canonical model name
        resolved_model = self._resolve_model_name(model_name)

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Prepare user message with text and potentially images
        user_content = []
        user_content.append({"type": "text", "text": prompt})

        # Add images if provided and model supports vision
        if images and capabilities and capabilities.supports_images:
            for image_path in images:
                try:
                    image_content = self._process_image(image_path)
                    if image_content:
                        user_content.append(image_content)
                except Exception as e:
                    logging.warning(f"Failed to process image {image_path}: {e}")
                    # Continue with other images and text
                    continue
        elif images and (not capabilities or not capabilities.supports_images):
            logging.warning(f"Model {resolved_model} does not support images, ignoring {len(images)} image(s)")

        # Add user message
        if len(user_content) == 1:
            # Only text content, use simple string format for compatibility
            messages.append({"role": "user", "content": prompt})
        else:
            # Text + images, use content array format
            messages.append({"role": "user", "content": user_content})

        # Prepare completion parameters
        # Always disable streaming for OpenRouter
        # MCP doesn't use streaming, and this avoids issues with O3 model access
        completion_params = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
        }

        # Use the effective temperature we calculated earlier
        supports_sampling = effective_temperature is not None

        if supports_sampling:
            completion_params["temperature"] = effective_temperature

        # Add max tokens if specified and model supports it
        # O3/O4 models that don't support temperature also don't support max_tokens
        if max_output_tokens and supports_sampling:
            completion_params["max_tokens"] = max_output_tokens

        # Add any additional OpenAI-specific parameters
        # Use capabilities to filter parameters for reasoning models
        for key, value in kwargs.items():
            if key in ["top_p", "frequency_penalty", "presence_penalty", "seed", "stop", "stream"]:
                # Reasoning models (those that don't support temperature) also don't support these parameters
                if not supports_sampling and key in ["top_p", "frequency_penalty", "presence_penalty", "stream"]:
                    continue  # Skip unsupported parameters for reasoning models
                completion_params[key] = value

        # Check if this model needs the Responses API endpoint
        # Prefer capability metadata; fall back to static map when capabilities unavailable
        use_responses_api = False
        if capabilities is not None:
            use_responses_api = getattr(capabilities, "use_openai_response_api", False)
        else:
            static_capabilities = self.get_all_model_capabilities().get(resolved_model)
            if static_capabilities is not None:
                use_responses_api = getattr(static_capabilities, "use_openai_response_api", False)

        if use_responses_api:
            # These models require the /v1/responses endpoint for stateful context
            # If it fails, we should not fall back to chat/completions
            return self._generate_with_responses_endpoint(
                model_name=resolved_model,
                messages=messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                capabilities=capabilities,
                **kwargs,
            )

        # Retry logic with progressive delays
        max_retries = 4  # Total of 4 attempts
        retry_delays = [1, 3, 5, 8]  # Progressive delays: 1s, 3s, 5s, 8s
        attempt_counter = {"value": 0}

        def _attempt() -> ModelResponse:
            attempt_counter["value"] += 1
            response = self.client.chat.completions.create(**completion_params)

            content = response.choices[0].message.content
            usage = self._extract_usage(response)

            # A reasoning model can answer 200 OK with content="" and
            # finish_reason="length" when reasoning consumes the entire output
            # budget. Returning that as a success is worse than failing: callers
            # test `if model_response.content:`, so an empty string is falsy and
            # the tool completes having produced nothing, with no error anywhere
            # to explain it. Raising surfaces the cause instead.
            if not (content or "").strip():
                finish_reason = response.choices[0].finish_reason
                reasoning_tokens = None
                try:
                    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                except AttributeError:
                    pass
                raise EmptyContentError(
                    f"{self.FRIENDLY_NAME} returned empty content for {resolved_model} "
                    f"(finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}). "
                    f"For reasoning models this usually means the output budget was spent "
                    f"on reasoning; reduce the input size or raise max_tokens.",
                    finish_reason=finish_reason,
                    reasoning_tokens=reasoning_tokens,
                )

            return ModelResponse(
                content=content,
                usage=usage,
                model_name=resolved_model,
                friendly_name=self.FRIENDLY_NAME,
                provider=self.get_provider_type(),
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                    "id": response.id,
                    "created": response.created,
                },
            )

        try:
            return self._run_with_retries(
                operation=_attempt,
                max_attempts=max_retries,
                delays=retry_delays,
                log_prefix=f"{self.FRIENDLY_NAME} API ({resolved_model})",
            )
        except Exception as exc:
            attempts = max(attempt_counter["value"], 1)
            error_msg = (
                f"{self.FRIENDLY_NAME} API error for model {resolved_model} after {attempts} attempt"
                f"{'s' if attempts > 1 else ''}: {exc}"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg) from exc

    def generate_content_stream(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int | None = None,
        images: list[str] | None = None,
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """Stream content using the OpenAI-compatible ``stream=True`` API.

        Yields :class:`StreamChunk` objects for each ``ChatCompletionChunk``
        delta received.  The final chunk (where ``finish_reason`` is set) has
        ``is_final=True`` and includes usage data when available.

        Subclasses (``openai.py``, ``azure_openai.py``, ``xai.py``) inherit
        this implementation unless they explicitly override it.
        """
        if not self.validate_model_name(model_name):
            raise ValueError(f"Model '{model_name}' not in allowed models list. Allowed models: {self.allowed_models}")

        capabilities: ModelCapabilities | None
        try:
            capabilities = self.get_capabilities(model_name)
        except Exception as exc:
            logging.debug(f"Falling back to generic capabilities for {model_name}: {exc}")
            capabilities = None

        if capabilities:
            effective_temperature = capabilities.get_effective_temperature(temperature)
            if effective_temperature is not None and effective_temperature != temperature:
                logging.debug(
                    f"Adjusting temperature from {temperature} to {effective_temperature} for model {model_name}"
                )
        else:
            effective_temperature = temperature

        if effective_temperature is not None:
            self.validate_parameters(model_name, effective_temperature)

        resolved_model = self._resolve_model_name(model_name)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = [{"type": "text", "text": prompt}]
        if images and capabilities and capabilities.supports_images:
            for image_path in images:
                try:
                    image_content = self._process_image(image_path)
                    if image_content:
                        user_content.append(image_content)
                except Exception as e:
                    logging.warning(f"Failed to process image {image_path}: {e}")
                    continue

        if len(user_content) == 1:
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": user_content})

        # Build streaming completion params
        supports_sampling = effective_temperature is not None
        completion_params = {
            "model": resolved_model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if supports_sampling:
            completion_params["temperature"] = effective_temperature
        if max_output_tokens and supports_sampling:
            completion_params["max_tokens"] = max_output_tokens

        # Models that require the Responses API fall back to the default
        # single-chunk wrapper since streaming on that endpoint is not
        # implemented here.
        use_responses_api = False
        if capabilities is not None:
            use_responses_api = getattr(capabilities, "use_openai_response_api", False)
        else:
            static_capabilities = self.get_all_model_capabilities().get(resolved_model)
            if static_capabilities is not None:
                use_responses_api = getattr(static_capabilities, "use_openai_response_api", False)

        if use_responses_api:
            yield from super().generate_content_stream(
                prompt,
                model_name,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                images=images,
                **kwargs,
            )
            return

        # Circuit breaker: the streaming path bypasses _run_with_retries, so
        # enforce and update breaker health here too (fail fast when down; record
        # the outcome so a dead provider is fail-fasted next time).
        if not self._circuit_breaker.allow_request():
            from utils.circuit_breaker import ProviderUnavailable

            raise ProviderUnavailable(
                provider_name=self._circuit_breaker._provider_name,
                circuit_state=self._circuit_breaker.state.value,
            )

        try:
            stream = self.client.chat.completions.create(**completion_params)
            usage = {}
            for chunk in stream:
                # Extract usage from the final stream message (when include_usage=True)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage = self._extract_usage(chunk)

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                delta_content = getattr(delta, "content", None) or ""
                finish_reason = chunk.choices[0].finish_reason

                if finish_reason is not None:
                    self._circuit_breaker.record_success()
                    yield StreamChunk(text=delta_content, is_final=True, usage=usage or None)
                    return
                elif delta_content:
                    yield StreamChunk(text=delta_content, is_final=False)

            # If we exit the loop without a finish_reason, yield a final chunk
            self._circuit_breaker.record_success()
            yield StreamChunk(text="", is_final=True, usage=usage or None)

        except Exception as exc:
            if self._is_provider_unhealthy_error(exc):
                self._circuit_breaker.record_failure()
            error_msg = f"{self.FRIENDLY_NAME} streaming error for model {resolved_model}: {exc}"
            raise RuntimeError(error_msg) from exc

    def validate_parameters(self, model_name: str, temperature: float, **kwargs) -> None:
        """Validate model parameters.

        For proxy providers, this may use generic capabilities.

        Args:
            model_name: Canonical model name or its alias
            temperature: Temperature to validate
            **kwargs: Additional parameters to validate
        """
        try:
            capabilities = self.get_capabilities(model_name)

            # Check if we're using generic capabilities
            if hasattr(capabilities, "_is_generic"):
                logging.debug(
                    f"Using generic parameter validation for {model_name}. Actual model constraints may differ."
                )

            # Validate temperature using parent class method
            super().validate_parameters(model_name, temperature, **kwargs)

        except Exception as e:
            # For proxy providers, we might not have accurate capabilities
            # Log warning but don't fail
            logging.warning(f"Parameter validation limited for {model_name}: {e}")

    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from OpenAI response.

        Args:
            response: OpenAI API response object

        Returns:
            Dictionary with usage statistics
        """
        usage = {}

        if hasattr(response, "usage") and response.usage:
            u = response.usage
            # Distinguish shapes by which field EXISTS: chat-completions has
            # prompt_tokens/completion_tokens; the Responses API has
            # input_tokens/output_tokens. Checking existence (not value) keeps the
            # chat-completions None->0 semantics while still reading usage for
            # responses-API models (previously always zero for them).
            if hasattr(u, "prompt_tokens"):
                usage["input_tokens"] = getattr(u, "prompt_tokens", 0) or 0
                usage["output_tokens"] = getattr(u, "completion_tokens", 0) or 0
                usage["total_tokens"] = getattr(u, "total_tokens", 0) or 0
            else:
                input_tokens = getattr(u, "input_tokens", 0) or 0
                output_tokens = getattr(u, "output_tokens", 0) or 0
                total_tokens = getattr(u, "total_tokens", 0) or 0
                usage["input_tokens"] = input_tokens
                usage["output_tokens"] = output_tokens
                usage["total_tokens"] = total_tokens or (input_tokens + output_tokens)

        return usage

    # Class-level cache for tiktoken encoding objects
    _encoding_cache: dict = {}

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens using tiktoken with encoding cache, falling back to base class."""

        resolved_model = self._resolve_model_name(model_name)

        try:
            import tiktoken

            if resolved_model not in self._encoding_cache:
                try:
                    self._encoding_cache[resolved_model] = tiktoken.encoding_for_model(resolved_model)
                except KeyError:
                    self._encoding_cache[resolved_model] = tiktoken.get_encoding("cl100k_base")

            return len(self._encoding_cache[resolved_model].encode(text))

        except (ImportError, Exception) as exc:
            logging.debug("tiktoken unavailable for %s: %s", resolved_model, exc)

        return super().count_tokens(text, model_name)

    def _is_error_retryable(self, error: Exception) -> bool:
        """Determine if an error should be retried based on structured error codes.

        Uses OpenAI API error structure instead of text pattern matching for reliability.

        Args:
            error: Exception from OpenAI API call

        Returns:
            True if error should be retried, False otherwise
        """
        error_str = str(error).lower()

        # Empty-content decisions read the finish_reason field, never the
        # message. This has to come first: the message embeds a token count,
        # and every classifier below inspects the string, so a reasoning_tokens
        # value of 1500 or 8503 would otherwise match the "500"/"503" retry
        # indicators and turn retry behaviour into a function of arbitrary
        # digits — the same failure the structured-status-code comment below
        # was added to fix.
        if isinstance(error, EmptyContentError):
            # finish_reason="length" means the output budget was exhausted.
            # That is a function of input size and model configuration, both
            # unchanged on a retry, so retrying spends an expensive reasoning
            # call to reproduce the result. Any other finish_reason with empty
            # content is unexplained and plausibly transient, so allow it.
            return error.finish_reason != "length"

        # Timeouts / connection errors are always retryable regardless of code.
        if _OpenAITimeoutError is not None and isinstance(error, _OpenAITimeoutError):
            return True
        if _OpenAIConnectionError is not None and isinstance(error, _OpenAIConnectionError):
            return True

        # Prefer a STRUCTURED status code over string matching. Without this, any
        # error whose message merely contains the digits "429" (token counts,
        # request IDs, byte sizes) was misrouted into the rate-limit branch and
        # retried even when it was a permanent 4xx.
        structured_code = None
        if _OpenAIStatusError is not None and isinstance(error, _OpenAIStatusError):
            structured_code = getattr(error, "status_code", None)
        if not isinstance(structured_code, int):
            structured_code = self._extract_status_code(error)

        # It is a rate-limit 429 only if the structured code says so; fall back to
        # the substring heuristic ONLY when no structured status code is available.
        is_rate_limit = (structured_code == 429) if isinstance(structured_code, int) else ("429" in error_str)

        if is_rate_limit:
            # Try to extract structured error information
            error_type = None
            error_code = None

            # Parse structured error from OpenAI API response
            # Format: "Error code: 429 - {'error': {'type': 'tokens', 'code': 'rate_limit_exceeded', ...}}"
            try:
                import ast
                import json
                import re

                # Extract JSON part from error string using regex
                # Look for pattern: {...} (from first { to last })
                json_match = re.search(r"\{.*\}", str(error))
                if json_match:
                    json_like_str = json_match.group(0)

                    # First try: parse as Python literal (handles single quotes safely)
                    try:
                        error_data = ast.literal_eval(json_like_str)
                    except (ValueError, SyntaxError):
                        # Fallback: try JSON parsing with simple quote replacement
                        # (for cases where it's already valid JSON or simple replacements work)
                        json_str = json_like_str.replace("'", '"')
                        error_data = json.loads(json_str)

                    if "error" in error_data:
                        error_info = error_data["error"]
                        error_type = error_info.get("type")
                        error_code = error_info.get("code")

            except (json.JSONDecodeError, ValueError, SyntaxError, AttributeError):
                # Fall back to checking hasattr for OpenAI SDK exception objects
                if hasattr(error, "response") and hasattr(error.response, "json"):
                    try:
                        response_data = error.response.json()
                        if "error" in response_data:
                            error_info = response_data["error"]
                            error_type = error_info.get("type")
                            error_code = error_info.get("code")
                    except Exception:
                        pass

            # Determine if 429 is retryable based on structured error codes
            if error_type == "tokens":
                # Token-related 429s are typically non-retryable (request too large)
                logging.debug(f"Non-retryable 429: token-related error (type={error_type}, code={error_code})")
                return False
            elif error_code in ["invalid_request_error", "context_length_exceeded"]:
                # These are permanent failures
                logging.debug(f"Non-retryable 429: permanent failure (type={error_type}, code={error_code})")
                return False
            else:
                # Other 429s (like requests per minute) are retryable
                logging.debug(f"Retryable 429: rate limiting (type={error_type}, code={error_code})")
                return True

        # Non-429 structured status codes.
        if isinstance(structured_code, int):
            if structured_code in self._RETRYABLE_STATUS_CODES:
                return True
            if structured_code in self._NON_RETRYABLE_STATUS_CODES:
                return False

        # Delegate to base class three-tier classification
        return super()._is_error_retryable(error)

    def _process_image(self, image_path: str) -> dict | None:
        """Process an image for OpenAI-compatible API."""
        try:
            if image_path.startswith("data:"):
                # Validate the data URL
                validate_image(image_path)
                # Handle data URL: data:image/png;base64,iVBORw0...
                return {"type": "image_url", "image_url": {"url": image_path}}
            else:
                # Use base class validation
                image_bytes, mime_type = validate_image(image_path)

                # Read and encode the image
                import base64

                image_data = base64.b64encode(image_bytes).decode()
                logging.debug(f"Processing image '{image_path}' as MIME type '{mime_type}'")

                # Create data URL for OpenAI API
                data_url = f"data:{mime_type};base64,{image_data}"

                return {"type": "image_url", "image_url": {"url": data_url}}

        except ValueError as e:
            logging.warning(str(e))
            return None
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return None
