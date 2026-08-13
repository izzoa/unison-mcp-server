"""
MCP tool handler implementations for the Unison MCP Server.

Contains the list_tools and call_tool handler logic. build_handlers()
assembles legacy-shaped impls (the 1.x handler signatures, re-exported by
server.py for tests and callers) plus the on_* adapters that server.py
passes to the mcp 2.x Server constructor.
"""

import logging
from typing import Any, NamedTuple

import jsonschema
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool, ToolAnnotations

from config import DEFAULT_MODEL
from tools.models import ToolOutput
from tools.shared.exceptions import ToolExecutionError
from utils.env import get_env
from utils.mcp_context import reset_current_request_context, set_current_request_context
from utils.model_resolution import parse_model_option
from utils.observability import tool_span
from utils.request_helpers import get_follow_up_instructions

logger = logging.getLogger(__name__)


def _sanitize_for_log(value: Any, max_len: int = 128) -> str:
    """Neutralize log-injection: escape CR/LF (and other control chars) and cap
    length before writing an untrusted value (tool name, continuation_id,
    clientInfo) to the logs. Prevents a crafted value from forging log lines
    (CWE-117)."""
    text = str(value)
    text = text.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    # Strip any remaining non-printable control characters.
    text = "".join(ch if ch.isprintable() else "?" for ch in text)
    if len(text) > max_len:
        text = text[:max_len] + "...(truncated)"
    return text


class ToolHandlers(NamedTuple):
    """Handler bundle: legacy-shaped impls plus mcp 2.x constructor adapters."""

    handle_list_tools: Any  # () -> list[Tool]
    handle_call_tool: Any  # (name, arguments) -> list[TextContent]
    on_list_tools: Any  # 2.x adapter: (ctx, params) -> ListToolsResult
    on_call_tool: Any  # 2.x adapter: (ctx, params) -> CallToolResult


def build_handlers(tool_registry):
    """
    Build list_tools and call_tool handlers for the mcp 2.x server.

    The legacy-shaped impls keep the 1.x handler signatures; the on_* adapters
    wrap them with the behavior the 1.x SDK decorators used to provide —
    request-context binding, input-schema validation, result wrapping, and
    exception→isError conversion — and are what the Server is constructed with.

    Args:
        tool_registry: A ToolRegistry providing tool lookup and schemas.

    Returns:
        ToolHandlers namedtuple.
    """

    async def handle_list_tools() -> list[Tool]:
        """List all available tools with their descriptions and input schemas."""
        logger.debug("MCP client requested tool list")

        # Try to log client info if available
        try:
            from utils.client_info import format_client_info, get_client_info_from_context

            client_info = get_client_info_from_context()
            if client_info:
                formatted = format_client_info(client_info)
                logger.info("MCP Client Connected: %s", formatted)

                try:
                    mcp_activity_logger = logging.getLogger("mcp_activity")
                    # clientInfo is attacker-influenced; sanitize before logging.
                    friendly_name = _sanitize_for_log(client_info.get("friendly_name", "CLI Agent"))
                    raw_name = _sanitize_for_log(client_info.get("name", "Unknown"))
                    version = _sanitize_for_log(client_info.get("version", "Unknown"))
                    mcp_activity_logger.info("MCP_CLIENT_INFO: %s (raw=%s v%s)", friendly_name, raw_name, version)
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Could not log client info during list_tools: %s", e)

        tools = []
        available = tool_registry.get_available_tools()

        for tool in available.values():
            annotations = tool.get_annotations()
            tool_annotations = ToolAnnotations(**annotations) if annotations else None
            tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.get_input_schema(),
                    annotations=tool_annotations,
                )
            )

        # Log cache efficiency info
        openrouter_key_for_cache = get_env("OPENROUTER_API_KEY")
        if openrouter_key_for_cache and openrouter_key_for_cache != "your_openrouter_api_key_here":
            logger.debug("OpenRouter registry cache used efficiently across all tool schemas")

        logger.debug("Returning %d tools to MCP client", len(tools))
        return tools

    async def _handle_call_tool_inner(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """
        Handle incoming tool execution requests from MCP clients.

        Routes tool calls to their appropriate handlers, managing model resolution,
        conversation thread reconstruction, and file size validation at the MCP boundary.
        """
        # Untrusted values (tool name, continuation_id) are logged before any
        # validation; sanitize them so a crafted value cannot forge log lines.
        safe_name = _sanitize_for_log(name)
        logger.info("MCP tool call: %s", safe_name)
        logger.debug("MCP tool arguments: %s", list(arguments.keys()))

        try:
            mcp_activity_logger = logging.getLogger("mcp_activity")
            mcp_activity_logger.info("TOOL_CALL: %s with %d arguments", safe_name, len(arguments))
        except Exception:
            pass

        # Cheap gate BEFORE any stateful work: reject unknown/disabled tools up
        # front so a continuation_id call to a nonexistent tool does not persist
        # an orphan user turn (via reconstruct_thread_context -> add_turn) and
        # burn a turn-budget slot before failing.
        if not tool_registry.is_available(name):
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        # Handle thread context reconstruction if continuation_id is present
        if "continuation_id" in arguments and arguments["continuation_id"]:
            continuation_id = arguments["continuation_id"]
            safe_cid = _sanitize_for_log(continuation_id)
            logger.debug("Resuming conversation thread: %s", safe_cid)
            logger.debug(
                "[CONVERSATION_DEBUG] Tool '%s' resuming thread %s with %d arguments",
                safe_name,
                safe_cid,
                len(arguments),
            )
            logger.debug("[CONVERSATION_DEBUG] Original arguments keys: %s", list(arguments.keys()))

            try:
                mcp_activity_logger = logging.getLogger("mcp_activity")
                mcp_activity_logger.info("CONVERSATION_RESUME: %s resuming thread %s", safe_name, safe_cid)
            except Exception:
                pass

            arguments = await reconstruct_thread_context(arguments, tool_registry)
            logger.debug(
                "[CONVERSATION_DEBUG] After thread reconstruction, arguments keys: %s",
                list(arguments.keys()),
            )
            _reconstructed_ctx = None
            if logger.isEnabledFor(logging.DEBUG):
                from utils.tool_execution_context import ToolExecutionContext as _TEC

                _reconstructed_ctx = _TEC.from_arguments(arguments)
                if _reconstructed_ctx is not None:
                    logger.debug(
                        "[CONVERSATION_DEBUG] Remaining token budget: %s",
                        f"{_reconstructed_ctx.remaining_tokens:,}",
                    )

        # Route to registered tools
        if tool_registry.is_available(name):
            logger.info("Executing tool '%s' with %d parameter(s)", name, len(arguments))
            try:
                tool = tool_registry.get_tool_instance(name)
            except Exception as exc:
                # A dynamic tool can be quarantined between the availability
                # gate and here (instance-level validation runs on first
                # instantiation). Map the controlled registry error to a
                # tool-level response instead of surfacing a raw traceback.
                from tools.registry import ToolQuarantinedError

                if isinstance(exc, ToolQuarantinedError):
                    logger.warning("Tool '%s' unavailable: %s", safe_name, exc.reason)
                    return [TextContent(type="text", text=f"Tool '{name}' is unavailable: {exc.reason}")]
                raise

            # EARLY MODEL RESOLUTION AT MCP BOUNDARY
            from providers.registry import get_default_registry
            from utils.file_utils import check_total_file_size
            from utils.model_context import ModelContext

            registry = get_default_registry()

            model_name = arguments.get("model") or DEFAULT_MODEL
            logger.debug("Initial model for %s: %s", name, model_name)

            model_name, model_option = parse_model_option(model_name)
            if model_option:
                logger.info("Parsed model format - model: '%s', option: '%s'", model_name, model_option)
            else:
                logger.info("Parsed model format - model: '%s'", model_name)

            # Skip model resolution for tools that don't require models
            if not tool.requires_model():
                logger.debug("Tool %s doesn't require model resolution - skipping model validation", name)
                return await tool.execute(arguments)

            # Handle auto mode at MCP boundary
            if model_name.lower() == "auto":
                tool_category = tool.get_model_category()
                resolved_model = registry.get_preferred_fallback_model(tool_category)
                logger.info(
                    "Auto mode resolved to %s for %s (category: %s)",
                    resolved_model,
                    name,
                    tool_category.value,
                )
                model_name = resolved_model
                arguments["model"] = model_name

            # Validate model availability at MCP boundary
            provider = registry.get_provider_for_model(model_name)
            if not provider:
                available_models = list(registry.get_available_models(respect_restrictions=True).keys())
                tool_category = tool.get_model_category()
                suggested_model = registry.get_preferred_fallback_model(tool_category)

                error_message = (
                    f"Model '{model_name}' is not available with current API keys. "
                    f"Available models: {', '.join(available_models)}. "
                    f"Suggested model for {name}: '{suggested_model}' "
                    f"(category: {tool_category.value})"
                )
                error_output = ToolOutput(
                    status="error",
                    content=error_message,
                    content_type="text",
                    metadata={"tool_name": name, "requested_model": model_name},
                )
                raise ToolExecutionError(error_output.model_dump_json())

            # Create model context and typed execution context
            model_context = ModelContext(model_name, model_option)
            import dataclasses

            from utils.tool_execution_context import ToolExecutionContext

            # Preserve continuation state that reconstruct_thread_context already
            # computed (remaining_tokens, original_user_prompt). Rebuilding the
            # context from scratch here would reset them to their defaults
            # (0 / ""), which makes continuation size-validation run against the
            # full enhanced prompt (spurious "prompt too large") and collapses the
            # workflow expert-analysis file budget to ~1k tokens. Only overlay the
            # freshly resolved model information onto the existing context.
            _existing_ctx = ToolExecutionContext.from_arguments(arguments)
            if _existing_ctx is not None and dataclasses.is_dataclass(_existing_ctx):
                arguments["_context"] = dataclasses.replace(
                    _existing_ctx,
                    model_context=model_context,
                    resolved_model_name=model_name,
                    registry=registry,
                )
            else:
                arguments["_context"] = ToolExecutionContext(
                    model_context=model_context,
                    resolved_model_name=model_name,
                    registry=registry,
                )
            logger.debug(
                "Model context created for %s with %d token capacity",
                model_name,
                model_context.capabilities.context_window,
            )
            if model_option:
                logger.debug("Model option stored in context: '%s'", model_option)

            # EARLY FILE SIZE VALIDATION AT MCP BOUNDARY
            argument_files = arguments.get("absolute_file_paths")
            if argument_files:
                logger.debug("Checking file sizes for %d files with model %s", len(argument_files), model_name)
                file_size_check = check_total_file_size(argument_files, model_name)
                if file_size_check:
                    logger.warning("File size check failed for %s with model %s", name, model_name)
                    raise ToolExecutionError(ToolOutput(**file_size_check).model_dump_json())

            # Execute tool with pre-resolved model context
            result = await tool.execute(arguments)
            logger.info("Tool '%s' execution completed", name)

            try:
                mcp_activity_logger = logging.getLogger("mcp_activity")
                mcp_activity_logger.info("TOOL_COMPLETED: %s", name)
            except Exception:
                pass
            return result

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """MCP entry point: one parent observability span wraps the full tool lifecycle.

        `tool_span` is a no-op unless UNISON_OTEL_ENABLED=true; when enabled it
        records lifecycle attributes, aggregates provider-call token/model data
        propagated via ContextVar, and increments the tool-call counter exactly
        once per invocation (provider fan-out and retries never inflate it).
        """
        with tool_span(name, arguments):
            return await _handle_call_tool_inner(name, arguments)

    # ------------------------------------------------------------------
    # mcp 2.x adapters — replicate what the 1.x SDK decorators provided
    # ------------------------------------------------------------------
    _tool_cache: dict[str, Tool] = {}

    def _error_result(message: str) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)

    async def _refresh_tool_cache() -> None:
        tools = await handle_list_tools()
        _tool_cache.clear()
        for tool in tools:
            _tool_cache[tool.name] = tool

    async def _get_cached_tool_definition(tool_name: str) -> Tool | None:
        # Mirrors the 1.x SDK: a cache miss triggers a list_tools refresh, and
        # an unlisted tool skips validation (the impl produces its own
        # "Unknown tool" response) rather than erroring here.
        if tool_name not in _tool_cache:
            await _refresh_tool_cache()
        tool = _tool_cache.get(tool_name)
        if tool is None:
            logger.warning("Tool '%s' not listed, no validation will be performed", tool_name)
        return tool

    async def on_list_tools(ctx, params) -> ListToolsResult:
        token = set_current_request_context(ctx)
        try:
            tools = await handle_list_tools()
        finally:
            reset_current_request_context(token)
        _tool_cache.clear()
        for tool in tools:
            _tool_cache[tool.name] = tool
        return ListToolsResult(tools=tools)

    async def on_call_tool(ctx, params) -> CallToolResult:
        token = set_current_request_context(ctx)
        try:
            name = params.name
            arguments = params.arguments or {}

            # Input-schema validation, exactly as the 1.x decorator did it
            # (same jsonschema mechanism and error text).
            tool = await _get_cached_tool_definition(name)
            if tool is not None:
                try:
                    jsonschema.validate(instance=arguments, schema=tool.input_schema)
                except jsonschema.ValidationError as e:
                    return _error_result(f"Input validation error: {e.message}")

            results = await handle_call_tool(name, arguments)

            # Our tools return list[TextContent]; pass a full CallToolResult
            # through untouched if one ever appears.
            if isinstance(results, CallToolResult):
                return results
            return CallToolResult(content=list(results), is_error=False)
        except Exception as e:
            # 1.x converted handler exceptions to an error result of str(e);
            # ToolExecutionError payloads round-trip through this unchanged.
            return _error_result(str(e))
        finally:
            reset_current_request_context(token)

    return ToolHandlers(
        handle_list_tools=handle_list_tools,
        handle_call_tool=handle_call_tool,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def reconstruct_thread_context(arguments: dict[str, Any], tool_registry: Any) -> dict[str, Any]:
    """
    Reconstruct conversation context for stateless-to-stateful thread continuation.

    Loads persistent conversation state from storage and rebuilds complete
    conversation context using the dual prioritization strategy.

    Args:
        arguments: Original request arguments containing continuation_id.
        tool_registry: ToolRegistry for looking up tool instances.

    Returns:
        Enhanced arguments with conversation history and token budget info.
    """
    from utils.conversation_memory import add_turn, build_conversation_history, get_thread

    continuation_id = arguments["continuation_id"]

    logger.debug("[CONVERSATION_DEBUG] Looking up thread %s in storage", continuation_id)
    context = get_thread(continuation_id)
    if not context:
        logger.warning("Thread not found: %s", continuation_id)
        logger.debug("[CONVERSATION_DEBUG] Thread %s not found in storage or expired", continuation_id)

        try:
            mcp_activity_logger = logging.getLogger("mcp_activity")
            mcp_activity_logger.info("CONVERSATION_ERROR: Thread %s not found or expired", continuation_id)
        except Exception:
            pass

        raise ValueError(
            f"Conversation thread '{continuation_id}' was not found or has expired. "
            f"This may happen if the conversation was created more than 3 hours ago or if the "
            f"server was restarted. "
            f"Please restart the conversation by providing your full question/prompt without the "
            f"continuation_id parameter. "
            f"This will create a new conversation thread that can continue with follow-up exchanges."
        )

    # Add user's new input to the conversation
    user_prompt = arguments.get("prompt", "")
    if user_prompt:
        user_files = arguments.get("absolute_file_paths") or []
        logger.debug("[CONVERSATION_DEBUG] Adding user turn to thread %s", continuation_id)
        user_prompt_tokens = len(user_prompt) // 4
        logger.debug(
            "[CONVERSATION_DEBUG] User prompt length: %d chars (~%s tokens)",
            len(user_prompt),
            f"{user_prompt_tokens:,}",
        )
        logger.debug("[CONVERSATION_DEBUG] User files: %s", user_files)
        success = add_turn(continuation_id, "user", user_prompt, files=user_files)
        if not success:
            logger.warning("Failed to add user turn to thread %s", continuation_id)
            logger.debug("[CONVERSATION_DEBUG] Failed to add user turn - thread may be at turn limit or expired")
        else:
            logger.debug("[CONVERSATION_DEBUG] Successfully added user turn to thread %s", continuation_id)

    # Create model context early to use for history building
    from utils.model_context import ModelContext

    tool = tool_registry.get_tool_instance(context.tool_name) if tool_registry.is_available(context.tool_name) else None
    requires_model = tool.requires_model() if tool else True

    # Check if we should use the model from the previous conversation turn
    model_from_args = arguments.get("model")
    if requires_model and not model_from_args and context.turns:
        for turn in reversed(context.turns):
            if turn.role == "assistant" and turn.model_name:
                arguments["model"] = turn.model_name
                logger.debug("[CONVERSATION_DEBUG] Using model from previous turn: %s", turn.model_name)
                break

    # Resolve an effective model for context reconstruction when DEFAULT_MODEL=auto
    from utils.tool_execution_context import ToolExecutionContext

    existing_ctx = ToolExecutionContext.from_arguments(arguments)
    model_context = existing_ctx.model_context if existing_ctx else None

    from utils.model_resolution import resolve_fallback_model

    if requires_model:
        if model_context is None:
            try:
                model_context = ModelContext.from_arguments(arguments)
            except ValueError as exc:
                fallback_model = resolve_fallback_model(tool, f"context reconstruction after error: {exc}")
                logger.debug(
                    "[CONVERSATION_DEBUG] Falling back to model '%s' for context reconstruction after error: %s",
                    fallback_model,
                    exc,
                )
                model_context = ModelContext(fallback_model)

        from providers.registry import get_default_registry

        provider = get_default_registry().get_provider_for_model(model_context.model_name)
        if provider is None:
            fallback_model = resolve_fallback_model(
                tool, f"model '{model_context.model_name}' is not available with current API keys"
            )
            logger.debug(
                "[CONVERSATION_DEBUG] Model '%s' unavailable; swapping to '%s' for context reconstruction",
                model_context.model_name,
                fallback_model,
            )
            model_context = ModelContext(fallback_model)
    else:
        if model_context is None:
            fallback_model = resolve_fallback_model(tool, "no available models detected for context reconstruction")
            logger.debug(
                "[CONVERSATION_DEBUG] Using fallback model '%s' for context reconstruction "
                "of tool without model requirement",
                fallback_model,
            )
            model_context = ModelContext(fallback_model)

    # Build conversation history with model-specific limits
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[CONVERSATION_DEBUG] Building conversation history for thread %s", continuation_id)
        logger.debug("[CONVERSATION_DEBUG] Thread has %d turns, tool: %s", len(context.turns), context.tool_name)
        logger.debug("[CONVERSATION_DEBUG] Using model: %s", model_context.model_name)

    def _tool_formatter_fn(tool_name, turn):
        """Look up tool-specific turn formatting from the registry."""
        if tool_registry.is_available(tool_name):
            t = tool_registry.get_tool_instance(tool_name)
            try:
                return t.format_conversation_turn(turn)
            except AttributeError:
                pass
        return None

    conversation_history, conversation_tokens = build_conversation_history(
        context, model_context, tool_formatter_fn=_tool_formatter_fn
    )
    logger.debug("[CONVERSATION_DEBUG] Conversation history built: %s tokens", f"{conversation_tokens:,}")
    logger.debug(
        "[CONVERSATION_DEBUG] Conversation history length: %d chars (~%s tokens)",
        len(conversation_history),
        f"{conversation_tokens:,}",
    )

    # Add dynamic follow-up instructions based on turn count
    follow_up_instructions = get_follow_up_instructions(len(context.turns))
    logger.debug("[CONVERSATION_DEBUG] Follow-up instructions added for turn %d", len(context.turns))

    # All tools now use standardized 'prompt' field
    original_prompt = arguments.get("prompt", "")
    logger.debug("[CONVERSATION_DEBUG] Extracting user input from 'prompt' field")
    original_prompt_tokens = model_context.estimate_tokens(original_prompt) if original_prompt else 0
    logger.debug(
        "[CONVERSATION_DEBUG] User input length: %d chars (~%s tokens)",
        len(original_prompt),
        f"{original_prompt_tokens:,}",
    )

    # Merge original context with new prompt and follow-up instructions
    if conversation_history:
        enhanced_prompt = (
            f"{conversation_history}\n\n=== NEW USER INPUT ===\n{original_prompt}\n\n{follow_up_instructions}"
        )
    else:
        enhanced_prompt = f"{original_prompt}\n\n{follow_up_instructions}"

    # Update arguments with enhanced context and remaining token budget
    enhanced_arguments = arguments.copy()
    enhanced_arguments["prompt"] = enhanced_prompt
    logger.debug("[CONVERSATION_DEBUG] Storing enhanced prompt in 'prompt' field")

    # Calculate remaining token budget based on current model
    token_allocation = model_context.calculate_token_allocation()
    remaining_tokens = token_allocation.content_tokens - conversation_tokens

    # Inject typed execution context with all server-resolved state
    from providers.registry import get_default_registry as _get_registry

    enhanced_arguments["_context"] = ToolExecutionContext(
        model_context=model_context,
        resolved_model_name=model_context.model_name,
        remaining_tokens=max(0, remaining_tokens),
        original_user_prompt=original_prompt,
        registry=_get_registry(),
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[CONVERSATION_DEBUG] Token budget calculation:")
        logger.debug("[CONVERSATION_DEBUG]   Model: %s", model_context.model_name)
        logger.debug("[CONVERSATION_DEBUG]   Total capacity: %s", f"{token_allocation.total_tokens:,}")
        logger.debug("[CONVERSATION_DEBUG]   Content allocation: %s", f"{token_allocation.content_tokens:,}")
        logger.debug("[CONVERSATION_DEBUG]   Conversation tokens: %s", f"{conversation_tokens:,}")
        logger.debug("[CONVERSATION_DEBUG]   Remaining tokens: %s", f"{remaining_tokens:,}")

    # Merge original context parameters (files, etc.) with new request
    if context.initial_context:
        logger.debug(
            "[CONVERSATION_DEBUG] Merging initial context with %d parameters",
            len(context.initial_context),
        )
        for key, value in context.initial_context.items():
            if key not in enhanced_arguments and key not in ["temperature", "thinking_mode", "model"]:
                enhanced_arguments[key] = value
                logger.debug("[CONVERSATION_DEBUG] Merged initial context param: %s", key)

    logger.info("Reconstructed context for thread %s (turn %d)", continuation_id, len(context.turns))
    logger.debug("[CONVERSATION_DEBUG] Final enhanced arguments keys: %s", list(enhanced_arguments.keys()))

    if "absolute_file_paths" in enhanced_arguments:
        logger.debug(
            "[CONVERSATION_DEBUG] Final files in enhanced arguments: %s",
            enhanced_arguments["absolute_file_paths"],
        )

    try:
        mcp_activity_logger = logging.getLogger("mcp_activity")
        mcp_activity_logger.info(
            "CONVERSATION_CONTINUATION: Thread %s turn %d - %d previous turns loaded",
            continuation_id,
            len(context.turns),
            len(context.turns),
        )
    except Exception:
        pass

    return enhanced_arguments
