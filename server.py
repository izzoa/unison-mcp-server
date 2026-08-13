"""
Unison MCP Server - Main server wiring module

This module creates the MCP server, assembles its dependencies, registers
handler modules, and starts the event loop. All handler implementations
live in the ``handlers/`` package; tool definitions and instantiation are
managed by ``tools.registry.ToolRegistry``.
"""

import asyncio
import logging
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import PromptsCapability, ServerCapabilities, ToolsCapability

from config import DEFAULT_MODEL, __version__
from handlers import prompt_handlers, tool_handlers
from providers.configure import configure_providers
from tools.registry import ToolRegistry
from utils.env import env_override_enabled, get_env
from utils.logging_setup import configure_logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_level = (get_env("LOG_LEVEL", "INFO") or "INFO").upper()
_, _mcp_activity_logger = configure_logging(log_level)

logger = logging.getLogger(__name__)

if env_override_enabled():
    logger.info("UNISON_MCP_FORCE_ENV_OVERRIDE enabled - .env file values will override system environment variables")
    logger.debug("Environment override prevents conflicts between different AI tools passing cached API keys")
else:
    logger.debug("UNISON_MCP_FORCE_ENV_OVERRIDE disabled - system environment variables take precedence")

# ---------------------------------------------------------------------------
# Server & registry setup
# ---------------------------------------------------------------------------
tool_registry = ToolRegistry()
# Load dynamic tool sources (entry-point plugins, opt-in local scan, pending
# @register_tool decorations) before handlers are wired, then the registry is
# frozen. With no plugins installed and UNISON_TOOL_AUTODISCOVERY unset this
# is a no-op beyond a DEBUG log line.
tool_registry.load_dynamic_sources()

# Initialize opt-in observability (no-op unless UNISON_OTEL_ENABLED=true; the
# OTel packages are an optional extra and their absence downgrades gracefully).
from utils.observability import init_observability  # noqa: E402

init_observability()

# Build handlers first, then construct the server from them: mcp 2.x replaces
# decorator registration with constructor callbacks, so the handler modules
# supply the callables the Server is created with (server.py stays wiring-only).
_tool_handlers = tool_handlers.build_handlers(tool_registry)
_on_list_prompts, _on_get_prompt = prompt_handlers.build_handlers(tool_registry)

server: Server = Server(
    "unison-server",
    on_list_tools=_tool_handlers.on_list_tools,
    on_call_tool=_tool_handlers.on_call_tool,
    on_list_prompts=_on_list_prompts,
    on_get_prompt=_on_get_prompt,
)

# Legacy-shaped handler references (1.x signatures) plus the call adapter,
# re-exported for tests and callers.
handle_list_tools = _tool_handlers.handle_list_tools
handle_call_tool = _tool_handlers.handle_call_tool
on_call_tool = _tool_handlers.on_call_tool


# ---------------------------------------------------------------------------
# Backwards-compatible module-level references
# ---------------------------------------------------------------------------
TOOLS = tool_registry.get_available_tools()

# Backward-compatible wrapper: binds the module-level tool_registry so
# callers can use the old single-argument signature.
from handlers.tool_handlers import reconstruct_thread_context as _reconstruct  # noqa: E402


async def reconstruct_thread_context(arguments):
    """Backward-compatible wrapper that passes the module-level tool_registry."""
    return await _reconstruct(arguments, tool_registry)


# Re-export disabled-tools helpers for test_disabled_tools.py
from tools.registry import apply_tool_filter, parse_disabled_tools_env, validate_disabled_tools  # noqa: E402, F401


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    """
    Main entry point for the MCP server.

    Creates the provider registry, configures providers, and starts the
    server using stdio transport.
    """
    from providers.registry import ModelProviderRegistry, set_default_registry

    registry = ModelProviderRegistry(config={})
    set_default_registry(registry)

    configure_providers(registry)

    logger.info("Unison MCP Server starting up...")
    logger.info("Log level: %s", log_level)

    from config import IS_AUTO_MODE

    if IS_AUTO_MODE:
        logger.info("Model mode: AUTO (CLI will select the best model for each task)")
    else:
        logger.info("Model mode: Fixed model '%s'", DEFAULT_MODEL)

    from config import DEFAULT_THINKING_MODE_THINKDEEP

    logger.info("Default thinking mode (ThinkDeep): %s", DEFAULT_THINKING_MODE_THINKDEEP)

    logger.info("Available tools: %s", list(tool_registry.get_available_tools().keys()))
    logger.info("Server ready - waiting for tool requests...")

    # Prepare dynamic instructions for the MCP client based on model mode
    if IS_AUTO_MODE:
        handshake_instructions = (
            "When the user names a specific model (e.g. 'use chat with gpt5'), send that exact model in the tool call. "
            "When no model is mentioned, first use the `listmodels` tool from Unison to obtain available models to choose the best one from."
        )
    else:
        handshake_instructions = (
            "When the user names a specific model (e.g. 'use chat with gpt5'), send that exact model in the tool call. "
            f"When no model is mentioned, default to '{DEFAULT_MODEL}'."
        )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="Unison",
                server_version=__version__,
                instructions=handshake_instructions,
                capabilities=ServerCapabilities(
                    tools=ToolsCapability(),
                    prompts=PromptsCapability(),
                ),
            ),
        )


def _resolve_pathological_cwd(cwd: Path | None = None) -> Path | None:
    """Return the directory the server should switch into, or None to stay put.

    GUI-launched MCP hosts spawn servers from non-workspace directories —
    Claude Desktop uses C:\\Windows\\System32 on Windows and / on macOS. Every
    downstream consumer of the process cwd (clink subagent spawns, clink's
    read-only snapshot verifier, version's installation path) then points at a
    system tree: the repo becomes invisible and the verifier attributes OS
    writes (e.g. Windows event logs) to the model. Fall back to the server's
    own directory for those cwds; a deliberate workspace cwd (Claude Code
    launches the server from the project root) is preserved.
    """
    current = cwd if cwd is not None else Path.cwd()
    server_root = Path(__file__).resolve().parent
    if current == server_root:
        return None
    if current == Path(current.anchor):  # filesystem root: "/" or a bare drive
        return server_root
    system_root = os.environ.get("SystemRoot") or os.environ.get("windir")
    if system_root:
        base = Path(system_root)
        if current in (base, base / "System32", base / "SysWOW64"):
            return server_root
    return None


def run():
    """Console script entry point for unison-mcp-server."""
    target = _resolve_pathological_cwd()
    if target is not None:
        logger.info(f"Launched from non-workspace directory {Path.cwd()}; working directory set to {target}")
        os.chdir(target)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
