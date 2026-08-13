import json
from types import SimpleNamespace

import pytest
from mcp.types import CallToolRequestParams

from providers.registry import ModelProviderRegistry
from server import on_call_tool


def _install_dummy_provider(monkeypatch):
    """Ensure preflight model checks succeed without real provider configuration."""

    class DummyProvider:
        def get_provider_type(self):
            return SimpleNamespace(value="dummy")

        def get_capabilities(self, model_name):
            return SimpleNamespace(
                supports_extended_thinking=False,
                allow_code_generation=False,
                supports_images=False,
                context_window=1_000_000,
                max_image_size_mb=10,
            )

    monkeypatch.setattr(
        ModelProviderRegistry,
        "get_provider_for_model",
        lambda self, model_name: DummyProvider(),
    )
    monkeypatch.setattr(
        ModelProviderRegistry,
        "get_available_models",
        lambda self, respect_restrictions=False: {"gemini-2.5-flash": None},
    )


@pytest.mark.asyncio
async def test_tool_execution_error_sets_is_error_flag_for_mcp_response(monkeypatch):
    """Ensure ToolExecutionError surfaces as CallToolResult with isError=True."""

    _install_dummy_provider(monkeypatch)

    arguments = {
        "prompt": "Trigger working_directory_absolute_path validation failure",
        "working_directory_absolute_path": "relative/path",  # Not absolute -> ToolExecutionError from ChatTool
        "absolute_file_paths": [],
        "model": "gemini-2.5-flash",
    }

    # The 2.x call adapter replaces the 1.x request_handlers registry: invoke
    # it directly with typed params (ctx=None is valid outside a live session).
    result = await on_call_tool(None, CallToolRequestParams(name="chat", arguments=arguments))

    assert result.is_error is True
    assert result.content, "Expected error response content"

    payload = result.content[0].text
    data = json.loads(payload)
    assert data["status"] == "error"
    assert "absolute" in data["content"].lower()


@pytest.mark.asyncio
async def test_unknown_tool_returns_text_response_not_error():
    """Unknown tools keep the 1.x contract: plain text response, isError False."""
    result = await on_call_tool(None, CallToolRequestParams(name="definitely_not_a_tool", arguments={}))
    assert result.is_error is False
    assert result.content[0].text == "Unknown tool: definitely_not_a_tool"


@pytest.mark.asyncio
async def test_input_validation_error_matches_1x_decorator_shape():
    """The adapter replicates the 1.x SDK's jsonschema input validation."""
    result = await on_call_tool(None, CallToolRequestParams(name="chat", arguments={}))
    assert result.is_error is True
    assert result.content[0].text.startswith("Input validation error: ")
    assert "'prompt' is a required property" in result.content[0].text


@pytest.mark.asyncio
async def test_prompts_round_trip_through_adapters():
    """list_prompts and get_prompt work through the 2.x constructor adapters."""
    from handlers import prompt_handlers
    from server import tool_registry

    on_list_prompts, on_get_prompt = prompt_handlers.build_handlers(tool_registry)

    listing = await on_list_prompts(None, None)
    names = [p.name for p in listing.prompts]
    assert "continue" in names

    result = await on_get_prompt(None, SimpleNamespace(name="continue", arguments={}))
    assert result.messages and result.messages[0].role == "user"
    assert "continuation_id" in result.messages[0].content.text


def test_client_info_read_from_bound_request_context():
    """Client info resolves via the ContextVar the adapters bind (2.x path)."""
    import utils.client_info as ci
    from utils.mcp_context import reset_current_request_context, set_current_request_context

    ctx = SimpleNamespace(
        session=SimpleNamespace(
            client_params=SimpleNamespace(client_info=SimpleNamespace(name="test-client", version="9.9"))
        )
    )
    ci._client_info_cache = None
    token = set_current_request_context(ctx)
    try:
        info = ci.get_client_info_from_context()
    finally:
        reset_current_request_context(token)
        ci._client_info_cache = None
    assert info is not None
    assert info["name"] == "test-client"
    assert info["version"] == "9.9"


@pytest.mark.asyncio
async def test_modern_protocol_initialize_smoke():
    """The server completes an initialize handshake on the current protocol revision."""
    import os
    import subprocess
    import sys

    from mcp.types import LATEST_PROTOCOL_VERSION

    env = os.environ.copy()
    env.update(
        {
            "GEMINI_API_KEY": "dummy",
            "OPENAI_API_KEY": "",
            "XAI_API_KEY": "",
            "OPENROUTER_API_KEY": "",
            "DIAL_API_KEY": "",
            "CUSTOM_API_URL": "",
            "LOG_LEVEL": "INFO",
        }
    )
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "modern-smoke", "version": "1.0"},
        },
    }
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    try:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
    finally:
        proc.kill()
    resp = json.loads(line)
    assert resp["id"] == 1 and "result" in resp
    # Offered the latest revision, the server counters with its highest
    # supported one (per-spec negotiation) — it must be a modern revision,
    # newer than the legacy 2024-11-05 the simulator/wheel tests pin.
    negotiated = resp["result"]["protocolVersion"]
    assert negotiated and negotiated > "2024-11-05"
