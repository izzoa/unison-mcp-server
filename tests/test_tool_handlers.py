"""Tests for handlers.tool_handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from handlers import tool_handlers
from tools.registry import ToolRegistry


@pytest.fixture
def mock_tool():
    """Create a mock tool with standard interface."""
    tool = MagicMock()
    tool.name = "chat"
    tool.description = "Chat tool"
    tool.get_input_schema.return_value = {"type": "object", "properties": {}}
    tool.get_annotations.return_value = None
    tool.requires_model.return_value = True
    tool.get_model_category.return_value = MagicMock(value="general")
    tool.execute = AsyncMock(return_value=[MagicMock(type="text", text="response")])
    return tool


@pytest.fixture
def mock_registry(mock_tool):
    """Create a mock ToolRegistry."""
    registry = MagicMock(spec=ToolRegistry)
    registry.get_available_tools.return_value = {"chat": mock_tool}
    registry.get_tool_instance.return_value = mock_tool
    registry.is_available.return_value = True
    return registry


@pytest.fixture
def mock_server():
    """Create a mock MCP server that captures registered handlers."""
    server = MagicMock()
    handlers = {}

    def list_tools_decorator():
        def decorator(fn):
            handlers["list_tools"] = fn
            return fn

        return decorator

    def call_tool_decorator():
        def decorator(fn):
            handlers["call_tool"] = fn
            return fn

        return decorator

    server.list_tools = list_tools_decorator
    server.call_tool = call_tool_decorator
    server._handlers = handlers
    return server


class TestListToolsHandler:
    """Tests for the list_tools handler."""

    @pytest.mark.asyncio
    async def test_returns_correct_tool_count(self, mock_server, mock_registry, mock_tool):
        """list_tools returns one Tool per available tool."""
        tool_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["list_tools"]

        tools = await handler()
        assert len(tools) == 1
        assert tools[0].name == "chat"

    @pytest.mark.asyncio
    async def test_schema_format(self, mock_server, mock_registry, mock_tool):
        """list_tools returns tools with correct schema format."""
        tool_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["list_tools"]

        tools = await handler()
        assert tools[0].name == "chat"
        assert tools[0].description == "Chat tool"

    @pytest.mark.asyncio
    async def test_filters_by_availability(self, mock_server, mock_registry):
        """list_tools only returns tools from get_available_tools."""
        mock_registry.get_available_tools.return_value = {}
        tool_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["list_tools"]

        tools = await handler()
        assert len(tools) == 0


class TestCallToolHandler:
    """Tests for the call_tool handler."""

    @pytest.mark.asyncio
    @patch("utils.tool_execution_context.ToolExecutionContext")
    @patch("utils.model_context.ModelContext")
    @patch("providers.registry.get_default_registry")
    async def test_successful_dispatch(
        self,
        mock_get_registry,
        mock_model_ctx_cls,
        mock_exec_ctx_cls,
        mock_server,
        mock_registry,
        mock_tool,
    ):
        """call_tool dispatches to the correct tool and returns result."""
        mock_provider = MagicMock()
        mock_get_registry.return_value.get_provider_for_model.return_value = mock_provider
        mock_model_ctx = MagicMock()
        mock_model_ctx.capabilities.context_window = 100000
        mock_model_ctx_cls.return_value = mock_model_ctx

        tool_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["call_tool"]

        result = await handler("chat", {"prompt": "hello", "model": "gemini-2.0-flash"})

        mock_tool.execute.assert_awaited_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, mock_server, mock_registry):
        """call_tool returns error for unknown tool names."""
        mock_registry.is_available.return_value = False
        tool_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["call_tool"]

        result = await handler("nonexistent", {"prompt": "test"})

        assert len(result) == 1
        assert "Unknown tool" in result[0].text
