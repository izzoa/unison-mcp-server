"""Tests for handlers.prompt_handlers."""

from unittest.mock import MagicMock

import pytest

from handlers import prompt_handlers
from tools.registry import ToolRegistry


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    tool = MagicMock()
    tool.name = "chat"
    tool.description = "Chat tool"
    return tool


@pytest.fixture
def mock_registry(mock_tool):
    """Create a mock ToolRegistry."""
    registry = MagicMock(spec=ToolRegistry)
    registry.get_available_tools.return_value = {"chat": mock_tool}
    registry.is_available.return_value = True
    return registry


@pytest.fixture
def mock_server():
    """Create a mock MCP server that captures registered handlers."""
    server = MagicMock()
    handlers = {}

    def list_prompts_decorator():
        def decorator(fn):
            handlers["list_prompts"] = fn
            return fn

        return decorator

    def get_prompt_decorator():
        def decorator(fn):
            handlers["get_prompt"] = fn
            return fn

        return decorator

    server.list_prompts = list_prompts_decorator
    server.get_prompt = get_prompt_decorator
    server._handlers = handlers
    return server


class TestListPromptsHandler:
    """Tests for the list_prompts handler."""

    @pytest.mark.asyncio
    async def test_returns_prompts_for_tools(self, mock_server, mock_registry):
        """list_prompts returns at least one prompt per tool plus 'continue'."""
        prompt_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["list_prompts"]

        prompts = await handler()
        assert len(prompts) >= 2
        names = [p.name for p in prompts]
        assert "continue" in names

    @pytest.mark.asyncio
    async def test_empty_registry_still_has_continue(self, mock_server, mock_registry):
        """list_prompts returns 'continue' even with no tools."""
        mock_registry.get_available_tools.return_value = {}
        prompt_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["list_prompts"]

        prompts = await handler()
        assert len(prompts) == 1
        assert prompts[0].name == "continue"


class TestGetPromptHandler:
    """Tests for the get_prompt handler."""

    @pytest.mark.asyncio
    async def test_continue_prompt(self, mock_server, mock_registry):
        """get_prompt handles 'continue' prompt."""
        prompt_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["get_prompt"]

        result = await handler("continue", None)
        assert result.prompt.name == "continue"
        assert len(result.messages) == 1
        content = result.messages[0].content
        text = content["text"] if isinstance(content, dict) else content.text
        assert "continuation_id" in text

    @pytest.mark.asyncio
    async def test_direct_tool_name(self, mock_server, mock_registry):
        """get_prompt handles direct tool name lookup."""
        prompt_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["get_prompt"]

        result = await handler("chat", None)
        assert result.prompt.name == "chat"

    @pytest.mark.asyncio
    async def test_unknown_prompt_raises(self, mock_server, mock_registry):
        """get_prompt raises ValueError for unknown prompt names."""
        mock_registry.get_available_tools.return_value = {}
        prompt_handlers.register(mock_server, mock_registry)
        handler = mock_server._handlers["get_prompt"]

        with pytest.raises(ValueError, match="Unknown prompt"):
            await handler("nonexistent", None)
