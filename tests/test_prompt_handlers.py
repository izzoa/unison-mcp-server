"""Tests for handlers.prompt_handlers."""

from types import SimpleNamespace
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


def _build(mock_registry):
    """Build the mcp 2.x constructor adapters for the mock registry."""
    return prompt_handlers.build_handlers(mock_registry)


class TestListPromptsHandler:
    """Tests for the list_prompts handler."""

    @pytest.mark.asyncio
    async def test_returns_prompts_for_tools(self, mock_registry):
        """list_prompts returns at least one prompt per tool plus 'continue'."""
        on_list_prompts, _ = _build(mock_registry)

        result = await on_list_prompts(None, None)
        prompts = result.prompts
        assert len(prompts) >= 2
        names = [p.name for p in prompts]
        assert "continue" in names

    @pytest.mark.asyncio
    async def test_empty_registry_still_has_continue(self, mock_registry):
        """list_prompts returns 'continue' even with no tools."""
        mock_registry.get_available_tools.return_value = {}
        on_list_prompts, _ = _build(mock_registry)

        result = await on_list_prompts(None, None)
        assert len(result.prompts) == 1
        assert result.prompts[0].name == "continue"


class TestGetPromptHandler:
    """Tests for the get_prompt handler."""

    @pytest.mark.asyncio
    async def test_continue_prompt(self, mock_registry):
        """get_prompt handles 'continue' prompt."""
        _, on_get_prompt = _build(mock_registry)

        result = await on_get_prompt(None, SimpleNamespace(name="continue", arguments=None))
        # mcp 2.x GetPromptResult carries the spec's top-level description
        # (the 1.x models leaked a nonstandard "prompt" object instead).
        assert result.description == "Continue the previous conversation"
        assert len(result.messages) == 1
        content = result.messages[0].content
        text = content["text"] if isinstance(content, dict) else content.text
        assert "continuation_id" in text

    @pytest.mark.asyncio
    async def test_direct_tool_name(self, mock_registry, mock_tool):
        """get_prompt handles direct tool name lookup (no marketing template)."""
        # "chat" has a PROMPT_TEMPLATES entry that takes precedence; use a
        # tool name without one to exercise the direct-name fallback path.
        mock_registry.get_available_tools.return_value = {"mytool": mock_tool}
        _, on_get_prompt = _build(mock_registry)

        result = await on_get_prompt(None, SimpleNamespace(name="mytool", arguments=None))
        assert result.description == "Use mytool tool"

    @pytest.mark.asyncio
    async def test_unknown_prompt_raises(self, mock_registry):
        """get_prompt raises ValueError for unknown prompt names."""
        mock_registry.get_available_tools.return_value = {}
        _, on_get_prompt = _build(mock_registry)

        with pytest.raises(ValueError, match="Unknown prompt"):
            await on_get_prompt(None, SimpleNamespace(name="nonexistent", arguments=None))
