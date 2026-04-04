"""Tests for tools.registry.ToolRegistry."""

import os
from unittest.mock import MagicMock, patch

import pytest

from tools.registry import ESSENTIAL_TOOLS, TOOL_DEFINITIONS, ToolRegistry


class TestToolRegistryDefinitions:
    """Tests for registry tool definitions."""

    def test_all_18_tools_defined(self):
        """Registry contains definitions for all 18 tools."""
        assert len(TOOL_DEFINITIONS) == 18

    def test_definition_format(self):
        """Each definition is a (module_path, class_name, description) tuple."""
        for name, defn in TOOL_DEFINITIONS.items():
            assert len(defn) == 3, f"Tool '{name}' definition should be a 3-tuple"
            module_path, class_name, description = defn
            assert isinstance(module_path, str)
            assert isinstance(class_name, str)
            assert isinstance(description, str)
            assert module_path.startswith("tools.")

    def test_essential_tools_defined(self):
        """Essential tools (version, listmodels) are in the definitions."""
        for name in ESSENTIAL_TOOLS:
            assert name in TOOL_DEFINITIONS


class TestToolRegistryCreation:
    """Tests for registry instantiation."""

    @patch("tools.registry.ToolRegistry._import_tool")
    def test_creation_does_not_import_tools(self, mock_import):
        """Creating a ToolRegistry should not eagerly import any tool modules."""
        registry = ToolRegistry()
        mock_import.assert_not_called()
        assert len(registry._instances) == 0

    def test_get_tool_names(self):
        """get_tool_names returns all registered tool names."""
        registry = ToolRegistry()
        names = registry.get_tool_names()
        assert set(names) == set(TOOL_DEFINITIONS.keys())


class TestToolRegistryLazyInstantiation:
    """Tests for lazy tool import and caching."""

    @patch("tools.registry.ToolRegistry._import_tool")
    def test_get_tool_instance_imports_lazily(self, mock_import):
        """get_tool_instance imports the tool module on first call."""
        mock_tool = MagicMock()
        mock_import.return_value = mock_tool
        registry = ToolRegistry()

        result = registry.get_tool_instance("chat")
        assert result is mock_tool
        mock_import.assert_called_once_with("tools.chat", "ChatTool")

    @patch("tools.registry.ToolRegistry._import_tool")
    def test_get_tool_instance_caches(self, mock_import):
        """Second call to get_tool_instance returns cached instance."""
        mock_tool = MagicMock()
        mock_import.return_value = mock_tool
        registry = ToolRegistry()

        first = registry.get_tool_instance("chat")
        second = registry.get_tool_instance("chat")
        assert first is second
        assert mock_import.call_count == 1

    def test_get_tool_instance_unknown_raises(self):
        """get_tool_instance raises KeyError for unknown tool names."""
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="Unknown tool: 'nonexistent_tool'"):
            registry.get_tool_instance("nonexistent_tool")


class TestToolRegistryAvailability:
    """Tests for tool availability filtering via DISABLED_TOOLS."""

    @patch("tools.registry.ToolRegistry._import_tool")
    @patch.dict(os.environ, {"DISABLED_TOOLS": ""})
    def test_all_available_when_no_disabled(self, mock_import):
        """All tools available when DISABLED_TOOLS is not set."""
        mock_import.return_value = MagicMock()
        registry = ToolRegistry()
        available = registry.get_available_tools()
        assert set(available.keys()) == set(TOOL_DEFINITIONS.keys())

    @patch("tools.registry.ToolRegistry._import_tool")
    @patch.dict(os.environ, {"DISABLED_TOOLS": "debug,analyze"})
    def test_disabled_tools_excluded(self, mock_import):
        """Disabled tools are excluded from available tools."""
        mock_import.return_value = MagicMock()
        registry = ToolRegistry()
        available = registry.get_available_tools()
        assert "debug" not in available
        assert "analyze" not in available
        assert "chat" in available

    @patch("tools.registry.ToolRegistry._import_tool")
    @patch.dict(os.environ, {"DISABLED_TOOLS": "version,listmodels"})
    def test_essential_tools_cannot_be_disabled(self, mock_import):
        """Essential tools remain available even if listed in DISABLED_TOOLS."""
        mock_import.return_value = MagicMock()
        registry = ToolRegistry()
        available = registry.get_available_tools()
        assert "version" in available
        assert "listmodels" in available

    def test_is_available(self):
        """is_available returns correct status."""
        registry = ToolRegistry()
        assert registry.is_available("chat") is True
        assert registry.is_available("nonexistent") is False

    @patch.dict(os.environ, {"DISABLED_TOOLS": "debug"})
    def test_is_available_disabled(self):
        """is_available returns False for disabled tools."""
        registry = ToolRegistry()
        assert registry.is_available("debug") is False
        assert registry.is_available("version") is True  # essential


class TestToolRegistrySchema:
    """Tests for tool schema generation."""

    @patch("tools.registry.ToolRegistry._import_tool")
    def test_get_tool_schema_format(self, mock_import):
        """get_tool_schema returns name, description, and inputSchema."""
        mock_tool = MagicMock()
        mock_tool.name = "chat"
        mock_tool.description = "Chat tool"
        mock_tool.get_input_schema.return_value = {"type": "object", "properties": {}}
        mock_tool.get_annotations.return_value = None
        mock_import.return_value = mock_tool

        registry = ToolRegistry()
        schema = registry.get_tool_schema("chat")

        assert schema["name"] == "chat"
        assert schema["description"] == "Chat tool"
        assert "inputSchema" in schema
        assert "annotations" not in schema

    @patch("tools.registry.ToolRegistry._import_tool")
    def test_get_tool_schema_with_annotations(self, mock_import):
        """get_tool_schema includes annotations when present."""
        mock_tool = MagicMock()
        mock_tool.name = "chat"
        mock_tool.description = "Chat tool"
        mock_tool.get_input_schema.return_value = {"type": "object"}
        mock_tool.get_annotations.return_value = {"readOnlyHint": True}
        mock_import.return_value = mock_tool

        registry = ToolRegistry()
        schema = registry.get_tool_schema("chat")

        assert schema["annotations"] == {"readOnlyHint": True}
