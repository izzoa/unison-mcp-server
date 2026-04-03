"""
Tests for the ConversationHandler class extracted from BaseTool.
"""

from unittest.mock import MagicMock, patch

import pytest

from tools.shared.conversation_handler import ConversationHandler


class TestConversationHandler:
    """Test suite for the ConversationHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a ConversationHandler instance for testing."""
        return ConversationHandler(tool_name="test_tool")

    def test_independent_instantiation(self):
        """Test that ConversationHandler can be instantiated with just a tool_name."""
        handler = ConversationHandler(tool_name="my_tool")
        assert handler.tool_name == "my_tool"

    def test_independent_instantiation_different_names(self):
        """Test instantiation with various tool names."""
        for name in ["chat", "codereview", "thinkdeep", "debug"]:
            handler = ConversationHandler(tool_name=name)
            assert handler.tool_name == name

    def test_format_conversation_turn_user_no_files(self, handler):
        """Test formatting a user turn with no files attached."""
        turn = MagicMock()
        turn.role = "user"
        turn.content = "Please review this code."
        turn.files = None

        result = handler.format_conversation_turn(turn)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "Please review this code."

    def test_format_conversation_turn_user_empty_files(self, handler):
        """Test formatting a user turn with an empty files list."""
        turn = MagicMock()
        turn.role = "user"
        turn.content = "Analyze this."
        turn.files = []

        result = handler.format_conversation_turn(turn)

        # Empty list is falsy, so no files line should appear
        assert len(result) == 1
        assert result[0] == "Analyze this."

    def test_format_conversation_turn_assistant_with_files(self, handler):
        """Test formatting an assistant turn that references files."""
        turn = MagicMock()
        turn.role = "assistant"
        turn.content = "I found several issues in the code."
        turn.files = ["/path/to/file1.py", "/path/to/file2.py"]

        result = handler.format_conversation_turn(turn)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == "Files used in this turn: /path/to/file1.py, /path/to/file2.py"
        assert result[1] == ""  # Empty line for readability
        assert result[2] == "I found several issues in the code."

    def test_format_conversation_turn_single_file(self, handler):
        """Test formatting a turn with a single file."""
        turn = MagicMock()
        turn.role = "assistant"
        turn.content = "Here is my analysis."
        turn.files = ["/src/main.py"]

        result = handler.format_conversation_turn(turn)

        assert len(result) == 3
        assert result[0] == "Files used in this turn: /src/main.py"
        assert result[1] == ""
        assert result[2] == "Here is my analysis."

    @patch("tools.shared.conversation_handler.MCP_PROMPT_SIZE_LIMIT", 100)
    def test_check_prompt_size_within_limit(self, handler):
        """Test that content within the limit returns None."""
        short_text = "This is a short prompt."
        result = handler.check_prompt_size(short_text)
        assert result is None

    @patch("tools.shared.conversation_handler.MCP_PROMPT_SIZE_LIMIT", 100)
    def test_check_prompt_size_at_exact_limit(self, handler):
        """Test that content exactly at the limit returns None."""
        text = "x" * 100
        result = handler.check_prompt_size(text)
        assert result is None

    @patch("tools.shared.conversation_handler.MCP_PROMPT_SIZE_LIMIT", 100)
    def test_check_prompt_size_exceeds_limit(self, handler):
        """Test that content exceeding the limit returns a resend_prompt response."""
        long_text = "x" * 101
        result = handler.check_prompt_size(long_text)

        assert result is not None
        assert isinstance(result, dict)
        assert result["status"] == "resend_prompt"
        assert "content" in result
        assert "MANDATORY ACTION REQUIRED" in result["content"]
        assert "prompt.txt" in result["content"]
        assert result["content_type"] == "text"
        assert result["metadata"]["prompt_size"] == 101
        assert result["metadata"]["limit"] == 100

    @patch("tools.shared.conversation_handler.MCP_PROMPT_SIZE_LIMIT", 50)
    def test_check_prompt_size_well_over_limit(self, handler):
        """Test with content far exceeding the limit."""
        huge_text = "a" * 10000
        result = handler.check_prompt_size(huge_text)

        assert result is not None
        assert result["status"] == "resend_prompt"
        assert result["metadata"]["prompt_size"] == 10000
        assert result["metadata"]["limit"] == 50

    def test_check_prompt_size_empty_string(self, handler):
        """Test that an empty string returns None."""
        result = handler.check_prompt_size("")
        assert result is None

    def test_check_prompt_size_none(self, handler):
        """Test that None input returns None."""
        result = handler.check_prompt_size(None)
        assert result is None

    def test_get_prompt_content_for_size_validation_passthrough(self, handler):
        """Test that the default implementation returns user_content unchanged."""
        content = "This is some user content for validation."
        result = handler.get_prompt_content_for_size_validation(content)
        assert result == content

    def test_get_prompt_content_for_size_validation_empty(self, handler):
        """Test passthrough with empty string."""
        result = handler.get_prompt_content_for_size_validation("")
        assert result == ""

    def test_get_prompt_content_for_size_validation_large_content(self, handler):
        """Test passthrough with large content."""
        large_content = "x" * 100000
        result = handler.get_prompt_content_for_size_validation(large_content)
        assert result == large_content
        assert len(result) == 100000
