"""
Test suite for context_reconstructor module.

Tests file/image collection, token-budget history building,
and tool-specific formatting via callback.
"""

from unittest.mock import Mock

from utils.context_reconstructor import (
    _default_turn_formatting,
    _get_tool_formatted_content,
    get_conversation_file_list,
    get_conversation_image_list,
)
from utils.conversation_store import ConversationTurn, ThreadContext


def _make_context(turns=None, thread_id="test-id", tool_name="chat"):
    return ThreadContext(
        thread_id=thread_id,
        created_at="2024-01-01T00:00:00+00:00",
        last_updated_at="2024-01-01T00:00:00+00:00",
        tool_name=tool_name,
        turns=turns or [],
        initial_context={},
    )


def _make_turn(role="user", content="test", files=None, images=None, tool_name=None):
    return ConversationTurn(
        role=role,
        content=content,
        timestamp="2024-01-01T00:00:00+00:00",
        files=files,
        images=images,
        tool_name=tool_name,
    )


class TestGetConversationFileList:
    """Test file collection with newest-first deduplication."""

    def test_empty_turns(self):
        context = _make_context(turns=[])
        assert get_conversation_file_list(context) == []

    def test_single_turn_files(self):
        turns = [_make_turn(files=["/a.py", "/b.py"])]
        context = _make_context(turns=turns)
        assert get_conversation_file_list(context) == ["/a.py", "/b.py"]

    def test_newest_first_deduplication(self):
        turns = [
            _make_turn(files=["/main.py", "/utils.py"]),  # Turn 1 (oldest)
            _make_turn(files=["/test.py"]),  # Turn 2
            _make_turn(files=["/main.py", "/config.py"]),  # Turn 3 (newest)
        ]
        context = _make_context(turns=turns)
        result = get_conversation_file_list(context)
        # main.py from Turn 3 takes precedence; utils.py from Turn 1 still included
        assert result == ["/main.py", "/config.py", "/test.py", "/utils.py"]

    def test_no_files_in_turns(self):
        turns = [_make_turn(files=None), _make_turn(files=None)]
        context = _make_context(turns=turns)
        assert get_conversation_file_list(context) == []

    def test_mixed_files_and_none(self):
        turns = [
            _make_turn(files=["/a.py"]),
            _make_turn(files=None),
            _make_turn(files=["/b.py"]),
        ]
        context = _make_context(turns=turns)
        result = get_conversation_file_list(context)
        assert result == ["/b.py", "/a.py"]


class TestGetConversationImageList:
    """Test image collection with newest-first deduplication."""

    def test_empty_turns(self):
        context = _make_context(turns=[])
        assert get_conversation_image_list(context) == []

    def test_newest_first_deduplication(self):
        turns = [
            _make_turn(images=["/diagram.png", "/flow.jpg"]),
            _make_turn(images=["/error.png"]),
            _make_turn(images=["/diagram.png", "/updated.png"]),
        ]
        context = _make_context(turns=turns)
        result = get_conversation_image_list(context)
        assert result == ["/diagram.png", "/updated.png", "/error.png", "/flow.jpg"]


class TestGetToolFormattedContent:
    """Test tool-specific formatting via callback."""

    def test_no_formatter_uses_default(self):
        turn = _make_turn(content="Hello", tool_name="chat")
        result = _get_tool_formatted_content(turn, tool_formatter_fn=None)
        assert result == ["Hello"]

    def test_formatter_called_with_tool_name(self):
        turn = _make_turn(content="Hello", tool_name="analyze")
        formatter = Mock(return_value=["Custom format"])
        result = _get_tool_formatted_content(turn, tool_formatter_fn=formatter)
        formatter.assert_called_once_with("analyze", turn)
        assert result == ["Custom format"]

    def test_formatter_returns_none_falls_back(self):
        turn = _make_turn(content="Hello", tool_name="analyze")
        formatter = Mock(return_value=None)
        result = _get_tool_formatted_content(turn, tool_formatter_fn=formatter)
        assert result == ["Hello"]

    def test_formatter_exception_falls_back(self):
        turn = _make_turn(content="Hello", tool_name="analyze")
        formatter = Mock(side_effect=RuntimeError("oops"))
        result = _get_tool_formatted_content(turn, tool_formatter_fn=formatter)
        assert result == ["Hello"]

    def test_no_tool_name_uses_default(self):
        turn = _make_turn(content="Hello", tool_name=None)
        formatter = Mock()
        result = _get_tool_formatted_content(turn, tool_formatter_fn=formatter)
        formatter.assert_not_called()
        assert result == ["Hello"]


class TestDefaultTurnFormatting:
    """Test default turn formatting."""

    def test_content_only(self):
        turn = _make_turn(content="Hello world")
        assert _default_turn_formatting(turn) == ["Hello world"]

    def test_with_files(self):
        turn = _make_turn(content="Analysis", files=["/a.py", "/b.py"])
        result = _default_turn_formatting(turn)
        assert result[0] == "Files used in this turn: /a.py, /b.py"
        assert result[1] == ""
        assert result[2] == "Analysis"
