"""
Tests for the ResponseFormatter class.

Validates the base response formatting and parsing behavior extracted
from BaseTool into tools/shared/response_formatter.py.
"""

import pytest

from tools.shared.response_formatter import ResponseFormatter


class TestResponseFormatterInstantiation:
    """Test that ResponseFormatter can be independently instantiated."""

    def test_instantiation_with_tool_name(self):
        """ResponseFormatter should be created with just a tool_name."""
        formatter = ResponseFormatter(tool_name="chat")
        assert formatter.tool_name == "chat"

    def test_instantiation_with_different_tool_names(self):
        """ResponseFormatter should store any tool name provided."""
        for name in ("debug", "codereview", "thinkdeep", "analyze"):
            formatter = ResponseFormatter(tool_name=name)
            assert formatter.tool_name == name


class TestFormatResponse:
    """Test the format_response passthrough method."""

    def test_returns_response_unchanged(self):
        """format_response should return the response string as-is."""
        formatter = ResponseFormatter(tool_name="chat")
        response = "This is a test response from the model."
        result = formatter.format_response(response, request=None)
        assert result == response

    def test_returns_empty_string(self):
        """format_response should handle empty strings."""
        formatter = ResponseFormatter(tool_name="chat")
        result = formatter.format_response("", request=None)
        assert result == ""

    def test_returns_multiline_response(self):
        """format_response should preserve multiline content."""
        formatter = ResponseFormatter(tool_name="debug")
        response = "Line 1\nLine 2\nLine 3\n"
        result = formatter.format_response(response, request=None)
        assert result == response

    def test_with_model_info_none(self):
        """format_response should work when model_info is None (default)."""
        formatter = ResponseFormatter(tool_name="chat")
        response = "Some response"
        result = formatter.format_response(response, request=None, model_info=None)
        assert result == response

    def test_with_model_info_dict(self):
        """format_response should accept model_info without affecting output."""
        formatter = ResponseFormatter(tool_name="chat")
        response = "Some response"
        model_info = {
            "model": "gemini-2.5-flash",
            "provider": "google",
            "tokens_used": 150,
        }
        result = formatter.format_response(response, request=None, model_info=model_info)
        assert result == response

    def test_with_request_object(self):
        """format_response should accept any request object without affecting output."""
        formatter = ResponseFormatter(tool_name="codereview")
        response = "Code review results here."

        class MockRequest:
            prompt = "Review this code"
            model = "gemini-2.5-flash"

        result = formatter.format_response(response, request=MockRequest())
        assert result == response

    def test_preserves_unicode(self):
        """format_response should preserve unicode characters."""
        formatter = ResponseFormatter(tool_name="chat")
        response = "Unicode: \u00e9\u00e8\u00ea \u00fc\u00f6\u00e4 \u4f60\u597d \ud83d\ude00"
        result = formatter.format_response(response, request=None)
        assert result == response


class TestParseResponse:
    """Test the _parse_response method raises NotImplementedError."""

    def test_raises_not_implemented_error(self):
        """_parse_response should raise NotImplementedError."""
        formatter = ResponseFormatter(tool_name="chat")
        with pytest.raises(NotImplementedError, match="Subclasses must implement _parse_response method"):
            formatter._parse_response("raw text", request=None)

    def test_raises_not_implemented_with_model_info(self):
        """_parse_response should raise NotImplementedError even with model_info."""
        formatter = ResponseFormatter(tool_name="debug")
        with pytest.raises(NotImplementedError):
            formatter._parse_response("raw text", request=None, model_info={"model": "gpt-4o"})

    def test_subclass_can_override(self):
        """A subclass should be able to override _parse_response."""

        class CustomFormatter(ResponseFormatter):
            def _parse_response(self, raw_text, request, model_info=None):
                return f"[parsed] {raw_text}"

        formatter = CustomFormatter(tool_name="custom")
        result = formatter._parse_response("hello", request=None)
        assert result == "[parsed] hello"
