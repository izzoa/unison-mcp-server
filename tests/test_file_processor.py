"""
Tests for the FileProcessor component extracted from BaseTool.

Tests cover:
- FileProcessor instantiation
- handle_prompt_file with and without prompt.txt
- filter_new_files deduplication logic
- get_conversation_embedded_files with and without continuation_id
- _validate_token_limit within and exceeding limits
- _validate_image_limits for various model capability scenarios
"""

from unittest.mock import MagicMock, patch

import pytest

from tools.shared.file_processor import FileProcessor


class TestFileProcessorInstantiation:
    """Test FileProcessor can be instantiated with minimal configuration."""

    def test_instantiation_with_tool_name_only(self):
        """FileProcessor should be instantiable with just a tool_name."""
        fp = FileProcessor(tool_name="chat")
        assert fp.tool_name == "chat"
        assert fp.include_line_numbers is True  # default

    def test_instantiation_with_line_numbers_false(self):
        """FileProcessor should accept include_line_numbers=False."""
        fp = FileProcessor(tool_name="codereview", include_line_numbers=False)
        assert fp.tool_name == "codereview"
        assert fp.include_line_numbers is False

    def test_instantiation_with_custom_tool_name(self):
        """FileProcessor should store any tool_name string."""
        fp = FileProcessor(tool_name="my-custom-tool")
        assert fp.tool_name == "my-custom-tool"


class TestHandlePromptFile:
    """Tests for handle_prompt_file method."""

    @pytest.fixture
    def fp(self):
        return FileProcessor(tool_name="test_tool")

    def test_none_files_returns_none(self, fp):
        """When files is None, should return (None, None)."""
        result = fp.handle_prompt_file(None)
        assert result == (None, None)

    def test_empty_files_returns_none(self, fp):
        """When files is empty list, should return (None, [])."""
        result = fp.handle_prompt_file([])
        assert result == (None, [])

    @patch("tools.shared.file_processor.read_file_content")
    def test_prompt_file_with_markers(self, mock_read, fp):
        """Should extract content between BEGIN/END FILE markers."""
        mock_read.return_value = (
            "--- BEGIN FILE: prompt.txt ---\nHello world\nSecond line\n--- END FILE: prompt.txt ---",
            None,
        )
        prompt, files = fp.handle_prompt_file(["/tmp/prompt.txt"])
        assert prompt == "Hello world\nSecond line"
        assert files is None  # No other files remain

    @patch("tools.shared.file_processor.read_file_content")
    def test_prompt_file_raw_content(self, mock_read, fp):
        """Should use raw content when no markers present."""
        mock_read.return_value = ("This is raw prompt content", None)
        prompt, files = fp.handle_prompt_file(["/tmp/prompt.txt"])
        assert prompt == "This is raw prompt content"
        assert files is None

    @patch("tools.shared.file_processor.read_file_content")
    def test_prompt_file_with_error_marker(self, mock_read, fp):
        """Should return None for prompt content when error markers present."""
        mock_read.return_value = ("\n--- ERROR: Could not read file ---", None)
        prompt, files = fp.handle_prompt_file(["/tmp/prompt.txt"])
        assert prompt is None
        assert files is None

    @patch("tools.shared.file_processor.read_file_content")
    def test_prompt_file_with_other_files(self, mock_read, fp):
        """Should separate prompt.txt from other files."""
        mock_read.return_value = ("Prompt content here", None)
        prompt, files = fp.handle_prompt_file(["/tmp/code.py", "/tmp/prompt.txt", "/tmp/other.js"])
        assert prompt == "Prompt content here"
        assert files == ["/tmp/code.py", "/tmp/other.js"]

    def test_no_prompt_file_in_list(self, fp):
        """When no prompt.txt is present, should return None and all files."""
        prompt, files = fp.handle_prompt_file(["/tmp/code.py", "/tmp/other.js"])
        assert prompt is None
        assert files == ["/tmp/code.py", "/tmp/other.js"]

    def test_similar_named_files_not_matched(self, fp):
        """Files like myprompt.txt or prompt.txt.bak should not be matched."""
        prompt, files = fp.handle_prompt_file(["/tmp/myprompt.txt", "/tmp/prompt.txt.bak"])
        assert prompt is None
        assert files == ["/tmp/myprompt.txt", "/tmp/prompt.txt.bak"]

    @patch("tools.shared.file_processor.read_file_content")
    def test_prompt_file_read_exception(self, mock_read, fp):
        """Should handle read exceptions gracefully."""
        mock_read.side_effect = OSError("File not found")
        prompt, files = fp.handle_prompt_file(["/tmp/prompt.txt", "/tmp/code.py"])
        # prompt.txt read failed, so prompt is None; code.py remains
        assert prompt is None
        assert files == ["/tmp/code.py"]


class TestFilterNewFiles:
    """Tests for filter_new_files deduplication."""

    @pytest.fixture
    def fp(self):
        return FileProcessor(tool_name="test_tool")

    def test_no_continuation_id_returns_all(self, fp):
        """Without continuation_id, all files should be returned."""
        files = ["/tmp/a.py", "/tmp/b.py"]
        result = fp.filter_new_files(files, continuation_id=None)
        assert result == files

    @patch("tools.shared.file_processor.get_conversation_file_list")
    @patch("tools.shared.file_processor.get_thread")
    def test_filters_already_embedded_files(self, mock_get_thread, mock_get_file_list, fp):
        """Should filter out files that are already in conversation history."""
        mock_get_thread.return_value = MagicMock()  # Thread exists
        mock_get_file_list.return_value = ["/tmp/a.py"]  # a.py already embedded

        result = fp.filter_new_files(["/tmp/a.py", "/tmp/b.py", "/tmp/c.py"], continuation_id="thread-123")
        assert result == ["/tmp/b.py", "/tmp/c.py"]

    @patch("tools.shared.file_processor.get_conversation_file_list")
    @patch("tools.shared.file_processor.get_thread")
    def test_all_files_already_embedded(self, mock_get_thread, mock_get_file_list, fp):
        """When all files are already embedded, should return empty list."""
        mock_get_thread.return_value = MagicMock()
        mock_get_file_list.return_value = ["/tmp/a.py", "/tmp/b.py"]

        result = fp.filter_new_files(["/tmp/a.py", "/tmp/b.py"], continuation_id="thread-123")
        assert result == []

    @patch("tools.shared.file_processor.get_conversation_file_list")
    @patch("tools.shared.file_processor.get_thread")
    def test_empty_embedded_files_returns_all(self, mock_get_thread, mock_get_file_list, fp):
        """When conversation has no embedded files, should return all requested files."""
        mock_get_thread.return_value = MagicMock()
        mock_get_file_list.return_value = []  # No files in conversation

        result = fp.filter_new_files(["/tmp/a.py", "/tmp/b.py"], continuation_id="thread-123")
        assert result == ["/tmp/a.py", "/tmp/b.py"]

    @patch("tools.shared.file_processor.get_thread")
    def test_exception_returns_all_files(self, mock_get_thread, fp):
        """On exception, should conservatively return all files."""
        mock_get_thread.side_effect = RuntimeError("Database error")

        result = fp.filter_new_files(["/tmp/a.py", "/tmp/b.py"], continuation_id="thread-123")
        assert result == ["/tmp/a.py", "/tmp/b.py"]


class TestGetConversationEmbeddedFiles:
    """Tests for get_conversation_embedded_files."""

    @pytest.fixture
    def fp(self):
        return FileProcessor(tool_name="test_tool")

    def test_no_continuation_id(self, fp):
        """Without continuation_id, should return empty list."""
        result = fp.get_conversation_embedded_files(None)
        assert result == []

    @patch("tools.shared.file_processor.get_thread")
    def test_thread_not_found(self, mock_get_thread, fp):
        """When thread is not found, should return empty list."""
        mock_get_thread.return_value = None
        result = fp.get_conversation_embedded_files("nonexistent-thread")
        assert result == []

    @patch("tools.shared.file_processor.get_conversation_file_list")
    @patch("tools.shared.file_processor.get_thread")
    def test_thread_with_files(self, mock_get_thread, mock_get_file_list, fp):
        """When thread exists with files, should return them."""
        mock_thread = MagicMock()
        mock_get_thread.return_value = mock_thread
        mock_get_file_list.return_value = ["/tmp/a.py", "/tmp/b.py"]

        result = fp.get_conversation_embedded_files("thread-456")
        assert result == ["/tmp/a.py", "/tmp/b.py"]
        mock_get_thread.assert_called_once_with("thread-456")
        mock_get_file_list.assert_called_once_with(mock_thread)

    @patch("tools.shared.file_processor.get_conversation_file_list")
    @patch("tools.shared.file_processor.get_thread")
    def test_thread_with_no_files(self, mock_get_thread, mock_get_file_list, fp):
        """When thread exists but has no files, should return empty list."""
        mock_get_thread.return_value = MagicMock()
        mock_get_file_list.return_value = []

        result = fp.get_conversation_embedded_files("thread-789")
        assert result == []


class TestValidateTokenLimit:
    """Tests for _validate_token_limit."""

    @pytest.fixture
    def fp(self):
        return FileProcessor(tool_name="test_tool")

    def test_empty_content_passes(self, fp):
        """Empty or None content should pass without error."""
        fp._validate_token_limit("", "Code")
        fp._validate_token_limit(None, "Code")

    def test_small_content_passes(self, fp):
        """Content within limits should not raise."""
        small_content = "Hello world"
        fp._validate_token_limit(small_content, "Code")

    @patch("tools.shared.file_processor.MCP_PROMPT_SIZE_LIMIT", 100)
    @patch("tools.shared.file_processor.estimate_tokens", return_value=50)
    def test_content_exceeding_limit_raises(self, mock_estimate, fp):
        """Content exceeding MCP_PROMPT_SIZE_LIMIT should raise ValueError."""
        large_content = "x" * 101  # Exceeds the mocked limit of 100
        with pytest.raises(ValueError, match="too large"):
            fp._validate_token_limit(large_content, "Content")

    @patch("tools.shared.file_processor.MCP_PROMPT_SIZE_LIMIT", 1000)
    @patch("tools.shared.file_processor.estimate_tokens", return_value=10)
    def test_content_within_limit_passes(self, mock_estimate, fp):
        """Content within the limit should pass without error."""
        content = "x" * 500  # Under the mocked limit of 1000
        fp._validate_token_limit(content, "Prompt")


class TestValidateImageLimits:
    """Tests for _validate_image_limits."""

    @pytest.fixture
    def fp(self):
        return FileProcessor(tool_name="test_tool")

    def test_no_images_returns_none(self, fp):
        """When no images provided, should return None (valid)."""
        assert fp._validate_image_limits(None) is None
        assert fp._validate_image_limits([]) is None

    def test_no_model_context_returns_none(self, fp):
        """Without model_context, should return None (skip validation)."""
        result = fp._validate_image_limits(["/tmp/img.png"], model_context=None)
        assert result is None

    def test_model_does_not_support_images(self, fp):
        """Should return error when model doesn't support images."""
        mock_context = MagicMock()
        mock_context.model_name = "text-only-model"
        mock_context.capabilities.supports_images = False

        result = fp._validate_image_limits(["/tmp/img.png"], model_context=mock_context)
        assert result is not None
        assert result["status"] == "error"
        assert "does not support image processing" in result["content"]
        assert result["metadata"]["supports_images"] is False

    def test_too_many_images(self, fp):
        """Should return error when too many images provided."""
        mock_context = MagicMock()
        mock_context.model_name = "gpt-4-vision"
        mock_context.capabilities.supports_images = True
        mock_context.capabilities.max_image_size_mb = 20.0

        images = [f"/tmp/img{i}.png" for i in range(6)]  # 6 images, max is 5
        result = fp._validate_image_limits(images, model_context=mock_context)
        assert result is not None
        assert result["status"] == "error"
        assert "Too many images" in result["content"]
        assert result["metadata"]["image_count"] == 6
        assert result["metadata"]["max_images"] == 5

    def test_images_within_limits(self, fp):
        """Should return None when images are within all limits."""
        mock_context = MagicMock()
        mock_context.model_name = "gpt-4-vision"
        mock_context.capabilities.supports_images = True
        mock_context.capabilities.max_image_size_mb = 20.0

        # Path is imported inside the method, so patch pathlib.Path
        with patch("pathlib.Path") as mock_path_cls:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.stat.return_value = MagicMock(st_size=1024 * 1024)  # 1MB
            mock_path_cls.return_value = mock_path_instance

            result = fp._validate_image_limits(["/tmp/img1.png", "/tmp/img2.png"], model_context=mock_context)
            assert result is None

    def test_image_size_exceeded(self, fp):
        """Should return error when total image size exceeds limit."""
        mock_context = MagicMock()
        mock_context.model_name = "gpt-4-vision"
        mock_context.capabilities.supports_images = True
        mock_context.capabilities.max_image_size_mb = 5.0

        # Path is imported inside the method, so patch pathlib.Path
        with patch("pathlib.Path") as mock_path_cls:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.stat.return_value = MagicMock(st_size=3 * 1024 * 1024)  # 3MB each
            mock_path_cls.return_value = mock_path_instance

            result = fp._validate_image_limits(
                ["/tmp/img1.png", "/tmp/img2.png"],  # 6MB total > 5MB limit
                model_context=mock_context,
            )
            assert result is not None
            assert result["status"] == "error"
            assert "size limit exceeded" in result["content"]

    def test_capabilities_access_failure(self, fp):
        """Should return error dict when capabilities cannot be accessed."""
        mock_context = MagicMock(spec=[])  # Empty spec so no attributes exist
        mock_context.model_name = "broken-model"
        # Make capabilities access raise an exception
        type(mock_context).capabilities = property(lambda self: (_ for _ in ()).throw(AttributeError("no caps")))

        result = fp._validate_image_limits(["/tmp/img.png"], model_context=mock_context)
        assert result is not None
        assert result["status"] == "error"
        assert result["metadata"]["supports_images"] is None
