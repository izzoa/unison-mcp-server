"""Model identifiers reach the provider unmodified (openspec: fix-openrouter-model-variants)."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from handlers import tool_handlers
from tools.registry import ToolRegistry
from utils import model_resolution
from utils.model_context import ModelContext
from utils.model_resolution import resolution_metadata, trim_model_identifier

# Provider-published variant segments (OpenRouter's catalog plus an Ollama tag).
PROVIDER_VARIANTS = [
    "openrouter/google/gemini-3.8-flash:batch",
    "openai/gpt-3.5-turbo:free",
    "anthropic/claude-opus-4.1:extended",
    "openai/gpt-4o:exacto",
    "anthropic/claude-sonnet-4:thinking",
    "openai/gpt-4o:online",
    "meta-llama/llama-3.3-70b-instruct:nitro",
    "openai/gpt-4o:floor",
    "llama3.2:3b",
    "llama3.2:latest",
]


class TestTrimModelIdentifier:
    @pytest.mark.parametrize("identifier", PROVIDER_VARIANTS)
    def test_variant_suffix_survives_intact(self, identifier):
        assert trim_model_identifier(identifier) == identifier

    @pytest.mark.parametrize(
        "identifier",
        [
            # Segments the old allowlist (free/beta/preview) never recognized, and ones no
            # provider has published yet: none may be stripped.
            "openai/gpt-4:custom-tag",
            "vendor/model:some-variant-shipped-next-year",
            "vendor/model:v2:experimental",
            "o3:for",
            "gemini-2.5-pro:against",
        ],
    )
    def test_unrecognized_suffix_is_not_stripped(self, identifier):
        assert trim_model_identifier(identifier) == identifier

    @pytest.mark.parametrize("identifier", ["gpt-4", "http://localhost:8080"])
    def test_plain_identifiers_unchanged(self, identifier):
        assert trim_model_identifier(identifier) == identifier

    def test_surrounding_whitespace_is_trimmed_and_the_rest_preserved(self):
        assert trim_model_identifier("  openai/gpt-3.5-turbo:free  ") == "openai/gpt-3.5-turbo:free"
        assert trim_model_identifier("\tllama3.2:3b\n") == "llama3.2:3b"

    def test_case_is_preserved(self):
        assert trim_model_identifier("openai/gpt-3.5-turbo:FREE") == "openai/gpt-3.5-turbo:FREE"

    def test_no_suffix_allowlist_remains(self):
        """Acceptance of an identifier must not depend on a hardcoded suffix list."""
        source = inspect.getsource(model_resolution)
        assert "parse_model_option" not in source
        for stale in ('"free"', '"beta"', '"preview"'):
            assert stale not in source


class TestResolutionMetadata:
    def test_difference_names_both_identifiers(self):
        assert resolution_metadata("flash", "gemini-2.5-flash") == {
            "model_requested": "flash",
            "model_resolved": "gemini-2.5-flash",
        }

    def test_identical_identifiers_report_nothing(self):
        assert resolution_metadata("gemini-2.5-flash", "gemini-2.5-flash") == {}

    def test_case_only_difference_is_not_a_different_model(self):
        assert resolution_metadata("Gemini-2.5-Flash", "gemini-2.5-flash") == {}

    def test_dropped_variant_is_reported(self):
        meta = resolution_metadata("openai/gpt-4o:batch", "openai/gpt-4o")
        assert meta == {"model_requested": "openai/gpt-4o:batch", "model_resolved": "openai/gpt-4o"}

    @pytest.mark.parametrize("requested,resolved", [(None, "x"), ("x", None), ("", "x"), ("x", "")])
    def test_missing_side_reports_nothing(self, requested, resolved):
        assert resolution_metadata(requested, resolved) == {}


class TestModelContextHasNoOption:
    def test_model_option_is_gone(self):
        ctx = ModelContext("openai/gpt-4o:batch")
        assert ctx.model_name == "openai/gpt-4o:batch"
        assert not hasattr(ctx, "model_option")
        assert "model_option" not in inspect.signature(ModelContext.__init__).parameters


@pytest.fixture
def mock_tool():
    tool = MagicMock()
    tool.name = "chat"
    tool.requires_model.return_value = True
    tool.get_model_category.return_value = MagicMock(value="general")
    tool.execute = AsyncMock(return_value=[MagicMock(type="text", text="response")])
    return tool


@pytest.fixture
def mock_registry(mock_tool):
    registry = MagicMock(spec=ToolRegistry)
    registry.get_available_tools.return_value = {"chat": mock_tool}
    registry.get_tool_instance.return_value = mock_tool
    registry.is_available.return_value = True
    return registry


class TestHandlerPreservesIdentifier:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("supplied", PROVIDER_VARIANTS + ["  openai/gpt-4o:batch  "])
    @patch("providers.registry.get_default_registry")
    async def test_identifier_reaching_model_context_is_the_one_supplied(
        self, mock_get_registry, supplied, mock_registry, mock_tool
    ):
        mock_get_registry.return_value.get_provider_for_model.return_value = MagicMock()
        seen = {}

        class CapturingContext:
            def __init__(self, model_name):
                seen["model_name"] = model_name
                self.model_name = model_name
                self.capabilities = MagicMock(context_window=100_000)

        with patch("utils.model_context.ModelContext", CapturingContext):
            handlers = tool_handlers.build_handlers(mock_registry)
            arguments = {"prompt": "hello", "model": supplied}
            await handlers.handle_call_tool("chat", arguments)

        expected = supplied.strip()
        assert seen["model_name"] == expected
        # The provider is validated against, and the tool is handed, that same identifier.
        mock_get_registry.return_value.get_provider_for_model.assert_called_once_with(expected)
        assert arguments["model"] == expected

    @pytest.mark.asyncio
    @patch("providers.registry.get_default_registry")
    async def test_tools_without_a_model_requirement_keep_their_string_verbatim(
        self, mock_get_registry, mock_registry, mock_tool
    ):
        """clink forwards model strings to a CLI subprocess byte-for-byte."""
        mock_tool.requires_model.return_value = False
        handlers = tool_handlers.build_handlers(mock_registry)
        arguments = {"prompt": "hello", "model": "  vendor/model:variant  "}
        await handlers.handle_call_tool("chat", arguments)
        assert arguments["model"] == "  vendor/model:variant  "
