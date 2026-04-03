"""Tests for ToolExecutionContext dataclass."""

from unittest.mock import MagicMock

from utils.model_context import ModelContext
from utils.tool_execution_context import ToolExecutionContext


class TestToolExecutionContext:
    def test_instantiation_all_fields(self):
        mc = MagicMock(spec=ModelContext)
        ctx = ToolExecutionContext(
            model_context=mc,
            resolved_model_name="gemini-2.5-flash",
            remaining_tokens=5000,
            original_user_prompt="Hello",
        )
        assert ctx.model_context is mc
        assert ctx.resolved_model_name == "gemini-2.5-flash"
        assert ctx.remaining_tokens == 5000
        assert ctx.original_user_prompt == "Hello"

    def test_instantiation_defaults(self):
        mc = MagicMock(spec=ModelContext)
        ctx = ToolExecutionContext(model_context=mc, resolved_model_name="flash")
        assert ctx.remaining_tokens == 0
        assert ctx.original_user_prompt == ""

    def test_from_arguments_new_style(self):
        mc = MagicMock(spec=ModelContext)
        ctx = ToolExecutionContext(model_context=mc, resolved_model_name="flash")
        args = {"prompt": "test", "_context": ctx}
        result = ToolExecutionContext.from_arguments(args)
        assert result is ctx

    def test_from_arguments_legacy_style(self):
        mc = MagicMock(spec=ModelContext)
        args = {
            "prompt": "test",
            "_model_context": mc,
            "_resolved_model_name": "flash",
            "_remaining_tokens": 1000,
            "_original_user_prompt": "Hello",
        }
        result = ToolExecutionContext.from_arguments(args)
        assert result is not None
        assert result.model_context is mc
        assert result.resolved_model_name == "flash"
        assert result.remaining_tokens == 1000
        assert result.original_user_prompt == "Hello"

    def test_from_arguments_legacy_partial(self):
        mc = MagicMock(spec=ModelContext)
        args = {"_model_context": mc}
        result = ToolExecutionContext.from_arguments(args)
        assert result is not None
        assert result.resolved_model_name == ""
        assert result.remaining_tokens == 0

    def test_from_arguments_no_context(self):
        args = {"prompt": "test"}
        result = ToolExecutionContext.from_arguments(args)
        assert result is None

    def test_from_arguments_prefers_new_over_legacy(self):
        mc1 = MagicMock(spec=ModelContext)
        mc2 = MagicMock(spec=ModelContext)
        new_ctx = ToolExecutionContext(model_context=mc1, resolved_model_name="new")
        args = {"_context": new_ctx, "_model_context": mc2, "_resolved_model_name": "old"}
        result = ToolExecutionContext.from_arguments(args)
        assert result.resolved_model_name == "new"
