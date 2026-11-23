"""Tests for batch handle / wait / cancel on the originating tool (add-batch-model-support, 6.1-6.4, 6.9).

No network, no API keys. Tools get an injected ``BatchExecutor`` over fake providers and a
temporary durable store; "restarts" are simulated by closing the store and building new store,
provider, executor and tool objects over the same file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import jsonschema
import pytest
from mcp.types import CallToolRequestParams
from pydantic import ValidationError

import utils.model_restrictions as model_restrictions
from handlers.tool_handlers import build_handlers
from providers.openrouter import OpenRouterProvider
from providers.shared import BatchFailed, BatchIncompleteResults, BatchState
from tests.batch_test_helpers import (
    FakeClock,
    FakeProvider,
    FakeRemote,
    RemoteBatch,
    consensus_payload,
    consensus_plans,
    make_executor,
    make_plan,
)
from tests.test_openrouter_batch_adapter import _accepted, _completed, _ok
from tools import (
    AnalyzeTool,
    ChallengeTool,
    ChatTool,
    CodeReviewTool,
    ConsensusTool,
    DebugIssueTool,
    DocgenTool,
    ListModelsTool,
    LookupTool,
    PlannerTool,
    PrecommitTool,
    RefactorTool,
    TestGenTool,
    ThinkDeepTool,
    VersionTool,
)
from tools.chat import ChatRequest
from tools.clink import CLinkTool
from tools.consensus import ConsensusRequest
from tools.shared.batch_support import (
    BATCH_CONTROL_KEYS,
    BatchResumableToolMixin,
    batch_control_schema_properties,
    is_batch_control_call,
)
from tools.shared.exceptions import ToolExecutionError
from tools.simple.base import SimpleTool
from utils import batch_store as bs
from utils.batch_store import BatchStore

SNAPSHOT = Path(__file__).parent / "fixtures" / "tool_schemas_pre_batch.json"

FIVE_WORKFLOW_FIELDS = ["step", "step_number", "total_steps", "next_step_required", "findings"]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("BATCH_STORE_PATH", "BATCH_RETENTION_DAYS"):
        monkeypatch.delenv(name, raising=False)
    model_restrictions._restriction_service = None
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()
    model_restrictions._restriction_service = None


def _chat() -> ChatTool:
    return ChatTool()


def _consensus() -> ConsensusTool:
    return ConsensusTool()


def _payload(blocks) -> dict:
    return json.loads(blocks[0].text)


def _message(excinfo) -> str:
    """The validator's own message, without pydantic's echo of the input."""

    return " | ".join(error["msg"] for error in excinfo.value.errors())


def _error_payload(excinfo) -> dict:
    return json.loads(excinfo.value.payload)


# ----------------------------------------------------------------------
# 6.1 published schema
# ----------------------------------------------------------------------


class TestPublishedSchema:
    @pytest.mark.parametrize("tool_factory", [_chat, _consensus])
    def test_schema_exposes_handle_wait_and_cancel(self, tool_factory):
        properties = tool_factory().get_input_schema()["properties"]
        assert properties["batch_handle"]["type"] == "string"
        assert properties["wait_seconds"]["type"] == "number" and properties["wait_seconds"]["minimum"] == 0
        assert properties["cancel"]["type"] == "boolean"
        assert set(batch_control_schema_properties()) == set(BATCH_CONTROL_KEYS)

    def test_chat_schema_lists_no_start_work_input_as_required(self):
        schema = _chat().get_input_schema()
        assert "prompt" not in schema["required"]
        assert "working_directory_absolute_path" not in schema["required"]
        assert schema["required"] == []

    def test_consensus_schema_lists_none_of_the_five_workflow_fields_as_required(self):
        schema = _consensus().get_input_schema()
        for name in FIVE_WORKFLOW_FIELDS:
            assert name not in schema["required"]
            assert name in schema["properties"]  # still published, with consensus' own descriptions
        assert "models" in schema["properties"]
        assert schema["properties"]["step"]["description"].startswith("Consensus prompt")

    @pytest.mark.parametrize("tool_factory", [_chat, _consensus])
    def test_a_handle_only_call_passes_schema_validation(self, tool_factory):
        schema = tool_factory().get_input_schema()
        jsonschema.validate(instance={"batch_handle": "batch_abc"}, schema=schema)
        jsonschema.validate(instance={"batch_handle": "batch_abc", "wait_seconds": 12.5}, schema=schema)
        jsonschema.validate(instance={"batch_handle": "batch_abc", "cancel": True}, schema=schema)

    @pytest.mark.parametrize("tool_factory", [_chat, _consensus])
    def test_a_handle_only_call_validates_in_auto_mode_too(self, tool_factory):
        """The default install runs in auto mode; the boundary validates against this exact schema."""
        with patch("config.DEFAULT_MODEL", "auto"), patch("config.IS_AUTO_MODE", True):
            tool = tool_factory()
            assert tool.is_effective_auto_mode()
            schema = tool.get_input_schema()
        assert schema["required"] == []
        jsonschema.validate(instance={"batch_handle": "batch_abc"}, schema=schema)
        jsonschema.validate(instance={"batch_handle": "batch_abc", "cancel": True}, schema=schema)

    def test_wait_seconds_above_the_maximum_is_clamped_not_rejected(self):
        schema = _chat().get_input_schema()
        jsonschema.validate(instance={"batch_handle": "h", "wait_seconds": 100000}, schema=schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={"batch_handle": "h", "wait_seconds": -1}, schema=schema)

    def test_other_tools_schemas_are_byte_for_byte_unchanged(self):
        """The batch work changed no other tool's schema.

        The fixture is not a frozen copy of the very first schemas: it was updated narrowly, and only for the
        intentional published-schema edits listed in tests/test_workflow_published_fields.py
        (REMOVED_PUBLISHED_FIELDS: properties dropped because the tool never reads them, planner / docgen no
        longer advertising ``model`` / ``use_assistant_model``, precommit's total_steps minimum of 1, and the
        tool-specific descriptions that now reach the schema) and for the model / temperature / thinking_mode
        text. Any other difference, in any tool, still fails here.
        """
        snapshot = json.loads(SNAPSHOT.read_text())
        tools = [
            AnalyzeTool,
            ChallengeTool,
            CodeReviewTool,
            DebugIssueTool,
            DocgenTool,
            LookupTool,
            PlannerTool,
            PrecommitTool,
            RefactorTool,
            TestGenTool,
            ThinkDeepTool,
            ListModelsTool,
            VersionTool,
            CLinkTool,
        ]
        assert {cls().get_name() for cls in tools} == set(snapshot)
        for cls in tools:
            tool = cls()
            schema = json.loads(json.dumps(tool.get_input_schema()))
            properties = schema.get("properties", {})
            if "model" in properties:  # environment dependent text, masked identically in the snapshot
                properties["model"] = {"__env_dependent__": True}
            assert json.dumps(schema) == json.dumps(snapshot[tool.get_name()]), tool.get_name()
            if cls is not CLinkTool:  # clink publishes its own job-control cancel / wait_seconds, unchanged
                assert "batch_handle" not in properties, f"{tool.get_name()} must not publish batch fields"

    def test_only_chat_and_consensus_are_batch_resumable(self):
        resumable = {
            cls().get_name()
            for cls in (
                AnalyzeTool,
                ChallengeTool,
                ChatTool,
                CLinkTool,
                CodeReviewTool,
                ConsensusTool,
                DebugIssueTool,
                DocgenTool,
                LookupTool,
                PlannerTool,
                PrecommitTool,
                RefactorTool,
                TestGenTool,
                ThinkDeepTool,
            )
            if isinstance(cls(), BatchResumableToolMixin)
        }
        assert resumable == {"chat", "consensus"}


# ----------------------------------------------------------------------
# 6.2 request models
# ----------------------------------------------------------------------


class TestRequestModels:
    def test_chat_handle_only_validates(self):
        request = ChatRequest(batch_handle="batch_abc")
        assert request.is_resuming and request.prompt is None

    def test_chat_start_work_missing_inputs_fails_naming_each_by_name(self):
        with pytest.raises(ValidationError) as both:
            ChatRequest()
        text = _message(both)
        assert "'prompt'" in text and "'working_directory_absolute_path'" in text

        with pytest.raises(ValidationError) as one:
            ChatRequest(prompt="hello")
        assert "'working_directory_absolute_path'" in _message(one) and "'prompt'" not in _message(one)

    def test_chat_with_both_inputs_validates_exactly_as_before(self):
        request = ChatRequest(prompt="hello", working_directory_absolute_path="/tmp")
        assert not request.is_resuming
        assert set(request.model_dump()) == {
            "model",
            "temperature",
            "thinking_mode",
            "continuation_id",
            "images",
            "prompt",
            "absolute_file_paths",
            "working_directory_absolute_path",
        }  # the batch fields never appear in a serialised request

    def test_cancel_without_a_handle_is_rejected(self):
        with pytest.raises(ValidationError, match="'cancel' requires 'batch_handle'"):
            ChatRequest(prompt="x", working_directory_absolute_path="/tmp", cancel=True)

    def test_consensus_handle_only_validates(self):
        assert ConsensusRequest(batch_handle="batch_abc").is_resuming

    def test_consensus_start_work_fails_naming_the_missing_workflow_fields(self):
        with pytest.raises(ValidationError) as excinfo:
            ConsensusRequest()
        text = _message(excinfo)
        for name in FIVE_WORKFLOW_FIELDS:
            assert f"'{name}'" in text

        with pytest.raises(ValidationError) as partial:
            ConsensusRequest(step="Evaluate X", step_number=1, total_steps=1)
        assert "'next_step_required'" in _message(partial) and "'findings'" in _message(partial)
        assert "'step'" not in _message(partial)

    def test_consensus_step_one_still_requires_models_and_unique_pairs(self):
        base = {"step": "Evaluate X", "step_number": 1, "total_steps": 1, "next_step_required": False, "findings": "f"}
        with pytest.raises(ValidationError, match="Step 1 requires 'models'"):
            ConsensusRequest(**base)
        with pytest.raises(ValidationError, match="Duplicate model \\+ stance"):
            ConsensusRequest(**base, models=[{"model": "a"}, {"model": "a"}])
        assert ConsensusRequest(**base, models=[{"model": "a"}, {"model": "b"}]).step_number == 1

    def test_consensus_later_steps_do_not_need_models(self):
        base = {"step": "note", "step_number": 2, "total_steps": 3, "next_step_required": True, "findings": "f"}
        assert ConsensusRequest(**base).models is None

    def test_a_handle_skips_the_step_one_models_requirement(self):
        assert ConsensusRequest(batch_handle="h", step_number=1).is_resuming


# ----------------------------------------------------------------------
# 6.4 precedence, plus dispatch of resume / cancel
# ----------------------------------------------------------------------


@pytest.fixture
def world(tmp_path):
    """A durable store, a fake provider's remote, and a factory for fresh executors over them."""

    clock = FakeClock()
    remote = FakeRemote()
    store = BatchStore(str(tmp_path / "durable" / "batches.db"))

    class World:
        pass

    state = World()
    state.clock, state.remote, state.store, state.path = clock, remote, store, str(tmp_path / "durable" / "batches.db")
    state.provider = FakeProvider(remote)
    state.executor = make_executor(store, {"openrouter": state.provider}, clock)
    yield state
    state.store.close()


def _submit_chat(world, *, wait=0) -> str:
    return world.executor.submit_and_wait(
        "chat",
        [make_plan(world.provider)],
        wait_seconds=wait,
        payload={"working_directory_absolute_path": "/tmp", "model": "anthropic/claude-opus-4.5"},
    ).handle


def _tool(tool, world):
    tool._batch_executor = world.executor
    return tool


def _submit_consensus(world, *models: str, wait=0) -> str:
    """A running consensus-shaped record: one batch per model, each owning its roster entry's request id."""

    return world.executor.submit_and_wait(
        "consensus",
        consensus_plans(world.provider, *models),
        wait_seconds=wait,
        payload=consensus_payload(*models),
        fetch=False,
    ).handle


class _GenericBatchTool(BatchResumableToolMixin):
    """A batch-capable tool that keeps the mixin's default rendering (consensus and chat now render their own)."""

    def get_name(self) -> str:
        return "generic"


class TestHandlePath:
    def test_is_batch_control_call(self):
        assert is_batch_control_call({"batch_handle": "batch_x"})
        assert not is_batch_control_call({"batch_handle": "   "})
        assert not is_batch_control_call({"prompt": "x", "cancel": True})

    @pytest.mark.asyncio
    async def test_running_handle_returns_the_standard_payload(self, world):
        handle = _submit_chat(world)
        out = _payload(await _tool(_chat(), world).execute({"batch_handle": handle, "wait_seconds": 2}))

        assert out["status"] == "success"
        body = json.loads(out["content"])
        assert body["status"] == "running" and body["batch_handle"] == handle
        assert body["resume_with"] == {"batch_handle": handle}
        assert "still running" in body["message"] and "NOT finished" in body["message"]
        assert out["metadata"]["batch_handle"] == handle
        assert "prov_batch" not in out["content"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_factory", [_chat, _consensus])
    async def test_the_resume_advice_asks_for_only_the_handle_not_a_placeholder_model(self, world, tool_factory):
        """The schema used to demand a model in auto mode, so chat told callers to send 'model: auto'."""

        tool = _tool(tool_factory(), world)
        assert tool.batch_resume_hint() == ""
        handle = _submit_chat(world) if isinstance(tool, ChatTool) else _submit_consensus(world, "m/one")

        out = _payload(await tool.execute({"batch_handle": handle, "wait_seconds": 0}))

        message = json.loads(out["content"])["message"]
        assert f"ONLY batch_handle='{handle}'" in message
        assert "ignored" not in message and "any value" not in message and "'auto'" not in message

    @pytest.mark.asyncio
    async def test_completed_chat_batch_is_rendered_like_a_synchronous_reply(self, world):
        handle = _submit_chat(world)
        world.remote.finish("prov_batch_1", results={"chat": "The answer."})

        out = _payload(await _tool(_chat(), world).execute({"batch_handle": handle}))

        assert out["status"] == "success"
        assert out["content"].startswith("The answer.")
        assert "AGENT'S TURN" in out["content"]
        assert out["metadata"]["model_used"] == "anthropic/claude-opus-4.5"  # the recorded model
        assert out["metadata"]["batch_execution"] is True

    @pytest.mark.asyncio
    async def test_a_handle_beats_start_work_inputs_and_submits_nothing(self, world):
        handle = _submit_chat(world)
        submissions_before = world.remote.submissions
        tool = _tool(_chat(), world)

        with patch.object(SimpleTool, "execute", side_effect=AssertionError("start-work path must not run")):
            blocks = await tool.execute(
                {
                    "batch_handle": handle,
                    "prompt": "ask something new",
                    "working_directory_absolute_path": "/tmp",
                    "model": "flash",
                    "wait_seconds": 0,
                }
            )

        out = _payload(blocks)
        body = json.loads(out["content"])
        assert body["batch_handle"] == handle
        assert set(body["ignored_inputs"]) == {"prompt", "working_directory_absolute_path"}
        assert world.remote.submissions == submissions_before
        assert world.provider.calls.count("submit_batch") == submissions_before

    @pytest.mark.asyncio
    async def test_a_call_without_batch_fields_reaches_the_unchanged_base_path(self, world):
        tool = _tool(_chat(), world)
        arguments = {"prompt": "hi", "working_directory_absolute_path": "/tmp"}
        with patch.object(SimpleTool, "execute", return_value=["sentinel"]) as base:
            assert await tool.execute(arguments) == ["sentinel"]
        assert base.call_args.args == (arguments,)  # untouched, nothing added or removed

    @pytest.mark.asyncio
    async def test_cancel_through_the_originating_tool(self, world):
        handle = _submit_chat(world)

        out = _payload(await _tool(_chat(), world).execute({"batch_handle": handle, "cancel": True}))

        body = json.loads(out["content"])
        assert body["status"] == "cancelled"
        assert body["provider_batches_continue"] is True  # FakeProvider has no cancel mechanism
        assert "continues and will still be billed" in body["message"]
        assert "NOT stopped" in body["message"]
        assert world.store.get(handle).state.value == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_without_a_handle_is_an_error(self, world):
        with pytest.raises(ToolExecutionError) as excinfo:
            await _tool(_chat(), world).execute({"cancel": True})
        assert "requires 'batch_handle'" in _error_payload(excinfo)["content"]

    @pytest.mark.asyncio
    async def test_unknown_and_wrong_tool_handles_are_distinct_errors(self, world):
        handle = _submit_chat(world)
        with pytest.raises(ToolExecutionError) as unknown:
            await _tool(_chat(), world).execute({"batch_handle": "batch_made_up"})
        assert "Unknown batch handle" in _error_payload(unknown)["content"]
        assert _error_payload(unknown)["metadata"]["batch_error"] == "unknown_handle"

        with pytest.raises(ToolExecutionError) as wrong:
            await _tool(_consensus(), world).execute({"batch_handle": handle})
        assert "issued by the 'chat' tool" in _error_payload(wrong)["content"]

    @pytest.mark.asyncio
    async def test_expired_handle_is_not_reported_as_unknown(self, tmp_path):
        wall = [20_000_000.0]
        store = BatchStore(str(tmp_path / "e" / "batches.db"), retention_days=1, clock=lambda: wall[0])
        clock, remote = FakeClock(), FakeRemote()
        provider = FakeProvider(remote)
        executor = make_executor(store, {"openrouter": provider}, clock)
        handle = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0).handle
        wall[0] += 3 * 86400
        tool = _chat()
        tool._batch_executor = executor

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute({"batch_handle": handle})

        error = _error_payload(excinfo)
        assert error["metadata"]["batch_error"] == "handle_expired"
        assert "have expired" in error["content"] and "Unknown" not in error["content"]
        store.close()

    @pytest.mark.asyncio
    async def test_consensus_handle_resumes_through_consensus(self, world):
        handle = _submit_consensus(world, "m/one", "m/two")
        tool = _tool(_consensus(), world)

        running = _payload(await tool.execute({"batch_handle": handle, "wait_seconds": 0}))
        assert json.loads(running["content"])["status"] == "running"

        world.remote.finish("prov_batch_1")
        world.remote.finish("prov_batch_2")
        body = json.loads((await tool.execute({"batch_handle": handle}))[0].text)
        # consensus renders its own response (the one a synchronous run returns), not the generic listing
        assert body["status"] == "consensus_workflow_complete" and body["metadata"]["batch_handle"] == handle
        assert {r["model"]: r["verdict"] for r in body["accumulated_responses"]} == {
            "m/one": "answer:consensus-0",
            "m/two": "answer:consensus-1",
        }
        assert body["complete_consensus"]["initial_prompt"] == "Evaluate X"

    @pytest.mark.asyncio
    async def test_a_batch_that_finishes_inside_the_wait_renders_without_asking_to_resume(self, world):
        original = world.provider.submit_batch

        def submit_and_finish(model, requests):
            batch_id = original(model, requests)
            world.remote.finish(batch_id)
            return batch_id

        world.provider.submit_batch = submit_and_finish
        result = world.executor.submit_and_wait("generic", [make_plan(world.provider, "m/one", ("a",))])
        assert result.status.value == "completed"

        for tool in (_tool(_GenericBatchTool(), world), _tool(_chat(), world)):
            blocks = tool.render_batch_result(result)
            text = _payload(blocks)["content"].lower()
            assert "resume" not in text and "still running" not in text and "not finished" not in text

    @pytest.mark.asyncio
    async def test_batch_failures_reach_the_caller_without_the_provider_batch_id(self, world):
        handle = world.executor.submit_and_wait(
            "generic",
            [make_plan(world.provider, "m/one", ("a",)), make_plan(world.provider, "m/two", ("b",))],
            wait_seconds=0,
        ).handle
        world.remote.finish("prov_batch_1")
        world.remote.batches["prov_batch_2"].script = [BatchState.FAILED]
        world.remote.batches["prov_batch_2"].fetch_error = BatchFailed(
            "OpenRouter batch prov_batch_2 failed: overloaded", state=BatchState.FAILED
        )

        out = _payload(await _tool(_GenericBatchTool(), world).execute({"batch_handle": handle}))

        body = json.loads(out["content"])
        assert [r["model"] for r in body["results"]] == ["m/one"]
        assert body["failures"] and "overloaded" in body["failures"][0]["error"]
        assert "prov_batch" not in out["content"]

    @pytest.mark.asyncio
    async def test_incomplete_provider_results_are_flagged_not_silently_dropped(self, world):
        handle = world.executor.submit_and_wait(
            "generic", [make_plan(world.provider, "m/one", ("a", "b"))], wait_seconds=0
        ).handle
        world.remote.finish("prov_batch_1")
        original = world.provider.fetch_batch

        def partial(batch_id, expected_custom_ids=None):
            outcomes = original(batch_id, expected_custom_ids=expected_custom_ids)
            raise BatchIncompleteResults("1 of 2 returned", outcomes=[o for o in outcomes if o.custom_id == "a"])

        world.provider.fetch_batch = partial

        body = json.loads(
            _payload(await _tool(_GenericBatchTool(), world).execute({"batch_handle": handle}))["content"]
        )

        assert [r["custom_id"] for r in body["results"]] == ["a"]
        assert body["failures"] and "fewer results" in body["failures"][0]["error"]

    @pytest.mark.asyncio
    async def test_consensus_start_work_without_inputs_names_what_is_missing(self, world):
        with pytest.raises(ValidationError) as excinfo:
            await _tool(_consensus(), world).execute({})
        for name in FIVE_WORKFLOW_FIELDS:
            assert f"'{name}'" in _message(excinfo)


# ----------------------------------------------------------------------
# 6.1 / 6.2 / 6.3 through the real MCP boundary (handle_list_tools + on_call_tool's jsonschema check)
# ----------------------------------------------------------------------


@pytest.fixture(params=["auto", "gemini-2.5-flash"], ids=["auto-default", "concrete-default"])
def default_model(request, monkeypatch):
    """Run under the default install (DEFAULT_MODEL=auto) and under a concrete default."""

    value = request.param
    monkeypatch.setattr("config.DEFAULT_MODEL", value)
    monkeypatch.setattr("config.IS_AUTO_MODE", value == "auto")
    monkeypatch.setattr("handlers.tool_handlers.DEFAULT_MODEL", value)
    return value


def _boundary():
    from server import tool_registry

    return build_handlers(tool_registry), tool_registry  # fresh handlers: no stale published-schema cache


class TestSchemaThroughTheServerBoundary:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["chat", "consensus"])
    async def test_the_listed_schema_accepts_a_handle_only_call(self, default_model, name):
        handlers, _ = _boundary()
        listed = {tool.name: tool for tool in await handlers.handle_list_tools()}

        schema = listed[name].input_schema

        assert schema["required"] == []
        jsonschema.validate(instance={"batch_handle": "batch_abc"}, schema=schema)
        jsonschema.validate(instance={"batch_handle": "batch_abc", "cancel": True}, schema=schema)
        assert {"batch_handle", "wait_seconds", "cancel"} <= set(schema["properties"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["chat", "consensus"])
    async def test_on_call_tool_lets_a_handle_only_call_through_validation_and_resumes_it(
        self, default_model, name, world, monkeypatch
    ):
        """Before the fix a handle-only chat call died in auto mode with "'model' is a required property"."""

        handlers, registry = _boundary()
        tool = registry.get_tool_instance(name)
        monkeypatch.setattr(tool, "_batch_executor", world.executor, raising=False)
        if name == "chat":
            handle = _submit_chat(world)
        else:
            handle = _submit_consensus(world, "m/one")

        result = await handlers.on_call_tool(None, CallToolRequestParams(name=name, arguments={"batch_handle": handle}))

        text = result.content[0].text
        assert not text.startswith("Input validation error"), text
        assert result.is_error is False, text
        body = json.loads(json.loads(text)["content"])
        assert body["status"] == "running" and body["batch_handle"] == handle

    @pytest.mark.asyncio
    async def test_a_schema_violation_is_still_rejected_at_the_boundary(self, default_model):
        handlers, _ = _boundary()

        result = await handlers.on_call_tool(None, CallToolRequestParams(name="chat", arguments={"batch_handle": 123}))

        assert result.is_error is True
        assert result.content[0].text.startswith("Input validation error: ")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name, arguments, expected",
        [
            ("chat", {"prompt": "hello", "model": "gemini-2.5-flash"}, "working_directory_absolute_path"),
            ("chat", {"working_directory_absolute_path": "/tmp", "model": "gemini-2.5-flash"}, "prompt"),
            ("consensus", {}, "step_number"),
        ],
    )
    async def test_start_work_without_its_inputs_fails_at_runtime_naming_them(
        self, default_model, name, arguments, expected
    ):
        handlers, _ = _boundary()

        result = await handlers.on_call_tool(None, CallToolRequestParams(name=name, arguments=arguments))

        assert result.is_error is True
        assert expected in result.content[0].text
        assert not result.content[0].text.startswith("Input validation error")


# ----------------------------------------------------------------------
# 6.9 resume across a process restart
# ----------------------------------------------------------------------


class _OpenRouterService:
    """OpenRouter's side of the wire behind ``OpenRouterProvider._batch_send``: outlives any adapter object."""

    def __init__(self) -> None:
        self.posts = 0
        self.calls: list[tuple[str, str]] = []
        self.custom_ids: list[str] = []
        self.results: dict[str, str] | None = None

    def finish(self, results: dict[str, str]) -> None:
        self.results = results

    def __call__(self, method: str, path: str, content: str | None = None) -> httpx.Response:
        self.calls.append((method, path))
        if method == "POST":
            self.posts += 1
            self.custom_ids = _collect_custom_ids(json.loads(content or "{}"))
            counts = {"total": len(self.custom_ids), "completed": 0, "failed": 0}
            return httpx.Response(202, json=_accepted("or_batch_1", request_counts=counts))
        assert method == "GET" and path == "/batches/or_batch_1", (method, path)
        if self.results is None:
            counts = {"total": len(self.custom_ids), "completed": 0, "failed": 0}
            return httpx.Response(200, json=_accepted("or_batch_1", status="in_progress", request_counts=counts))
        return httpx.Response(200, json=_completed([_ok(cid, text) for cid, text in self.results.items()]))


def _collect_custom_ids(node) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        if isinstance(node.get("custom_id"), str):
            found.append(node["custom_id"])
        for value in node.values():
            found.extend(_collect_custom_ids(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_collect_custom_ids(value))
    return found


class TestResumeAcrossRestart:
    @pytest.mark.asyncio
    async def test_fresh_tool_instance_resumes_a_handle_from_a_discarded_process(self, tmp_path):
        db_path = str(tmp_path / "restart" / "batches.db")
        remote = FakeRemote()

        # --- process one: submit, persist, hand back a handle ------------------------------
        clock_one = FakeClock()
        store_one = BatchStore(db_path)
        provider_one = FakeProvider(remote)
        executor_one = make_executor(store_one, {"openrouter": provider_one}, clock_one)
        first = executor_one.submit_and_wait(
            "chat",
            [make_plan(provider_one, "anthropic/claude-opus-4.5:batch", ("chat",))],
            continuation_id="thread-1",
            payload={"working_directory_absolute_path": str(tmp_path)},
            wait_seconds=0,
        )
        handle = first.handle
        assert first.status.value == "running"

        # --- the process dies: every in-process object is discarded -----------------------------
        store_one.close()
        del store_one, provider_one, executor_one
        remote.finish("prov_batch_1", results={"chat": "Resumed answer."})

        # --- process two: new store instance, new provider adapter, new tool instance -----------
        store_two = BatchStore(db_path)
        provider_two = FakeProvider(remote)  # a fresh adapter that remembers nothing it submitted
        tool = ChatTool()
        tool._batch_executor = make_executor(store_two, {"openrouter": provider_two}, FakeClock())

        out = _payload(await tool.execute({"batch_handle": handle}))

        assert out["status"] == "success"
        assert out["content"].startswith("Resumed answer.")
        assert out["metadata"]["model_used"] == "anthropic/claude-opus-4.5"  # recorded slug, not :batch
        assert provider_two.fetch_expected_ids == [("chat",)]  # the persisted custom_ids reached the adapter
        assert provider_two.calls.count("submit_batch") == 0 and remote.submissions == 1

        again = _payload(await tool.execute({"batch_handle": handle}))
        assert again["content"] == out["content"]  # repeated resumption: same results, nothing resubmitted
        assert remote.submissions == 1
        store_two.close()

    @pytest.mark.asyncio
    async def test_restart_with_the_real_openrouter_adapter_resumes_and_never_resubmits(self, tmp_path):
        """The real adapter and wire parser (only the HTTP send is stubbed) across a discarded process."""

        db_path = str(tmp_path / "restart-or" / "batches.db")
        service = _OpenRouterService()
        model = "anthropic/claude-opus-4.5"

        store_one = BatchStore(db_path)
        adapter_one = OpenRouterProvider(api_key="test-key")
        adapter_one._batch_send = service
        executor_one = make_executor(store_one, {"openrouter": adapter_one}, FakeClock())
        first = executor_one.submit_and_wait(
            "chat",
            [make_plan(adapter_one, model, ("chat",))],
            payload={"working_directory_absolute_path": str(tmp_path)},
            wait_seconds=0,
        )
        handle = first.handle
        assert first.status.value == "running" and service.posts == 1
        store_one.close()
        del store_one, adapter_one, executor_one  # the process is gone; OpenRouter still holds the batch
        service.finish({"chat": "Resumed through the real wire parser."})

        store_two = BatchStore(db_path)
        adapter_two = OpenRouterProvider(api_key="test-key")  # a fresh adapter: it remembers nothing it submitted
        adapter_two._batch_send = service
        tool = ChatTool()
        tool._batch_executor = make_executor(store_two, {"openrouter": adapter_two}, FakeClock())

        out = _payload(await tool.execute({"batch_handle": handle}))

        assert out["status"] == "success"
        assert out["content"].startswith("Resumed through the real wire parser.")
        assert out["metadata"]["model_used"] == model and out["metadata"]["provider_used"] == "openrouter"
        assert "or_batch_1" not in out["content"]  # the caller sees the handle, never the provider id

        again = _payload(await tool.execute({"batch_handle": handle}))
        assert again["content"] == out["content"]
        assert service.posts == 1  # two resumptions, one billable submission
        store_two.close()

    @pytest.mark.asyncio
    async def test_a_consensus_payload_round_trips_through_a_restart_and_rebuilds_the_response(self, tmp_path):
        """The persisted proposal, findings, roster and accumulated responses survive a restart byte for byte."""

        db_path = str(tmp_path / "restart-consensus" / "batches.db")
        payload = {
            "original_proposal": "Should we migrate the billing service \u2014 \u00e9\u00e8 \u65e5\u672c\u8a9e? \U0001f680",
            "findings": 'Step 1 findings:\nline two with "quotes" and a tab\t.',
            "models": [
                {"model": "m/one", "stance": "for", "execution_mode": "batch"},
                {"model": "m/sync", "stance": "against"},
                {"model": "m/one", "stance": "neutral", "execution_mode": "batch"},
                {"model": "m/two", "stance": "against", "execution_mode": "batch"},
            ],
            "accumulated_responses": [
                {
                    "model": "m/sync",
                    "stance": "against",
                    "status": "success",
                    "verdict": "ship it",
                    "metadata": {"n": 1},
                }
            ],
        }
        remote = FakeRemote()
        store_one = BatchStore(db_path)
        provider_one = FakeProvider(remote)
        executor_one = make_executor(store_one, {"openrouter": provider_one}, FakeClock())
        handle = executor_one.submit_and_wait(
            "consensus",
            [
                make_plan(provider_one, "m/one", ("consensus-0", "consensus-2")),
                make_plan(provider_one, "m/two", ("consensus-3",)),
            ],
            continuation_id="thread-9",
            payload=payload,
            wait_seconds=0,
            fetch=False,
        ).handle
        store_one.close()
        del store_one, provider_one, executor_one
        remote.finish("prov_batch_1")
        remote.finish("prov_batch_2")

        store_two = BatchStore(db_path)
        provider_two = FakeProvider(remote)
        tool = ConsensusTool()
        tool._batch_executor = make_executor(store_two, {"openrouter": provider_two}, FakeClock())

        result = tool._batch_executor.resume(handle, tool_name="consensus", fetch=False)
        assert result.record.payload == payload  # byte-for-byte the structure that was persisted
        assert result.record.continuation_id == "thread-9"

        body = json.loads((await tool.execute({"batch_handle": handle}))[0].text)
        assert body["status"] == "consensus_workflow_complete" and body["metadata"]["batch_handle"] == handle
        assert body["complete_consensus"]["initial_prompt"] == payload["original_proposal"]
        assert body["agent_analysis"]["findings"] == payload["findings"]
        assert [(r["model"], r["stance"]) for r in body["accumulated_responses"]] == [
            ("m/one", "for"),
            ("m/sync", "against"),
            ("m/one", "neutral"),
            ("m/two", "against"),
        ]
        assert body["accumulated_responses"][1]["verdict"] == "ship it"  # the persisted synchronous verdict
        assert provider_two.calls.count("submit_batch") == 0 and remote.submissions == 2
        store_two.close()

    @pytest.mark.asyncio
    async def test_a_running_handle_and_a_cancel_survive_a_restart_too(self, tmp_path):
        db_path = str(tmp_path / "restart2" / "batches.db")
        remote = FakeRemote()
        store_one = BatchStore(db_path)
        provider_one = FakeProvider(remote)
        handle = (
            make_executor(store_one, {"openrouter": provider_one}, FakeClock())
            .submit_and_wait(
                "consensus",
                consensus_plans(provider_one, "m/one"),
                wait_seconds=0,
                payload=consensus_payload("m/one"),
                fetch=False,
            )
            .handle
        )
        store_one.close()

        store_two = BatchStore(db_path)
        provider_two = FakeProvider(remote)
        tool = ConsensusTool()
        tool._batch_executor = make_executor(store_two, {"openrouter": provider_two}, FakeClock())

        running = json.loads(_payload(await tool.execute({"batch_handle": handle, "wait_seconds": 0}))["content"])
        assert running["status"] == "running"

        cancelled = json.loads(_payload(await tool.execute({"batch_handle": handle, "cancel": True}))["content"])
        assert cancelled["status"] == "cancelled" and cancelled["provider_batches_continue"] is True
        assert remote.batches["prov_batch_1"].script == [BatchState.VALIDATING]  # nothing was deleted or altered
        store_two.close()

    @pytest.mark.asyncio
    async def test_resume_works_in_a_genuinely_separate_process(self, tmp_path):
        """Process one is a real interpreter that exits; this process resumes the handle it left behind."""

        db_path = str(tmp_path / "xproc" / "batches.db")
        repo_root = Path(__file__).resolve().parent.parent
        script = (
            "import sys\n"
            "from tests.batch_test_helpers import FakeClock, FakeProvider, FakeRemote, make_executor, make_plan\n"
            "from utils.batch_store import BatchStore\n"
            "store = BatchStore(sys.argv[1])\n"
            "provider = FakeProvider(FakeRemote())\n"
            "executor = make_executor(store, {'openrouter': provider}, FakeClock())\n"
            "result = executor.submit_and_wait('chat', [make_plan(provider, 'anthropic/claude-opus-4.5:batch')],\n"
            "    wait_seconds=0, payload={'working_directory_absolute_path': sys.argv[2]})\n"
            "store.close()\n"
            "print(result.handle)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script, db_path, str(tmp_path)],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root)},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        handle = proc.stdout.strip().splitlines()[-1]
        assert handle.startswith("batch_")

        # The provider service outlives the dead process; this process has never seen the submission.
        remote = FakeRemote()
        remote.submissions = 1
        remote.batches["prov_batch_1"] = RemoteBatch(model="anthropic/claude-opus-4.5", custom_ids=["chat"])
        remote.finish("prov_batch_1", results={"chat": "Answer from another life."})
        store = BatchStore(db_path)
        provider = FakeProvider(remote)
        tool = ChatTool()
        tool._batch_executor = make_executor(store, {"openrouter": provider}, FakeClock())

        out = _payload(await tool.execute({"batch_handle": handle}))

        assert out["content"].startswith("Answer from another life.")
        assert provider.fetch_expected_ids == [("chat",)]
        assert provider.calls.count("submit_batch") == 0
        store.close()
