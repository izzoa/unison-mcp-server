"""Chat with ``execution_mode="batch"``, end to end (add-batch-model-support, 9.1-9.3).

These are in-process integration tests through the real ``ChatTool`` entry: a real ``BatchStore`` on disk,
the real ``OpenRouterProvider`` (only ``_batch_send``, the HTTP edge, is stubbed by an
``OpenRouterBatchService``), the real executor and coordinator. No network and no API keys.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import utils.model_restrictions as model_restrictions
from handlers.tool_handlers import build_handlers
from providers.openrouter import OpenRouterProvider
from tests.batch_test_helpers import OpenRouterBatchService, batch_world_factory
from tools import AnalyzeTool, CodeReviewTool, DebugIssueTool, PrecommitTool, RefactorTool, TestGenTool, ThinkDeepTool
from tools.shared.exceptions import ToolExecutionError
from utils import batch_store as bs

OPUS = "opus"
#: 14.13: under batch `opus` is the latest batch-listed Opus of the fixture catalog, not the curated 4.5 pin.
OPUS_SLUG = "anthropic/claude-opus-5.5"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("BATCH_STORE_PATH", "BATCH_RETENTION_DAYS", "OPENROUTER_ALLOWED_MODELS"):
        monkeypatch.delenv(name, raising=False)
    model_restrictions._restriction_service = None
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()
    model_restrictions._restriction_service = None


@pytest.fixture
def service() -> OpenRouterBatchService:
    return OpenRouterBatchService()


@pytest.fixture
def boot(tmp_path, service):
    started = []
    factory = batch_world_factory(tmp_path, service)

    def make():
        world = factory()
        started.append(world)
        return world

    yield make
    for world in started:
        try:
            world.close()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _no_synchronous_openrouter_call(monkeypatch):
    """A batched chat call must never reach the provider's synchronous generate path."""

    def boom(self, *args, **kwargs):
        raise AssertionError("the synchronous provider path ran for a batch request")

    monkeypatch.setattr(OpenRouterProvider, "generate_content", boom)


def _arguments(tmp_path, **extra: Any) -> dict[str, Any]:
    return {
        "prompt": "my question about the cache",
        "working_directory_absolute_path": str(tmp_path),
        "model": OPUS,
        "execution_mode": "batch",
        **extra,
    }


def _content(blocks) -> dict[str, Any]:
    return json.loads(blocks[0].text)


class TestChatBatchEndToEnd:
    @pytest.mark.asyncio
    async def test_a_slow_batch_returns_a_handle_and_a_fast_one_returns_the_completed_content(
        self, boot, service, tmp_path
    ):
        world = boot()

        # --- slow: the provider has not finished when the bounded wait ends --------------------------
        slow = _content(await world.chat.execute(_arguments(tmp_path, wait_seconds=3)))

        assert slow["status"] == "success"
        body = json.loads(slow["content"])
        assert body["status"] == "running" and body["batch_handle"].startswith("batch_")
        assert body["resume_with"] == {"batch_handle": body["batch_handle"]}
        assert "NOT finished" in body["message"]
        assert len(service.posts) == 1 and "or_batch_1" not in slow["content"]

        # --- it finishes later; resuming with only the handle returns the reply ---------------------
        service.finish("or_batch_1", {"chat": "The cache should be write-through."})
        done = _content(await world.chat.execute({"batch_handle": body["batch_handle"]}))
        assert done["status"] == "success"
        assert done["content"].startswith("The cache should be write-through.")
        assert done["metadata"]["batch_execution"] is True and done["metadata"]["model_used"] == OPUS_SLUG
        assert len(service.posts) == 1  # resuming never submits again

        # --- fast: the batch completes inside the wait, so the caller never learns a batch existed -----
        service.on_get = lambda batch_id: service.finish(batch_id, {"chat": "Answered at once."})
        fast = _content(await world.chat.execute(_arguments(tmp_path, wait_seconds=30)))
        assert fast["status"] == "success" and fast["content"].startswith("Answered at once.")
        assert "batch_handle" not in fast["content"].split("---")[0]
        assert "resume" not in fast["content"].lower()
        assert len(service.posts) == 2

    @pytest.mark.asyncio
    async def test_the_batch_request_carries_what_a_synchronous_call_would_send(self, boot, service, tmp_path):
        world = boot()
        (tmp_path / "notes.txt").write_text("FILE BODY the model must see")

        await world.chat.execute(
            _arguments(tmp_path, absolute_file_paths=[str(tmp_path / "notes.txt")], wait_seconds=0)
        )

        envelope = service.posts[0]["envelope"]
        assert list(envelope)[:2] == ["endpoint", "model"] and envelope["model"] == OPUS_SLUG
        (entry,) = envelope["requests"]
        assert entry["custom_id"] == "chat"
        messages = entry["body"]["messages"]
        assert messages[0]["role"] == "system" and messages[0]["content"].strip()
        user = messages[-1]["content"]
        assert "my question about the cache" in user and "FILE BODY the model must see" in user
        assert "model" not in entry["body"]  # the batch-level model governs every request

    @pytest.mark.asyncio
    async def test_a_new_conversation_gets_the_follow_up_instructions_a_synchronous_call_gets(
        self, boot, service, tmp_path
    ):
        from utils.request_helpers import get_follow_up_instructions

        world = boot()

        await world.chat.execute(_arguments(tmp_path, wait_seconds=0))

        user = service.posts[0]["envelope"]["requests"][0]["body"]["messages"][-1]["content"]
        assert user.rstrip().endswith(get_follow_up_instructions(0).rstrip())

    @pytest.mark.asyncio
    async def test_a_continuation_sends_the_conversation_history_with_the_new_input(self, boot, service, tmp_path):
        from utils.conversation_memory import add_turn, create_thread

        world = boot()
        thread = create_thread("chat", {"prompt": "earlier question"})
        add_turn(thread, "user", "EARLIER USER TURN about sharding")
        add_turn(thread, "assistant", "EARLIER ASSISTANT ANSWER about sharding", tool_name="chat")

        await world.chat.execute(_arguments(tmp_path, continuation_id=thread, wait_seconds=0))

        user = service.posts[0]["envelope"]["requests"][0]["body"]["messages"][-1]["content"]
        assert "=== CONVERSATION HISTORY" in user
        assert "EARLIER ASSISTANT ANSWER about sharding" in user
        assert "=== NEW USER INPUT ===" in user and "my question about the cache" in user
        record = world.store.list_records()[0]
        assert record.continuation_id == thread

    @pytest.mark.asyncio
    async def test_the_handle_survives_a_discarded_process(self, boot, service, tmp_path):
        first = boot()
        body = json.loads(_content(await first.chat.execute(_arguments(tmp_path, wait_seconds=0)))["content"])
        first.close()
        del first
        service.finish("or_batch_1", {"chat": "Still here after the restart."})

        second = boot()  # a new store, registry, adapter, executor and tool
        out = _content(await second.chat.execute({"batch_handle": body["batch_handle"]}))

        assert out["content"].startswith("Still here after the restart.")
        assert len(service.posts) == 1

    @pytest.mark.asyncio
    async def test_a_failed_batch_is_reported_as_an_error_not_as_content(self, boot, service, tmp_path):
        world = boot()
        service.on_get = lambda batch_id: service.fail(batch_id, "provider could not process the batch")

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.chat.execute(_arguments(tmp_path, wait_seconds=30))

        payload = json.loads(excinfo.value.payload)
        assert payload["status"] == "error" and "or_batch_1" not in payload["content"]

    @pytest.mark.asyncio
    async def test_a_model_that_cannot_batch_is_refused_before_anything_is_submitted(self, boot, service, tmp_path):
        world = boot()

        # 14.13: `flash` now floats to the latest batch-listed Flash, so a genuinely unbatchable model stands in.
        with pytest.raises(ToolExecutionError) as excinfo:
            await world.chat.execute(_arguments(tmp_path, model="mistral"))

        message = json.loads(excinfo.value.payload)["content"]
        assert "mistral" in message and "without execution_mode" in message
        assert service.posts == []

    @pytest.mark.asyncio
    async def test_a_refused_submission_surfaces_as_an_error_with_no_handle(self, boot, service, tmp_path):
        import httpx

        world = boot()
        service.post_error = httpx.Response(402, json={"error": {"message": "insufficient credits"}})

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.chat.execute(_arguments(tmp_path))

        payload = json.loads(excinfo.value.payload)
        assert "insufficient credits" in payload["content"] and "batch_handle" not in payload["metadata"]
        assert world.store.list_records() == []  # nothing retained for a submission that never happened

    @pytest.mark.asyncio
    async def test_a_batch_call_without_a_working_directory_still_names_it(self, boot, service):
        world = boot()

        with pytest.raises(Exception) as excinfo:
            await world.chat.execute({"prompt": "p", "model": OPUS, "execution_mode": "batch"})

        assert "working_directory_absolute_path" in str(excinfo.value)
        assert service.posts == []


class TestThroughTheServerBoundary:
    """The same calls through the real ``handle_call_tool`` entry the MCP adapters use."""

    @pytest.mark.asyncio
    async def test_a_batched_chat_call_through_the_boundary_returns_a_handle_then_the_reply(
        self, boot, service, tmp_path, monkeypatch
    ):
        from server import tool_registry

        world = boot()
        chat = tool_registry.get_tool_instance("chat")
        monkeypatch.setattr(chat, "_batch_executor", world.executor, raising=False)
        handlers = build_handlers(tool_registry)

        started = await handlers.handle_call_tool("chat", _arguments(tmp_path, wait_seconds=1))
        body = json.loads(json.loads(started[0].text)["content"])
        assert body["status"] == "running"

        service.finish("or_batch_1", {"chat": "Through the boundary."})
        done = await handlers.handle_call_tool("chat", {"batch_handle": body["batch_handle"]})
        assert json.loads(done[0].text)["content"].startswith("Through the boundary.")
        assert len(service.posts) == 1


# Section 15 (D8), updating the assertions that assumed an alias borrows the curated entry's temperature: ``opus``
# now resolves to claude-opus-5.5, whose own batch listing takes no ``temperature``, so a request that must carry a
# temperature names a model whose entry does (the curated claude-opus-4.5, by its full slug).
TEMPERATURE_MODEL = "anthropic/claude-opus-4.5"


class TestBatchedChatRequestFidelity:
    @pytest.mark.asyncio
    async def test_the_temperature_a_caller_gives_reaches_the_batched_request(self, boot, service, tmp_path):
        world = boot()

        await world.chat.execute(_arguments(tmp_path, model=TEMPERATURE_MODEL, temperature=0.9, wait_seconds=0))
        await world.chat.execute(_arguments(tmp_path, model=TEMPERATURE_MODEL, wait_seconds=0))

        explicit = service.posts[0]["envelope"]["requests"][0]["body"]["temperature"]
        default = service.posts[1]["envelope"]["requests"][0]["body"]["temperature"]
        assert explicit == 0.9
        assert default == world.chat.get_default_temperature() != explicit

    @pytest.mark.asyncio
    async def test_the_temperature_is_the_one_validated_for_the_batched_model(
        self, boot, service, tmp_path, monkeypatch
    ):
        world = boot()
        seen: list[Any] = []

        def validated(request: Any, model_context: Any) -> tuple[float, list[str]]:
            seen.append(model_context.model_name)
            return 0.77, []

        monkeypatch.setattr(world.chat, "get_validated_temperature", validated)

        await world.chat.execute(_arguments(tmp_path, model=TEMPERATURE_MODEL, wait_seconds=0))

        assert seen == [TEMPERATURE_MODEL]
        assert service.posts[0]["envelope"]["requests"][0]["body"]["temperature"] == 0.77


class TestBatchedChatLeavesAConsistentConversation:
    @staticmethod
    def _roles(thread_id: str) -> list[tuple[str, str]]:
        from utils.conversation_memory import get_thread

        thread = get_thread(thread_id)
        assert thread is not None
        return [(turn.role, turn.content) for turn in thread.turns]

    @staticmethod
    def _thread() -> str:
        from utils.conversation_memory import add_turn, create_thread

        thread = create_thread("chat", {"prompt": "earlier question"})
        add_turn(thread, "user", "EARLIER USER TURN")
        add_turn(thread, "assistant", "EARLIER ASSISTANT ANSWER", tool_name="chat")
        return thread

    @pytest.mark.asyncio
    async def test_the_reply_is_recorded_once_so_the_thread_never_ends_on_an_unanswered_user_turn(
        self, boot, service, tmp_path
    ):
        world = boot()
        thread = self._thread()
        started = _content(await world.chat.execute(_arguments(tmp_path, continuation_id=thread, wait_seconds=0)))
        handle = json.loads(started["content"])["batch_handle"]

        # submission recorded the user turn, exactly as a synchronous call does before the model call
        assert self._roles(thread)[-1] == ("user", "my question about the cache")

        service.finish("or_batch_1", {"chat": "The cache should be write-through."})
        first = _content(await world.chat.execute({"batch_handle": handle}))
        second = _content(await world.chat.execute({"batch_handle": handle}))  # resuming again records nothing more

        roles = self._roles(thread)
        assert [role for role, _ in roles] == ["user", "assistant", "user", "assistant"]
        assert roles[-1] == ("assistant", "The cache should be write-through.")
        assert not any(a == b == "user" for (a, _), (b, _) in zip(roles, roles[1:], strict=False))
        for done in (first, second):
            assert done["status"] == "continuation_available"
            assert done["continuation_offer"]["continuation_id"] == thread
            assert done["content"].startswith("The cache should be write-through.")

    @pytest.mark.asyncio
    async def test_the_reply_is_recorded_when_the_handle_is_resumed_by_a_fresh_process(self, boot, service, tmp_path):
        first = boot()
        thread = self._thread()
        started = _content(await first.chat.execute(_arguments(tmp_path, continuation_id=thread, wait_seconds=0)))
        handle = json.loads(started["content"])["batch_handle"]
        first.close()
        del first
        service.finish("or_batch_1", {"chat": "Recorded after the restart."})

        second = boot()
        await second.chat.execute({"batch_handle": handle})
        await second.chat.execute({"batch_handle": handle})

        assert self._roles(thread)[-2:] == [
            ("user", "my question about the cache"),
            ("assistant", "Recorded after the restart."),
        ]
        assert len(self._roles(thread)) == 4

    @pytest.mark.asyncio
    async def test_a_batch_that_fails_leaves_no_assistant_turn(self, boot, service, tmp_path):
        world = boot()
        thread = self._thread()
        service.on_get = lambda batch_id: service.fail(batch_id, "provider could not process the batch")

        with pytest.raises(ToolExecutionError):
            await world.chat.execute(_arguments(tmp_path, continuation_id=thread, wait_seconds=30))

        assert [role for role, _ in self._roles(thread)] == ["user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_a_new_conversation_batch_offers_no_continuation(self, boot, service, tmp_path):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id, {"chat": "Answered at once."})

        done = _content(await world.chat.execute(_arguments(tmp_path, wait_seconds=30)))

        assert done["status"] == "success"
        assert not done.get("continuation_offer")  # there is no conversation to offer a continuation of


class TestSynchronousPathIsUntouched:
    """9.2: with ``execution_mode`` absent the synchronous path runs exactly as before; no batch machinery is reached."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("absent", [{}, {"execution_mode": None}])
    async def test_without_execution_mode_chat_never_reaches_batch_submission(
        self, boot, service, tmp_path, monkeypatch, absent
    ):
        from tools.simple.base import SimpleTool
        from utils.batch_execution import BatchExecutor

        world = boot()
        reached = []

        async def sentinel(self_, arguments):
            reached.append(arguments)
            return ["synchronous"]

        def no_submit(*args, **kwargs):
            raise AssertionError("batch submission ran for a synchronous call")

        monkeypatch.setattr(SimpleTool, "execute", sentinel)
        monkeypatch.setattr(BatchExecutor, "submit_and_wait", no_submit)
        arguments = {"prompt": "p", "working_directory_absolute_path": str(tmp_path), "model": OPUS, **absent}

        assert await world.chat.execute(arguments) == ["synchronous"]
        assert reached == [arguments] and service.posts == []


class TestWorkflowToolsNeverReachBatchSubmission:
    """9.3: workflow tools other than consensus reject ``execution_mode`` and never submit a batch."""

    @pytest.fixture
    def trap(self, monkeypatch):
        """Make any batch submission, executor construction or workflow run fail the test."""

        from tools.workflow.workflow_mixin import BaseWorkflowMixin
        from utils.batch_coordinator import BatchCoordinator
        from utils.batch_execution import BatchExecutor

        calls: list[str] = []

        def trap_call(name):
            def fail(*args, **kwargs):
                calls.append(name)
                raise AssertionError(f"{name} was reached by a workflow tool asked for batch")

            return fail

        monkeypatch.setattr(BatchCoordinator, "submit", trap_call("BatchCoordinator.submit"))
        monkeypatch.setattr(BatchExecutor, "submit_and_wait", trap_call("BatchExecutor.submit_and_wait"))
        monkeypatch.setattr("tools.shared.batch_support.build_default_executor", trap_call("build_default_executor"))

        async def no_workflow(self_, arguments):
            calls.append("execute_workflow")
            raise AssertionError("the workflow ran for a call that asked for batch")

        monkeypatch.setattr(BaseWorkflowMixin, "execute_workflow", no_workflow)
        return calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_factory",
        [CodeReviewTool, AnalyzeTool, DebugIssueTool, PrecommitTool, RefactorTool, TestGenTool, ThinkDeepTool],
    )
    async def test_a_workflow_tool_asked_for_batch_is_rejected_and_submits_nothing(self, tool_factory, trap):
        tool = tool_factory()
        arguments = {
            "step": "Review the module",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": "nothing yet",
            "model": "opus",
            "execution_mode": "batch",
        }

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(arguments)

        payload = json.loads(excinfo.value.payload)
        assert payload["metadata"]["batch_error"] == "execution_mode_unsupported"
        assert "execution_mode" in payload["content"]
        assert trap == []

    @pytest.mark.asyncio
    async def test_through_the_server_boundary_codereview_is_rejected_before_any_state_is_written(self, trap):
        from server import tool_registry

        handlers = build_handlers(tool_registry)
        arguments = {
            "step": "Review the module",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": "nothing yet",
            "relevant_files": ["/tmp/x.py"],
            "model": "opus",
            "execution_mode": "batch",
        }

        with pytest.raises(Exception) as excinfo:
            await handlers.handle_call_tool("codereview", arguments)

        assert "execution_mode" in str(excinfo.value)
        assert trap == []

    @pytest.mark.asyncio
    async def test_the_guard_is_what_stops_it(self, trap, monkeypatch):
        """Mutation proof inside the suite: without the rejection the workflow would run (and the trap fires)."""

        from tools.shared import batch_support

        monkeypatch.setattr(batch_support, "execution_mode_rejection", lambda *a, **k: None)
        tool = CodeReviewTool()
        arguments = {
            "step": "s",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": "f",
            "model": "opus",
            "execution_mode": "batch",
        }

        with pytest.raises(AssertionError):
            await tool.execute(arguments)

        assert trap == ["execute_workflow"]
