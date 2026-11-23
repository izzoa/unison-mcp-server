"""Execution selection (add-batch-model-support, section 7).

Specs: batch-execution-selection, clink-runtime-model-selection, batch-job-lifecycle (a resumption
is not gated on model resolution). No network and no API keys: providers are real adapter classes
with dummy keys (selection and capability checks only), or in-process fakes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import jsonschema
import pytest
from mcp.types import CallToolRequestParams
from pydantic import ValidationError

import utils.model_restrictions as model_restrictions
from handlers.tool_handlers import build_handlers
from providers.gemini import GeminiModelProvider
from providers.openai import OpenAIModelProvider
from providers.openrouter import OpenRouterProvider
from providers.registry import ModelProviderRegistry, set_default_registry
from providers.shared import ModelResponse, ProviderType
from tests.batch_test_helpers import (
    FakeClock,
    FakeProvider,
    FakeRemote,
    consensus_payload,
    consensus_plans,
    make_executor,
    make_plan,
)
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
    ACCEPTED_EXECUTION_MODES,
    BatchRequestRejected,
    BatchResumableToolMixin,
    batch_schema_properties,
    execution_mode_rejection,
    non_text_argument_names,
    normalize_execution_mode,
    reject_non_text_arguments,
    select_batch_provider,
)
from tools.shared.exceptions import ToolExecutionError
from utils import batch_store as bs
from utils.batch_store import BatchStore

OPUS = "anthropic/claude-opus-4.5"  # catalogued with a batch listing, aliases: opus
BATCHABLE_ALIAS = "gpt5"  # served by native OpenAI (no batch adapter) AND by OpenRouter (batch listing)
NOT_BATCHABLE = "mistral"  # served by OpenRouter, which has no batch listing for it (14.13: `flash` now floats)
SYNC_ONLY_NATIVE = "flash"  # still served by native Google synchronously; under batch it floats on OpenRouter


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("BATCH_STORE_PATH", "BATCH_RETENTION_DAYS", "OPENROUTER_ALLOWED_MODELS"):
        monkeypatch.delenv(name, raising=False)
    model_restrictions._restriction_service = None
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()
    model_restrictions._restriction_service = None


def _registry(*provider_types: ProviderType) -> ModelProviderRegistry:
    """A registry of real provider classes with dummy keys (nothing here touches the network)."""

    registry = ModelProviderRegistry(config={"OPENAI_API_KEY": "k", "GEMINI_API_KEY": "k", "OPENROUTER_API_KEY": "k"})
    classes = {
        ProviderType.OPENAI: OpenAIModelProvider,
        ProviderType.GOOGLE: GeminiModelProvider,
        ProviderType.OPENROUTER: OpenRouterProvider,
    }
    for provider_type in provider_types:
        registry.register_provider(provider_type, classes[provider_type])
    return registry


@pytest.fixture
def full_registry() -> ModelProviderRegistry:
    registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)
    set_default_registry(registry)
    return registry


def _payload(error: ToolExecutionError) -> dict[str, Any]:
    return json.loads(error.payload)


class _Recorder:
    """Stands in for ``execute_batch_start`` and records what routing handed it."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def __call__(self, arguments: dict[str, Any], providers: Any) -> list[Any]:
        self.calls.append((dict(arguments), dict(providers)))
        return ["batch-started"]


def _chat_with_recorder(monkeypatch) -> tuple[ChatTool, _Recorder]:
    tool = ChatTool()
    recorder = _Recorder()
    monkeypatch.setattr(tool, "execute_batch_start", recorder)
    return tool, recorder


def _consensus_args(models: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "step": "Evaluate the proposal",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": "initial analysis",
        "models": models,
        **extra,
    }


# ----------------------------------------------------------------------
# 7.1 the execution_mode field and what it routes
# ----------------------------------------------------------------------


class TestExecutionModeField:
    def test_the_accepted_values_are_exactly_batch(self):
        assert ACCEPTED_EXECUTION_MODES == ("batch",)
        assert normalize_execution_mode("batch") == "batch"
        assert normalize_execution_mode(None) is None

    @pytest.mark.parametrize("junk", ["fast", "Batch", "BATCH", "", " batch", "sync", 1, True, ["batch"]])
    def test_an_unrecognized_value_is_rejected_naming_the_accepted_values(self, junk):
        with pytest.raises(ValueError) as excinfo:
            normalize_execution_mode(junk)

        message = str(excinfo.value)
        assert '"batch"' in message and "null" in message

    @pytest.mark.parametrize("tool_factory", [ChatTool, ConsensusTool])
    def test_the_published_schema_documents_the_field_its_tradeoff_and_accepts_null(self, tool_factory):
        schema = tool_factory().get_input_schema()
        prop = schema["properties"]["execution_mode"]

        assert prop["enum"] == ["batch", None] and prop["type"] == ["string", "null"]
        description = prop["description"].lower()
        assert "batch" in description and "less" in description and "longer" in description
        assert "may not be available within this call" in description
        assert prop == batch_schema_properties()["execution_mode"]
        assert "execution_mode" not in schema["required"]
        jsonschema.validate({"batch_handle": "h", "execution_mode": "batch"}, schema)
        jsonschema.validate({"batch_handle": "h", "execution_mode": None}, schema)
        with pytest.raises(jsonschema.ValidationError) as excinfo:
            jsonschema.validate({"batch_handle": "h", "execution_mode": "fast"}, schema)
        assert "batch" in excinfo.value.message  # the schema error names the accepted value too

    @pytest.mark.parametrize("junk", ["fast", "Batch", ""])
    def test_the_request_models_reject_an_unrecognized_value_naming_the_accepted_ones(self, junk):
        with pytest.raises(ValidationError) as chat_error:
            ChatRequest(prompt="p", working_directory_absolute_path="/tmp", execution_mode=junk)
        with pytest.raises(ValidationError) as consensus_error:
            ConsensusRequest(**_consensus_args([{"model": "a"}, {"model": "b"}], execution_mode=junk))

        for excinfo in (chat_error, consensus_error):
            text = " | ".join(error["msg"] for error in excinfo.value.errors())
            assert '"batch"' in text

    def test_the_field_stays_out_of_the_serialised_request(self):
        """A request that does not use it serialises exactly as it did before the field existed."""

        request = ChatRequest(prompt="p", working_directory_absolute_path="/tmp", execution_mode="batch")

        assert request.requests_batch is True
        assert "execution_mode" not in request.model_dump()
        assert ChatRequest(prompt="p", working_directory_absolute_path="/tmp").requests_batch is False

    @pytest.mark.asyncio
    async def test_a_tool_call_with_an_unrecognized_value_fails_naming_the_accepted_values(
        self, monkeypatch, full_registry
    ):
        tool, recorder = _chat_with_recorder(monkeypatch)

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute({"prompt": "p", "working_directory_absolute_path": "/tmp", "execution_mode": "fast"})

        payload = _payload(excinfo.value)
        assert payload["status"] == "error" and payload["metadata"]["batch_error"] == "invalid_execution_mode"
        assert '"batch"' in payload["content"]
        assert recorder.calls == []  # nothing was run, as a batch or otherwise

    @pytest.mark.asyncio
    async def test_through_the_server_boundary_a_typo_is_rejected_before_anything_runs(self, full_registry):
        from server import tool_registry

        handlers = build_handlers(tool_registry)
        result = await handlers.on_call_tool(
            None,
            CallToolRequestParams(
                name="chat",
                arguments={
                    "prompt": "p",
                    "working_directory_absolute_path": "/tmp",
                    "model": BATCHABLE_ALIAS,
                    "execution_mode": "fast",
                },
            ),
        )

        assert result.is_error is True
        assert "batch" in result.content[0].text


class TestBatchRouting:
    @pytest.mark.asyncio
    async def test_execution_mode_batch_routes_a_chat_call_to_the_batch_path_and_not_the_synchronous_one(
        self, monkeypatch, full_registry, tmp_path
    ):
        tool, recorder = _chat_with_recorder(monkeypatch)

        async def must_not_run(self_, arguments):
            raise AssertionError("the synchronous path ran for a batch request")

        monkeypatch.setattr("tools.simple.base.SimpleTool.execute", must_not_run)
        arguments = {
            "prompt": "p",
            "working_directory_absolute_path": str(tmp_path),
            "model": BATCHABLE_ALIAS,
            "execution_mode": "batch",
        }

        result = await tool.execute(arguments)

        assert result == ["batch-started"]
        assert len(recorder.calls) == 1
        seen_arguments, providers = recorder.calls[0]
        assert seen_arguments["execution_mode"] == "batch" and seen_arguments["model"] == BATCHABLE_ALIAS
        assert set(providers) == {BATCHABLE_ALIAS}
        assert providers[BATCHABLE_ALIAS].get_provider_type() is ProviderType.OPENROUTER

    @pytest.mark.asyncio
    @pytest.mark.parametrize("absent", [{}, {"execution_mode": None}])
    async def test_omitting_the_field_or_passing_null_keeps_the_synchronous_path(
        self, monkeypatch, full_registry, absent
    ):
        tool, recorder = _chat_with_recorder(monkeypatch)
        reached: list[dict[str, Any]] = []

        async def sentinel(self_, arguments):
            reached.append(arguments)
            return ["sync"]

        monkeypatch.setattr("tools.simple.base.SimpleTool.execute", sentinel)
        arguments = {"prompt": "p", "working_directory_absolute_path": "/tmp", "model": BATCHABLE_ALIAS, **absent}

        assert await tool.execute(arguments) == ["sync"]
        assert reached == [arguments] and recorder.calls == []

    @pytest.mark.asyncio
    async def test_a_consensus_request_selecting_batch_routes_to_the_batch_path(self, monkeypatch, full_registry):
        tool = ConsensusTool()
        recorder = _Recorder()
        monkeypatch.setattr(tool, "execute_batch_start", recorder)

        async def must_not_run(self_, arguments):
            raise AssertionError("the synchronous path ran for a batch request")

        monkeypatch.setattr("tools.workflow.base.WorkflowTool.execute", must_not_run)
        models = [{"model": "opus", "stance": "for"}, {"model": BATCHABLE_ALIAS, "stance": "against"}]

        assert await tool.execute(_consensus_args(models, execution_mode="batch")) == ["batch-started"]
        _, providers = recorder.calls[0]
        assert set(providers) == {"opus", BATCHABLE_ALIAS}

    @pytest.mark.asyncio
    async def test_a_tool_with_no_batch_submission_yet_refuses_instead_of_running_at_full_price(self, full_registry):
        """The mixin's default ``execute_batch_start`` is a loud refusal, never a quiet synchronous fallback.

        chat and consensus now implement ``execute_batch_start`` (sections 8 and 9), so the default seam is
        exercised through a minimal tool that, like any future batch-capable tool, has not implemented it yet.
        """

        ran_synchronously: list[dict[str, Any]] = []

        # The mixin's fallthrough is ``super().execute``: give it one so a leak would be visible.
        class _Base:
            async def execute(self, arguments: dict[str, Any]) -> list[Any]:
                ran_synchronously.append(arguments)
                return ["synchronous"]

        class _Tool(BatchResumableToolMixin, _Base):
            def get_name(self) -> str:
                return "seam-only"

        with pytest.raises(ToolExecutionError) as excinfo:
            await _Tool().execute({"model": BATCHABLE_ALIAS, "execution_mode": "batch"})

        payload = _payload(excinfo.value)
        assert payload["metadata"]["batch_error"] == "batch_start_not_implemented"
        assert "Nothing was submitted and nothing was run" in payload["content"]
        assert ran_synchronously == []

    def test_chat_and_consensus_implement_the_batch_start(self):
        """Neither tool relies on the refusing default any more."""

        for tool in (ChatTool, ConsensusTool):
            assert tool.execute_batch_start is not BatchResumableToolMixin.execute_batch_start

    @pytest.mark.asyncio
    async def test_a_handle_still_takes_precedence_over_execution_mode(self, monkeypatch, full_registry):
        tool, recorder = _chat_with_recorder(monkeypatch)
        seen: list[dict[str, Any]] = []

        async def control(arguments):
            seen.append(arguments)
            return ["resumed"]

        monkeypatch.setattr(tool, "execute_batch_control", control)

        assert await tool.execute({"batch_handle": "batch_x", "execution_mode": "batch"}) == ["resumed"]
        assert recorder.calls == [] and len(seen) == 1


class TestModelIdentifiersAreNeverReadForExecutionMode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("identifier", [f"{OPUS}:batch", f"{OPUS}:free", "openai/gpt-5:online"])
    async def test_a_suffixed_identifier_without_the_field_runs_synchronously_and_is_forwarded_byte_for_byte(
        self, identifier, tmp_path, monkeypatch
    ):
        """7.1 + 7.2: a ``:batch`` suffix is a provider's own identifier, never an execution hint."""

        registry = _registry(ProviderType.OPENROUTER)
        set_default_registry(registry)
        forwarded: list[str] = []

        def fake_generate(self, prompt, model_name, **kwargs):
            forwarded.append(model_name)
            return ModelResponse(
                content="synchronous answer",
                usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                model_name=model_name,
                friendly_name="OpenRouter",
                provider=ProviderType.OPENROUTER,
                metadata={},
            )

        monkeypatch.setattr(OpenRouterProvider, "generate_content", fake_generate)
        monkeypatch.setattr("handlers.tool_handlers.DEFAULT_MODEL", identifier)
        from server import tool_registry

        chat = tool_registry.get_tool_instance("chat")

        async def boom(arguments, providers):
            raise AssertionError("a model suffix must not select batch execution")

        monkeypatch.setattr(chat, "execute_batch_start", boom)
        monkeypatch.setattr(chat, "_batch_executor", SimpleNamespace(), raising=False)  # any use would fail

        handlers = build_handlers(tool_registry)
        result = await handlers.handle_call_tool(
            "chat",
            {"prompt": "hello", "working_directory_absolute_path": str(tmp_path), "model": identifier},
        )

        assert forwarded == [identifier]  # exactly as supplied: suffix intact, nothing consumed
        body = json.loads(result[0].text)
        assert body["status"] in ("success", "continuation_available")
        assert "synchronous answer" in body["content"]

    def test_the_batch_adapter_strips_the_suffix_only_from_what_it_submits_never_from_the_selection(
        self, full_registry
    ):
        """With the field set, the identifier is still looked up as supplied; the envelope slug is the adapter's concern."""

        provider = select_batch_provider(f"{OPUS}:batch", full_registry)

        assert provider.get_provider_type() is ProviderType.OPENROUTER
        assert provider.batch_submission_model(f"{OPUS}:batch") == OPUS == provider.batch_submission_model(OPUS)

    @pytest.mark.asyncio
    async def test_execution_intent_and_model_identity_are_independently_expressible(self, monkeypatch, full_registry):
        tool, recorder = _chat_with_recorder(monkeypatch)
        identifier = f"{OPUS}:batch"

        await tool.execute(
            {
                "prompt": "p",
                "working_directory_absolute_path": "/tmp",
                "model": identifier,
                "execution_mode": "batch",
            }
        )

        seen_arguments, providers = recorder.calls[0]
        assert seen_arguments["model"] == identifier  # untouched
        assert list(providers) == [identifier]  # keyed by the full identifier including the variant segment


# ----------------------------------------------------------------------
# 7.3 per-entry execution_mode on consensus models[]
# ----------------------------------------------------------------------


class TestConsensusEntryExecutionMode:
    def test_entries_accept_the_same_field_name_and_values_alongside_stance(self):
        request = ConsensusRequest(
            **_consensus_args(
                [
                    {"model": "opus", "stance": "for", "execution_mode": "batch"},  # batched, with a stance
                    {"model": "gpt5", "stance": "against"},  # unbatched
                    {"model": "flash", "execution_mode": None},  # explicit null: unbatched
                ]
            )
        )

        assert [m.get("execution_mode") for m in request.models] == ["batch", None, None]
        assert request.models[0]["stance"] == "for"

    def test_an_invalid_entry_value_names_the_entry_and_the_accepted_values(self):
        with pytest.raises(ValidationError) as excinfo:
            ConsensusRequest(**_consensus_args([{"model": "a"}, {"model": "b", "execution_mode": "slow"}]))

        text = " | ".join(error["msg"] for error in excinfo.value.errors())
        assert "models[1].execution_mode" in text and '"batch"' in text

    def test_the_published_entry_schema_carries_the_field(self):
        items = ConsensusTool().get_input_schema()["properties"]["models"]["items"]["properties"]

        assert items["execution_mode"] == batch_schema_properties()["execution_mode"]
        assert {"model", "stance", "stance_prompt", "execution_mode"} <= set(items)

    def test_entry_modes_fall_back_to_the_request_level_value(self, monkeypatch):
        tool = ConsensusTool()
        models = [{"model": "a", "execution_mode": "batch"}, {"model": "b"}, {"model": "c", "execution_mode": None}]

        assert tool.entry_execution_modes({"models": models}) == ["batch", None, None]
        assert tool.entry_execution_modes({"models": models, "execution_mode": "batch"}) == ["batch"] * 3

    @pytest.mark.asyncio
    async def test_one_batched_entry_routes_the_call_to_batch_and_only_that_entry_is_admitted(
        self, monkeypatch, full_registry
    ):
        """A mixed roster: the synchronous entry's model need not be batchable (it is not batched)."""

        tool = ConsensusTool()
        recorder = _Recorder()
        monkeypatch.setattr(tool, "execute_batch_start", recorder)
        models = [
            {"model": "opus", "stance": "for", "execution_mode": "batch"},
            {"model": NOT_BATCHABLE, "stance": "against"},  # synchronous entry: never asked to batch
        ]

        await tool.execute(_consensus_args(models))

        _, providers = recorder.calls[0]
        assert list(providers) == ["opus"]  # the unbatched entry's model was not even looked up for batch
        assert providers["opus"].get_provider_type() is ProviderType.OPENROUTER

    @pytest.mark.asyncio
    async def test_no_entry_selecting_batch_keeps_the_synchronous_path(self, monkeypatch, full_registry):
        tool = ConsensusTool()
        recorder = _Recorder()
        monkeypatch.setattr(tool, "execute_batch_start", recorder)
        reached: list[Any] = []

        async def sentinel(self_, arguments):
            reached.append(arguments)
            return ["sync"]

        monkeypatch.setattr("tools.workflow.base.WorkflowTool.execute", sentinel)
        models = [{"model": "opus"}, {"model": "gpt5", "execution_mode": None}]

        assert await tool.execute(_consensus_args(models)) == ["sync"]
        assert recorder.calls == [] and len(reached) == 1

    @pytest.mark.asyncio
    async def test_a_batched_entry_whose_model_cannot_batch_is_rejected_by_name(self, monkeypatch, full_registry):
        tool = ConsensusTool()
        recorder = _Recorder()
        monkeypatch.setattr(tool, "execute_batch_start", recorder)
        models = [{"model": "opus"}, {"model": NOT_BATCHABLE, "execution_mode": "batch"}]

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(_consensus_args(models))

        content = _payload(excinfo.value)["content"]
        assert NOT_BATCHABLE in content and "without execution_mode" in content
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_an_entry_keeps_its_model_identifier_unchanged(self, monkeypatch, full_registry):
        tool = ConsensusTool()
        recorder = _Recorder()
        monkeypatch.setattr(tool, "execute_batch_start", recorder)
        identifier = f"{OPUS}:batch"

        await tool.execute(_consensus_args([{"model": identifier, "execution_mode": "batch"}, {"model": "opus"}]))

        seen_arguments, providers = recorder.calls[0]
        assert seen_arguments["models"][0]["model"] == identifier
        assert list(providers) == [identifier]

    def test_the_stance_never_comes_from_the_model_string_or_the_mode(self):
        request = ConsensusRequest(
            **_consensus_args([{"model": "opus", "execution_mode": "batch"}, {"model": "gpt5", "stance": "against"}])
        )

        assert request.models[0].get("stance", "neutral") == "neutral"  # not influenced by the mode


# ----------------------------------------------------------------------
# 7.4 a model that cannot batch is rejected, never silently run at full price
# ----------------------------------------------------------------------


class TestUnbatchableModels:
    @pytest.mark.asyncio
    async def test_the_error_names_the_model_and_points_at_the_call_without_execution_mode(
        self, monkeypatch, full_registry
    ):
        tool, recorder = _chat_with_recorder(monkeypatch)

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(
                {
                    "prompt": "p",
                    "working_directory_absolute_path": "/tmp",
                    "model": NOT_BATCHABLE,
                    "execution_mode": "batch",
                }
            )

        payload = _payload(excinfo.value)
        assert payload["metadata"]["batch_error"] == "batch_unsupported_model"
        assert f"'{NOT_BATCHABLE}'" in payload["content"]
        assert "without execution_mode" in payload["content"]
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_a_model_no_provider_serves_is_reported_as_unavailable_not_as_unbatchable(
        self, monkeypatch, full_registry
    ):
        tool, recorder = _chat_with_recorder(monkeypatch)

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(
                {
                    "prompt": "p",
                    "working_directory_absolute_path": "/tmp",
                    "model": "no-such-model-anywhere",
                    "execution_mode": "batch",
                }
            )

        payload = _payload(excinfo.value)
        assert payload["metadata"]["batch_error"] == "model_unavailable"
        assert "no-such-model-anywhere" in payload["content"] and recorder.calls == []

    @pytest.mark.asyncio
    async def test_a_batch_request_without_any_model_is_refused_with_the_way_out(self, monkeypatch, full_registry):
        tool, recorder = _chat_with_recorder(monkeypatch)

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute({"prompt": "p", "working_directory_absolute_path": "/tmp", "execution_mode": "batch"})

        assert _payload(excinfo.value)["metadata"]["batch_error"] == "model_required"
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_the_boundary_also_refuses_it_by_name_end_to_end(self, full_registry):
        from server import tool_registry

        handlers = build_handlers(tool_registry)
        with pytest.raises(ToolExecutionError) as excinfo:
            await handlers.handle_call_tool(
                "chat",
                {
                    "prompt": "p",
                    "working_directory_absolute_path": "/tmp",
                    "model": NOT_BATCHABLE,
                    "execution_mode": "batch",
                },
            )

        content = _payload(excinfo.value)["content"]
        assert f"'{NOT_BATCHABLE}'" in content and "without execution_mode" in content

    def test_no_registry_at_all_is_a_clear_error(self):
        tool = ChatTool()
        tool.get_batch_registry = lambda: (_ for _ in ()).throw(RuntimeError("Registry not initialized"))  # type: ignore[method-assign]

        with pytest.raises(ToolExecutionError) as excinfo:
            tool.admit_batch_request({"model": "opus"})

        assert _payload(excinfo.value)["metadata"]["batch_error"] == "registry_unavailable"


# ----------------------------------------------------------------------
# 7.5 non-text content is refused before submission
# ----------------------------------------------------------------------


class _SpyRegistry:
    """Fails the test if provider selection is reached at all."""

    def get_provider_for_model(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a provider was consulted before the non-text check")


class TestNonTextContent:
    def test_names_of_non_text_keys(self):
        assert non_text_argument_names({"images": ["/a.png"], "audio": [], "video": ["v"], "prompt": "p"}) == [
            "images",
            "video",
        ]
        # Local text files are not non-text content.
        assert non_text_argument_names({"absolute_file_paths": ["/a.py"], "relevant_files": ["/b.py"]}) == []

    def test_the_rejection_names_the_content_and_the_synchronous_alternative(self):
        with pytest.raises(BatchRequestRejected) as excinfo:
            reject_non_text_arguments("chat", {"images": ["/a.png"]})

        message = str(excinfo.value)
        assert "text-only" in message and "'images'" in message
        assert "without execution_mode" in message and "synchronously" in message
        assert "absolute_file_paths" in message  # and says local files are fine

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["images", "audio", "video", "attachments", "documents"])
    async def test_chat_refuses_before_any_provider_is_consulted(self, monkeypatch, key):
        tool, recorder = _chat_with_recorder(monkeypatch)
        monkeypatch.setattr(tool, "get_batch_registry", lambda: _SpyRegistry())

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(
                {
                    "prompt": "p",
                    "working_directory_absolute_path": "/tmp",
                    "model": "opus",
                    "execution_mode": "batch",
                    key: ["/tmp/x.bin"],
                }
            )

        payload = _payload(excinfo.value)
        assert payload["metadata"]["batch_error"] == "non_text_content"
        assert "text-only" in payload["content"] and f"'{key}'" in payload["content"]
        assert recorder.calls == []  # no submission was attempted

    @pytest.mark.asyncio
    async def test_consensus_refuses_images_before_any_provider_is_consulted(self, monkeypatch):
        tool = ConsensusTool()
        recorder = _Recorder()
        monkeypatch.setattr(tool, "execute_batch_start", recorder)
        monkeypatch.setattr(tool, "get_batch_registry", lambda: _SpyRegistry())

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(
                _consensus_args([{"model": "opus", "execution_mode": "batch"}, {"model": "gpt5"}], images=["/a.png"])
            )

        assert _payload(excinfo.value)["metadata"]["batch_error"] == "non_text_content"
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_local_files_read_into_the_prompt_as_text_are_accepted(self, monkeypatch, full_registry, tmp_path):
        source = tmp_path / "module.py"
        source.write_text("print('hi')\n")
        tool, recorder = _chat_with_recorder(monkeypatch)

        await tool.execute(
            {
                "prompt": "review this",
                "working_directory_absolute_path": str(tmp_path),
                "absolute_file_paths": [str(source)],
                "images": [],  # an empty list carries no content
                "model": "opus",
                "execution_mode": "batch",
            }
        )

        assert len(recorder.calls) == 1  # accepted and handed on for submission

    @pytest.mark.asyncio
    async def test_the_synchronous_call_with_images_is_not_touched(self, monkeypatch, full_registry):
        tool, recorder = _chat_with_recorder(monkeypatch)
        reached: list[dict[str, Any]] = []

        async def sentinel(self_, arguments):
            reached.append(arguments)
            return ["sync"]

        monkeypatch.setattr("tools.simple.base.SimpleTool.execute", sentinel)
        arguments = {"prompt": "p", "working_directory_absolute_path": "/tmp", "model": "opus", "images": ["/a.png"]}

        assert await tool.execute(arguments) == ["sync"]
        assert reached[0]["images"] == ["/a.png"] and recorder.calls == []


# ----------------------------------------------------------------------
# 7.6 clink (and the other tools with no batch path)
# ----------------------------------------------------------------------


class TestExecutionModeOnToolsWithNoBatchPath:
    CLINK_TEXT = "server-side batch execution does not apply to CLI bridge tools"

    @pytest.mark.asyncio
    async def test_clink_rejects_execution_mode_with_the_cli_bridge_reason(self):
        with pytest.raises(ToolExecutionError) as excinfo:
            await CLinkTool().execute({"prompt": "hi", "cli_name": "opencode", "execution_mode": "batch"})

        assert self.CLINK_TEXT in _payload(excinfo.value)["content"]

    @pytest.mark.asyncio
    async def test_clink_rejects_it_on_a_job_poll_too(self):
        with pytest.raises(ToolExecutionError) as excinfo:
            await CLinkTool().execute({"job_id": "job_1", "execution_mode": "batch"})

        assert self.CLINK_TEXT in _payload(excinfo.value)["content"]

    @pytest.mark.asyncio
    async def test_through_the_mcp_boundary_the_reason_replaces_the_generic_schema_error(self):
        from server import tool_registry

        handlers = build_handlers(tool_registry)
        result = await handlers.on_call_tool(
            None,
            CallToolRequestParams(name="clink", arguments={"prompt": "hi", "execution_mode": "batch"}),
        )

        assert result.is_error is True
        text = result.content[0].text
        assert self.CLINK_TEXT in text and "Additional properties" not in text and "Input validation" not in text

    @pytest.mark.asyncio
    async def test_through_handle_call_tool_it_is_rejected_before_any_state_is_written(self):
        from server import tool_registry

        handlers = build_handlers(tool_registry)
        with pytest.raises(ToolExecutionError) as excinfo:
            await handlers.handle_call_tool(
                "clink", {"prompt": "hi", "execution_mode": "batch", "continuation_id": "no-such-thread"}
            )

        # The thread lookup would have failed first had the gate not run ahead of it.
        assert self.CLINK_TEXT in _payload(excinfo.value)["content"]

    def test_null_and_absent_are_not_rejected(self):
        assert execution_mode_rejection("clink", {}, accepts_execution_mode=False) is None
        assert execution_mode_rejection("clink", {"execution_mode": None}, accepts_execution_mode=False) is None
        assert execution_mode_rejection("chat", {"execution_mode": "batch"}, accepts_execution_mode=True) is None

    def test_the_clink_schema_does_not_publish_the_field(self):
        assert "execution_mode" not in CLinkTool().get_input_schema()["properties"]

    def test_model_strings_still_forward_to_the_cli_verbatim(self):
        from clink import get_registry
        from clink.agents.opencode import OpencodeAgent

        agent = OpencodeAgent(get_registry().get_client("opencode"))
        suffixed = "openrouter/openai/gpt-6-astra-pro:batch"

        assert agent.render_model_args(suffixed) == ["-m", suffixed]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_factory",
        [
            AnalyzeTool,
            CodeReviewTool,
            DebugIssueTool,
            DocgenTool,
            PlannerTool,
            PrecommitTool,
            RefactorTool,
            TestGenTool,
            ThinkDeepTool,
        ],
    )
    async def test_workflow_tools_other_than_consensus_reject_it_without_reaching_their_workflow(
        self, tool_factory, monkeypatch
    ):
        tool = tool_factory()

        async def must_not_run(arguments):
            raise AssertionError("the workflow ran for a batch request")

        monkeypatch.setattr(tool, "execute_workflow", must_not_run)
        assert not isinstance(tool, BatchResumableToolMixin)

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute({"step": "s", "step_number": 1, "execution_mode": "batch", "model": "opus"})

        payload = _payload(excinfo.value)
        assert payload["metadata"]["batch_error"] == "execution_mode_unsupported"
        assert tool.get_name() in payload["content"] and "chat and consensus" in payload["content"]

    @pytest.mark.parametrize(
        "tool_factory",
        [
            AnalyzeTool,
            ChallengeTool,
            CodeReviewTool,
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
        ],
    )
    def test_no_other_tool_schema_publishes_execution_mode(self, tool_factory):
        """Only chat and consensus carry the field; the byte-for-byte snapshot test covers the rest of each schema."""

        assert "execution_mode" not in tool_factory().get_input_schema().get("properties", {})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["codereview", "analyze", "thinkdeep", "planner", "challenge", "listmodels"])
    async def test_the_boundary_rejects_it_for_every_tool_that_has_no_batch_path(self, name):
        from server import tool_registry

        handlers = build_handlers(tool_registry)
        result = await handlers.on_call_tool(
            None, CallToolRequestParams(name=name, arguments={"execution_mode": "batch"})
        )

        assert result.is_error is True
        payload = json.loads(result.content[0].text)
        assert payload["metadata"]["batch_error"] == "execution_mode_unsupported"
        assert f"'{name}'" in payload["content"]

    @pytest.mark.asyncio
    async def test_null_reaches_the_tool_normally(self, monkeypatch):
        from server import tool_registry

        tool = tool_registry.get_tool_instance("codereview")
        reached: list[dict[str, Any]] = []

        async def sentinel(arguments):
            reached.append(arguments)
            return ["ran"]

        monkeypatch.setattr(tool, "execute_workflow", sentinel)
        handlers = build_handlers(tool_registry)

        # A null execution_mode on a tool that has no batch path is a synchronous call, not an error.
        assert await tool.execute({"step": "s", "execution_mode": None}) == ["ran"]
        assert handlers is not None and len(reached) == 1


# ----------------------------------------------------------------------
# 7.7 batch-aware provider selection
# ----------------------------------------------------------------------


class TestBatchAwareProviderSelection:
    def test_an_alias_served_by_a_native_provider_and_openrouter_resolves_to_openrouter_under_batch(self):
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)

        native = registry.get_provider_for_model(BATCHABLE_ALIAS)
        batching = registry.get_provider_for_model(BATCHABLE_ALIAS, require_batch=True)

        assert native is not None and native.get_provider_type() is ProviderType.OPENAI  # the preferred one
        assert batching is not None and batching.get_provider_type() is ProviderType.OPENROUTER

    def test_synchronous_selection_is_unchanged(self):
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)

        for model, expected in [
            (BATCHABLE_ALIAS, ProviderType.OPENAI),
            ("o3", ProviderType.OPENAI),
            (SYNC_ONLY_NATIVE, ProviderType.GOOGLE),
            ("opus", ProviderType.OPENROUTER),
            ("grok", ProviderType.OPENROUTER),
        ]:
            assert registry.get_provider_for_model(model).get_provider_type() is expected, model
            # require_batch left at False: the default, spelled out, behaves identically
            assert registry.get_provider_for_model(model, require_batch=False).get_provider_type() is expected

        assert registry.get_provider_for_model("no-such-model") is None

    def test_a_model_no_provider_can_batch_still_fails(self):
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)

        assert registry.get_provider_for_model(NOT_BATCHABLE) is not None  # it is served...
        assert registry.get_provider_for_model(NOT_BATCHABLE, require_batch=True) is None  # ...but nobody batches it
        assert registry.get_provider_for_model("llama", require_batch=True) is None  # OpenRouter listing, no batch
        assert registry.get_provider_for_model("no-such-model", require_batch=True) is None

    def test_priority_order_still_decides_between_two_providers_that_can_both_batch(self):
        registry = ModelProviderRegistry(config={})

        class _Batcher:
            def __init__(self, kind: ProviderType, batches: bool) -> None:
                self.kind, self._batches = kind, batches

            def validate_model_name(self, name: str) -> bool:
                return name == "shared"

            def get_capabilities(self, name: str) -> Any:
                return SimpleNamespace(supports_batch=self._batches)

            def get_provider_type(self) -> ProviderType:
                return self.kind

        first, second, third = (
            _Batcher(ProviderType.OPENAI, False),
            _Batcher(ProviderType.DIAL, True),
            _Batcher(ProviderType.OPENROUTER, True),
        )
        for provider in (first, second, third):
            registry._providers[provider.kind] = type(provider)
            registry._initialized_providers[provider.kind] = provider  # type: ignore[assignment]

        assert registry.get_provider_for_model("shared") is first
        assert registry.get_provider_for_model("shared", require_batch=True) is second  # DIAL precedes OPENROUTER

    def test_a_provider_whose_capabilities_cannot_be_read_does_not_claim_to_batch(self):
        class _Broken:
            def get_capabilities(self, name: str) -> Any:
                raise ValueError("unsupported model")

        assert ModelProviderRegistry.provider_can_batch(_Broken(), "x") is False  # type: ignore[arg-type]

    def test_select_batch_provider_picks_the_batching_provider_for_the_alias(self, full_registry):
        provider = select_batch_provider("  gpt5 ", full_registry)  # whitespace trimmed, nothing else touched

        assert provider.get_provider_type() is ProviderType.OPENROUTER

    def test_select_batch_provider_rejects_the_unbatchable_naming_the_model_and_the_way_out(self, full_registry):
        with pytest.raises(BatchRequestRejected) as excinfo:
            select_batch_provider(NOT_BATCHABLE, full_registry)

        assert excinfo.value.code == "batch_unsupported_model"
        assert f"'{NOT_BATCHABLE}'" in str(excinfo.value) and "without execution_mode" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_the_alias_end_to_end_reaches_openrouter_not_the_native_provider(self, monkeypatch, full_registry):
        tool, recorder = _chat_with_recorder(monkeypatch)

        await tool.execute(
            {
                "prompt": "p",
                "working_directory_absolute_path": "/tmp",
                "model": BATCHABLE_ALIAS,
                "execution_mode": "batch",
            }
        )

        provider = recorder.calls[0][1][BATCHABLE_ALIAS]
        assert provider.get_provider_type() is ProviderType.OPENROUTER
        assert (
            provider.batch_submission_model(BATCHABLE_ALIAS) == "openai/gpt-5.5"
        )  # 14.13: the latest batch-listed GPT-5


# ----------------------------------------------------------------------
# 7.4 / 7.7 admission agrees with the adapter: variant identifiers are rejected up front
# ----------------------------------------------------------------------

#: Provider variant identifiers. ``_lookup_capabilities`` lends each its base model's ``supports_batch=True``,
#: but the adapter refuses them at submit (the batch API takes the model's normal slug).
VARIANTS = [
    f"{OPUS}:free",
    f"{OPUS}:online",
    f"{OPUS}:thinking",
    f"{OPUS}:batch:batch",
    "openai/gpt-5:free",
    "openai/gpt-5:online",
]
#: Spellings the adapter does accept: the normal slug, an alias, and a single trailing ``:batch``.
ADMISSIBLE = [OPUS, f"{OPUS}:batch", "opus", BATCHABLE_ALIAS]


class TestAdmissionAgreesWithTheAdapter:
    @pytest.mark.parametrize("identifier", VARIANTS)
    def test_the_capability_flag_alone_would_have_waved_a_variant_through(self, identifier):
        """The premise of the defect: the flag says yes, so a flag-only check cannot be what admission uses."""

        assert OpenRouterProvider("k").get_capabilities(identifier).supports_batch is True

    @pytest.mark.parametrize("identifier", VARIANTS)
    def test_registry_selection_under_batch_skips_a_variant_the_adapter_would_refuse(self, identifier):
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)

        assert registry.get_provider_for_model(identifier) is not None  # it is served synchronously...
        assert registry.get_provider_for_model(identifier, require_batch=True) is None  # ...and is not batchable

    @pytest.mark.parametrize("identifier", ADMISSIBLE)
    def test_what_the_adapter_accepts_is_still_admitted(self, identifier):
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)

        provider = registry.get_provider_for_model(identifier, require_batch=True)

        assert provider is not None and provider.get_provider_type() is ProviderType.OPENROUTER

    @pytest.mark.parametrize("identifier", VARIANTS)
    def test_select_batch_provider_rejects_a_variant_naming_it_and_the_synchronous_way_out(
        self, identifier, full_registry
    ):
        with pytest.raises(BatchRequestRejected) as excinfo:
            select_batch_provider(identifier, full_registry)

        assert excinfo.value.code == "batch_unsupported_model"
        assert f"'{identifier}'" in str(excinfo.value) and "without execution_mode" in str(excinfo.value)

    @pytest.mark.parametrize("identifier", [*VARIANTS, *ADMISSIBLE])
    def test_admission_and_submission_never_disagree(self, identifier, monkeypatch):
        """``can_batch_model`` is True exactly when ``submit_batch`` gets as far as the network."""

        import httpx

        from providers.shared import BatchRequest

        provider = OpenRouterProvider("k")
        sent: list[tuple[str, str]] = []

        def send(method: str, path: str, content: str | None = None) -> httpx.Response:
            sent.append((method, path))
            return httpx.Response(202, json={"id": "batch_1", "status": "validating"})

        monkeypatch.setattr(provider, "_batch_send", send)
        admitted = provider.can_batch_model(identifier)

        try:
            provider.submit_batch(identifier, [BatchRequest(custom_id="a", prompt="p")])
            refused = False
        except ValueError:
            refused = True

        assert admitted is not refused
        assert bool(sent) is admitted  # refused locally means nothing was sent

    @pytest.mark.parametrize("native", [GeminiModelProvider, OpenAIModelProvider])
    def test_a_provider_without_a_batch_adapter_reports_it_cannot_batch(self, native):
        provider = native("k")

        assert provider.can_batch_model("gpt5" if native is OpenAIModelProvider else "flash") is False
        assert provider.can_batch_model("no-such-model") is False

    @pytest.mark.asyncio
    async def test_chat_refuses_a_variant_up_front_and_submits_nothing(self, monkeypatch, full_registry):
        tool, recorder = _chat_with_recorder(monkeypatch)

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(
                {
                    "prompt": "p",
                    "working_directory_absolute_path": "/tmp",
                    "model": "openai/gpt-5:free",
                    "execution_mode": "batch",
                }
            )

        content = _payload(excinfo.value)["content"]
        assert "openai/gpt-5:free" in content and "without execution_mode" in content
        assert recorder.calls == []  # it never reached the batch-start seam

    @pytest.mark.asyncio
    async def test_a_consensus_roster_with_one_unbatchable_variant_admits_none_of_its_entries(
        self, monkeypatch, full_registry
    ):
        """The stranded-batch scenario: the first entry would have been submitted before the second was refused."""

        tool = ConsensusTool()
        recorder = _Recorder()
        monkeypatch.setattr(tool, "execute_batch_start", recorder)
        models = [{"model": OPUS, "execution_mode": "batch"}, {"model": "openai/gpt-5:free", "execution_mode": "batch"}]

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(_consensus_args(models))

        assert "openai/gpt-5:free" in _payload(excinfo.value)["content"]
        assert recorder.calls == []  # not even the admissible entry reached the batch-start seam


# ----------------------------------------------------------------------
# 7.8 the MCP boundary does not gate a resumption on model resolution
# ----------------------------------------------------------------------


@pytest.fixture
def world(tmp_path):
    clock = FakeClock()
    remote = FakeRemote()
    store = BatchStore(str(tmp_path / "durable" / "batches.db"))
    state = SimpleNamespace(clock=clock, remote=remote, store=store, provider=FakeProvider(remote))
    state.executor = make_executor(store, {"openrouter": state.provider}, clock)
    yield state
    store.close()


UNSERVED_DEFAULT = "model-nobody-serves"


@pytest.fixture
def server_with_no_provider_for_the_default(monkeypatch, world):
    """A server whose default model has no provider at all: resolving it can only fail."""

    # Other providers exist (and could serve other models); none serves the default.
    registry = _registry(ProviderType.OPENAI)
    set_default_registry(registry)
    monkeypatch.setattr("handlers.tool_handlers.DEFAULT_MODEL", UNSERVED_DEFAULT)
    monkeypatch.setattr("config.DEFAULT_MODEL", UNSERVED_DEFAULT)
    from server import tool_registry

    chat = tool_registry.get_tool_instance("chat")
    consensus = tool_registry.get_tool_instance("consensus")
    monkeypatch.setattr(chat, "_batch_executor", world.executor, raising=False)
    monkeypatch.setattr(consensus, "_batch_executor", world.executor, raising=False)
    return SimpleNamespace(handlers=build_handlers(tool_registry), registry=registry, chat=chat, consensus=consensus)


def _submit(world, tool_name: str = "chat") -> str:
    if tool_name == "consensus":
        return world.executor.submit_and_wait(
            "consensus",
            consensus_plans(world.provider, OPUS),
            wait_seconds=0,
            payload=consensus_payload(OPUS),
            fetch=False,
        ).handle
    return world.executor.submit_and_wait(
        tool_name,
        [make_plan(world.provider, OPUS)],
        wait_seconds=0,
        payload={"working_directory_absolute_path": "/tmp", "model": OPUS},
    ).handle


class TestResumptionBypassesBoundaryResolution:
    """Integration through the real ``handle_call_tool`` entry on a server with NO provider for the default model.

    In-process, with the provider registry cleared, rather than a spawned server: the boundary code under test
    is the same ``handle_call_tool`` the MCP adapters call.
    """

    @pytest.mark.asyncio
    async def test_a_handle_only_call_resumes_with_no_provider_for_the_default_model(
        self, server_with_no_provider_for_the_default, world
    ):
        handle = _submit(world)

        result = await server_with_no_provider_for_the_default.handlers.handle_call_tool(
            "chat", {"batch_handle": handle}
        )

        body = json.loads(json.loads(result[0].text)["content"])
        assert body["status"] == "running" and body["batch_handle"] == handle
        assert "not available" not in result[0].text
        assert world.provider.calls.count("submit_batch") == 1  # the original submission only

    @pytest.mark.asyncio
    async def test_the_same_server_still_fails_a_call_that_starts_work_proving_resolution_was_what_blocked_it(
        self, server_with_no_provider_for_the_default
    ):
        with pytest.raises(ToolExecutionError) as excinfo:
            await server_with_no_provider_for_the_default.handlers.handle_call_tool(
                "chat", {"prompt": "hello", "working_directory_absolute_path": "/tmp"}
            )

        content = _payload(excinfo.value)["content"]
        assert UNSERVED_DEFAULT in content and "not available with current API keys" in content

    @pytest.mark.asyncio
    async def test_resolution_validation_allocation_and_size_checks_do_not_run_for_a_resumption(
        self, server_with_no_provider_for_the_default, world, tmp_path
    ):
        handle = _submit(world)
        lookups: list[str] = []
        sizes: list[Any] = []
        contexts: list[str] = []
        real_lookup = ModelProviderRegistry.get_provider_for_model

        def counting_lookup(self, model_name, **kwargs):
            lookups.append(model_name)
            return real_lookup(self, model_name, **kwargs)

        import utils.model_context as model_context_module

        real_context = model_context_module.ModelContext

        class CountingContext(real_context):  # type: ignore[misc, valid-type]
            def __init__(self, model_name, *args, **kwargs):
                contexts.append(model_name)
                super().__init__(model_name, *args, **kwargs)

        with (
            patch.object(ModelProviderRegistry, "get_provider_for_model", counting_lookup),
            patch("utils.model_context.ModelContext", CountingContext),
            patch("utils.file_utils.check_total_file_size", lambda *a, **k: sizes.append(a)),
        ):
            await server_with_no_provider_for_the_default.handlers.handle_call_tool(
                "chat",
                {"batch_handle": handle, "absolute_file_paths": [str(tmp_path / "huge.bin")]},
            )

        assert lookups == [] and contexts == [] and sizes == []

    @pytest.mark.asyncio
    async def test_a_handle_with_start_work_inputs_still_lets_the_handle_win_and_resubmits_nothing(
        self, server_with_no_provider_for_the_default, world, tmp_path
    ):
        handle = _submit(world)
        submissions_before = world.remote.submissions

        result = await server_with_no_provider_for_the_default.handlers.handle_call_tool(
            "chat",
            {
                "batch_handle": handle,
                "prompt": "a brand new question",
                "working_directory_absolute_path": str(tmp_path),
                "model": "gemini-2.5-flash",
            },
        )

        body = json.loads(json.loads(result[0].text)["content"])
        assert body["status"] == "running" and body["batch_handle"] == handle
        assert "prompt" in body["ignored_inputs"]  # reported as ignored, the tool's own precedence
        assert world.remote.submissions == submissions_before  # nothing was submitted

    @pytest.mark.asyncio
    async def test_cancel_by_handle_works_without_a_provider_for_the_default(
        self, server_with_no_provider_for_the_default, world
    ):
        handle = _submit(world)

        result = await server_with_no_provider_for_the_default.handlers.handle_call_tool(
            "chat", {"batch_handle": handle, "cancel": True}
        )

        body = json.loads(json.loads(result[0].text)["content"])
        assert body["status"] == "cancelled"
        assert "still be billed" in body["message"] or "billed" in body["message"]

    @pytest.mark.asyncio
    async def test_consensus_resumption_is_unaffected(self, server_with_no_provider_for_the_default, world):
        handle = _submit(world, "consensus")

        result = await server_with_no_provider_for_the_default.handlers.handle_call_tool(
            "consensus", {"batch_handle": handle}
        )

        assert json.loads(json.loads(result[0].text)["content"])["status"] == "running"

    @pytest.mark.asyncio
    async def test_an_unknown_handle_is_still_reported_as_unknown_by_the_tool(
        self, server_with_no_provider_for_the_default
    ):
        with pytest.raises(ToolExecutionError) as excinfo:
            await server_with_no_provider_for_the_default.handlers.handle_call_tool(
                "chat", {"batch_handle": "batch_does_not_exist"}
            )

        assert _payload(excinfo.value)["metadata"]["batch_error"] == "unknown_handle"

    @pytest.mark.asyncio
    async def test_a_blank_handle_is_not_a_resumption_and_resolves_as_before(
        self, server_with_no_provider_for_the_default
    ):
        with pytest.raises(ToolExecutionError) as excinfo:
            await server_with_no_provider_for_the_default.handlers.handle_call_tool(
                "chat", {"batch_handle": "   ", "prompt": "p", "working_directory_absolute_path": "/tmp"}
            )

        assert "not available with current API keys" in _payload(excinfo.value)["content"]

    @pytest.mark.asyncio
    async def test_the_adapter_path_resumes_through_on_call_tool_on_such_a_server(
        self, server_with_no_provider_for_the_default, world
    ):
        """The full MCP adapter (schema validation included), not just the legacy-shaped impl."""

        handle = _submit(world)

        result = await server_with_no_provider_for_the_default.handlers.on_call_tool(
            None, CallToolRequestParams(name="chat", arguments={"batch_handle": handle})
        )

        assert result.is_error is False
        assert json.loads(json.loads(result.content[0].text)["content"])["status"] == "running"


class TestNullExecutionModeAtTheMcpBoundary:
    """``null`` means synchronous on every tool, including those whose closed schema does not list the field."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["listmodels", "version"])
    async def test_a_tool_that_does_not_publish_the_field_accepts_a_null_exactly_like_its_absence(self, name):
        from server import tool_registry

        handlers = build_handlers(tool_registry)

        with_null = await handlers.on_call_tool(
            None, CallToolRequestParams(name=name, arguments={"execution_mode": None})
        )
        without = await handlers.on_call_tool(None, CallToolRequestParams(name=name, arguments={}))

        assert with_null.is_error is False, with_null.content[0].text
        assert without.is_error is False

    @pytest.mark.asyncio
    async def test_clink_receives_the_call_without_the_null_field(self, monkeypatch):
        from mcp.types import TextContent

        from server import tool_registry

        tool = tool_registry.get_tool_instance("clink")
        reached: list[dict[str, Any]] = []

        async def sentinel(arguments):
            reached.append(dict(arguments))
            return [TextContent(type="text", text="ran")]

        monkeypatch.setattr(tool, "execute", sentinel)
        handlers = build_handlers(tool_registry)

        result = await handlers.on_call_tool(
            None,
            CallToolRequestParams(
                name="clink", arguments={"prompt": "hi", "working_dir": "/tmp", "execution_mode": None}
            ),
        )

        assert result.is_error is False, result.content[0].text
        assert len(reached) == 1 and "execution_mode" not in reached[0]

    @pytest.mark.asyncio
    async def test_the_legacy_shaped_entry_drops_the_null_too(self, monkeypatch):
        """``handle_call_tool`` is also called directly (not only through ``on_call_tool``), so it strips on its own."""

        from server import tool_registry

        tool = tool_registry.get_tool_instance("clink")
        reached: list[dict[str, Any]] = []

        async def sentinel(arguments):
            reached.append(dict(arguments))
            return ["ran"]

        monkeypatch.setattr(tool, "execute", sentinel)
        handlers = build_handlers(tool_registry)

        await handlers.handle_call_tool("clink", {"prompt": "hi", "execution_mode": None})

        assert len(reached) == 1 and "execution_mode" not in reached[0]

    @pytest.mark.asyncio
    async def test_a_non_null_value_is_still_refused_with_its_reason(self):
        from server import tool_registry

        handlers = build_handlers(tool_registry)
        result = await handlers.on_call_tool(
            None, CallToolRequestParams(name="listmodels", arguments={"execution_mode": "batch"})
        )

        assert result.is_error is True
        assert json.loads(result.content[0].text)["metadata"]["batch_error"] == "execution_mode_unsupported"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["chat", "consensus"])
    async def test_the_batch_capable_tools_still_receive_the_null_untouched(self, name, monkeypatch):
        """They publish the field, so a null is theirs to read (as 'absent'); the boundary must not rewrite it."""

        from server import tool_registry

        tool = tool_registry.get_tool_instance(name)
        reached: list[dict[str, Any]] = []

        async def sentinel(arguments):
            reached.append(dict(arguments))
            return ["ran"]

        monkeypatch.setattr(tool, "execute", sentinel)
        handlers = build_handlers(tool_registry)
        arguments = {"batch_handle": "h", "execution_mode": None}  # a handle: no model resolution in the way

        await handlers.handle_call_tool(name, arguments)

        assert reached == [arguments]


class TestTheHandleBypassIsLimitedToBatchCapableTools:
    @pytest.mark.asyncio
    async def test_a_tool_with_no_batch_path_is_not_bypassed_by_a_batch_handle(
        self, server_with_no_provider_for_the_default
    ):
        """Only chat and consensus honour a handle. ``analyze`` given one must still be resolved like any call, so it
        fails on the unserved default model here instead of skipping model resolution and context creation."""

        with pytest.raises(ToolExecutionError) as excinfo:
            await server_with_no_provider_for_the_default.handlers.handle_call_tool(
                "analyze",
                {
                    "batch_handle": "h",
                    "step": "s",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "f",
                },
            )

        assert "not available with current API keys" in _payload(excinfo.value)["content"]


class TestAHandleCallNeverReconstructsTheConversationThread:
    """A resumption carries a ``continuation_id`` sometimes (a host echoing earlier arguments); it must not decide the call."""

    @pytest.mark.asyncio
    async def test_an_expired_thread_does_not_fail_a_valid_handle(self, server_with_no_provider_for_the_default, world):
        handle = _submit(world)

        result = await server_with_no_provider_for_the_default.handlers.handle_call_tool(
            "chat", {"batch_handle": handle, "continuation_id": "thread-that-expired"}
        )

        body = json.loads(json.loads(result[0].text)["content"])
        assert body["status"] == "running" and body["batch_handle"] == handle

    @pytest.mark.asyncio
    async def test_a_cancel_with_a_continuation_id_and_a_prompt_works_and_records_no_turn(
        self, server_with_no_provider_for_the_default, world, monkeypatch
    ):
        handle = _submit(world)
        reconstructed: list[Any] = []

        async def spy(arguments, registry):
            reconstructed.append(arguments)
            raise AssertionError("thread reconstruction ran for a handle call")

        monkeypatch.setattr("handlers.tool_handlers.reconstruct_thread_context", spy)

        result = await server_with_no_provider_for_the_default.handlers.handle_call_tool(
            "chat",
            {
                "batch_handle": handle,
                "cancel": True,
                "prompt": "p",
                "working_directory_absolute_path": "/tmp",
                "continuation_id": "abc",
            },
        )

        assert json.loads(json.loads(result[0].text)["content"])["status"] == "cancelled"
        assert reconstructed == []

    @pytest.mark.asyncio
    async def test_a_call_that_starts_work_still_reconstructs_its_thread(self, server_with_no_provider_for_the_default):
        with pytest.raises(ValueError, match="thread-that-expired"):
            await server_with_no_provider_for_the_default.handlers.handle_call_tool(
                "chat",
                {"prompt": "p", "working_directory_absolute_path": "/tmp", "continuation_id": "thread-that-expired"},
            )

    @pytest.mark.asyncio
    async def test_a_non_batch_tool_with_a_handle_and_a_continuation_id_still_reconstructs_its_thread(
        self, server_with_no_provider_for_the_default
    ):
        with pytest.raises(ValueError, match="thread-that-expired"):
            await server_with_no_provider_for_the_default.handlers.handle_call_tool(
                "analyze", {"batch_handle": "h", "continuation_id": "thread-that-expired"}
            )


class TestStartWorkStillResolvesAtTheBoundary:
    """A call without a handle keeps today's boundary behaviour (7.8: byte-for-byte unchanged)."""

    @pytest.mark.asyncio
    async def test_a_normal_call_is_resolved_validated_and_given_its_context(self, monkeypatch, full_registry):
        from server import tool_registry

        chat = tool_registry.get_tool_instance("chat")
        seen: list[dict[str, Any]] = []

        async def capture(arguments):
            seen.append(arguments)
            return ["ran"]

        monkeypatch.setattr(chat, "execute", capture)
        handlers = build_handlers(tool_registry)

        result = await handlers.handle_call_tool(
            "chat", {"prompt": "p", "working_directory_absolute_path": "/tmp", "model": "  gemini-2.5-flash "}
        )

        assert result == ["ran"]
        arguments = seen[0]
        assert arguments["model"] == "gemini-2.5-flash"  # trimmed, nothing else
        context = arguments["_context"]
        assert context.resolved_model_name == "gemini-2.5-flash" and context.model_context is not None

    @pytest.mark.asyncio
    async def test_a_start_work_call_without_a_model_in_auto_mode_is_resolved_to_the_preferred_model(
        self, monkeypatch, full_registry
    ):
        """The boundary, not the tool, fills in a missing model: it is neither refused by the schema nor at runtime."""

        from server import tool_registry
        from tools.models import ToolModelCategory

        monkeypatch.setattr("handlers.tool_handlers.DEFAULT_MODEL", "auto")
        chat = tool_registry.get_tool_instance("chat")
        seen: list[dict[str, Any]] = []

        async def capture(arguments):
            seen.append(arguments)
            return ["ran"]

        monkeypatch.setattr(chat, "execute", capture)
        handlers = build_handlers(tool_registry)

        await handlers.handle_call_tool("chat", {"prompt": "p", "working_directory_absolute_path": "/tmp"})

        expected = full_registry.get_preferred_fallback_model(chat.get_model_category())
        assert expected and seen[0]["model"] == expected
        assert ToolModelCategory  # the category came from the tool, not from a hardcoded name


# ----------------------------------------------------------------------
# helpers shared by the mutation checks below are intentionally absent: each test above stands alone
# ----------------------------------------------------------------------
