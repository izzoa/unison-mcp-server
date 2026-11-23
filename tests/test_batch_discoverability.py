"""Batch discoverability (add-batch-model-support, section 11).

Specs: batch-execution-selection ("discoverable from tool schemas", "server instructions steer
natural-language batch requests"), clink-runtime-model-selection.

A calling agent can only ask for batch if it can map the word onto an argument, so the published schema
text and the MCP handshake instructions are part of the deliverable and are pinned here. The last class
exercises the two motivating phrasings the way an agent following those instructions would act on them:
the tool call it would send, driven through the real server boundary with only the network (OpenRouter's
batch HTTP edge) and the operating-system process launch replaced.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from typing import Any
from unittest.mock import patch

import pytest

import server as server_module
import utils.model_restrictions as model_restrictions
from handlers.tool_handlers import build_handlers
from tests.batch_test_helpers import OpenRouterBatchService, batch_world_factory
from tools.chat import ChatTool
from tools.clink import CLinkTool
from tools.consensus import ConsensusTool
from tools.shared.batch_support import EXECUTION_MODE_DESCRIPTION
from tools.shared.exceptions import ToolExecutionError
from utils import batch_store as bs

CLINK_MODEL = "openrouter/openai/gpt-6-astra-pro:batch"

# Wording that would tell a caller to express batch through the model identifier.
SUFFIX_ADVICE = re.compile(r"append|suffix|add\s+[`'\"]?:?batch|\bmodel\b[^.]{0,40}:batch|:batch", re.IGNORECASE)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("BATCH_STORE_PATH", "BATCH_RETENTION_DAYS", "OPENROUTER_ALLOWED_MODELS"):
        monkeypatch.delenv(name, raising=False)
    model_restrictions._restriction_service = None
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()
    model_restrictions._restriction_service = None


def _schema_variants(tool_class):
    """The published schema in a fixed-model install and in auto mode (the default install)."""

    fixed = tool_class().get_input_schema()
    with patch("config.DEFAULT_MODEL", "auto"), patch("config.IS_AUTO_MODE", True):
        auto = tool_class().get_input_schema()
    return {"fixed": fixed, "auto": auto}


def _model_descriptions(schema: dict[str, Any]) -> list[str]:
    """Every description published for a model field: ``model`` and the consensus ``models`` list and entries."""

    props = schema["properties"]
    found = []
    if "model" in props:
        found.append(props["model"].get("description", ""))
    if "models" in props:
        found.append(props["models"].get("description", ""))
        entry = props["models"].get("items", {}).get("properties", {}).get("model", {})
        found.append(entry.get("description", ""))
    return [text for text in found if text]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# A sentence that talks about the model's name/suffix/variant in connection with batch.
_NAME_AND_BATCH = re.compile(r"(model\s+(name|string|identifier|variant)|suffix|append|variant|:batch)", re.IGNORECASE)
_BATCH_WORD = re.compile(r"batch", re.IGNORECASE)
# Sentences that are allowed to mention both: the explicit prohibition and the clink verbatim-forwarding example.
_ALLOWED_NAME_STATEMENT = re.compile(r"\bnever\b|\bkeep each model exactly\b|\bclink\b|\bverbatim\b", re.IGNORECASE)


def _suffix_advice_sentences(text: str) -> list[str]:
    """Sentences relating batch to the model name that are neither a prohibition nor the clink example."""

    return [
        sentence
        for sentence in _SENTENCE_SPLIT.split(text.strip())
        if _BATCH_WORD.search(sentence)
        and _NAME_AND_BATCH.search(sentence)
        and not _ALLOWED_NAME_STATEMENT.search(sentence)
    ]


# ----------------------------------------------------------------------
# 11.1 the shared field description
# ----------------------------------------------------------------------


class TestExecutionModeDescription:
    @pytest.mark.parametrize("tool_class", [ChatTool, ConsensusTool])
    def test_the_published_description_explains_the_field_the_tradeoff_and_the_price(self, tool_class):
        for variant, schema in _schema_variants(tool_class).items():
            text = schema["properties"]["execution_mode"]["description"]
            lowered = text.lower()
            assert 'set to "batch"' in lowered, variant
            assert "costs less" in lowered and "takes longer" in lowered, variant
            assert "about half the standard price" in lowered, variant
            assert "may not be available within this call" in lowered, variant
            assert "batch_handle" in text, variant  # and how the result is collected

    def test_consensus_entries_carry_the_same_description(self):
        entry = ConsensusTool().get_input_schema()["properties"]["models"]["items"]["properties"]["execution_mode"]

        assert entry["description"] == EXECUTION_MODE_DESCRIPTION

    def test_the_description_is_not_self_contradictory_on_consensus(self):
        """consensus is itself a multi-step workflow tool: the text must say batch applies to its step 1 only."""

        text = EXECUTION_MODE_DESCRIPTION
        assert "On consensus it applies to step 1 only" in text
        assert "later steps are refused" in text
        # The blanket exclusion must be scoped to the *other* workflow tools, not include consensus.
        assert "other multi-step workflow tools" in text
        assert "the multi-step workflow tools" not in text

    def test_the_consensus_step_the_description_names_is_the_one_the_tool_enforces(self):
        """The description's claim ('later steps are refused') is true of the tool, not just of the text."""

        import inspect

        source = inspect.getsource(ConsensusTool)
        assert 'code="batch_applies_to_step_one"' in source and "request.step_number != 1" in source

    def test_the_description_ties_the_phrase_batch_models_to_the_argument(self):
        assert "batch execution or batch models" in EXECUTION_MODE_DESCRIPTION
        assert "independent of the model" in EXECUTION_MODE_DESCRIPTION


# ----------------------------------------------------------------------
# 11.2 model fields never advertise a suffix convention
# ----------------------------------------------------------------------


class TestModelDescriptionsAdvertiseNoSuffix:
    @pytest.mark.parametrize("tool_class", [ChatTool, ConsensusTool])
    def test_no_model_field_description_tells_the_caller_to_append_an_execution_suffix(self, tool_class):
        for variant, schema in _schema_variants(tool_class).items():
            descriptions = _model_descriptions(schema)
            assert descriptions, f"{tool_class.__name__} ({variant}) published no model description"
            for text in descriptions:
                assert not SUFFIX_ADVICE.search(text), f"{tool_class.__name__} ({variant}): {text}"

    @pytest.mark.parametrize("tool_class", [ChatTool, ConsensusTool])
    def test_model_field_descriptions_do_not_mention_batch_at_all(self, tool_class):
        """Keyword-free advice ("pick the batch variant of the model name") would slip past SUFFIX_ADVICE.

        Execution intent lives only in ``execution_mode``; a model field has no business saying the word.
        """

        for variant, schema in _schema_variants(tool_class).items():
            for text in _model_descriptions(schema):
                assert "batch" not in text.lower(), f"{tool_class.__name__} ({variant}): {text}"

    def test_clink_describes_verbatim_forwarding_without_telling_a_caller_to_add_a_suffix(self):
        """clink may name suffixes (it explains they are forwarded as given) but must not advise adding one."""

        advice = re.compile(r"append|add\s+(a\s+)?[`'\"]?:?(batch|suffix)|use\s+(a\s+)?[`'\"]?:batch", re.IGNORECASE)
        for variant, schema in _schema_variants(CLinkTool).items():
            for text in _model_descriptions(schema):
                assert not advice.search(text), f"clink ({variant}): {text}"
                assert "forwarded to the external CLI verbatim" in text
                for sentence in re.split(r"(?<=[.;])\s+", text):
                    if re.search(r"batch|suffix", sentence, re.IGNORECASE):
                        assert re.search(
                            r"verbatim|does not|do not|never|up to the CLI|matches exactly", sentence, re.IGNORECASE
                        ), f"clink ({variant}): sentence neither states forwarding nor non-application: {sentence}"

    @pytest.mark.parametrize("tool_class", [ChatTool, ConsensusTool])
    def test_the_only_place_that_mentions_naming_says_never_to_edit_the_model_to_request_batch(self, tool_class):
        schema = tool_class().get_input_schema()
        assert "never edit the model name to request batch" in schema["properties"]["execution_mode"]["description"]

    def test_the_regex_used_here_would_catch_the_advice_it_guards_against(self):
        for advice in (
            "Append :batch to the model name to batch it",
            "add a :batch suffix",
            "use the model with a batch suffix",
        ):
            assert SUFFIX_ADVICE.search(advice), advice


# ----------------------------------------------------------------------
# 11.1 / 11.2 / 10.1 through the tools/list handler an MCP client actually calls
# ----------------------------------------------------------------------


class TestPublishedThroughToolsList:
    """The schema tests above call ``get_input_schema()``; a client sees what ``handle_list_tools`` returns."""

    @pytest.fixture
    async def listed(self) -> dict[str, Any]:
        from server import tool_registry

        tools = await build_handlers(tool_registry).handle_list_tools()
        return {tool.name: tool.input_schema for tool in tools}

    @pytest.mark.asyncio
    async def test_chat_and_consensus_publish_execution_mode_with_the_tradeoff_and_price(self, listed):
        for name in ("chat", "consensus"):
            text = listed[name]["properties"]["execution_mode"]["description"]
            assert text == EXECUTION_MODE_DESCRIPTION, name
            assert "costs less" in text and "takes longer" in text and "about half the standard price" in text, name
        entry = listed["consensus"]["properties"]["models"]["items"]["properties"]["execution_mode"]
        assert entry["description"] == EXECUTION_MODE_DESCRIPTION

    @pytest.mark.asyncio
    async def test_model_fields_in_the_listing_never_mention_batch(self, listed):
        for name in ("chat", "consensus"):
            descriptions = _model_descriptions(listed[name])
            assert descriptions, name
            for text in descriptions:
                assert "batch" not in text.lower(), f"{name}: {text}"

    @pytest.mark.asyncio
    async def test_clink_in_the_listing_forwards_verbatim_and_has_no_execution_mode(self, listed):
        properties = listed["clink"]["properties"]

        assert "execution_mode" not in properties
        description = properties["model"]["description"]
        assert "forwarded to the external CLI verbatim" in description
        assert "Server-side batch execution does not apply to clink" in description

    @pytest.mark.asyncio
    async def test_no_other_workflow_tool_advertises_execution_mode(self, listed):
        for name in ("codereview", "debug", "analyze", "planner", "thinkdeep"):
            assert "execution_mode" not in listed[name]["properties"], name


# ----------------------------------------------------------------------
# 11.3 the MCP handshake instructions
# ----------------------------------------------------------------------


class TestServerInstructions:
    @pytest.fixture(params=[True, False], ids=["auto", "fixed"])
    def instructions(self, request) -> str:
        return server_module.build_handshake_instructions(is_auto_mode=request.param, default_model="flash")

    def test_a_natural_language_batch_request_maps_to_execution_mode(self, instructions):
        assert "When the user asks for batch execution" in instructions
        assert 'set `execution_mode` to "batch"' in instructions
        assert "the only way to request batch" in instructions
        assert "each consensus `models` entry" in instructions
        assert "use unison consensus with gemini and opus batch models" in instructions

    def test_a_model_the_user_named_is_still_passed_through_exactly(self, instructions):
        assert "send that exact model in the tool call" in instructions  # the pre-existing instruction survives
        assert "Keep each model exactly as the user named it" in instructions
        assert "never add, remove or change a suffix" in instructions

    def test_no_sentence_of_the_instructions_advises_expressing_batch_through_the_model_name(self, instructions):
        """A sentence contradicting the verbatim rule must fail this test, not merely sit beside it."""

        offenders = _suffix_advice_sentences(instructions)
        assert offenders == [], offenders

    @pytest.mark.parametrize(
        "contradiction",
        [
            'Alternatively append ":batch" to the model name.',
            "For cheaper runs pick the batch variant of the model name.",
            "You can also add a :batch suffix to the model.",
            "Use the batch model name to get batch pricing.",
        ],
    )
    def test_the_instruction_guard_catches_a_contradicting_sentence(self, instructions, contradiction):
        assert _suffix_advice_sentences(f"{instructions} {contradiction}") == [contradiction]

    def test_provider_backed_tools_are_distinguished_from_cli_bridge_tools(self, instructions):
        assert "applies to Unison's provider-backed tools (`chat`, `consensus`)" in instructions
        assert "CLI-bridge tools (`clink`) do not support `execution_mode`" in instructions
        assert "forward the model string you send to an external CLI verbatim" in instructions

    def test_the_agent_is_told_not_to_retry_synchronously_at_full_price(self, instructions):
        assert "tell the user instead of retrying without `execution_mode`" in instructions
        assert "full price" in instructions

    def test_the_model_selection_part_still_depends_on_the_mode(self):
        auto = server_module.build_handshake_instructions(is_auto_mode=True, default_model="flash")
        fixed = server_module.build_handshake_instructions(is_auto_mode=False, default_model="flash")

        assert "use the `listmodels` tool" in auto and "default to" not in auto
        assert "default to 'flash'" in fixed and "listmodels" not in fixed.split("When the user asks for batch")[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("auto_mode", [True, False], ids=["auto", "fixed"])
    async def test_the_running_server_hands_exactly_these_instructions_to_the_client(self, monkeypatch, auto_mode):
        """Retrieval: ``main()`` passes the built text into the handshake options, nothing else replaces it."""

        captured: dict[str, Any] = {}

        class _Stdio:
            async def __aenter__(self):
                return object(), object()

            async def __aexit__(self, *exc):
                return False

        async def fake_run(read_stream, write_stream, options, *args, **kwargs):
            captured["instructions"] = options.instructions

        monkeypatch.setattr(server_module, "stdio_server", lambda: _Stdio())
        monkeypatch.setattr(server_module.server, "run", fake_run)
        monkeypatch.setattr(server_module, "_start_batch_reconciliation", lambda: None)
        monkeypatch.setattr(server_module, "configure_providers", lambda registry: None)
        # main() reads both from module state; pin them so the expectation does not depend on other tests' config.
        monkeypatch.setattr("config.IS_AUTO_MODE", auto_mode)
        monkeypatch.setattr(server_module, "DEFAULT_MODEL", "flash")

        await server_module.main()

        assert captured["instructions"] == server_module.build_handshake_instructions(
            is_auto_mode=auto_mode, default_model="flash"
        )
        assert 'set `execution_mode` to "batch"' in captured["instructions"]
        assert ("listmodels" in captured["instructions"]) is auto_mode


# ----------------------------------------------------------------------
# 11.4 the motivating phrasings, end to end as far as a unit test can go
# ----------------------------------------------------------------------


def _consensus_call(models: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "step": "Should we adopt event sourcing for billing?",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": "initial analysis",
        "models": models,
        "wait_seconds": 0,
        **extra,
    }


@pytest.fixture
def service() -> OpenRouterBatchService:
    return OpenRouterBatchService()


@pytest.fixture
def consensus_server(tmp_path, service, monkeypatch):
    """The real server boundary over a fresh batch world; consensus is wired to that world's executor."""

    from server import tool_registry

    world = batch_world_factory(tmp_path, service)()
    consensus = tool_registry.get_tool_instance("consensus")
    monkeypatch.setattr(consensus, "_batch_executor", world.executor, raising=False)
    yield SimpleHandlers(build_handlers(tool_registry), world)
    world.close()


class SimpleHandlers:
    def __init__(self, handlers: Any, world: Any) -> None:
        self.handlers = handlers
        self.world = world

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self.handlers.handle_call_tool(name, arguments)


def _running_body(blocks) -> dict[str, Any]:
    outer = json.loads(blocks[0].text)
    assert outer["status"] == "success", outer
    body = json.loads(outer["content"])
    assert body["status"] == "running", body
    return body


class TestUseUnisonConsensusWithGeminiAndOpusBatchModels:
    """'use unison consensus with gemini and opus batch models' -> execution_mode batch on those entries."""

    ENTRIES = [
        {"model": "gemini", "stance": "for", "execution_mode": "batch"},
        {"model": "opus", "stance": "against", "execution_mode": "batch"},
    ]

    @pytest.mark.asyncio
    async def test_both_entries_are_submitted_as_batches_with_their_names_untouched(self, consensus_server, service):
        # 14.12: `gemini` is a family alias (the Pro tier), so under batch it resolves to the latest Pro model
        # OpenRouter lists as ':batch' (the fixture catalog's google/gemini-3.1-pro-preview) and the call succeeds
        # with no flag patched; `opus` likewise resolves to the latest Opus. One batch per distinct model.
        body = _running_body(await consensus_server.call("consensus", _consensus_call(self.ENTRIES)))

        assert body["batch_handle"].startswith("batch_")
        assert sorted(service.submitted_models) == ["anthropic/claude-opus-5.5", "google/gemini-3.1-pro-preview"]
        assert len(service.posts) == 2  # one batch per distinct model
        stored = consensus_server.world.store.get(body["batch_handle"]).payload
        assert [m["model"] for m in stored["models"]] == ["gemini", "opus"]  # the user's names, verbatim
        assert [m["stance"] for m in stored["models"]] == ["for", "against"]

    @pytest.mark.asyncio
    async def test_the_request_level_field_batches_every_entry_the_same_way(self, consensus_server, service):
        models = [{"model": "opus"}, {"model": "sonnet"}]

        _running_body(await consensus_server.call("consensus", _consensus_call(models, execution_mode="batch")))

        assert sorted(service.submitted_models) == ["anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5.5"]

    @pytest.mark.asyncio
    async def test_a_model_that_cannot_batch_is_refused_by_name_with_nothing_submitted(self, consensus_server, service):
        """A genuinely unbatchable model refuses the whole call: never run at full price, never half-batched."""

        entries = [self.ENTRIES[1], {"model": "mistral", "stance": "for", "execution_mode": "batch"}]
        with pytest.raises(ToolExecutionError) as excinfo:
            await consensus_server.call("consensus", _consensus_call(entries))

        payload = json.loads(excinfo.value.payload)
        assert payload["metadata"]["batch_error"] == "batch_unsupported_model"
        assert "'mistral'" in payload["content"] and "without execution_mode" in payload["content"]
        assert service.posts == []  # opus was not submitted either: no half-batched run, no silent sync fallback

    @pytest.mark.asyncio
    async def test_the_published_schema_accepts_the_call_the_instructions_tell_an_agent_to_send(self):
        import jsonschema

        schema = ConsensusTool().get_input_schema()

        jsonschema.validate(_consensus_call(self.ENTRIES), schema)


class _Stdin:
    def write(self, chunk: bytes) -> None:
        pass

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


def _reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


class _StubProcess:
    returncode = 0
    pid = 0

    def __init__(self) -> None:
        self.stdin = _Stdin()
        self.stdout = _reader(json.dumps({"type": "text", "text": "ok"}).encode())
        self.stderr = _reader(b"")

    async def communicate(self, stdin_data: bytes | None = None):
        return json.dumps({"type": "text", "text": "ok"}).encode(), b""

    async def wait(self) -> int:
        return 0

    def kill(self) -> None:
        pass


class TestClinkWithOpencodeUsingASuffixedModel:
    """'clink with opencode using model openrouter/openai/gpt-6-astra-pro:batch' -> verbatim forward, no batch path."""

    @pytest.fixture
    def launched(self, monkeypatch) -> list[list[str]]:
        argv: list[list[str]] = []

        async def fake_exec(*args, **_kwargs):
            argv.append(list(args))
            return _StubProcess()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        return argv

    @pytest.mark.asyncio
    async def test_the_string_reaches_opencode_verbatim_and_no_batch_is_created(
        self, launched, tmp_path, service, monkeypatch
    ):
        from server import tool_registry

        world = batch_world_factory(tmp_path, service)()
        try:
            handlers = build_handlers(tool_registry)
            arguments = {"prompt": "hi", "cli_name": "opencode", "model": CLINK_MODEL}

            result = await handlers.handle_call_tool("clink", arguments)

            assert len(launched) == 1
            command = launched[0]
            assert any(command[i] == "-m" and command[i + 1] == CLINK_MODEL for i in range(len(command) - 1))
            assert json.loads(result[0].text)["metadata"]["model_requested"] == CLINK_MODEL
            assert service.posts == []  # no provider batch was submitted for the CLI call
            assert world.store.list_records() == []  # and no batch record exists
        finally:
            world.close()

    @pytest.mark.asyncio
    async def test_adding_execution_mode_is_refused_and_the_cli_is_never_launched(self, launched):
        from server import tool_registry

        handlers = build_handlers(tool_registry)
        arguments = {"prompt": "hi", "cli_name": "opencode", "model": CLINK_MODEL, "execution_mode": "batch"}

        with pytest.raises(ToolExecutionError) as excinfo:
            await handlers.handle_call_tool("clink", arguments)

        assert (
            "server-side batch execution does not apply to CLI bridge tools"
            in json.loads(excinfo.value.payload)["content"]
        )
        assert launched == []
