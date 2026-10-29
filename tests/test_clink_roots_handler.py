"""Tests for resolving the host's MCP roots in the request handler (clink-working-dir-derivation, tasks 5.3 and 5.4).

``handlers.tool_handlers.on_call_tool`` asks the host for its roots for a clink call that names no ``working_dir`` and
no ``job_id``, in the live request and before ``reconstruct_thread_context`` writes anything, and hands the result to the
tool as the private argument ``_mcp_roots``. Most tests run the real handler wiring (``build_handlers`` -> low-level
``Server``) against the SDK's in-memory ``Client`` in both protocol eras, with a stub ``clink`` tool that records the
arguments it receives. The era decision and the loop guard are also tested directly on the extracted helper with fake
sessions, because the in-memory client always retries a well-formed input request.
"""

from __future__ import annotations

import asyncio
import time
import warnings
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.client import Client
from mcp.server import Server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    InputRequiredResult,
    ListRootsRequest,
    ListRootsResult,
    Root,
    TextContent,
)
from mcp.types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import ValidationError

from handlers import tool_handlers
from handlers.tool_handlers import ROOTS_REQUEST_STATE, build_handlers, resolve_clink_roots

LEGACY_VERSION = "2025-11-25"
MODERN_VERSION = MODERN_PROTOCOL_VERSIONS[0]
ROOT_A = "file:///tmp/project-a"
ROOT_B = "file:///tmp/project-b"


# ---------------------------------------------------------------------------
# Stub registry / tools / server
# ---------------------------------------------------------------------------


class _StubTool:
    """A tool that records the arguments it receives; needs no model."""

    def __init__(self, name: str):
        self.name = name
        self.description = f"stub {name}"
        self.received: list[dict[str, Any]] = []

    def get_input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": True}

    def get_annotations(self):
        return None

    def requires_model(self) -> bool:
        return False

    def get_model_category(self):  # pragma: no cover - never reached for a model-less tool
        raise AssertionError("model category requested for a model-less tool")

    def format_conversation_turn(self, turn):
        return None

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        self.received.append(dict(arguments))
        return [TextContent(type="text", text="ok")]


class _StubRegistry:
    def __init__(self, *tools: _StubTool):
        self.tools = {tool.name: tool for tool in tools}

    def get_available_tools(self):
        return dict(self.tools)

    def get_tool_instance(self, name: str):
        return self.tools[name]

    def is_available(self, name: str) -> bool:
        return name in self.tools


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Never inherit the developer's CLINK_MCP_ROOTS; every test that depends on a mode states it (setenv/delenv).
    monkeypatch.delenv("CLINK_MCP_ROOTS", raising=False)


@pytest.fixture
def clink_tool():
    return _StubTool("clink")


@pytest.fixture
def chat_tool():
    return _StubTool("chat")


@pytest.fixture
def server(clink_tool, chat_tool):
    handlers = build_handlers(_StubRegistry(clink_tool, chat_tool))
    return Server("roots-test", on_list_tools=handlers.on_list_tools, on_call_tool=handlers.on_call_tool)


class _RootsHost:
    """A roots callback that counts how often the server asked."""

    def __init__(self, *uris: str, delay: float | None = None, raw: Any = None):
        self.uris = uris
        self.delay = delay
        self.raw = raw
        self.calls = 0

    async def __call__(self, ctx):
        self.calls += 1
        if self.delay is not None:
            await asyncio.sleep(self.delay)
        if self.raw is not None:
            return self.raw
        return ListRootsResult(roots=[Root(uri=uri) for uri in self.uris])


async def _call(server: Server, arguments: dict[str, Any], *, mode: str, host: _RootsHost | None = None, tool="clink"):
    kwargs: dict[str, Any] = {"mode": mode}
    if host is not None:
        kwargs["list_roots_callback"] = host
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        async with Client(server, **kwargs) as client:
            return await client.call_tool(tool, arguments)


# ---------------------------------------------------------------------------
# Server-request path (legacy-era connection, real SDK client)
# ---------------------------------------------------------------------------


class TestServerRequestPath:
    @pytest.mark.asyncio
    async def test_no_capability_sends_no_request_and_no_warning(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        result = await _call(server, {"prompt": "hi"}, mode="legacy")

        assert not result.is_error
        assert len(clink_tool.received) == 1
        assert "_mcp_roots" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_answered_roots_are_passed_in_order(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        host = _RootsHost(ROOT_B, ROOT_A)

        result = await _call(server, {"prompt": "hi"}, mode="legacy", host=host)

        assert not result.is_error
        assert host.calls == 1
        assert clink_tool.received[0]["_mcp_roots"] == {"uris": [ROOT_B, ROOT_A], "warnings": []}

    @pytest.mark.asyncio
    async def test_unanswered_request_returns_within_the_bound_with_a_warning(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        monkeypatch.setattr(tool_handlers, "ROOTS_REQUEST_TIMEOUT_SECONDS", 0.3)
        host = _RootsHost(delay=30)

        started = time.monotonic()
        result = await _call(server, {"prompt": "hi"}, mode="legacy", host=host)
        elapsed = time.monotonic() - started

        assert not result.is_error
        assert elapsed < 5
        roots = clink_tool.received[0]["_mcp_roots"]
        assert roots["uris"] == []
        assert len(roots["warnings"]) == 1 and "unavailable" in roots["warnings"][0]

    @pytest.mark.asyncio
    async def test_reply_with_a_non_file_root_is_rejected_whole_with_a_warning(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        # The SDK model refuses non-file URIs, so a misbehaving host is simulated with an unvalidated reply.
        raw = ListRootsResult.model_construct(
            roots=[SimpleNamespace(uri=ROOT_A, name=None), SimpleNamespace(uri="https://example.com/repo", name=None)]
        )
        host = _RootsHost(raw=raw)

        result = await _call(server, {"prompt": "hi"}, mode="legacy", host=host)

        assert not result.is_error
        roots = clink_tool.received[0]["_mcp_roots"]
        assert roots["uris"] == []
        assert len(roots["warnings"]) == 1 and "unavailable" in roots["warnings"][0]

    @pytest.mark.asyncio
    async def test_empty_list_is_silent(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        host = _RootsHost()

        result = await _call(server, {"prompt": "hi"}, mode="legacy", host=host)

        assert not result.is_error
        assert host.calls == 1
        assert "_mcp_roots" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_off_sends_nothing_on_either_era(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "off")
        for mode in ("legacy", "auto"):
            host = _RootsHost(ROOT_A)
            result = await _call(server, {"prompt": "hi"}, mode=mode, host=host)
            assert not result.is_error
            assert host.calls == 0
        assert all("_mcp_roots" not in received for received in clink_tool.received)
        assert len(clink_tool.received) == 2

    @pytest.mark.asyncio
    async def test_unrecognised_mode_behaves_as_legacy_and_warns(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "sometimes")
        host = _RootsHost(ROOT_A)

        await _call(server, {"prompt": "hi"}, mode="legacy", host=host)

        roots = clink_tool.received[0]["_mcp_roots"]
        assert roots["uris"] == [ROOT_A]
        assert len(roots["warnings"]) == 1 and "sometimes" in roots["warnings"][0]

    @pytest.mark.asyncio
    async def test_roots_are_requested_fresh_on_every_call(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        host = _RootsHost(ROOT_A)

        await _call(server, {"prompt": "one"}, mode="legacy", host=host)
        host.uris = (ROOT_B,)
        await _call(server, {"prompt": "two"}, mode="legacy", host=host)

        assert host.calls == 2
        assert [r["_mcp_roots"]["uris"] for r in clink_tool.received] == [[ROOT_A], [ROOT_B]]


# ---------------------------------------------------------------------------
# Input-request path (modern-era connection, real SDK client)
# ---------------------------------------------------------------------------


class TestInputRequestPath:
    @pytest.mark.asyncio
    async def test_unset_means_auto_and_a_modern_client_round_trips_the_exchange(self, server, clink_tool, monkeypatch):
        monkeypatch.delenv("CLINK_MCP_ROOTS", raising=False)
        host = _RootsHost(ROOT_A, ROOT_B)

        result = await _call(server, {"prompt": "hi"}, mode="auto", host=host)

        assert not result.is_error
        assert host.calls == 1
        assert len(clink_tool.received) == 1  # the first pass returned the input request without executing
        assert clink_tool.received[0]["_mcp_roots"] == {"uris": [ROOT_A, ROOT_B], "warnings": []}

    @pytest.mark.parametrize("blank", ["", "   "])
    @pytest.mark.asyncio
    async def test_a_blank_value_means_auto_too(self, server, clink_tool, monkeypatch, blank):
        monkeypatch.setenv("CLINK_MCP_ROOTS", blank)
        host = _RootsHost(ROOT_A)

        await _call(server, {"prompt": "hi"}, mode="auto", host=host)

        assert host.calls == 1
        assert clink_tool.received[0]["_mcp_roots"] == {"uris": [ROOT_A], "warnings": []}

    @pytest.mark.asyncio
    async def test_explicit_legacy_setting_sends_no_input_request_and_stays_silent(
        self, server, clink_tool, monkeypatch
    ):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        host = _RootsHost(ROOT_A)

        result = await _call(server, {"prompt": "hi"}, mode="auto", host=host)

        assert not result.is_error
        assert host.calls == 0
        assert len(clink_tool.received) == 1
        assert "_mcp_roots" not in clink_tool.received[0]  # roots unavailable, and no warning either

    @pytest.mark.asyncio
    async def test_off_sends_no_input_request(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "off")
        host = _RootsHost(ROOT_A)

        result = await _call(server, {"prompt": "hi"}, mode="auto", host=host)

        assert not result.is_error
        assert host.calls == 0
        assert len(clink_tool.received) == 1
        assert "_mcp_roots" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_unrecognised_value_behaves_as_legacy_on_a_modern_client_and_warns(
        self, server, clink_tool, monkeypatch
    ):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "sometimes")
        host = _RootsHost(ROOT_A)

        result = await _call(server, {"prompt": "hi"}, mode="auto", host=host)

        assert not result.is_error
        assert host.calls == 0  # no input request: an unrecognised value is the conservative mode, not the default
        assert len(clink_tool.received) == 1
        roots = clink_tool.received[0]["_mcp_roots"]
        assert roots["uris"] == []
        assert len(roots["warnings"]) == 1
        assert "sometimes" in roots["warnings"][0] and "legacy" in roots["warnings"][0]

    @pytest.mark.asyncio
    async def test_auto_round_trips_and_runs_the_tool_once(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        host = _RootsHost(ROOT_A, ROOT_B)

        result = await _call(server, {"prompt": "hi"}, mode="auto", host=host)

        assert not result.is_error
        assert host.calls == 1
        assert len(clink_tool.received) == 1  # the first pass returned the input request without executing
        assert clink_tool.received[0]["_mcp_roots"] == {"uris": [ROOT_A, ROOT_B], "warnings": []}

    @pytest.mark.asyncio
    async def test_auto_without_capability_returns_no_input_request(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")

        result = await _call(server, {"prompt": "hi"}, mode="auto")

        assert not result.is_error
        assert len(clink_tool.received) == 1
        assert "_mcp_roots" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_auto_with_an_empty_reply_is_silent(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        host = _RootsHost()

        await _call(server, {"prompt": "hi"}, mode="auto", host=host)

        assert host.calls == 1
        assert len(clink_tool.received) == 1
        assert "_mcp_roots" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_auto_records_a_continuation_turn_once(self, server, clink_tool, monkeypatch):
        from utils.conversation_memory import create_thread, get_thread

        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        thread_id = create_thread("clink", {"prompt": "first"})
        host = _RootsHost(ROOT_A)

        result = await _call(
            server,
            {"prompt": "second turn", "continuation_id": thread_id, "absolute_file_paths": ["/tmp/project-a/x.py"]},
            mode="auto",
            host=host,
        )

        assert not result.is_error
        assert host.calls == 1
        assert len(clink_tool.received) == 1
        received = clink_tool.received[0]
        assert received["_mcp_roots"] == {"uris": [ROOT_A], "warnings": []}
        assert received["_passed_file_paths"] == ["/tmp/project-a/x.py"]
        turns = get_thread(thread_id).turns
        assert [(t.role, t.content) for t in turns] == [("user", "second turn")]

    @pytest.mark.asyncio
    async def test_the_default_records_a_continuation_turn_once(self, server, clink_tool, monkeypatch):
        from utils.conversation_memory import create_thread, get_thread

        monkeypatch.delenv("CLINK_MCP_ROOTS", raising=False)
        thread_id = create_thread("clink", {"prompt": "first"})
        host = _RootsHost(ROOT_A)

        result = await _call(
            server,
            {"prompt": "second turn", "continuation_id": thread_id, "absolute_file_paths": ["/tmp/project-a/x.py"]},
            mode="auto",
            host=host,
        )

        assert not result.is_error
        assert host.calls == 1
        assert len(clink_tool.received) == 1
        received = clink_tool.received[0]
        assert received["_mcp_roots"] == {"uris": [ROOT_A], "warnings": []}
        assert received["_passed_file_paths"] == ["/tmp/project-a/x.py"]
        turns = get_thread(thread_id).turns
        assert [(t.role, t.content) for t in turns] == [("user", "second turn")]

    @pytest.mark.parametrize("setting", [None, "", "auto", "legacy", "off", "sometimes"])
    @pytest.mark.asyncio
    async def test_legacy_client_never_receives_an_input_request_result(self, server, clink_tool, monkeypatch, setting):
        if setting is None:
            monkeypatch.delenv("CLINK_MCP_ROOTS", raising=False)
        else:
            monkeypatch.setenv("CLINK_MCP_ROOTS", setting)
        host = _RootsHost(ROOT_A)

        result = await _call(server, {"prompt": "hi"}, mode="legacy", host=host)

        assert isinstance(result, CallToolResult) and not result.is_error
        assert len(clink_tool.received) == 1
        if setting == "off":
            assert host.calls == 0
            assert "_mcp_roots" not in clink_tool.received[0]
        else:  # answered through the server request, never the exchange
            assert host.calls == 1
            assert clink_tool.received[0]["_mcp_roots"]["uris"] == [ROOT_A]

    @pytest.mark.asyncio
    async def test_explicit_working_dir_and_job_polls_never_request_roots_under_the_default(
        self, server, clink_tool, monkeypatch
    ):
        monkeypatch.delenv("CLINK_MCP_ROOTS", raising=False)
        host = _RootsHost(ROOT_A)

        for mode in ("legacy", "auto"):
            await _call(server, {"prompt": "hi", "working_dir": "/tmp/somewhere"}, mode=mode, host=host)
            await _call(server, {"job_id": "abc123"}, mode=mode, host=host)

        assert host.calls == 0
        assert len(clink_tool.received) == 4
        assert all("_mcp_roots" not in received for received in clink_tool.received)


# ---------------------------------------------------------------------------
# The helper and the handler, driven with fake contexts
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(
        self,
        *,
        protocol_version: str,
        can_send_request: bool,
        declared: bool = True,
        reply: Any = None,
        error: BaseException | None = None,
    ):
        self.protocol_version = protocol_version
        self.can_send_request = can_send_request
        self.declared = declared
        self.reply = reply
        self.error = error
        self.sent: list[tuple[Any, float | None]] = []

    def check_client_capability(self, capability) -> bool:
        return self.declared

    async def send_request(self, request, result_type, request_read_timeout_seconds=None):
        self.sent.append((request, request_read_timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.reply if self.reply is not None else ListRootsResult(roots=[Root(uri=ROOT_A)])


def _ctx(session: _FakeSession | None) -> SimpleNamespace:
    return SimpleNamespace(session=session)


def _params(**kwargs: Any) -> CallToolRequestParams:
    return CallToolRequestParams(name="clink", arguments={"prompt": "hi"}, **kwargs)


class TestResolveHelper:
    @pytest.mark.parametrize("setting", [None, "", "auto", "legacy", "off"])
    @pytest.mark.asyncio
    async def test_legacy_era_without_a_back_channel_never_uses_the_input_request_path(self, monkeypatch, setting):
        if setting is None:
            monkeypatch.delenv("CLINK_MCP_ROOTS", raising=False)
        else:
            monkeypatch.setenv("CLINK_MCP_ROOTS", setting)
        session = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=False)

        outcome = await resolve_clink_roots(_ctx(session), _params())

        assert outcome is None
        assert session.sent == []

    @pytest.mark.asyncio
    async def test_legacy_era_without_a_back_channel_with_a_bad_value_only_warns(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "bogus")
        session = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=False)

        outcome = await resolve_clink_roots(_ctx(session), _params())

        assert not isinstance(outcome, InputRequiredResult)
        assert outcome is not None and outcome["uris"] == [] and "bogus" in outcome["warnings"][0]

    @pytest.mark.asyncio
    async def test_legacy_era_ignores_a_roots_reply_in_the_params(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        session = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=False)
        params = _params(input_responses={"roots": ListRootsResult(roots=[Root(uri=ROOT_A)])})

        assert await resolve_clink_roots(_ctx(session), params) is None

    @pytest.mark.asyncio
    async def test_modern_era_without_a_back_channel_asks_through_the_exchange_under_auto(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)

        outcome = await resolve_clink_roots(_ctx(session), _params())

        assert isinstance(outcome, InputRequiredResult)
        assert outcome.request_state == ROOTS_REQUEST_STATE
        assert list(outcome.input_requests or {}) == ["roots"]
        assert isinstance(outcome.input_requests["roots"], ListRootsRequest)
        assert session.sent == []

    @pytest.mark.asyncio
    async def test_modern_era_without_a_back_channel_asks_through_the_exchange_by_default(self, monkeypatch):
        monkeypatch.delenv("CLINK_MCP_ROOTS", raising=False)
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)

        outcome = await resolve_clink_roots(_ctx(session), _params())

        assert isinstance(outcome, InputRequiredResult)
        assert session.sent == []

    @pytest.mark.parametrize("setting", ["legacy", "LEGACY", "off", "bogus"])
    @pytest.mark.asyncio
    async def test_modern_era_without_a_back_channel_never_asks_unless_auto(self, monkeypatch, setting):
        monkeypatch.setenv("CLINK_MCP_ROOTS", setting)
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)

        outcome = await resolve_clink_roots(_ctx(session), _params())

        assert not isinstance(outcome, InputRequiredResult)

    @pytest.mark.asyncio
    async def test_modern_era_without_the_capability_never_asks(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False, declared=False)

        assert await resolve_clink_roots(_ctx(session), _params()) is None

    @pytest.mark.asyncio
    async def test_retry_reads_the_roots_reply(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)
        params = _params(
            input_responses={"roots": ListRootsResult(roots=[Root(uri=ROOT_B), Root(uri=ROOT_A)])},
            request_state=ROOTS_REQUEST_STATE,
        )

        assert await resolve_clink_roots(_ctx(session), params) == {"uris": [ROOT_B, ROOT_A], "warnings": []}

    @pytest.mark.asyncio
    async def test_retry_reads_a_mapping_shaped_reply(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)
        params = SimpleNamespace(input_responses={"roots": {"roots": [{"uri": ROOT_A}]}}, request_state="roots1")

        assert await resolve_clink_roots(_ctx(session), params) == {"uris": [ROOT_A], "warnings": []}

    @pytest.mark.asyncio
    async def test_retry_without_a_roots_reply_is_unavailable_with_a_warning_and_no_second_request(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)

        for responses in (None, {}, {"other": ListRootsResult(roots=[])}):
            params = _params(input_responses=responses, request_state=ROOTS_REQUEST_STATE)
            outcome = await resolve_clink_roots(_ctx(session), params)
            assert not isinstance(outcome, InputRequiredResult)
            assert outcome is not None and outcome["uris"] == [] and len(outcome["warnings"]) == 1

    @pytest.mark.asyncio
    async def test_unusable_retry_reply_is_unavailable_with_a_warning(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)
        for reply in ({"roots": [{"uri": "https://example.com/x"}]}, {"nope": 1}, "garbage"):
            params = SimpleNamespace(input_responses={"roots": reply}, request_state=ROOTS_REQUEST_STATE)
            outcome = await resolve_clink_roots(_ctx(session), params)
            assert outcome is not None and outcome["uris"] == [] and len(outcome["warnings"]) == 1

    @pytest.mark.asyncio
    async def test_retry_with_an_empty_reply_is_silent(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)
        params = _params(input_responses={"roots": ListRootsResult(roots=[])}, request_state=ROOTS_REQUEST_STATE)

        assert await resolve_clink_roots(_ctx(session), params) is None

    @pytest.mark.asyncio
    async def test_server_request_uses_the_bounded_timeout(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        session = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=True)

        outcome = await resolve_clink_roots(_ctx(session), _params())

        assert outcome == {"uris": [ROOT_A], "warnings": []}
        request, timeout = session.sent[0]
        assert isinstance(request, ListRootsRequest)
        assert timeout == tool_handlers.ROOTS_REQUEST_TIMEOUT_SECONDS <= 3

    @pytest.mark.asyncio
    async def test_server_request_wins_on_a_modern_connection_that_can_carry_it(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=True)

        outcome = await resolve_clink_roots(_ctx(session), _params())

        assert outcome == {"uris": [ROOT_A], "warnings": []}
        assert len(session.sent) == 1

    @pytest.mark.parametrize(
        "error",
        [
            asyncio.TimeoutError(),
            TimeoutError(),
            ValueError("bad"),
            RuntimeError("boom"),
        ],
    )
    @pytest.mark.asyncio
    async def test_request_failures_degrade_to_unavailable_with_a_warning(self, error, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        session = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=True, error=error)

        outcome = await resolve_clink_roots(_ctx(session), _params())

        assert outcome is not None and outcome["uris"] == [] and len(outcome["warnings"]) == 1

    @pytest.mark.asyncio
    async def test_sdk_errors_degrade_to_unavailable_with_a_warning(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        from mcp.shared.exceptions import MCPError, NoBackChannelError

        try:
            ListRootsResult.model_validate({"roots": [{"uri": "https://example.com/x"}]})
        except ValidationError as exc:
            validation_error = exc
        for error in (MCPError(code=-32603, message="nope"), NoBackChannelError("roots/list"), validation_error):
            session = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=True, error=error)

            outcome = await resolve_clink_roots(_ctx(session), _params())

            assert outcome is not None and outcome["uris"] == [] and len(outcome["warnings"]) == 1

    @pytest.mark.asyncio
    async def test_no_back_channel_and_no_session_are_silent(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        assert await resolve_clink_roots(_ctx(None), _params()) is None
        quiet = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=False)
        assert await resolve_clink_roots(_ctx(quiet), _params()) is None

    @pytest.mark.asyncio
    async def test_cancellation_is_not_swallowed(self, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        session = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=True, error=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await resolve_clink_roots(_ctx(session), _params())


class TestHandlerReturnsInputRequiredResult:
    @pytest.mark.asyncio
    async def test_on_call_tool_returns_the_input_request_before_any_state_is_written(self, clink_tool, monkeypatch):
        from utils.conversation_memory import create_thread, get_thread

        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        thread_id = create_thread("clink", {"prompt": "first"})
        handlers = build_handlers(_StubRegistry(clink_tool))
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)
        params = CallToolRequestParams(name="clink", arguments={"prompt": "again", "continuation_id": thread_id})

        result = await handlers.on_call_tool(_ctx(session), params)

        assert isinstance(result, InputRequiredResult)
        assert clink_tool.received == []
        assert get_thread(thread_id).turns == []

    @pytest.mark.asyncio
    async def test_on_call_tool_gives_up_after_a_retry_without_roots(self, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        handlers = build_handlers(_StubRegistry(clink_tool))
        session = _FakeSession(protocol_version=MODERN_VERSION, can_send_request=False)
        params = CallToolRequestParams(name="clink", arguments={"prompt": "again"}, request_state=ROOTS_REQUEST_STATE)

        result = await handlers.on_call_tool(_ctx(session), params)

        assert isinstance(result, CallToolResult) and not result.is_error
        roots = clink_tool.received[0]["_mcp_roots"]
        assert roots["uris"] == [] and len(roots["warnings"]) == 1

    @pytest.mark.asyncio
    async def test_a_resolver_crash_never_fails_the_call(self, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")

        async def explode(ctx, params):
            raise RuntimeError("boom")

        monkeypatch.setattr(tool_handlers, "_resolve_clink_roots", explode)
        handlers = build_handlers(_StubRegistry(clink_tool))
        session = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=True)

        result = await handlers.on_call_tool(_ctx(session), _params())

        assert isinstance(result, CallToolResult) and not result.is_error
        assert clink_tool.received[0]["_mcp_roots"]["uris"] == []
        assert len(clink_tool.received[0]["_mcp_roots"]["warnings"]) == 1


# ---------------------------------------------------------------------------
# Which calls resolve roots, and private arguments
# ---------------------------------------------------------------------------


class TestWhichCallsAskForRoots:
    @pytest.mark.parametrize(
        "arguments",
        [
            {"prompt": "hi", "working_dir": "/tmp/somewhere"},
            {"job_id": "abc123"},
            {"job_id": "abc123", "working_dir": ""},
            {"job_id": "abc123", "wait_seconds": 5},
        ],
    )
    @pytest.mark.asyncio
    async def test_explicit_working_dir_and_job_polls_never_request_roots(
        self, server, clink_tool, arguments, monkeypatch
    ):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        host = _RootsHost(ROOT_A)

        result = await _call(server, arguments, mode="legacy", host=host)

        assert not result.is_error
        assert host.calls == 0
        assert "_mcp_roots" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_explicit_working_dir_never_requests_roots_on_the_modern_era(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        host = _RootsHost(ROOT_A)

        await _call(server, {"prompt": "hi", "working_dir": "/tmp/somewhere"}, mode="auto", host=host)
        await _call(server, {"job_id": "abc123"}, mode="auto", host=host)

        assert host.calls == 0
        assert len(clink_tool.received) == 2

    @pytest.mark.parametrize("blank", ["", "   ", None])
    @pytest.mark.asyncio
    async def test_blank_working_dir_counts_as_absent(self, server, clink_tool, blank, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        host = _RootsHost(ROOT_A)

        await _call(server, {"prompt": "hi", "working_dir": blank}, mode="legacy", host=host)

        assert host.calls == 1
        assert clink_tool.received[0]["_mcp_roots"]["uris"] == [ROOT_A]

    @pytest.mark.asyncio
    async def test_other_tools_are_untouched(self, server, chat_tool, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "auto")
        for mode in ("legacy", "auto"):
            host = _RootsHost(ROOT_A)
            result = await _call(server, {"prompt": "hi"}, mode=mode, host=host, tool="chat")
            assert not result.is_error
            assert host.calls == 0
        assert len(chat_tool.received) == 2
        assert all("_mcp_roots" not in received for received in chat_tool.received)
        assert clink_tool.received == []

    @pytest.mark.asyncio
    async def test_a_disabled_clink_tool_does_not_request_roots(self, chat_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        handlers = build_handlers(_StubRegistry(chat_tool))
        host = _RootsHost(ROOT_A)
        server = Server("roots-test", on_list_tools=handlers.on_list_tools, on_call_tool=handlers.on_call_tool)

        result = await _call(server, {"prompt": "hi"}, mode="legacy", host=host)

        assert "Unknown tool" in result.content[0].text
        assert host.calls == 0


class TestPrivateArguments:
    @pytest.mark.asyncio
    async def test_client_supplied_roots_are_ignored_when_none_are_resolved(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        spoofed = {"uris": ["file:///etc"], "warnings": []}

        result = await _call(server, {"prompt": "hi", "_mcp_roots": spoofed}, mode="legacy")

        assert not result.is_error
        assert "_mcp_roots" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_client_supplied_roots_never_survive_off(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "off")

        await _call(server, {"prompt": "hi", "_mcp_roots": {"uris": ["file:///etc"], "warnings": []}}, mode="legacy")

        assert "_mcp_roots" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_the_handler_value_replaces_a_client_supplied_one(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        host = _RootsHost(ROOT_A)

        await _call(
            server, {"prompt": "hi", "_mcp_roots": {"uris": ["file:///etc"], "warnings": []}}, mode="legacy", host=host
        )

        assert clink_tool.received[0]["_mcp_roots"] == {"uris": [ROOT_A], "warnings": []}

    @pytest.mark.asyncio
    async def test_client_supplied_roots_are_dropped_on_explicit_working_dir_calls(
        self, server, clink_tool, monkeypatch
    ):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        await _call(
            server,
            {"prompt": "hi", "working_dir": "/tmp/x", "_mcp_roots": {"uris": ["file:///etc"], "warnings": []}},
            mode="legacy",
        )

        assert "_mcp_roots" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_client_supplied_passed_file_paths_are_dropped_on_a_fresh_call(self, server, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        await _call(server, {"prompt": "hi", "_passed_file_paths": ["/etc/passwd"]}, mode="legacy")

        assert "_passed_file_paths" not in clink_tool.received[0]

    @pytest.mark.asyncio
    async def test_continuation_captures_the_files_of_this_call_not_a_client_supplied_list(
        self, server, clink_tool, monkeypatch
    ):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        from utils.conversation_memory import create_thread

        thread_id = create_thread("clink", {"prompt": "first"})

        await _call(
            server,
            {
                "prompt": "next",
                "continuation_id": thread_id,
                "absolute_file_paths": ["/tmp/real/a.py"],
                "_passed_file_paths": ["/etc/passwd"],
            },
            mode="legacy",
        )

        assert clink_tool.received[0]["_passed_file_paths"] == ["/tmp/real/a.py"]

    @pytest.mark.asyncio
    async def test_the_caller_mapping_is_not_mutated(self, clink_tool, monkeypatch):
        monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
        handlers = build_handlers(_StubRegistry(clink_tool))
        session = _FakeSession(protocol_version=LEGACY_VERSION, can_send_request=True)
        arguments = {"prompt": "hi", "_mcp_roots": {"uris": [], "warnings": []}}
        params = CallToolRequestParams(name="clink", arguments=arguments)

        await handlers.on_call_tool(_ctx(session), params)

        assert params.arguments == {"prompt": "hi", "_mcp_roots": {"uris": [], "warnings": []}}
