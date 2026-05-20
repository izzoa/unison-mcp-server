"""Smoke tests for the Amp clink integration.

Validates the contract between Unison and Amp (https://ampcode.com).
Fixtures are real Amp JSONL stdout captured during the implementation spike
(amp 0.0.1775837683-g6ddb8e). Mocked binary, no real Amp invocation in CI.

Includes a recursion-guard test that verifies the cross-cutting Phase 0
guard fires for Amp invocations (Amp is MCP-aware via ``amp mcp``, so it's
the most likely target for a loop scenario).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from clink import get_registry
from clink.agents import create_agent
from clink.agents.amp import AmpAgent
from clink.constants import CLINK_DEPTH_ENV_VAR
from clink.parsers.amp import AmpJSONLParser
from clink.parsers.base import ParserError
from tools.clink import _check_recursion_guard
from tools.shared.exceptions import ToolExecutionError

# ---------------------------------------------------------------------------
# Fixtures captured from real Amp invocations
# ---------------------------------------------------------------------------

# Real Amp stdout from spike: `amp -x --stream-json "What is 2+2? Answer in
# one number only."` — JSONL with system/user/assistant/result events.
AMP_HAPPY_PATH_FIXTURE = """{"type":"system","subtype":"init","cwd":"/Users/test/repo","session_id":"T-019e47a8-5090-7538-9035-f2fde26859b7","tools":["Bash","chart","create_file","edit_file","find_thread","finder","glob","Grep","handoff","librarian","look_at","mermaid","oracle","painter","Read","read_mcp_resource","read_thread","read_web_page","skill","Task","task_list","undo_edit","web_search"],"mcp_servers":[]}
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"What is 2+2? Answer in one number only."}]},"parent_tool_use_id":null,"session_id":"T-019e47a8-5090-7538-9035-f2fde26859b7"}
{"type":"assistant","message":{"type":"message","role":"assistant","content":[{"type":"text","text":"4"}],"stop_reason":"end_turn","usage":{"input_tokens":7,"cache_creation_input_tokens":3,"cache_read_input_tokens":14009,"output_tokens":13,"max_tokens":300000,"service_tier":"standard"}},"parent_tool_use_id":null,"session_id":"T-019e47a8-5090-7538-9035-f2fde26859b7"}
{"type":"result","subtype":"success","duration_ms":2442,"is_error":false,"num_turns":1,"result":"4","session_id":"T-019e47a8-5090-7538-9035-f2fde26859b7"}
"""

# Multi-turn fixture: assistant emits multiple text blocks before the final
# result. Verifies the parser prefers the canonical `result` field but
# can fall back to concatenated assistant messages if absent.
AMP_MULTI_ASSISTANT_FIXTURE = """{"type":"system","subtype":"init","session_id":"T-abc"}
{"type":"assistant","message":{"type":"message","role":"assistant","content":[{"type":"text","text":"First, let me think about this."}],"stop_reason":"tool_use"}}
{"type":"assistant","message":{"type":"message","role":"assistant","content":[{"type":"text","text":"The answer is 4."}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":30}}}
{"type":"result","subtype":"success","result":"The answer is 4.","is_error":false,"session_id":"T-abc"}
"""

# Fallback case: result event missing, only assistant messages present.
AMP_FALLBACK_NO_RESULT_FIXTURE = """{"type":"system","subtype":"init","session_id":"T-xyz"}
{"type":"assistant","message":{"type":"message","role":"assistant","content":[{"type":"text","text":"Fallback message"}]}}
"""

# Synthetic auth error fixture (would be produced if AMP_API_KEY is missing
# or invalid). Amp typically writes to stderr in this case, with empty stdout.
AMP_AUTH_ERROR_STDERR = "Error: Not authenticated. Please run `amp login` or set AMP_API_KEY."


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestAmpParser:
    def test_happy_path_extracts_result_field(self) -> None:
        parsed = AmpJSONLParser().parse(AMP_HAPPY_PATH_FIXTURE, "")
        assert parsed.content == "4"

    def test_session_id_captured_in_metadata(self) -> None:
        parsed = AmpJSONLParser().parse(AMP_HAPPY_PATH_FIXTURE, "")
        assert parsed.metadata["session_id"] == "T-019e47a8-5090-7538-9035-f2fde26859b7"

    def test_usage_captured_in_metadata(self) -> None:
        parsed = AmpJSONLParser().parse(AMP_HAPPY_PATH_FIXTURE, "")
        assert parsed.metadata["usage"]["input_tokens"] == 7
        assert parsed.metadata["usage"]["output_tokens"] == 13

    def test_events_collected_in_metadata(self) -> None:
        parsed = AmpJSONLParser().parse(AMP_HAPPY_PATH_FIXTURE, "")
        types = [e.get("type") for e in parsed.metadata["events"]]
        assert types == ["system", "user", "assistant", "result"]

    def test_multi_assistant_prefers_result_field(self) -> None:
        # When both assistant messages and a result event exist, content
        # SHOULD be the result field (canonical answer), not the concatenation.
        parsed = AmpJSONLParser().parse(AMP_MULTI_ASSISTANT_FIXTURE, "")
        assert parsed.content == "The answer is 4."

    def test_fallback_to_assistant_when_no_result(self) -> None:
        parsed = AmpJSONLParser().parse(AMP_FALLBACK_NO_RESULT_FIXTURE, "")
        assert parsed.content == "Fallback message"

    def test_empty_stdout_with_stderr_raises_with_stderr(self) -> None:
        with pytest.raises(ParserError, match="Not authenticated"):
            AmpJSONLParser().parse("", AMP_AUTH_ERROR_STDERR)

    def test_empty_stdout_no_stderr_raises_generic(self) -> None:
        with pytest.raises(ParserError, match="no parseable response events"):
            AmpJSONLParser().parse("", "")

    def test_malformed_json_lines_skipped(self) -> None:
        # The parser tolerates non-JSON lines (just skips them) per the
        # codex_jsonl precedent.
        with_junk = "not json\n" + AMP_HAPPY_PATH_FIXTURE
        parsed = AmpJSONLParser().parse(with_junk, "")
        assert parsed.content == "4"

    def test_is_error_flag_propagates(self) -> None:
        error_fixture = (
            '{"type":"result","subtype":"error_max_turns","result":"Hit limit",' '"is_error":true,"session_id":"T-1"}\n'
        )
        parsed = AmpJSONLParser().parse(error_fixture, "")
        assert parsed.metadata["is_error"] is True
        assert parsed.metadata["error_subtype"] == "error_max_turns"


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class TestAmpAgentClass:
    def _make_agent(self) -> AmpAgent:
        from clink.models import ResolvedCLIClient, ResolvedCLIRole

        role = ResolvedCLIRole(
            name="default",
            prompt_path=Path("systemprompts/clink/default.txt").resolve(),
            role_args=[],
        )
        client = ResolvedCLIClient(
            name="amp",
            executable=["amp"],
            internal_args=[],
            config_args=["--execute", "--stream-json"],
            env={},
            timeout_seconds=30,
            parser="amp_jsonl",
            roles={"default": role},
            output_to_file=None,
            working_dir=None,
        )
        return AmpAgent(client)

    def test_text_only_returns_stdin_plan(self) -> None:
        agent = self._make_agent()
        plan = agent.prepare_invocation("hi", [], [])
        assert plan.kind == "stdin"

    def test_images_present_switches_to_stream_json(self) -> None:
        agent = self._make_agent()
        plan = agent.prepare_invocation("describe", [], ["/tmp/img.png"])
        assert plan.kind == "stream_json"

    def test_read_only_args_empty_no_native_flag(self) -> None:
        agent = self._make_agent()
        assert agent.get_read_only_args() == []

    def test_model_flag_aliases(self) -> None:
        agent = self._make_agent()
        assert "-m" in agent.model_flag_aliases
        assert "--mode" in agent.model_flag_aliases

    def test_render_model_args_uses_mode_flag(self) -> None:
        agent = self._make_agent()
        assert agent.render_model_args("deep") == ["--mode", "deep"]


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestAmpRegistryWiring:
    def test_amp_in_internal_defaults(self) -> None:
        from clink.constants import INTERNAL_DEFAULTS

        assert "amp" in INTERNAL_DEFAULTS
        assert INTERNAL_DEFAULTS["amp"].parser == "amp_jsonl"

    def test_amp_in_agent_factory(self) -> None:
        from clink.agents import _AGENTS  # type: ignore[attr-defined]

        assert "amp" in _AGENTS
        assert _AGENTS["amp"] is AmpAgent

    def test_amp_parser_registered(self) -> None:
        from clink.parsers import _PARSER_CLASSES  # type: ignore[attr-defined]

        assert "amp_jsonl" in _PARSER_CLASSES
        assert _PARSER_CLASSES["amp_jsonl"] is AmpJSONLParser

    def test_registry_loads_amp_manifest(self) -> None:
        registry = get_registry()
        assert "amp" in registry.list_clients()
        client = registry.get_client("amp")
        assert client.name == "amp"
        assert client.parser == "amp_jsonl"
        roles = registry.list_roles("amp")
        assert set(roles) >= {"default", "planner", "codereviewer"}

    def test_supported_models_allowlist_present(self) -> None:
        registry = get_registry()
        client = registry.get_client("amp")
        assert set(client.supported_models) == {"deep", "large", "rush", "smart"}


# ---------------------------------------------------------------------------
# End-to-end with mocked subprocess
# ---------------------------------------------------------------------------


class _StubProcess:
    def __init__(self, *, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.received_stdin: bytes | None = None

    async def communicate(self, stdin_data: bytes | None = None):
        self.received_stdin = stdin_data
        return self._stdout, self._stderr

    def kill(self) -> None:
        pass


class TestAmpEndToEnd:
    @pytest.mark.asyncio
    async def test_happy_path_e2e(self, monkeypatch) -> None:
        registry = get_registry()
        client = registry.get_client("amp")
        agent = create_agent(client)
        role = client.roles["default"]

        process = _StubProcess(stdout=AMP_HAPPY_PATH_FIXTURE.encode("utf-8"))
        captured_args: list[str] = []

        async def fake_exec(*args, **_kw):
            captured_args.extend(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        result = await agent.run(role=role, prompt="hi", files=[], images=[])

        # Text-only → stdin transport
        assert process.received_stdin == b"hi"
        # Manifest defaults
        assert "--execute" in captured_args
        assert "--stream-json" in captured_args
        # Parser pulled the canonical result
        assert result.parsed.content == "4"

    @pytest.mark.asyncio
    async def test_images_switch_to_stream_json_input(self, monkeypatch) -> None:
        registry = get_registry()
        client = registry.get_client("amp")
        agent = create_agent(client)
        role = client.roles["default"]

        process = _StubProcess(stdout=AMP_HAPPY_PATH_FIXTURE.encode("utf-8"))

        async def fake_exec(*args, **_kw):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        await agent.run(
            role=role,
            prompt="describe",
            files=[],
            images=["/tmp/screenshot.png"],
        )

        # stream_json transport serializes prompt+images to JSON over stdin
        assert process.received_stdin is not None
        import json as _json

        payload = _json.loads(process.received_stdin.decode("utf-8"))
        assert payload["messages"][0]["content"] == "describe"
        assert payload["images"] == ["/tmp/screenshot.png"]

    @pytest.mark.asyncio
    async def test_runtime_mode_selection(self, monkeypatch) -> None:
        registry = get_registry()
        client = registry.get_client("amp")
        agent = create_agent(client)
        role = client.roles["default"]

        process = _StubProcess(stdout=AMP_HAPPY_PATH_FIXTURE.encode("utf-8"))
        captured_args: list[str] = []

        async def fake_exec(*args, **_kw):
            captured_args.extend(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        await agent.run(role=role, prompt="hi", files=[], images=[], model="deep")

        assert "--mode" in captured_args
        idx = captured_args.index("--mode")
        assert captured_args[idx + 1] == "deep"


# ---------------------------------------------------------------------------
# Recursion guard: Amp is MCP-aware so this is the most likely target
# ---------------------------------------------------------------------------


class TestAmpRecursionGuard:
    """Verify the cross-cutting Phase 0 guard fires when Amp is the spawn target.

    The guard itself is implemented at CLinkTool.execute() entry (covered by
    test_clink_recursion_guard.py). These tests confirm that pattern catches
    the Amp-specific scenario where a user wires Unison as an MCP server in
    Amp's config AND invokes clink with cli_name='amp' from a Unison-aware CLI.
    """

    def test_guard_fires_at_depth_2_simulating_amp_loop(self, monkeypatch) -> None:
        """Top-level call → Amp spawned (depth=1) → Amp's MCP-server Unison
        re-invokes clink → guard at depth=2 trips."""
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "2")
        with pytest.raises(ToolExecutionError, match="recursion limit"):
            _check_recursion_guard()

    def test_amp_normal_invocation_passes_guard(self, monkeypatch) -> None:
        """A normal `clink with cli_name='amp'` call (not wired into a loop) succeeds."""
        monkeypatch.delenv(CLINK_DEPTH_ENV_VAR, raising=False)
        _check_recursion_guard()  # passes

    def test_amp_agent_env_propagates_depth(self) -> None:
        """Verify AmpAgent's _build_environment inherits the Phase 0 depth-propagation
        behavior from BaseCLIAgent (no Amp-specific override needed)."""
        from clink.models import ResolvedCLIClient, ResolvedCLIRole

        role = ResolvedCLIRole(
            name="default",
            prompt_path=Path("systemprompts/clink/default.txt").resolve(),
            role_args=[],
        )
        client = ResolvedCLIClient(
            name="amp",
            executable=["amp"],
            internal_args=[],
            config_args=[],
            env={},
            timeout_seconds=30,
            parser="amp_jsonl",
            roles={"default": role},
            output_to_file=None,
            working_dir=None,
        )
        agent = AmpAgent(client)
        env = agent._build_environment()
        # The depth was unset in the parent process (or 0), so the spawned
        # subprocess sees 1.
        assert env[CLINK_DEPTH_ENV_VAR] == "1"
