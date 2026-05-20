"""Smoke tests for the Crush clink integration.

Validates the contract between Unison and Crush (https://github.com/charmbracelet/crush).
Fixtures are real Crush stdout/stderr captured during the implementation spike
(crush v0.70.0 + openai/gpt-4o-mini). Mocked binary, no real Crush invocation in CI.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from clink import get_registry
from clink.agents import create_agent
from clink.agents.crush import CrushAgent
from clink.parsers.base import ParserError
from clink.parsers.crush import CrushTextParser

# ---------------------------------------------------------------------------
# Fixtures captured from real Crush invocations
# ---------------------------------------------------------------------------

# Real Crush stdout from spike: `crush run --quiet --model openai/gpt-4o-mini
# "What is 2+2? Answer in one number only."` — Crush's --quiet output is the
# response text alone, no preamble, no token footer.
CRUSH_HAPPY_PATH_FIXTURE = "4\n"

# A longer response — verifies the parser doesn't truncate or modify content.
CRUSH_LONGER_RESPONSE = 'The file `test.md` contains a single header titled "scratch" and no additional content.\n'

# Real Crush stderr from an error case (invalid model). Crush prints a styled
# ERROR box; stdout is empty when the error fires before model invocation.
CRUSH_ERROR_STDERR = """
   ERROR

  Failed to override models: large model "openai/nonexistent-model" not found.
"""


# ---------------------------------------------------------------------------
# Parser tests against real fixtures
# ---------------------------------------------------------------------------


class TestCrushParser:
    def test_happy_path_returns_stdout_content(self) -> None:
        parsed = CrushTextParser().parse(CRUSH_HAPPY_PATH_FIXTURE, "")
        assert parsed.content == "4"
        assert parsed.metadata == {}

    def test_longer_response_preserved_verbatim(self) -> None:
        parsed = CrushTextParser().parse(CRUSH_LONGER_RESPONSE, "")
        assert parsed.content == CRUSH_LONGER_RESPONSE.strip()

    def test_stderr_captured_in_metadata_when_stdout_present(self) -> None:
        parsed = CrushTextParser().parse(CRUSH_HAPPY_PATH_FIXTURE, "some diagnostic")
        assert parsed.metadata["stderr"] == "some diagnostic"

    def test_empty_stderr_omitted_from_metadata(self) -> None:
        parsed = CrushTextParser().parse(CRUSH_HAPPY_PATH_FIXTURE, "")
        assert "stderr" not in parsed.metadata

    def test_empty_stdout_with_stderr_raises_with_stderr_message(self) -> None:
        with pytest.raises(ParserError, match="Failed to override models"):
            CrushTextParser().parse("", CRUSH_ERROR_STDERR)

    def test_empty_stdout_no_stderr_raises_generic(self) -> None:
        with pytest.raises(ParserError, match="no stdout content and no stderr"):
            CrushTextParser().parse("", "")

    def test_whitespace_only_stdout_treated_as_empty(self) -> None:
        with pytest.raises(ParserError):
            CrushTextParser().parse("   \n\n  \n", "")


# ---------------------------------------------------------------------------
# Agent class wiring
# ---------------------------------------------------------------------------


class TestCrushAgentClass:
    def _make_agent(self) -> CrushAgent:
        from clink.models import ResolvedCLIClient, ResolvedCLIRole

        role = ResolvedCLIRole(
            name="default",
            prompt_path=Path("systemprompts/clink/default.txt").resolve(),
            role_args=[],
        )
        client = ResolvedCLIClient(
            name="crush",
            executable=["crush"],
            internal_args=[],
            config_args=["run", "--quiet"],
            env={},
            timeout_seconds=30,
            parser="crush_text",
            roles={"default": role},
            output_to_file=None,
            working_dir=None,
        )
        return CrushAgent(client)

    def test_prepare_invocation_returns_default_stdin_plan(self) -> None:
        # Crush accepts stdin per Charm docs — use the default
        agent = self._make_agent()
        plan = agent.prepare_invocation("hi", [], [])
        assert plan.kind == "stdin"

    def test_read_only_args_empty_no_native_flag(self) -> None:
        agent = self._make_agent()
        assert agent.get_read_only_args() == []

    def test_model_flag_aliases_supports_short_and_long(self) -> None:
        agent = self._make_agent()
        assert "-m" in agent.model_flag_aliases
        assert "--model" in agent.model_flag_aliases

    def test_render_model_args_uses_long_flag(self) -> None:
        agent = self._make_agent()
        assert agent.render_model_args("openai/gpt-4o-mini") == ["--model", "openai/gpt-4o-mini"]

    def test_fs_bookkeeping_covers_crush_dir(self) -> None:
        agent = self._make_agent()
        assert ".crush/**" in agent.fs_violation_ignore_patterns


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestCrushRegistryWiring:
    def test_crush_in_internal_defaults(self) -> None:
        from clink.constants import INTERNAL_DEFAULTS

        assert "crush" in INTERNAL_DEFAULTS
        assert INTERNAL_DEFAULTS["crush"].parser == "crush_text"
        assert INTERNAL_DEFAULTS["crush"].runner == "crush"

    def test_crush_in_agent_factory(self) -> None:
        from clink.agents import _AGENTS  # type: ignore[attr-defined]

        assert "crush" in _AGENTS
        assert _AGENTS["crush"] is CrushAgent

    def test_crush_parser_registered(self) -> None:
        from clink.parsers import _PARSER_CLASSES  # type: ignore[attr-defined]

        assert "crush_text" in _PARSER_CLASSES
        assert _PARSER_CLASSES["crush_text"] is CrushTextParser

    def test_registry_loads_crush_manifest(self) -> None:
        registry = get_registry()
        assert "crush" in registry.list_clients()
        client = registry.get_client("crush")
        assert client.name == "crush"
        assert client.parser == "crush_text"
        roles = registry.list_roles("crush")
        assert set(roles) >= {"default", "planner", "codereviewer"}

    def test_create_agent_returns_crush_agent(self) -> None:
        registry = get_registry()
        client = registry.get_client("crush")
        agent = create_agent(client)
        assert isinstance(agent, CrushAgent)


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


class TestCrushEndToEnd:
    @pytest.mark.asyncio
    async def test_happy_path_e2e(self, monkeypatch) -> None:
        registry = get_registry()
        client = registry.get_client("crush")
        agent = create_agent(client)
        role = client.roles["default"]

        process = _StubProcess(stdout=CRUSH_HAPPY_PATH_FIXTURE.encode("utf-8"))
        captured_args: list[str] = []

        async def fake_exec(*args, **_kw):
            captured_args.extend(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        result = await agent.run(role=role, prompt="hi", files=[], images=[])

        # Crush uses stdin transport (default) — prompt sent via stdin
        assert process.received_stdin == b"hi"
        # Manifest's `run --quiet` propagated
        assert "run" in captured_args
        assert "--quiet" in captured_args
        # Default config_args from manifest do NOT include --model (user opts in)
        assert "--model" not in captured_args
        # Parser extracted "4"
        assert result.parsed.content == "4"

    @pytest.mark.asyncio
    async def test_runtime_model_appends_flag(self, monkeypatch) -> None:
        registry = get_registry()
        client = registry.get_client("crush")
        agent = create_agent(client)
        role = client.roles["default"]

        process = _StubProcess(stdout=CRUSH_HAPPY_PATH_FIXTURE.encode("utf-8"))
        captured_args: list[str] = []

        async def fake_exec(*args, **_kw):
            captured_args.extend(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        await agent.run(
            role=role,
            prompt="hi",
            files=[],
            images=[],
            model="anthropic/claude-sonnet-4-5",
        )

        assert "--model" in captured_args
        idx = captured_args.index("--model")
        assert captured_args[idx + 1] == "anthropic/claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_read_only_does_not_add_native_flag(self, monkeypatch) -> None:
        """Crush has no native dry-run; read_only=True should add no extra args."""
        registry = get_registry()
        client = registry.get_client("crush")
        agent = create_agent(client)
        role = client.roles["default"]

        process = _StubProcess(stdout=CRUSH_HAPPY_PATH_FIXTURE.encode("utf-8"))
        captured_args: list[str] = []

        async def fake_exec(*args, **_kw):
            captured_args.extend(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        before_count = len(captured_args)
        await agent.run(role=role, prompt="hi", files=[], images=[], read_only=False)
        baseline = len(captured_args) - before_count

        captured_args.clear()
        process2 = _StubProcess(stdout=CRUSH_HAPPY_PATH_FIXTURE.encode("utf-8"))

        async def fake_exec2(*args, **_kw):
            captured_args.extend(args)
            return process2

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec2)
        await agent.run(role=role, prompt="hi", files=[], images=[], read_only=True)
        ro_count = len(captured_args)

        # read_only should add 0 extra args for Crush (no native flag)
        assert ro_count == baseline
