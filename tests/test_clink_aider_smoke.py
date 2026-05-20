"""Smoke tests for the Aider clink integration.

Validates the contract between Unison and Aider — manifest, agent class,
parser, and registry wiring. Fixtures are real Aider stdout captured
during the implementation spike (gpt-4o-mini against a scratch repo).
Mocked binary, no real Aider invocation in CI.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from clink import get_registry
from clink.agents import create_agent
from clink.agents.aider import AiderAgent
from clink.parsers.aider import AiderTextParser
from clink.parsers.base import ParserError

# ---------------------------------------------------------------------------
# Fixtures captured from real Aider invocations
# ---------------------------------------------------------------------------

# Real Aider stdout from spike: `aider --no-pretty --no-stream --no-auto-commits
# --yes-always --message-file <prompt.txt> --model gpt-4o-mini README.md` with
# prompt "What is 2+2? Answer in one number only." against a scratch repo.
AIDER_NO_EDIT_FIXTURE = """Warning: Input is not a terminal (fd=0).

You can skip this check with --no-gitignore
Added .aider* to .gitignore
Aider v0.86.2
Model: gpt-4o-mini with whole edit format
Git repo: .git with 1 files
Repo-map: using 4096 tokens, auto refresh
Added README.md to the chat.


https://aider.chat/HISTORY.html#release-notes

4

Tokens: 815 sent, 1 received. Cost: $0.00012 message, $0.00012 session.
"""

# Same spike with prompt "Add a paragraph below the heading explaining this is
# a test repo." — produced an edit.
AIDER_EDIT_FIXTURE = """Warning: Input is not a terminal (fd=0).

Aider v0.86.2
Model: gpt-4o-mini with whole edit format
Git repo: .git with 1 files
Repo-map: using 4096 tokens, auto refresh
Added README.md to the chat.

I will add a paragraph below the heading in the README.md file to explain that
this is a test repository.



README.md

```diff
@@ -1 +1,2 @@
 # scratch
+This is a test repository to demonstrate functionality and features.
```


Tokens: 815 sent, 42 received. Cost: $0.00015 message, $0.00015 session.
Applied edit to README.md
Shell cwd was reset to /Users/test/repo
"""


# ---------------------------------------------------------------------------
# Parser tests against real fixtures
# ---------------------------------------------------------------------------


class TestAiderParser:
    def test_no_edit_case(self) -> None:
        parsed = AiderTextParser().parse(AIDER_NO_EDIT_FIXTURE, "")
        assert parsed.content == "4"
        assert parsed.metadata["edits"] == []
        assert "Tokens:" in parsed.metadata["usage_line"]

    def test_edit_case_extracts_response_and_edits(self) -> None:
        parsed = AiderTextParser().parse(AIDER_EDIT_FIXTURE, "")
        # Response prose contains the explanation and the diff block
        assert "test repository" in parsed.content
        assert "```diff" in parsed.content
        # Edit was extracted
        assert parsed.metadata["edits"] == ["README.md"]

    def test_preamble_stripped(self) -> None:
        parsed = AiderTextParser().parse(AIDER_NO_EDIT_FIXTURE, "")
        # Preamble lines must not appear in content
        assert "Aider v0.86.2" not in parsed.content
        assert "Git repo:" not in parsed.content
        assert "Added README.md to the chat" not in parsed.content
        assert "Input is not a terminal" not in parsed.content
        assert "https://aider.chat" not in parsed.content

    def test_release_notes_url_stripped(self) -> None:
        parsed = AiderTextParser().parse(AIDER_NO_EDIT_FIXTURE, "")
        # Release-notes URL is preamble noise, must not pollute the response
        assert "HISTORY" not in parsed.content
        assert "release-notes" not in parsed.content

    def test_missing_tokens_line_raises(self) -> None:
        with pytest.raises(ParserError, match="Tokens.*summary"):
            AiderTextParser().parse("just some random output\nno tokens line", "")

    def test_empty_body_with_no_edits_raises(self) -> None:
        # Tokens: present but body fully preamble + no edits
        empty_fixture = """Aider v0.86.2
Git repo: .git with 0 files

Tokens: 100 sent, 0 received. Cost: $0.00001 message.
"""
        with pytest.raises(ParserError, match="no response text and no applied edits"):
            AiderTextParser().parse(empty_fixture, "")

    def test_empty_body_with_edits_synthesizes_summary(self) -> None:
        # Edge case: response is empty prose but edits happened — content
        # falls back to a brief summary so callers always have non-empty content.
        edits_only_fixture = """Aider v0.86.2
Git repo: .git with 1 files
Added README.md to the chat.



Tokens: 100 sent, 10 received. Cost: $0.00001 message.
Applied edit to README.md
"""
        parsed = AiderTextParser().parse(edits_only_fixture, "")
        assert "README.md" in parsed.content
        assert parsed.metadata["edits"] == ["README.md"]

    def test_stderr_captured_in_metadata(self) -> None:
        parsed = AiderTextParser().parse(AIDER_NO_EDIT_FIXTURE, "some diagnostic on stderr")
        assert parsed.metadata.get("stderr") == "some diagnostic on stderr"

    def test_empty_stderr_omitted_from_metadata(self) -> None:
        parsed = AiderTextParser().parse(AIDER_NO_EDIT_FIXTURE, "")
        assert "stderr" not in parsed.metadata


# ---------------------------------------------------------------------------
# Agent class wiring
# ---------------------------------------------------------------------------


class TestAiderAgentClass:
    def _make_agent(self) -> AiderAgent:
        from clink.models import ResolvedCLIClient, ResolvedCLIRole

        role = ResolvedCLIRole(
            name="default",
            prompt_path=Path("systemprompts/clink/default.txt").resolve(),
            role_args=[],
        )
        client = ResolvedCLIClient(
            name="aider",
            executable=["aider"],
            internal_args=[],
            config_args=["--no-pretty", "--no-stream", "--no-auto-commits", "--yes-always"],
            env={},
            timeout_seconds=30,
            parser="aider_text",
            roles={"default": role},
            output_to_file=None,
            working_dir=None,
        )
        return AiderAgent(client)

    def test_prepare_invocation_returns_message_file_plan(self) -> None:
        agent = self._make_agent()
        plan = agent.prepare_invocation("hi", [], [])
        assert plan.kind == "message_file"
        assert plan.flag == "--message-file"

    def test_read_only_args_returns_dry_run(self) -> None:
        agent = self._make_agent()
        assert agent.get_read_only_args() == ["--dry-run"]

    def test_model_flag_aliases(self) -> None:
        agent = self._make_agent()
        assert agent.model_flag_aliases == ("--model",)

    def test_render_model_args(self) -> None:
        agent = self._make_agent()
        assert agent.render_model_args("gpt-4o-mini") == ["--model", "gpt-4o-mini"]

    def test_fs_bookkeeping_patterns_cover_aider_files(self) -> None:
        agent = self._make_agent()
        patterns = agent.fs_violation_ignore_patterns
        assert ".aider.chat.history.md" in patterns
        assert ".aider.input.history" in patterns
        assert ".aider.tags.cache.v4/**" in patterns


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestAiderRegistryWiring:
    def test_aider_in_internal_defaults(self) -> None:
        from clink.constants import INTERNAL_DEFAULTS

        assert "aider" in INTERNAL_DEFAULTS
        assert INTERNAL_DEFAULTS["aider"].parser == "aider_text"
        assert INTERNAL_DEFAULTS["aider"].runner == "aider"

    def test_aider_in_agent_factory(self) -> None:
        from clink.agents import _AGENTS  # type: ignore[attr-defined]

        assert "aider" in _AGENTS
        assert _AGENTS["aider"] is AiderAgent

    def test_aider_parser_registered(self) -> None:
        from clink.parsers import _PARSER_CLASSES  # type: ignore[attr-defined]

        assert "aider_text" in _PARSER_CLASSES
        assert _PARSER_CLASSES["aider_text"] is AiderTextParser

    def test_registry_loads_aider_manifest(self) -> None:
        registry = get_registry()
        # If the manifest, agent, parser, and registry wiring are all in place,
        # the registry should expose 'aider' as an available CLI.
        assert "aider" in registry.list_clients()
        client = registry.get_client("aider")
        assert client.name == "aider"
        assert client.parser == "aider_text"
        # Default + planner + codereviewer roles from the manifest
        roles = registry.list_roles("aider")
        assert set(roles) >= {"default", "planner", "codereviewer"}

    def test_create_agent_returns_aider_agent(self) -> None:
        registry = get_registry()
        client = registry.get_client("aider")
        agent = create_agent(client)
        assert isinstance(agent, AiderAgent)


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


class TestAiderEndToEnd:
    @pytest.mark.asyncio
    async def test_no_edit_case_e2e(self, monkeypatch) -> None:
        registry = get_registry()
        client = registry.get_client("aider")
        agent = create_agent(client)
        role = client.roles["default"]

        process = _StubProcess(stdout=AIDER_NO_EDIT_FIXTURE.encode("utf-8"))

        captured_args: list[str] = []

        async def fake_exec(*args, **_kw):
            captured_args.extend(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        result = await agent.run(role=role, prompt="hi", files=[], images=[])

        # Aider uses message_file plan: empty stdin, --message-file flag in command
        assert process.received_stdin == b""
        assert "--message-file" in captured_args
        # Default config_args from the manifest are present
        assert "--no-pretty" in captured_args
        assert "--no-stream" in captured_args
        assert "--no-auto-commits" in captured_args
        assert "--yes-always" in captured_args
        # Not in read-only mode → no --dry-run
        assert "--dry-run" not in captured_args
        # Parser extracted "4" as the response
        assert result.parsed.content == "4"

    @pytest.mark.asyncio
    async def test_read_only_adds_dry_run(self, monkeypatch) -> None:
        registry = get_registry()
        client = registry.get_client("aider")
        agent = create_agent(client)
        role = client.roles["default"]

        process = _StubProcess(stdout=AIDER_EDIT_FIXTURE.encode("utf-8"))

        captured_args: list[str] = []

        async def fake_exec(*args, **_kw):
            captured_args.extend(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        await agent.run(role=role, prompt="hi", files=[], images=[], read_only=True)

        assert "--dry-run" in captured_args

    @pytest.mark.asyncio
    async def test_runtime_model_override(self, monkeypatch) -> None:
        registry = get_registry()
        client = registry.get_client("aider")
        agent = create_agent(client)
        role = client.roles["default"]

        process = _StubProcess(stdout=AIDER_NO_EDIT_FIXTURE.encode("utf-8"))

        captured_args: list[str] = []

        async def fake_exec(*args, **_kw):
            captured_args.extend(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        await agent.run(role=role, prompt="hi", files=[], images=[], model="claude-sonnet-4-5")

        # --model + value appended
        assert "--model" in captured_args
        idx = captured_args.index("--model")
        assert captured_args[idx + 1] == "claude-sonnet-4-5"
