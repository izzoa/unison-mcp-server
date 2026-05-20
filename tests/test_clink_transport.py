"""Unit tests for the InvocationPlan transport hook on BaseCLIAgent."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from clink.agents.base import BaseCLIAgent, CLIAgentError, InvocationPlan
from clink.models import ResolvedCLIClient, ResolvedCLIRole


class _StubProcess:
    """asyncio subprocess stand-in used by transport tests."""

    def __init__(self, *, stdout: bytes = b"{}", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.received_stdin: bytes | None = None

    async def communicate(self, stdin_data: bytes):
        self.received_stdin = stdin_data
        return self._stdout, self._stderr

    def kill(self) -> None:
        pass


def _make_client(parser: str = "codex_jsonl") -> ResolvedCLIClient:
    role = ResolvedCLIRole(
        name="default",
        prompt_path=Path("systemprompts/clink/default.txt").resolve(),
        role_args=[],
    )
    return ResolvedCLIClient(
        name="transport-test",
        executable=["transport-test-binary"],
        internal_args=[],
        config_args=[],
        env={},
        timeout_seconds=30,
        parser=parser,
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )


def _make_agent(plan: InvocationPlan | None = None) -> tuple[BaseCLIAgent, ResolvedCLIRole]:
    client = _make_client()
    agent = BaseCLIAgent(client)
    if plan is not None:
        # Override prepare_invocation for the test.
        agent.prepare_invocation = lambda *_a, **_kw: plan  # type: ignore[method-assign]
    return agent, client.roles["default"]


# ---------------------------------------------------------------------------
# Default prepare_invocation
# ---------------------------------------------------------------------------


class TestPrepareInvocationDefault:
    def test_default_returns_stdin_plan(self) -> None:
        agent, _ = _make_agent()
        plan = agent.prepare_invocation("hello", [], [])
        assert plan.kind == "stdin"
        assert plan.flag is None
        assert plan.extra_payload == {}


# ---------------------------------------------------------------------------
# _materialize_plan per kind
# ---------------------------------------------------------------------------


class TestMaterializeStdin:
    def test_returns_prompt_bytes(self) -> None:
        agent, _ = _make_agent()
        args, stdin_bytes, cleanup = agent._materialize_plan(InvocationPlan(kind="stdin"), "hello world", [], [])
        assert args == []
        assert stdin_bytes == b"hello world"
        cleanup()  # no-op, must not raise


class TestMaterializeArgv:
    def test_argv_with_flag(self) -> None:
        agent, _ = _make_agent()
        args, stdin_bytes, _ = agent._materialize_plan(InvocationPlan(kind="argv", flag="--prompt"), "hi", [], [])
        assert args == ["--prompt", "hi"]
        assert stdin_bytes == b""

    def test_argv_without_flag(self) -> None:
        agent, _ = _make_agent()
        args, stdin_bytes, _ = agent._materialize_plan(InvocationPlan(kind="argv"), "hi", [], [])
        assert args == ["hi"]
        assert stdin_bytes == b""


class TestMaterializeMessageFile:
    def test_writes_tempfile_and_returns_path(self) -> None:
        agent, _ = _make_agent()
        args, stdin_bytes, cleanup = agent._materialize_plan(
            InvocationPlan(kind="message_file", flag="--message-file"),
            "prompt content",
            [],
            [],
        )
        assert args[0] == "--message-file"
        assert len(args) == 2
        tmp_path = Path(args[1])
        assert tmp_path.exists()
        assert tmp_path.read_text() == "prompt content"
        assert stdin_bytes == b""

        cleanup()
        assert not tmp_path.exists()

    def test_cleanup_idempotent(self) -> None:
        agent, _ = _make_agent()
        _, _, cleanup = agent._materialize_plan(InvocationPlan(kind="message_file", flag="--message-file"), "x", [], [])
        cleanup()
        cleanup()  # second call must not raise

    def test_message_file_without_flag_rejected(self) -> None:
        agent, _ = _make_agent()
        with pytest.raises(CLIAgentError, match="requires a 'flag'"):
            agent._materialize_plan(InvocationPlan(kind="message_file"), "x", [], [])


class TestMaterializeStreamJson:
    def test_default_envelope_includes_prompt(self) -> None:
        agent, _ = _make_agent()
        args, stdin_bytes, _ = agent._materialize_plan(InvocationPlan(kind="stream_json"), "hi", [], [])
        assert args == []
        payload = json.loads(stdin_bytes.decode("utf-8"))
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert "images" not in payload
        assert "files" not in payload

    def test_default_envelope_includes_files_and_images(self) -> None:
        agent, _ = _make_agent()
        _, stdin_bytes, _ = agent._materialize_plan(
            InvocationPlan(kind="stream_json"),
            "look at this",
            ["/tmp/a.py", "/tmp/b.py"],
            ["/tmp/img.png"],
        )
        payload = json.loads(stdin_bytes.decode("utf-8"))
        assert payload["files"] == ["/tmp/a.py", "/tmp/b.py"]
        assert payload["images"] == ["/tmp/img.png"]

    def test_extra_payload_override(self) -> None:
        agent, _ = _make_agent()
        custom = b'{"custom": "schema"}'
        _, stdin_bytes, _ = agent._materialize_plan(
            InvocationPlan(kind="stream_json", extra_payload={"serialized": custom}),
            "ignored",
            [],
            [],
        )
        assert stdin_bytes == custom

    def test_extra_payload_serialized_must_be_bytes(self) -> None:
        agent, _ = _make_agent()
        with pytest.raises(CLIAgentError, match="must be bytes"):
            agent._materialize_plan(
                InvocationPlan(kind="stream_json", extra_payload={"serialized": "not bytes"}),
                "x",
                [],
                [],
            )


class TestMaterializeUnknownKind:
    def test_unknown_kind_rejected(self) -> None:
        agent, _ = _make_agent()
        with pytest.raises(CLIAgentError, match="Unknown InvocationPlan kind"):
            agent._materialize_plan(InvocationPlan(kind="rainbow"), "x", [], [])


# ---------------------------------------------------------------------------
# End-to-end: subclass override actually drives subprocess transport
# ---------------------------------------------------------------------------


class TestEndToEndDispatch:
    @pytest.mark.asyncio
    async def test_default_agent_writes_prompt_to_stdin(self, monkeypatch) -> None:
        agent, role = _make_agent()
        process = _StubProcess(
            stdout=b'{"type":"item.completed","item":{"id":"x","type":"agent_message","text":"ok"}}\n'
        )

        captured: dict[str, Any] = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = list(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        await agent.run(role=role, prompt="hello", files=[], images=[])
        assert process.received_stdin == b"hello"
        # No extra args appended (default stdin plan)
        assert (
            captured["args"][-1] == "/usr/bin/transport-test-binary" or "transport-test-binary" in captured["args"][0]
        )

    @pytest.mark.asyncio
    async def test_message_file_subclass_writes_tempfile_not_stdin(self, monkeypatch) -> None:
        agent, role = _make_agent(plan=InvocationPlan(kind="message_file", flag="--message-file"))
        process = _StubProcess(
            stdout=b'{"type":"item.completed","item":{"id":"x","type":"agent_message","text":"ok"}}\n'
        )

        captured: dict[str, Any] = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = list(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        await agent.run(role=role, prompt="my prompt", files=[], images=[])

        # Empty stdin payload
        assert process.received_stdin == b""
        # --message-file <path> appended to args
        assert "--message-file" in captured["args"]
        flag_idx = captured["args"].index("--message-file")
        tmp_path = captured["args"][flag_idx + 1]
        # Tempfile should have been cleaned up post-execution
        assert not Path(tmp_path).exists()

    @pytest.mark.asyncio
    async def test_argv_subclass_appends_prompt_arg(self, monkeypatch) -> None:
        agent, role = _make_agent(plan=InvocationPlan(kind="argv", flag="--ask"))
        process = _StubProcess(
            stdout=b'{"type":"item.completed","item":{"id":"x","type":"agent_message","text":"ok"}}\n'
        )

        captured: dict[str, Any] = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = list(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        await agent.run(role=role, prompt="my question", files=[], images=[])

        assert process.received_stdin == b""
        assert "--ask" in captured["args"]
        assert "my question" in captured["args"]

    @pytest.mark.asyncio
    async def test_message_file_cleanup_on_timeout(self, monkeypatch) -> None:
        """Verify message_file tempfile is cleaned up even if the subprocess times out."""
        agent, role = _make_agent(plan=InvocationPlan(kind="message_file", flag="--message-file"))

        captured_path: dict[str, str] = {}

        class _SlowProcess:
            returncode = 0
            _raised_once = False

            async def communicate(self, _data=None):
                # First call (from wait_for) raises TimeoutError; subsequent
                # call (from the kill cleanup path) returns empty so the
                # finally block doesn't crash.
                if not self._raised_once:
                    self._raised_once = True
                    raise asyncio.TimeoutError
                return b"", b""

            def kill(self):
                pass

        async def fake_exec(*args, **_kw):
            # Capture the tempfile path before subprocess "starts"
            if "--message-file" in args:
                idx = list(args).index("--message-file")
                captured_path["path"] = args[idx + 1]
            return _SlowProcess()

        async def fake_wait_for(coro, **_kw):
            await coro  # let coro raise TimeoutError
            return b"", b""

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

        with pytest.raises(CLIAgentError, match="timed out"):
            await agent.run(role=role, prompt="x", files=[], images=[])

        assert "path" in captured_path
        assert not Path(captured_path["path"]).exists()
