"""Graceful-first termination: interrupt the CLI's process group before SIGKILL.

An agent that declares ``interrupt_signal`` gets the signal, a grace period to
exit, and only then the group SIGKILL. These tests drive real subprocesses (a
Python fake CLI that traps, ignores, or floods on SIGINT) because the whole
point is the signal semantics, not a mocked process object.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import anyio
import pytest

import clink.agents.opencode as opencode_agent
from clink.agents.base import BaseCLIAgent, CLIAgentError
from clink.agents.opencode import OpencodeAgent
from clink.models import ResolvedCLIClient, ResolvedCLIRole

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process-group signalling is POSIX-only")

# argv: mode ready_file trap_file exit_delay
#   trap   - SIGINT handler records the signal, waits exit_delay, exits 130
#   ignore - SIGINT is ignored (only SIGKILL stops it)
#   big    - SIGINT handler writes 1 MiB to stdout (more than a pipe holds) and exits 130
_FAKE_CLI = textwrap.dedent("""
    import os, pathlib, signal, sys, time

    mode, ready_file, trap_file, exit_delay = sys.argv[1:5]

    def handler(signum, frame):
        if mode == "big":
            sys.stdout.write("x" * (1 << 20))
            sys.stdout.flush()
        time.sleep(float(exit_delay))
        pathlib.Path(trap_file).write_text("SIGINT")
        sys.exit(130)

    if mode == "ignore":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGINT, handler)

    print('{"sessionID":"ses_fake"}', flush=True)
    pathlib.Path(ready_file).write_text(str(os.getpid()))
    while True:
        time.sleep(0.05)
    """)


class _InterruptAgent(BaseCLIAgent):
    interrupt_signal = signal.SIGINT
    interrupt_grace_seconds = 3.0

    def __init__(self, client: ResolvedCLIClient) -> None:
        super().__init__(client)
        self.force_kills: list[tuple[bytes, str | None]] = []

    def _on_force_kill(self, captured_stdout: bytes, cwd: str | None) -> None:
        self.force_kills.append((captured_stdout, cwd))


class _RecordingOpencodeAgent(OpencodeAgent):
    """The real opencode agent, recording what its force-kill hook receives."""

    def __init__(self, client: ResolvedCLIClient) -> None:
        super().__init__(client)
        self.force_kills: list[tuple[bytes, str | None]] = []

    def _on_force_kill(self, captured_stdout: bytes, cwd: str | None) -> None:
        self.force_kills.append((captured_stdout, cwd))
        super()._on_force_kill(captured_stdout, cwd)


@pytest.fixture(params=["generic", "opencode"])
def make_agent(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run the core cases against a generic declaring agent and the real opencode agent."""
    if request.param == "opencode":
        monkeypatch.setattr(opencode_agent, "_probe_opencode_version", lambda _path: "2.2.0")
        return _RecordingOpencodeAgent
    return _InterruptAgent


class _Harness:
    def __init__(self, tmp_path: Path, mode: str, exit_delay: float = 0.0, timeout: int = 30) -> None:
        script = tmp_path / "fake_cli.py"
        script.write_text(_FAKE_CLI)
        self.ready_file = tmp_path / "ready"
        self.trap_file = tmp_path / "trap"
        role = ResolvedCLIRole(
            name="default",
            prompt_path=Path("systemprompts/clink/default.txt").resolve(),
            role_args=[],
        )
        self.client = ResolvedCLIClient(
            name="stop-test",
            executable=[sys.executable, str(script), mode, str(self.ready_file), str(self.trap_file), str(exit_delay)],
            internal_args=[],
            config_args=[],
            env={},
            timeout_seconds=timeout,
            parser="codex_jsonl",
            roles={"default": role},
            output_to_file=None,
            working_dir=None,
        )
        self.role = role

    def run(self, agent: BaseCLIAgent) -> Any:
        return agent.run(role=self.role, prompt="hello", files=[], images=[])

    async def wait_ready(self, spawned: list[Any]) -> int:
        """Wait until the fake CLI has installed its handler AND clink holds its process object.

        The ready file can appear before ``create_subprocess_exec`` has returned (a loaded
        machine makes that gap real); cancelling inside it kills the bare child before any
        stop logic exists, which is not what these tests exercise.
        """
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if spawned and self.ready_file.exists() and self.ready_file.read_text():
                return int(self.ready_file.read_text())
            await asyncio.sleep(0.02)
        raise AssertionError("fake CLI never became ready")


@pytest.fixture()
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.subprocess.Process]:
    processes: list[asyncio.subprocess.Process] = []
    real = asyncio.create_subprocess_exec

    async def recording(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await real(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording)
    return processes


async def _wait_dead(process: asyncio.subprocess.Process, timeout: float = 5.0) -> bool:
    """True once the process is gone *and* asyncio has recorded its exit status.

    The pid can vanish before the loop's child watcher has set ``returncode``, so
    checking only ``os.kill(pid, 0)`` flakes on a loaded machine.
    """
    try:
        await asyncio.wait_for(process.wait(), timeout)
    except asyncio.TimeoutError:
        return False
    return True


@pytest.mark.asyncio
async def test_timeout_interrupts_before_killing(tmp_path: Path, spawned: list[Any], make_agent: Any) -> None:
    harness = _Harness(tmp_path, "trap", timeout=3)
    agent = make_agent(harness.client)

    with pytest.raises(CLIAgentError, match="timed out after"):
        await harness.run(agent)

    assert harness.trap_file.read_text() == "SIGINT"
    assert spawned[0].returncode == 130
    assert agent.force_kills == []


@pytest.mark.asyncio
async def test_ignored_interrupt_is_force_killed_after_the_grace_period(
    tmp_path: Path, spawned: list[Any], make_agent: Any
) -> None:
    harness = _Harness(tmp_path, "ignore", timeout=3)
    agent = make_agent(harness.client)
    agent.interrupt_grace_seconds = 0.5

    started = time.monotonic()
    with pytest.raises(CLIAgentError, match="timed out after"):
        await harness.run(agent)
    elapsed = time.monotonic() - started

    assert spawned[0].returncode == -signal.SIGKILL
    # timeout + grace + the bounded drain, with generous slack for a busy machine
    assert elapsed < 3 + 0.5 + 5 + 2
    assert len(agent.force_kills) == 1
    assert b"ses_fake" in agent.force_kills[0][0]


@pytest.mark.asyncio
async def test_force_kill_hook_receives_output_on_the_cancellation_path(
    tmp_path: Path, spawned: list[Any], make_agent: Any
) -> None:
    harness = _Harness(tmp_path, "ignore")
    agent = make_agent(harness.client)
    agent.interrupt_grace_seconds = 0.3

    task = asyncio.create_task(harness.run(agent))
    await harness.wait_ready(spawned)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await _wait_dead(spawned[0])
    assert spawned[0].returncode == -signal.SIGKILL
    assert len(agent.force_kills) == 1
    assert b"ses_fake" in agent.force_kills[0][0]


@pytest.mark.asyncio
async def test_native_cancel_lets_the_cli_exit_on_the_interrupt(
    tmp_path: Path, spawned: list[Any], make_agent: Any
) -> None:
    harness = _Harness(tmp_path, "trap", exit_delay=0.2)
    agent = make_agent(harness.client)

    task = asyncio.create_task(harness.run(agent))
    await harness.wait_ready(spawned)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert harness.trap_file.read_text() == "SIGINT"
    assert spawned[0].returncode == 130
    assert agent.force_kills == []


@pytest.mark.asyncio
async def test_second_native_cancel_does_not_skip_the_force_kill(
    tmp_path: Path, spawned: list[Any], make_agent: Any
) -> None:
    harness = _Harness(tmp_path, "ignore")
    agent = make_agent(harness.client)
    agent.interrupt_grace_seconds = 30.0

    task = asyncio.create_task(harness.run(agent))
    await harness.wait_ready(spawned)
    task.cancel()
    await asyncio.sleep(0.3)  # now waiting out the grace period
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await _wait_dead(spawned[0])
    assert time.monotonic() - started < 5
    assert spawned[0].returncode == -signal.SIGKILL
    assert len(agent.force_kills) == 1


@pytest.mark.asyncio
async def test_anyio_scope_cancellation_does_not_cut_the_grace_period_short(
    tmp_path: Path, spawned: list[Any], make_agent: Any
) -> None:
    """The MCP SDK cancels a request through an anyio CancelScope, which re-delivers on every await."""
    harness = _Harness(tmp_path, "trap", exit_delay=0.3)
    agent = make_agent(harness.client)
    holder: dict[str, anyio.CancelScope] = {}

    async def request() -> None:
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            await harness.run(agent)

    task = asyncio.create_task(request())
    await harness.wait_ready(spawned)
    holder["scope"].cancel()
    await task

    assert harness.trap_file.read_text() == "SIGINT"
    assert spawned[0].returncode == 130
    assert agent.force_kills == []


@pytest.mark.asyncio
async def test_agent_without_a_signal_is_killed_immediately(tmp_path: Path, spawned: list[Any]) -> None:
    harness = _Harness(tmp_path, "trap", exit_delay=0.2)
    agent = BaseCLIAgent(harness.client)  # declares no interrupt signal

    task = asyncio.create_task(harness.run(agent))
    await harness.wait_ready(spawned)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await _wait_dead(spawned[0])
    assert spawned[0].returncode == -signal.SIGKILL
    assert not harness.trap_file.exists()


@pytest.mark.asyncio
async def test_output_keeps_draining_during_the_grace_period(
    tmp_path: Path, spawned: list[Any], make_agent: Any
) -> None:
    """A CLI that writes more than a pipe holds in its interrupt handler must still be able to exit."""
    harness = _Harness(tmp_path, "big", timeout=3)
    agent = make_agent(harness.client)
    agent.interrupt_grace_seconds = 10.0

    with pytest.raises(CLIAgentError, match="timed out after"):
        await harness.run(agent)

    assert harness.trap_file.read_text() == "SIGINT"
    assert spawned[0].returncode == 130
    assert agent.force_kills == []


@pytest.mark.asyncio
async def test_normal_run_output_is_unchanged_by_live_capture(tmp_path: Path) -> None:
    script = tmp_path / "echo_cli.py"
    script.write_text(
        'import sys\nsys.stdin.read()\nprint(\'{"type":"item.completed","item":{"id":"x","type":"agent_message","text":"hi"}}\')\n'
    )
    role = ResolvedCLIRole(name="default", prompt_path=Path("systemprompts/clink/default.txt").resolve(), role_args=[])
    client = ResolvedCLIClient(
        name="echo-test",
        executable=[sys.executable, str(script)],
        internal_args=[],
        config_args=[],
        env={},
        timeout_seconds=30,
        parser="codex_jsonl",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    agent = _InterruptAgent(client)

    result = await agent.run(role=role, prompt="hello", files=[], images=[])

    assert result.returncode == 0
    assert "hi" in result.parsed.content
    assert agent.force_kills == []


# ---------------------------------------------------------------------------
# Background-job cancel and server shutdown reach the same graceful path
# ---------------------------------------------------------------------------

_JOB_ARGS = {"prompt": "hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []}


class _JobTools:
    @pytest.fixture(autouse=True)
    def _isolated_jobs(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        from tools.clink import CLinkTool
        from tools.clink_jobs import reset_clink_job_manager_for_tests

        reset_clink_job_manager_for_tests()
        monkeypatch.setattr(CLinkTool, "_DEFAULT_JOB_WAIT_SECONDS", 0.05)
        yield
        reset_clink_job_manager_for_tests()

    @staticmethod
    def _payload(result: Any) -> dict[str, Any]:
        import json

        return json.loads(json.loads(result[0].text)["content"])

    async def _start(
        self, monkeypatch: pytest.MonkeyPatch, harness: _Harness, grace: float, spawned: list[Any]
    ) -> tuple[_InterruptAgent, dict[str, Any]]:
        from tools.clink import CLinkTool

        agent = _InterruptAgent(harness.client)
        agent.interrupt_grace_seconds = grace
        monkeypatch.setattr("tools.clink.create_agent", lambda _client: agent)
        running = self._payload(await CLinkTool().execute(dict(_JOB_ARGS)))
        assert running["status"] == "running"
        await harness.wait_ready(spawned)
        return agent, running


class TestJobCancelAndShutdown(_JobTools):
    @pytest.mark.asyncio
    async def test_cancel_interrupts_the_cli_and_reports_cancelled(
        self, tmp_path: Path, spawned: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools.clink import CLinkTool

        harness = _Harness(tmp_path, "trap", exit_delay=0.2)
        agent, running = await self._start(monkeypatch, harness, grace=3.0, spawned=spawned)

        reply = self._payload(await CLinkTool().execute({"job_id": running["job_id"], "cancel": True}))

        assert reply["status"] == "cancelled"
        assert harness.trap_file.read_text() == "SIGINT"
        assert spawned[0].returncode == 130
        assert agent.force_kills == []

    @pytest.mark.asyncio
    async def test_cancel_of_a_cli_that_ignores_the_interrupt_is_never_reported_running(
        self, tmp_path: Path, spawned: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools.clink import CLinkTool

        harness = _Harness(tmp_path, "ignore")
        agent, running = await self._start(monkeypatch, harness, grace=0.5, spawned=spawned)

        reply = self._payload(await CLinkTool().execute({"job_id": running["job_id"], "cancel": True}))

        assert reply["status"] in {"cancelled", "cancelling"}
        assert "still working" not in reply.get("instructions", "")
        await asyncio.wait_for(spawned[0].wait(), 5)  # reaped asynchronously after the SIGKILL
        assert spawned[0].returncode == -signal.SIGKILL
        assert len(agent.force_kills) == 1

    @pytest.mark.asyncio
    async def test_cancel_reply_says_cancelling_while_the_grace_period_runs(
        self, tmp_path: Path, spawned: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools.clink import CLinkTool

        monkeypatch.setattr(CLinkTool, "_CANCEL_WAIT_SECONDS", 0.1)
        harness = _Harness(tmp_path, "ignore")
        _, running = await self._start(monkeypatch, harness, grace=2.0, spawned=spawned)

        reply = self._payload(await CLinkTool().execute({"job_id": running["job_id"], "cancel": True}))
        assert reply["status"] == "cancelling"
        assert "Do not resubmit" in reply["instructions"]

        # A later poll still reports it as cancelling, then cancelled — never "running".
        again = self._payload(await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 0}))
        assert again["status"] == "cancelling"
        final = self._payload(await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 5}))
        assert final["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_shutdown_interrupts_running_jobs(
        self, tmp_path: Path, spawned: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools.clink_jobs import shutdown_clink_jobs

        harness = _Harness(tmp_path, "trap", exit_delay=0.2)
        agent, _ = await self._start(monkeypatch, harness, grace=3.0, spawned=spawned)

        await shutdown_clink_jobs()

        assert harness.trap_file.read_text() == "SIGINT"
        assert spawned[0].returncode == 130
        assert agent.force_kills == []


# ---------------------------------------------------------------------------
# opencode: a force-killed run reports the session it may have left suspended
# ---------------------------------------------------------------------------


class TestOpencodeOrphanReport:
    @pytest.fixture(autouse=True)
    def _v2_binary(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setattr(opencode_agent, "_probe_opencode_version", lambda _path: "2.2.0")
        caplog.set_level(logging.WARNING, logger="clink.runner.opencode")

    @staticmethod
    def _orphan_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if "force-killed" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_timeout_path(self, tmp_path: Path, spawned: list[Any], caplog: pytest.LogCaptureFixture) -> None:
        harness = _Harness(tmp_path, "ignore", timeout=3)
        agent = OpencodeAgent(harness.client)
        agent.interrupt_grace_seconds = 0.5

        with pytest.raises(CLIAgentError, match="timed out after"):
            await harness.run(agent)

        (warning,) = self._orphan_warnings(caplog)
        assert "ses_fake" in warning
        assert "opencode session delete --standalone ses_fake" in warning

    @pytest.mark.asyncio
    async def test_cancellation_path(
        self, tmp_path: Path, spawned: list[Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = _Harness(tmp_path, "ignore")
        agent = OpencodeAgent(harness.client)
        agent.interrupt_grace_seconds = 0.3

        task = asyncio.create_task(harness.run(agent))
        await harness.wait_ready(spawned)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        (warning,) = self._orphan_warnings(caplog)
        assert "opencode session delete --standalone ses_fake" in warning

    @pytest.mark.asyncio
    async def test_a_clean_interrupt_reports_nothing(
        self, tmp_path: Path, spawned: list[Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = _Harness(tmp_path, "trap", timeout=3)
        agent = OpencodeAgent(harness.client)

        with pytest.raises(CLIAgentError, match="timed out after"):
            await harness.run(agent)

        assert self._orphan_warnings(caplog) == []


# ---------------------------------------------------------------------------
# Review follow-ups: idempotent job cancel, host-cancel semantics, non-POSIX, task cleanup
# ---------------------------------------------------------------------------


class TestJobCancelSemantics(_JobTools):
    """Reuses the job fixtures (isolated manager, short foreground wait) of the class above."""

    @pytest.mark.asyncio
    async def test_a_repeated_cancel_does_not_cut_the_grace_period_short(
        self, tmp_path: Path, spawned: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools.clink import CLinkTool

        harness = _Harness(tmp_path, "trap", exit_delay=2.0)  # exits cleanly, but only after 2 s
        agent, running = await self._start(monkeypatch, harness, grace=10.0, spawned=spawned)

        first = asyncio.create_task(CLinkTool().execute({"job_id": running["job_id"], "cancel": True}))
        await asyncio.sleep(0.3)  # the CLI is now inside its grace period
        second = self._payload(await CLinkTool().execute({"job_id": running["job_id"], "cancel": True}))
        first_reply = self._payload(await first)

        assert {first_reply["status"], second["status"]} <= {"cancelling", "cancelled"}
        assert harness.trap_file.read_text() == "SIGINT"
        assert spawned[0].returncode == 130  # not -9: the second cancel did not force-kill it
        assert agent.force_kills == []

    @pytest.mark.asyncio
    async def test_a_host_cancel_of_the_waiting_request_leaves_the_job_running(
        self, tmp_path: Path, spawned: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default hybrid mode: the host abandoning its wait must not stop the CLI; only cancel=true does."""
        from tools.clink import CLinkTool
        from tools.clink_jobs import get_clink_job_manager

        harness = _Harness(tmp_path, "trap", exit_delay=0.2)
        _, running = await self._start(monkeypatch, harness, grace=3.0, spawned=spawned)

        waiting = asyncio.create_task(CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 30}))
        await asyncio.sleep(0.3)
        waiting.cancel()  # the MCP host cancelling the request it was waiting on
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await asyncio.sleep(0.5)

        job = get_clink_job_manager().get(running["job_id"])
        assert job is not None and job.state == "running"
        assert spawned[0].returncode is None and not harness.trap_file.exists()

        reply = self._payload(await CLinkTool().execute({"job_id": running["job_id"], "cancel": True}))
        assert reply["status"] == "cancelled"
        assert harness.trap_file.read_text() == "SIGINT"  # now stopped through the graceful path


@pytest.mark.asyncio
async def test_a_second_cancel_during_the_stop_still_cancels_the_output_readers(
    tmp_path: Path, spawned: list[Any], make_agent: Any
) -> None:
    harness = _Harness(tmp_path, "ignore")
    agent = make_agent(harness.client)
    agent.interrupt_grace_seconds = 30.0

    task = asyncio.create_task(harness.run(agent))
    await harness.wait_ready(spawned)
    task.cancel()
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _wait_dead(spawned[0])
    await asyncio.sleep(0.2)

    leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leftover == []


@pytest.mark.asyncio
async def test_without_posix_process_groups_the_stop_is_an_immediate_kill_with_no_false_orphan_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On Windows nothing can be signalled, so it behaves like an agent that declares no signal."""
    import clink.agents.base as base_module

    class _NtOs:
        name = "nt"

        def __getattr__(self, attr: str) -> Any:
            return getattr(os, attr)

    class _Process:
        pid = 987654321
        returncode = None

        def __init__(self) -> None:
            self.killed = False
            self.signalled = False

        def kill(self) -> None:
            self.killed = True

        def send_signal(self, _sig: int) -> None:
            self.signalled = True

    monkeypatch.setattr(base_module, "os", _NtOs())
    harness_client = _Harness(tmp_path, "trap").client
    agent = _InterruptAgent(harness_client)
    process = _Process()

    await agent._stop_process(process, bytearray(b'{"sessionID":"ses_x"}'), None)  # type: ignore[arg-type]

    assert process.killed and not process.signalled
    assert agent.force_kills == []
