"""clink against a real opencode v2 binary, fully sandboxed (opt-in: ``-m integration``).

Every test runs opencode with private ``XDG_*`` dirs, a temp project whose
``opencode.json`` points at a local mock provider, and ``--standalone`` on every
helper command, so the user's opencode config, database, MCP servers and
background service are never touched and nothing is spent. Skipped cleanly when
no opencode v2 binary is on PATH.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from clink import get_registry
from clink.agents.base import CLIAgentError
from clink.agents.opencode import OpencodeAgent
from clink.agents.opencode import _probe_opencode_version as probe_version
from tests.opencode_support.sandbox import Sandbox, sessions_in_real_database
from utils.fs_snapshot import capture_snapshot, diff_snapshots

pytestmark = pytest.mark.integration


@pytest.fixture()
def make_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Sandbox]]:
    binary = shutil.which("opencode")
    if binary is None:
        pytest.skip("opencode is not installed or on PATH")
    version = probe_version(binary)
    if version is None or int(version.split(".")[0]) < 2:
        pytest.skip(f"opencode v2 is required (found {version})")

    started: list[Sandbox] = []

    def make(**kwargs: Any) -> Sandbox:
        box = Sandbox(tmp_path / f"sandbox{len(started)}", **kwargs).start()
        started.append(box)
        for name, value in box.env.items():
            monkeypatch.setenv(name, value)
        # The server "was launched" somewhere else: a working_dir that is ignored
        # would make opencode load that directory instead of the sandbox project.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir(exist_ok=True)
        monkeypatch.chdir(elsewhere)
        monkeypatch.setenv("PWD", str(elsewhere))
        return box

    yield make

    for box in started:
        box.stop()
        assert sessions_in_real_database(box.root) == 0, "sandbox sessions leaked into the user's opencode database"


def _agent() -> OpencodeAgent:
    return OpencodeAgent(get_registry().get_client("opencode"))


def _run(agent: OpencodeAgent, box: Sandbox, model: str, *, read_only: bool = False) -> Any:
    return agent.run(
        role=agent.client.roles["default"],
        prompt="Reply with the single word PONG.",
        files=[],
        images=[],
        read_only=read_only,
        model=model,
        working_dir=box.project,
    )


def _service_pids() -> set[int]:
    listing = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True, check=False).stdout
    return {int(line.split()[0]) for line in listing.splitlines()[1:] if "serve --service" in line}


async def _wait_for(condition: Callable[[], bool], timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_working_dir_is_honored(make_sandbox: Callable[..., Sandbox]) -> None:
    box = make_sandbox(tool_command="pwd")

    result = await _run(_agent(), box, "mock/tool")

    tool_events = [e for e in result.parsed.metadata["events"] if e.get("type") == "tool_use"]
    assert tool_events, "the mock model's shell call never ran"
    ran_in = Path(tool_events[0]["part"]["state"]["output"].strip())
    assert ran_in.resolve() == box.project.resolve()


@pytest.mark.asyncio
async def test_default_call_is_standalone_untitled_and_leaves_the_service_alone(
    make_sandbox: Callable[..., Sandbox],
) -> None:
    box = make_sandbox()
    services_before = _service_pids()

    result = await _run(_agent(), box, "mock/text")

    assert result.parsed.content == "PONG"
    assert "--standalone" in result.sanitized_command
    assert result.sanitized_command[-2:] == ["--title", "clink:default"]
    assert _service_pids() == services_before
    # With a title opencode makes no title-generation request: exactly one model request.
    assert len(box.chat_requests()) == 1


@pytest.mark.asyncio
async def test_environment_reaches_mcp_servers_and_stripped_secrets_do_not(
    make_sandbox: Callable[..., Sandbox], monkeypatch: pytest.MonkeyPatch
) -> None:
    box = make_sandbox()
    monkeypatch.setenv("UNISON_CLINK_STRIP_SECRETS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-marker-must-not-leak")

    await _run(_agent(), box, "mock/text")

    await _wait_for(lambda: box.mcp_env() is not None, timeout=30)
    mcp_env = box.mcp_env()
    assert mcp_env is not None
    assert mcp_env["UNISON_CLINK_DEPTH"] == "1"
    assert "OPENAI_API_KEY" not in mcp_env


@pytest.mark.asyncio
async def test_provider_requests_do_not_use_stripped_credentials(
    make_sandbox: Callable[..., Sandbox], monkeypatch: pytest.MonkeyPatch
) -> None:
    box = make_sandbox(api_key_var="OPENAI_API_KEY")  # the provider's key comes from a stripped variable
    monkeypatch.setenv("UNISON_CLINK_STRIP_SECRETS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-marker-must-not-leak")

    try:
        await _run(_agent(), box, "mock/text")
    except CLIAgentError:
        pass  # without its key the call may fail; what matters is what reached the provider

    assert all("sk-marker-must-not-leak" not in str(r.get("authorization")) for r in box.mock_events())


@pytest.mark.asyncio
async def test_read_only_call_leaves_no_bookkeeping(make_sandbox: Callable[..., Sandbox]) -> None:
    box = make_sandbox()
    before = capture_snapshot(box.project)

    await _run(_agent(), box, "mock/text", read_only=True)

    assert not diff_snapshots(before, capture_snapshot(box.project)).has_changes


@pytest.mark.asyncio
async def test_cancellation_interrupts_the_session_and_releases_its_claim(
    make_sandbox: Callable[..., Sandbox], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="clink.runner.opencode")
    box = make_sandbox()
    task = asyncio.create_task(_run(_agent(), box, "mock/slow"))
    await _wait_for(lambda: any(r.get("tool_names") for r in box.chat_requests()))  # the main request is streaming
    await asyncio.sleep(2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    ((session_id, suspended),) = box.sessions()
    assert suspended is None, "the session kept its execution claim: opencode's service would resume it"
    exported = box.cli("session", "export", session_id)
    assert exported.returncode == 0, exported.stderr
    assert "interrupted" in exported.stdout
    assert any(e.get("event") == "client_disconnected" for e in box.mock_events())
    assert not [r for r in caplog.records if "force-killed" in r.getMessage()]  # SIGINT sufficed


@pytest.mark.asyncio
async def test_two_concurrent_calls_both_succeed(make_sandbox: Callable[..., Sandbox]) -> None:
    box = make_sandbox()
    # Two standalone servers racing to initialize a brand-new opencode database can fail
    # ("Standalone server exited before reporting readiness": 1 of 2 cold trials). Real use
    # always has a database, where 6 of 6 concurrent calls succeeded, so warm it first.
    await _run(_agent(), box, "mock/text")
    first, second = await asyncio.gather(_run(_agent(), box, "mock/text"), _run(_agent(), box, "mock/text"))

    assert first.parsed.content == second.parsed.content == "PONG"
    assert len({sid for sid, _ in box.sessions()}) == 3  # the warm-up plus the two concurrent calls
