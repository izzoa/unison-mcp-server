"""The spawned CLI's PWD must follow its effective working directory.

opencode (v1.18+ and v2) roots its session at ``$PWD`` rather than the process
cwd. clink copies ``os.environ``, whose ``PWD`` names the directory the Unison
server was launched from, so a per-call ``working_dir`` was silently ignored
and the CLI read and wrote the wrong project.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from clink.agents.base import BaseCLIAgent
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from utils.fs_snapshot import capture_snapshot, classify_changes, diff_snapshots

_OK_STDOUT = b'{"type":"item.completed","item":{"id":"x","type":"agent_message","text":"ok"}}\n'


class _StubProcess:
    returncode = 0

    async def communicate(self, _data: bytes | None = None) -> tuple[bytes, bytes]:
        return _OK_STDOUT, b""


def _make_agent(
    *,
    executable: list[str] | None = None,
    working_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[BaseCLIAgent, ResolvedCLIRole]:
    role = ResolvedCLIRole(
        name="default",
        prompt_path=Path("systemprompts/clink/default.txt").resolve(),
        role_args=[],
    )
    client = ResolvedCLIClient(
        name="pwd-test",
        executable=executable or ["pwd-test-binary"],
        internal_args=[],
        config_args=[],
        env=env or {},
        timeout_seconds=30,
        parser="codex_jsonl",
        roles={"default": role},
        output_to_file=None,
        working_dir=working_dir,
    )
    return BaseCLIAgent(client), role


def _capture_spawn(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_exec(*_args: Any, **kwargs: Any) -> _StubProcess:
        captured.update(kwargs)
        return _StubProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    return captured


@pytest.mark.asyncio
async def test_per_call_working_dir_sets_pwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PWD", "/stale/launch/dir")
    agent, role = _make_agent()
    captured = _capture_spawn(monkeypatch)

    await agent.run(role=role, prompt="x", files=[], images=[], working_dir=tmp_path)

    assert captured["cwd"] == str(tmp_path)
    assert captured["env"]["PWD"] == str(tmp_path)


@pytest.mark.asyncio
async def test_manifest_working_dir_sets_pwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PWD", "/stale/launch/dir")
    agent, role = _make_agent(working_dir=tmp_path)
    captured = _capture_spawn(monkeypatch)

    await agent.run(role=role, prompt="x", files=[], images=[])

    assert captured["env"]["PWD"] == str(tmp_path)


@pytest.mark.asyncio
async def test_per_call_working_dir_beats_manifest_pwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent, role = _make_agent(env={"PWD": "/manifest/stale"})
    captured = _capture_spawn(monkeypatch)

    await agent.run(role=role, prompt="x", files=[], images=[], working_dir=tmp_path)

    assert captured["env"]["PWD"] == str(tmp_path)


@pytest.mark.asyncio
async def test_without_working_dir_pwd_matches_inherited_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The server chdir'd after launch: its inherited PWD no longer names its cwd.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/stale/launch/dir")
    agent, role = _make_agent()
    captured = _capture_spawn(monkeypatch)

    await agent.run(role=role, prompt="x", files=[], images=[])

    assert captured["cwd"] is None
    assert captured["env"]["PWD"] == os.getcwd()


def test_build_environment_without_argument_still_works(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    agent, _ = _make_agent()

    env = agent._build_environment()

    assert env["PWD"] == os.getcwd()


@pytest.mark.skipif(os.name == "nt", reason="a directory in use cannot be removed on Windows")
@pytest.mark.asyncio
async def test_deleted_server_cwd_removes_pwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent, role = _make_agent(env={"PWD": "/manifest/stale"})
    captured = _capture_spawn(monkeypatch)
    doomed = tmp_path / "doomed"
    doomed.mkdir()
    monkeypatch.chdir(doomed)
    monkeypatch.setenv("PWD", str(doomed))
    doomed.rmdir()

    await agent.run(role=role, prompt="x", files=[], images=[])

    assert "PWD" not in captured["env"]


# A CLI that reads the raw environment variable, the way opencode does. This
# must be a Python script: POSIX shells replace a stale inherited PWD with the
# physical cwd, so a shell-script fake would write into the right directory
# even without the fix and prove nothing.
_PWD_WRITER = textwrap.dedent("""
    import os, sys
    sys.stdin.read()
    with open(os.path.join(os.environ["PWD"], "written.txt"), "w") as fh:
        fh.write("x")
    print('{"type":"item.completed","item":{"id":"x","type":"agent_message","text":"ok"}}')
    """)


def _pwd_writer_agent(tmp_path: Path) -> tuple[BaseCLIAgent, ResolvedCLIRole]:
    script = tmp_path / "pwd_writer.py"
    script.write_text(_PWD_WRITER)
    return _make_agent(executable=[sys.executable, str(script)])


@pytest.mark.asyncio
async def test_read_only_snapshot_covers_directory_the_cli_writes_to(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work_dir = tmp_path / "work"
    launch_dir = tmp_path / "launch"
    work_dir.mkdir()
    launch_dir.mkdir()
    # The server was launched in launch_dir; the call asks for work_dir.
    monkeypatch.chdir(launch_dir)
    monkeypatch.setenv("PWD", str(launch_dir))
    agent, role = _pwd_writer_agent(tmp_path)

    before = capture_snapshot(work_dir)
    await agent.run(role=role, prompt="x", files=[], images=[], working_dir=work_dir)
    by_model, _ = classify_changes(
        diff_snapshots(before, capture_snapshot(work_dir)), agent.fs_violation_ignore_patterns
    )

    assert by_model.created == ["written.txt"]
    assert list(launch_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_negative_control_without_pwd_override_write_lands_elsewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Proves the test above detects the bug: with the old environment the file lands in the launch dir."""
    work_dir = tmp_path / "work"
    launch_dir = tmp_path / "launch"
    work_dir.mkdir()
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    monkeypatch.setenv("PWD", str(launch_dir))
    agent, role = _pwd_writer_agent(tmp_path)

    def old_environment(cwd: str | None = None) -> dict[str, str]:
        return {**os.environ, **agent.client.env}

    monkeypatch.setattr(agent, "_build_environment", old_environment)

    before = capture_snapshot(work_dir)
    await agent.run(role=role, prompt="x", files=[], images=[], working_dir=work_dir)
    diff = diff_snapshots(before, capture_snapshot(work_dir))

    assert diff.created == []
    assert (launch_dir / "written.txt").exists()
