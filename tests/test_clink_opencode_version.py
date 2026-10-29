"""opencode version gate: v2-only arguments appear only on opencode v2.

opencode v2 takes ``--standalone`` (private server, so clink's environment and
process lifetime apply) and benefits from ``--title`` (skips a paid
title-generation request). v1 accepts neither ``--standalone`` nor needs
``--title``, so the v1 command line must stay byte-for-byte what it was.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

import clink.agents.opencode as oc
from clink import get_registry
from clink.agents.base import BaseCLIAgent, CLIAgentError
from clink.agents.opencode import OpencodeAgent
from clink.agents.opencode import (
    _probe_opencode_version as real_probe,  # the conftest stub replaces the module attribute
)

LOGGER = "clink.runner.opencode"
ROLES = ["default", "planner", "codereviewer"]


def _fake_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "opencode"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return binary


def _prime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str | None) -> str:
    """Make ``shutil.which`` resolve to a real file whose probed version is already cached."""
    binary = _fake_binary(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: str(binary))
    key = oc._binary_key(str(binary))
    assert key is not None
    oc._VERSION_CACHE[key] = version
    return str(binary)


def _agent(**client_overrides: Any) -> OpencodeAgent:
    client = get_registry().get_client("opencode")
    if client_overrides:
        client = client.model_copy(update=client_overrides)
    return OpencodeAgent(client)


# ---------------------------------------------------------------------------
# The probe function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stdout,expected",
    [
        ("opencode v2.2.0\n", "2.2.0"),
        ("1.18.35\n", "1.18.35"),
        ("opencode v0.0.0-dev-176\n", "0.0.0"),
        ("no digits here\n", None),
        ("", None),
    ],
)
def test_probe_parses_version_output(monkeypatch: pytest.MonkeyPatch, stdout: str, expected: str | None) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **_k: subprocess.CompletedProcess(a, 0, stdout, ""))
    assert real_probe("/x/opencode") == expected


def test_probe_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **_k: subprocess.CompletedProcess(a, 1, "2.2.0", ""))
    assert real_probe("/x/opencode") is None

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("exec format error")

    monkeypatch.setattr(subprocess, "run", boom)
    assert real_probe("/x/opencode") is None

    def hang(*a: Any, **_k: Any) -> None:
        raise subprocess.TimeoutExpired(a, 5)

    monkeypatch.setattr(subprocess, "run", hang)
    assert real_probe("/x/opencode") is None


# ---------------------------------------------------------------------------
# Detection: once per binary, off the event loop, logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_version_is_probed_once_and_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    binary = str(_fake_binary(tmp_path))
    calls: list[str] = []
    monkeypatch.setattr(oc, "_probe_opencode_version", lambda path: calls.append(path) or "2.2.0")
    caplog.set_level(logging.INFO, logger=LOGGER)

    assert await oc._detect_version(binary) == "2.2.0"
    assert await oc._detect_version(binary) == "2.2.0"

    assert calls == [binary]
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert infos == [f"opencode at {binary} is v2.2.0"]


@pytest.mark.asyncio
async def test_concurrent_first_calls_probe_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = str(_fake_binary(tmp_path))
    calls: list[str] = []

    def slow_probe(path: str) -> str:
        calls.append(path)
        time.sleep(0.2)
        return "2.2.0"

    monkeypatch.setattr(oc, "_probe_opencode_version", slow_probe)

    results = await asyncio.gather(*(oc._detect_version(binary) for _ in range(5)))

    assert results == ["2.2.0"] * 5
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_failed_probe_is_cached_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    binary = str(_fake_binary(tmp_path))
    calls: list[str] = []
    monkeypatch.setattr(oc, "_probe_opencode_version", lambda path: calls.append(path))
    caplog.set_level(logging.INFO, logger=LOGGER)

    assert await oc._detect_version(binary) is None
    assert await oc._detect_version(binary) is None

    assert len(calls) == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert binary in warnings[0]
    assert "v2 isolation" in warnings[0] and "not applied" in warnings[0]


@pytest.mark.asyncio
async def test_binary_change_triggers_a_new_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    versions = iter(["1.18.35", "2.2.0"])
    monkeypatch.setattr(oc, "_probe_opencode_version", lambda _path: next(versions))

    assert await oc._detect_version(str(binary)) == "1.18.35"
    stat = binary.stat()
    os.utime(binary, (stat.st_atime, stat.st_mtime + 10))  # an in-place upgrade
    assert await oc._detect_version(str(binary)) == "2.2.0"


@pytest.mark.asyncio
async def test_probe_does_not_stall_other_coroutines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = str(_fake_binary(tmp_path))
    monkeypatch.setattr(oc, "_probe_opencode_version", lambda _path: time.sleep(0.4) or "2.2.0")
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await oc._detect_version(binary)
    task.cancel()

    assert ticks >= 10  # a blocking probe would have allowed none


@pytest.mark.asyncio
async def test_unstattable_path_skips_probe_and_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(oc, "_probe_opencode_version", lambda path: calls.append(path) or "2.2.0")
    caplog.set_level(logging.INFO, logger=LOGGER)

    assert await oc._detect_version("/nonexistent/opencode") is None

    assert calls == []
    assert caplog.records == []


@pytest.mark.asyncio
async def test_unresolvable_executable_fails_with_the_existing_error_and_no_version_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(oc, "_probe_opencode_version", lambda path: calls.append(path) or "2.2.0")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    caplog.set_level(logging.INFO, logger=LOGGER)
    agent = _agent()

    with pytest.raises(CLIAgentError, match="not found in PATH"):
        await agent.run(role=agent.client.roles["default"], prompt="x", files=[], images=[])

    assert calls == []
    assert caplog.records == []


# ---------------------------------------------------------------------------
# Command line: v1 unchanged, v2 gains --standalone and --title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role_name", ROLES)
@pytest.mark.parametrize("model", [None, "anthropic/claude-sonnet-4-5"])
@pytest.mark.parametrize("version", ["1.18.35", "0.18.4", "0.0.0", None])
def test_pre_v2_command_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role_name: str, model: str | None, version: str | None
) -> None:
    _prime(monkeypatch, tmp_path, version)
    agent = _agent()
    role = agent.client.roles[role_name]

    command = agent._build_command(role=role, system_prompt=None, model=model)

    assert command == BaseCLIAgent._build_command(agent, role=role, system_prompt=None, model=model)
    assert "--standalone" not in command and "--title" not in command
    assert agent._apply_read_only(list(command)) == command  # read-only stays byte-identical


@pytest.mark.parametrize("role_name", ROLES)
@pytest.mark.parametrize("model", [None, "anthropic/claude-sonnet-4-5"])
def test_v2_command_appends_standalone_and_title_after_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role_name: str, model: str | None
) -> None:
    _prime(monkeypatch, tmp_path, "2.2.0")
    agent = _agent()
    role = agent.client.roles[role_name]

    command = agent._build_command(role=role, system_prompt=None, model=model)

    base = BaseCLIAgent._build_command(agent, role=role, system_prompt=None, model=model)
    assert command == base + ["--standalone", "--title", f"clink:{role_name}"]
    if model:
        assert command[-5:-3] == ["-m", model]


def test_unprobed_binary_is_treated_as_pre_v2_without_probing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: str(binary))
    monkeypatch.setattr(oc, "_probe_opencode_version", lambda _path: pytest.fail("_build_command must never probe"))
    agent = _agent()

    command = agent._build_command(role=agent.client.roles["default"], system_prompt=None)

    assert "--standalone" not in command


def test_standalone_is_not_duplicated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prime(monkeypatch, tmp_path, "2.2.0")
    agent = _agent(config_args=["--standalone"])

    command = agent._build_command(role=agent.client.roles["default"], system_prompt=None)

    assert command.count("--standalone") == 1
    assert command[-2:] == ["--title", "clink:default"]


def test_manifest_title_is_respected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prime(monkeypatch, tmp_path, "2.2.0")
    agent = _agent(config_args=["--title", "mine"])

    command = agent._build_command(role=agent.client.roles["default"], system_prompt=None)

    assert command.count("--title") == 1
    assert "clink:default" not in command


@pytest.mark.parametrize("server_args", [["--server", "http://127.0.0.1:4096"], ["--server=http://127.0.0.1:4096"]])
def test_manifest_server_suppresses_standalone_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, server_args: list[str]
) -> None:
    _prime(monkeypatch, tmp_path, "2.2.0")
    caplog.set_level(logging.WARNING, logger=LOGGER)
    agent = _agent(config_args=server_args)
    role = agent.client.roles["default"]

    first = agent._build_command(role=role, system_prompt=None)
    agent._build_command(role=role, system_prompt=None)

    assert "--standalone" not in first
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1 and "the manifest passes --server" in warnings[0]


def test_service_mode_is_an_explicit_warned_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _prime(monkeypatch, tmp_path, "2.2.0")
    monkeypatch.setenv("CLINK_OPENCODE_MODE", "service")
    caplog.set_level(logging.WARNING, logger=LOGGER)
    agent = _agent()
    role = agent.client.roles["default"]

    command = agent._build_command(role=role, system_prompt=None)
    agent._build_command(role=role, system_prompt=None)

    assert "--standalone" not in command
    assert command[-2:] == ["--title", "clink:default"]
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1  # once per process
    text = warnings[0]
    for fragment in (
        "CLINK_OPENCODE_MODE=service",
        "recursion-depth propagation",
        "secret stripping",
        "cancellation is only best-effort",
        "auto-starts a detached service that outlives clink",
        "inherits clink's environment",
    ):
        assert fragment in text


def test_invalid_mode_falls_back_to_standalone_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _prime(monkeypatch, tmp_path, "2.2.0")
    monkeypatch.setenv("CLINK_OPENCODE_MODE", "bogus")
    caplog.set_level(logging.WARNING, logger=LOGGER)
    agent = _agent()

    command = agent._build_command(role=agent.client.roles["default"], system_prompt=None)

    assert "--standalone" in command
    assert any("'bogus'" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Interrupt declaration and orphan report
# ---------------------------------------------------------------------------


def test_opencode_declares_a_sigint_interrupt_that_the_job_cancel_reply_covers() -> None:
    from tools.clink import CLinkTool

    assert OpencodeAgent.interrupt_signal == signal.SIGINT
    assert OpencodeAgent.interrupt_grace_seconds == 5.0
    assert CLinkTool._CANCEL_WAIT_SECONDS >= OpencodeAgent.interrupt_grace_seconds + 2


def test_force_kill_reports_the_session_and_a_safe_cleanup_command(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    captured = b'{"type":"step_start","sessionID":"ses_abc123","part":{}}\n{"type":"text"'

    _agent()._on_force_kill(captured, "/work/project")

    (record,) = caplog.records
    message = record.getMessage()
    assert "ses_abc123" in message
    assert "opencode session delete --standalone ses_abc123" in message


def test_force_kill_without_a_session_id_names_the_working_directory(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)

    _agent()._on_force_kill(b"", "/work/project")

    (record,) = caplog.records
    message = record.getMessage()
    assert "/work/project" in message
    assert "opencode session delete --standalone" in message


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "executable,is_launcher",
    [
        (["opencode", "run"], True),
        (["/opt/homebrew/bin/opencode", "run"], True),
        (["opencode.exe", "run"], True),
        (["opencode2", "run"], True),
        (["npx", "-y", "opencode-ai", "run"], False),
        (["bunx", "opencode-ai", "run"], False),
        (["node", "opencode.js", "run"], False),
    ],
)
def test_only_an_opencode_launcher_is_version_probed(executable: list[str], is_launcher: bool) -> None:
    agent = _agent(executable=executable)

    assert agent._is_opencode_launcher() is is_launcher


def test_a_wrapper_command_never_gets_v2_arguments_from_the_wrappers_own_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `npx --version` prints npx's version (10.9.2): its major says nothing about opencode.
    npx = tmp_path / "npx"
    npx.write_text("#!/bin/sh\n")
    npx.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name: str(npx))
    key = oc._binary_key(str(npx))
    assert key is not None
    oc._VERSION_CACHE[key] = "10.9.2"  # even if it were cached, it must not be consulted
    monkeypatch.setattr(oc, "_probe_opencode_version", lambda _path: pytest.fail("a wrapper must not be probed"))
    agent = _agent(executable=["npx", "-y", "opencode-ai", "run"])

    command = agent._build_command(role=agent.client.roles["default"], system_prompt=None)

    assert agent._resolve_binary() is None
    assert "--standalone" not in command and "--title" not in command


def test_force_kill_on_v1_does_not_give_v2_cleanup_advice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _prime(monkeypatch, tmp_path, "1.18.35")
    caplog.set_level(logging.WARNING, logger=LOGGER)

    _agent()._on_force_kill(b'{"sessionID":"ses_abc123"}', "/work/project")

    (record,) = caplog.records
    message = record.getMessage()
    assert "ses_abc123" in message
    assert "--standalone" not in message and "resumed by your opencode service" not in message


def test_force_kill_report_survives_a_deleted_server_working_directory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _OsWithoutCwd:
        def getcwd(self) -> str:
            raise FileNotFoundError("cwd was deleted")

        def __getattr__(self, attr: str) -> Any:
            return getattr(os, attr)

    monkeypatch.setattr(oc, "os", _OsWithoutCwd())
    caplog.set_level(logging.WARNING, logger=LOGGER)

    _agent()._on_force_kill(b"", None)

    (record,) = caplog.records
    assert "working directory" in record.getMessage()
    assert "opencode session delete --standalone" in record.getMessage()


def test_the_model_parameter_description_mentions_variants_for_opencode_v2() -> None:
    from tools.clink import CLinkRequest, CLinkTool

    schema_description = CLinkTool().get_input_schema()["properties"]["model"]["description"]
    field_description = CLinkRequest.model_fields["model"].description or ""

    for description in (schema_description, field_description):
        assert "provider/model" in description
        assert "#variant" in description and "v2" in description
