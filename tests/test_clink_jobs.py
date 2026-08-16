"""Tests for the clink background-job API (clink_start / clink_poll / clink_cancel).

The job API exists because some MCP hosts cancel tool calls long before a
real CLI run finishes: no single MCP request may outlive the host's patience,
so long runs become a start + bounded polls.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import tools.clink_jobs as clink_jobs
from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from tools.clink_jobs import ClinkCancelTool, ClinkPollTool, ClinkStartTool, reset_clink_job_manager_for_tests
from tools.shared.exceptions import ToolExecutionError


@pytest.fixture(autouse=True)
def _fresh_manager():
    reset_clink_job_manager_for_tests()
    yield
    reset_clink_job_manager_for_tests()


def _slow_dummy_agent_factory(delay: float):
    class DummyAgent:
        fs_violation_ignore_patterns: tuple = ()

        def get_read_only_args(self):
            return []

        async def run(self, **kwargs):
            await asyncio.sleep(delay)
            return AgentOutput(
                parsed=ParsedCLIResponse(content="cli says hi", metadata={}),
                sanitized_command=["gemini"],
                returncode=0,
                stdout="{}",
                stderr="",
                duration_seconds=delay,
                parser_name="gemini_json",
                output_file_content=None,
            )

    return lambda c: DummyAgent()


def _payload(result) -> dict:
    return json.loads(json.loads(result[0].text)["content"])


_BASE_ARGS = {"prompt": "hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []}


@pytest.mark.asyncio
async def test_fast_run_completes_within_grace_window(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(0.01))

    result = await ClinkStartTool().execute(dict(_BASE_ARGS))
    payload = _payload(result)

    assert payload["status"] == "completed"
    assert payload["result"]["content"] == "cli says hi"
    assert "job_id" in payload


@pytest.mark.asyncio
async def test_slow_run_returns_job_and_poll_delivers_result(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(0.3))
    monkeypatch.setattr(clink_jobs, "GRACE_WINDOW_SECONDS", 0.05)

    start_payload = _payload(await ClinkStartTool().execute(dict(_BASE_ARGS)))
    assert start_payload["status"] == "running"
    assert "clink_poll" in start_payload["instructions"]
    job_id = start_payload["job_id"]

    poll_payload = _payload(await ClinkPollTool().execute({"job_id": job_id, "wait_ms": 5000}))
    assert poll_payload["status"] == "completed"
    assert poll_payload["result"]["content"] == "cli says hi"


@pytest.mark.asyncio
async def test_validation_error_surfaces_at_start(monkeypatch):
    start_payload = _payload(await ClinkStartTool().execute({**_BASE_ARGS, "cli_name": "gemini", "model": "-bad"}))
    assert start_payload["status"] == "failed"
    assert "-bad" in json.dumps(start_payload["error"])


@pytest.mark.asyncio
async def test_poll_unknown_job_is_clear_error():
    with pytest.raises(ToolExecutionError) as exc_info:
        await ClinkPollTool().execute({"job_id": "deadbeef"})
    payload = json.loads(exc_info.value.payload)
    assert "Unknown clink job" in payload["content"]
    assert "restarted" in payload["content"]


@pytest.mark.asyncio
async def test_cancel_running_job_reports_cancelled(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(5.0))
    monkeypatch.setattr(clink_jobs, "GRACE_WINDOW_SECONDS", 0.05)

    start_payload = _payload(await ClinkStartTool().execute(dict(_BASE_ARGS)))
    assert start_payload["status"] == "running"

    cancel_payload = _payload(await ClinkCancelTool().execute({"job_id": start_payload["job_id"]}))
    assert cancel_payload["status"] == "cancelled"

    poll_payload = _payload(await ClinkPollTool().execute({"job_id": start_payload["job_id"], "wait_ms": 0}))
    assert poll_payload["status"] == "cancelled"


@pytest.mark.asyncio
async def test_concurrency_cap_rejects_excess_jobs(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(5.0))
    monkeypatch.setattr(clink_jobs, "GRACE_WINDOW_SECONDS", 0.02)
    monkeypatch.setattr(clink_jobs, "MAX_CONCURRENT_JOBS", 2)

    p1 = _payload(await ClinkStartTool().execute(dict(_BASE_ARGS)))
    p2 = _payload(await ClinkStartTool().execute(dict(_BASE_ARGS)))
    assert p1["status"] == p2["status"] == "running"

    with pytest.raises(ToolExecutionError) as exc_info:
        await ClinkStartTool().execute(dict(_BASE_ARGS))
    assert "Too many concurrent clink jobs" in json.loads(exc_info.value.payload)["content"]

    for payload in (p1, p2):
        await ClinkCancelTool().execute({"job_id": payload["job_id"]})


@pytest.mark.asyncio
async def test_idempotency_key_reuses_running_job(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(5.0))
    monkeypatch.setattr(clink_jobs, "GRACE_WINDOW_SECONDS", 0.02)

    args = {**_BASE_ARGS, "idempotency_key": "review-42"}
    p1 = _payload(await ClinkStartTool().execute(dict(args)))
    p2 = _payload(await ClinkStartTool().execute(dict(args)))
    assert p1["job_id"] == p2["job_id"]

    await ClinkCancelTool().execute({"job_id": p1["job_id"]})


@pytest.mark.asyncio
async def test_expired_terminal_jobs_are_garbage_collected(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(0.01))
    monkeypatch.setattr(clink_jobs, "RESULT_TTL_SECONDS", 0.0)

    start_payload = _payload(await ClinkStartTool().execute(dict(_BASE_ARGS)))
    assert start_payload["status"] == "completed"

    with pytest.raises(ToolExecutionError):
        await ClinkPollTool().execute({"job_id": start_payload["job_id"]})


@pytest.mark.asyncio
async def test_background_job_suppresses_progress_heartbeat(monkeypatch):
    """A job runs after its start request completed; it must send no progress."""
    calls: list = []

    class _Session:
        async def report_progress(self, *args, **kwargs):
            calls.append(args)

    from types import SimpleNamespace

    monkeypatch.setattr("tools.clink.get_current_request_context", lambda: SimpleNamespace(session=_Session()))
    monkeypatch.setattr("tools.clink.CLinkTool._PROGRESS_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(0.3))
    monkeypatch.setattr(clink_jobs, "GRACE_WINDOW_SECONDS", 0.05)

    start_payload = _payload(await ClinkStartTool().execute(dict(_BASE_ARGS)))
    job_id = start_payload["job_id"]
    poll_payload = _payload(await ClinkPollTool().execute({"job_id": job_id, "wait_ms": 5000}))
    assert poll_payload["status"] == "completed"
    assert calls == []


def test_start_schema_extends_clink_schema():
    schema = ClinkStartTool().get_input_schema()
    assert "prompt" in schema["properties"]
    assert "working_dir" in schema["properties"]
    assert "idempotency_key" in schema["properties"]


def test_registry_exposes_job_tools():
    from tools.registry import TOOL_DEFINITIONS

    for name in ("clink_start", "clink_poll", "clink_cancel"):
        assert name in TOOL_DEFINITIONS
