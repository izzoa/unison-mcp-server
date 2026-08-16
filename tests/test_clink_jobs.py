"""Tests for clink's integrated job control (bounded waits, job_id, cancel).

Job control is folded into the clink tool itself: every call waits a bounded
time; long runs return status='running' with a job_id, and calling clink
again with only that job_id continues the wait. No single MCP request ever
outlives an impatient host's tool deadline.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import tools.clink_jobs as clink_jobs
from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from tools.clink import CLinkTool
from tools.clink_jobs import get_clink_job_manager, reset_clink_job_manager_for_tests
from tools.shared.exceptions import ToolExecutionError


@pytest.fixture(autouse=True)
def _fresh_manager():
    reset_clink_job_manager_for_tests()
    yield
    reset_clink_job_manager_for_tests()


@pytest.fixture(autouse=True)
def _short_wait(monkeypatch):
    """Keep foreground wait budgets tiny so tests never sleep for real."""
    monkeypatch.setattr(CLinkTool, "_DEFAULT_JOB_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(CLinkTool, "_MAX_JOB_WAIT_SECONDS", 5.0)


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


def _outer(result) -> dict:
    return json.loads(result[0].text)


def _job_payload(result) -> dict:
    return json.loads(_outer(result)["content"])


_BASE_ARGS = {"prompt": "hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []}


@pytest.mark.asyncio
async def test_fast_run_is_transparent(monkeypatch):
    """Runs finishing inside the wait budget look exactly like blocking calls."""
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(0.01))

    outer = _outer(await CLinkTool().execute(dict(_BASE_ARGS)))
    assert outer["status"] in ("success", "continuation_available")
    assert outer["content"] == "cli says hi"


@pytest.mark.asyncio
async def test_slow_run_returns_job_then_job_id_call_delivers_result(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(0.3))

    running = _job_payload(await CLinkTool().execute(dict(_BASE_ARGS)))
    assert running["status"] == "running"
    assert "job_id" in running
    assert "job_id" in running["instructions"]

    final = _outer(await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 5}))
    assert final["status"] in ("success", "continuation_available")
    assert final["content"] == "cli says hi"


@pytest.mark.asyncio
async def test_failed_run_raises_original_error_on_followup(monkeypatch):
    running = _job_payload(await CLinkTool().execute({**_BASE_ARGS, "model": "-bad", "wait_seconds": 0}))
    # Validation failures are fast; either the first call already surfaces
    # the failure or the follow-up does. Force the follow-up path.
    if running.get("status") == "running":
        with pytest.raises(ToolExecutionError) as exc_info:
            await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 5})
        assert "-bad" in json.loads(exc_info.value.payload)["content"]


@pytest.mark.asyncio
async def test_validation_error_within_budget_raises_immediately(monkeypatch):
    with pytest.raises(ToolExecutionError) as exc_info:
        await CLinkTool().execute({**_BASE_ARGS, "model": "-bad", "wait_seconds": 5})
    assert "-bad" in json.loads(exc_info.value.payload)["content"]


@pytest.mark.asyncio
async def test_missing_prompt_without_job_id_is_error():
    with pytest.raises(ToolExecutionError) as exc_info:
        await CLinkTool().execute({"cli_name": "gemini"})
    assert "prompt is required" in json.loads(exc_info.value.payload)["content"]


@pytest.mark.asyncio
async def test_unknown_job_id_is_clear_error():
    with pytest.raises(ToolExecutionError) as exc_info:
        await CLinkTool().execute({"job_id": "deadbeef"})
    content = json.loads(exc_info.value.payload)["content"]
    assert "Unknown clink job" in content
    assert "restarted" in content


@pytest.mark.asyncio
async def test_cancel_via_job_id(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(5.0))

    running = _job_payload(await CLinkTool().execute(dict(_BASE_ARGS)))
    assert running["status"] == "running"

    cancelled = _job_payload(await CLinkTool().execute({"job_id": running["job_id"], "cancel": True}))
    assert cancelled["status"] == "cancelled"

    followup = _job_payload(await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 0}))
    assert followup["status"] == "cancelled"


@pytest.mark.asyncio
async def test_blocking_mode_bypasses_job_manager(monkeypatch):
    monkeypatch.setenv("CLINK_EXECUTION_MODE", "blocking")
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(0.1))

    outer = _outer(await CLinkTool().execute(dict(_BASE_ARGS)))
    assert outer["content"] == "cli says hi"
    assert get_clink_job_manager().running_count() == 0


@pytest.mark.asyncio
async def test_concurrency_cap_rejects_excess_jobs(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(5.0))
    monkeypatch.setattr(clink_jobs, "MAX_CONCURRENT_JOBS", 2)

    p1 = _job_payload(await CLinkTool().execute(dict(_BASE_ARGS)))
    p2 = _job_payload(await CLinkTool().execute(dict(_BASE_ARGS)))
    assert p1["status"] == p2["status"] == "running"

    with pytest.raises(ToolExecutionError) as exc_info:
        await CLinkTool().execute(dict(_BASE_ARGS))
    assert "Too many concurrent clink jobs" in json.loads(exc_info.value.payload)["content"]

    for payload in (p1, p2):
        await CLinkTool().execute({"job_id": payload["job_id"], "cancel": True})


@pytest.mark.asyncio
async def test_idempotency_key_reuses_running_job(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(5.0))

    args = {**_BASE_ARGS, "idempotency_key": "review-42"}
    p1 = _job_payload(await CLinkTool().execute(dict(args)))
    p2 = _job_payload(await CLinkTool().execute(dict(args)))
    assert p1["job_id"] == p2["job_id"]

    await CLinkTool().execute({"job_id": p1["job_id"], "cancel": True})


@pytest.mark.asyncio
async def test_expired_terminal_jobs_are_garbage_collected(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(0.3))
    monkeypatch.setattr(clink_jobs, "RESULT_TTL_SECONDS", 0.0)

    running = _job_payload(await CLinkTool().execute(dict(_BASE_ARGS)))
    job = get_clink_job_manager().get(running["job_id"])
    assert job is not None
    await asyncio.wait_for(job.done_event.wait(), timeout=5)

    with pytest.raises(ToolExecutionError):
        await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 0})


@pytest.mark.asyncio
async def test_foreground_wait_emits_heartbeats(monkeypatch):
    """The waiting call (whose progress token is live) emits the heartbeats."""
    from types import SimpleNamespace

    calls: list = []

    class _Session:
        async def report_progress(self, *args, **kwargs):
            calls.append(args)

    monkeypatch.setattr("tools.clink.get_current_request_context", lambda: SimpleNamespace(session=_Session()))
    monkeypatch.setattr(CLinkTool, "_PROGRESS_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(CLinkTool, "_DEFAULT_JOB_WAIT_SECONDS", 0.1)
    monkeypatch.setattr("tools.clink.create_agent", _slow_dummy_agent_factory(0.06))

    outer = _outer(await CLinkTool().execute(dict(_BASE_ARGS)))
    assert outer["content"] == "cli says hi"
    assert len(calls) >= 1
    assert "gemini" in (calls[0][2] or "")


def test_schema_has_job_control_and_no_required_prompt():
    schema = CLinkTool().get_input_schema()
    for prop in ("prompt", "job_id", "cancel", "wait_seconds", "idempotency_key", "working_dir"):
        assert prop in schema["properties"]
    assert "prompt" not in schema.get("required", [])


def test_job_tools_removed_from_registry():
    from tools.registry import TOOL_DEFINITIONS

    for name in ("clink_start", "clink_poll", "clink_cancel"):
        assert name not in TOOL_DEFINITIONS
    assert "clink" in TOOL_DEFINITIONS
