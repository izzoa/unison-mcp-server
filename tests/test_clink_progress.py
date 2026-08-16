"""Tests for clink's MCP progress heartbeat during CLI execution.

Hosts that reset their tool timeout on progress notifications keep long CLI
runs alive; hosts that sent no progressToken get a guaranteed no-op. Either
way the heartbeat must stop before the tool response is returned and must
never break the call.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from tools.clink import CLinkTool


def _slow_dummy_agent(delay: float):
    class DummyAgent:
        fs_violation_ignore_patterns: tuple = ()

        def get_read_only_args(self):
            return []

        async def run(self, **kwargs):
            await asyncio.sleep(delay)
            return AgentOutput(
                parsed=ParsedCLIResponse(content="ok", metadata={}),
                sanitized_command=["gemini"],
                returncode=0,
                stdout="{}",
                stderr="",
                duration_seconds=delay,
                parser_name="gemini_json",
                output_file_content=None,
            )

    return DummyAgent()


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float | None, str | None]] = []

    async def report_progress(self, progress: float, total: float | None = None, message: str | None = None) -> None:
        self.calls.append((progress, total, message))


@pytest.mark.asyncio
async def test_heartbeat_emits_progress_during_long_run(monkeypatch):
    session = _RecordingSession()
    fake_ctx = SimpleNamespace(session=session)
    monkeypatch.setattr("tools.clink.get_current_request_context", lambda: fake_ctx)
    monkeypatch.setattr(CLinkTool, "_PROGRESS_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("tools.clink.create_agent", lambda c: _slow_dummy_agent(0.08))

    tool = CLinkTool()
    result = await tool.execute({"prompt": "hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []})

    payload = json.loads(result[0].text)
    assert payload["status"] in ("success", "continuation_available")
    assert len(session.calls) >= 2
    # Monotonically increasing progress values, message names the CLI
    progresses = [call[0] for call in session.calls]
    assert progresses == sorted(progresses)
    assert "gemini" in (session.calls[0][2] or "")

    # Heartbeat must stop once the response is returned
    count_after_return = len(session.calls)
    await asyncio.sleep(0.05)
    assert len(session.calls) == count_after_return


@pytest.mark.asyncio
async def test_no_request_context_is_harmless(monkeypatch):
    monkeypatch.setattr("tools.clink.get_current_request_context", lambda: None)
    monkeypatch.setattr(CLinkTool, "_PROGRESS_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("tools.clink.create_agent", lambda c: _slow_dummy_agent(0.03))

    tool = CLinkTool()
    result = await tool.execute({"prompt": "hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)
    assert payload["status"] in ("success", "continuation_available")


@pytest.mark.asyncio
async def test_report_progress_failure_does_not_break_call(monkeypatch):
    class _BrokenSession:
        async def report_progress(self, *args, **kwargs):
            raise RuntimeError("transport gone")

    fake_ctx = SimpleNamespace(session=_BrokenSession())
    monkeypatch.setattr("tools.clink.get_current_request_context", lambda: fake_ctx)
    monkeypatch.setattr(CLinkTool, "_PROGRESS_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("tools.clink.create_agent", lambda c: _slow_dummy_agent(0.05))

    tool = CLinkTool()
    result = await tool.execute({"prompt": "hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)
    assert payload["status"] in ("success", "continuation_available")
