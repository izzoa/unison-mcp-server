"""Tool-layer tests for clink's per-call working_dir parameter.

The spawned CLI's working directory decides which files it can see at all —
some CLIs (Copilot) root their file tools at their cwd and refuse absolute
paths outside it. The per-call override lets the calling agent point the CLI
at its own project or worktree root instead of inheriting the MCP server's
process cwd.
"""

from __future__ import annotations

import json
import os

import pytest

from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from tools.clink import CLinkTool
from tools.shared.exceptions import ToolExecutionError


def _dummy_agent(captured: dict):
    class DummyAgent:
        async def run(self, **kwargs):
            captured.update(kwargs)
            return AgentOutput(
                parsed=ParsedCLIResponse(content="ok", metadata={}),
                sanitized_command=["gemini"],
                returncode=0,
                stdout="{}",
                stderr="",
                duration_seconds=0.1,
                parser_name="gemini_json",
                output_file_content=None,
            )

    return DummyAgent()


def test_schema_exposes_working_dir():
    tool = CLinkTool()
    schema = tool.get_input_schema()
    assert "working_dir" in schema["properties"]
    assert schema["properties"]["working_dir"]["type"] == "string"
    assert "working_dir" not in schema["required"]


@pytest.mark.asyncio
async def test_relative_working_dir_rejected():
    tool = CLinkTool()
    with pytest.raises(ToolExecutionError) as exc_info:
        await tool.execute({"prompt": "hi", "cli_name": "gemini", "working_dir": "relative/path"})
    payload = json.loads(exc_info.value.payload)
    assert payload["status"] == "error"
    assert "absolute" in payload["content"]


@pytest.mark.asyncio
async def test_nonexistent_working_dir_rejected(tmp_path):
    tool = CLinkTool()
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ToolExecutionError) as exc_info:
        await tool.execute({"prompt": "hi", "cli_name": "gemini", "working_dir": str(missing)})
    payload = json.loads(exc_info.value.payload)
    assert payload["status"] == "error"
    assert "does not exist or is not a directory" in payload["content"]


@pytest.mark.asyncio
async def test_working_dir_passed_to_agent_and_reported(monkeypatch, tmp_path):
    captured: dict = {}
    monkeypatch.setattr("tools.clink.create_agent", lambda c: _dummy_agent(captured))

    tool = CLinkTool()
    result = await tool.execute(
        {
            "prompt": "hi",
            "cli_name": "gemini",
            "working_dir": str(tmp_path),
            "absolute_file_paths": [],
            "images": [],
        }
    )
    payload = json.loads(result[0].text)
    assert captured["working_dir"] == tmp_path
    assert payload["metadata"]["working_dir"] == str(tmp_path)


@pytest.mark.asyncio
async def test_omitted_working_dir_defaults_to_server_cwd(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("tools.clink.create_agent", lambda c: _dummy_agent(captured))

    tool = CLinkTool()
    result = await tool.execute({"prompt": "hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)
    assert captured["working_dir"] is None
    assert payload["metadata"]["working_dir"] == os.getcwd()
