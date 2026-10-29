"""OpencodeAgent recovery from non-zero exits.

opencode v2 reports failures only as a JSON ``error`` event on stdout (exit 1,
empty stderr), so without reading it callers saw a bare "exited with status 1".
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from clink import get_registry
from clink.agents.base import CLIAgentError
from clink.agents.opencode import OpencodeAgent
from tests.test_clink_opencode_parser import V1_ERROR, V2_ERROR_NO_ROUTE


def _agent() -> OpencodeAgent:
    return OpencodeAgent(get_registry().get_client("opencode"))


def _recover(agent: OpencodeAgent, *, returncode: int, stdout: str, stderr: str = "") -> Any:
    return agent._recover_from_error(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        sanitized_command=["opencode", "run"],
        duration_seconds=0.1,
        output_file_content=None,
    )


def _jsonl(*events: dict[str, Any]) -> str:
    return "\n".join(json.dumps(e) for e in events)


def test_error_only_output_raises_with_opencodes_own_reason_first() -> None:
    with pytest.raises(CLIAgentError) as exc_info:
        _recover(_agent(), returncode=1, stdout=V2_ERROR_NO_ROUTE)

    error = exc_info.value
    assert str(error) == "CLI 'opencode' error: provider.no-route: Model unavailable: nosuch/model"
    assert error.returncode == 1
    assert error.stdout == V2_ERROR_NO_ROUTE  # the original output stays attached
    assert error.stderr == ""


def test_v1_error_only_output_leads_with_the_error_name() -> None:
    with pytest.raises(CLIAgentError, match=r"^CLI 'opencode' error: UnknownError: Unexpected server error"):
        _recover(_agent(), returncode=1, stdout=V1_ERROR)


def test_text_plus_errors_returns_partial_output_carrying_the_errors() -> None:
    stdout = _jsonl(
        {"type": "text", "part": {"text": "half an answer"}},
        {"type": "error", "error": {"type": "provider.timeout", "message": "stream stalled"}},
    )

    output = _recover(_agent(), returncode=1, stdout=stdout)

    assert output.parsed.content == "half an answer"
    assert output.parsed.metadata["partial"] is True
    assert output.parsed.metadata["errors"] == ["provider.timeout: stream stalled"]
    assert output.returncode == 1


def test_text_only_exit_130_is_partial_without_errors() -> None:
    stdout = _jsonl({"type": "text", "part": {"text": "answer cut short by an interrupt"}})

    output = _recover(_agent(), returncode=130, stdout=stdout)

    assert output.parsed.content == "answer cut short by an interrupt"
    assert output.parsed.metadata["partial"] is True
    assert "errors" not in output.parsed.metadata
    assert output.returncode == 130


def test_unparseable_output_falls_through_to_the_generic_error() -> None:
    assert _recover(_agent(), returncode=1, stdout="not json at all") is None
    assert _recover(_agent(), returncode=1, stdout="") is None


# A fake CLI that replays a captured stdout and exits non-zero, so the whole
# path (spawn, live capture, recovery, error message) runs for real.
_REPLAY_CLI = textwrap.dedent("""
    import sys
    sys.stdin.read()
    sys.stdout.write(open(sys.argv[1]).read())
    sys.exit(int(sys.argv[2]))
    """)


def _replay_agent(tmp_path: Path, stdout: str, exit_code: int) -> OpencodeAgent:
    script = tmp_path / "replay_cli.py"
    script.write_text(_REPLAY_CLI)
    captured = tmp_path / "stdout.jsonl"
    captured.write_text(stdout)
    client = (
        get_registry()
        .get_client("opencode")
        .model_copy(
            update={"executable": [sys.executable, str(script), str(captured), str(exit_code)], "internal_args": []}
        )
    )
    return OpencodeAgent(client)


@pytest.mark.asyncio
async def test_run_surfaces_opencodes_reason_instead_of_a_bare_exit_status(tmp_path: Path) -> None:
    agent = _replay_agent(tmp_path, V2_ERROR_NO_ROUTE, exit_code=1)

    with pytest.raises(CLIAgentError) as exc_info:
        await agent.run(role=agent.client.roles["default"], prompt="hello", files=[], images=[])

    assert str(exc_info.value).startswith("CLI 'opencode' error: provider.no-route")
    assert "exited with status" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_returns_partial_output_for_a_non_zero_exit_with_text(tmp_path: Path) -> None:
    stdout = _jsonl({"type": "text", "part": {"text": "usable answer"}})
    agent = _replay_agent(tmp_path, stdout, exit_code=130)

    result = await agent.run(role=agent.client.roles["default"], prompt="hello", files=[], images=[])

    assert result.returncode == 130
    assert result.parsed.content == "usable answer"
    assert result.parsed.metadata["partial"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("captured", [V2_ERROR_NO_ROUTE, V1_ERROR])
async def test_error_only_output_is_a_failure_even_when_the_exit_status_is_zero(tmp_path: Path, captured: str) -> None:
    # opencode v1 exits 0 on session errors (upstream anomalyco/opencode#14551).
    agent = _replay_agent(tmp_path, captured, exit_code=0)

    with pytest.raises(CLIAgentError, match=r"^CLI 'opencode' error: ") as exc_info:
        await agent.run(role=agent.client.roles["default"], prompt="hello", files=[], images=[])

    assert exc_info.value.returncode == 0
    assert exc_info.value.stdout == captured


@pytest.mark.asyncio
async def test_a_successful_zero_exit_run_with_text_is_unchanged(tmp_path: Path) -> None:
    stdout = _jsonl({"type": "text", "part": {"text": "fine"}}, {"type": "error", "message": "late warning"})
    agent = _replay_agent(tmp_path, stdout, exit_code=0)

    result = await agent.run(role=agent.client.roles["default"], prompt="hello", files=[], images=[])

    assert result.parsed.content == "fine"
    assert "partial" not in result.parsed.metadata
