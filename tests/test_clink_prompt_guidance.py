"""Tests for CLI-agnostic prompt guidance in the clink tool.

These assert on the prompt clink actually hands to the spawned agent, captured
via a dummy agent at the ``execute()`` boundary. Driving them end-to-end rather
than calling the prompt assembler directly keeps them stable across the internal
signature changes this change makes.
"""

from __future__ import annotations

import pytest

from clink import get_registry
from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from tools.clink import CLinkTool


def _configured_clis() -> list[str]:
    return get_registry().list_clients()


class _PromptCapture:
    """Dummy agent that records the prompt it was invoked with."""

    def __init__(self) -> None:
        self.prompt: str | None = None

    def install(self, monkeypatch) -> _PromptCapture:
        capture = self

        class DummyAgent:
            #: Mirrors the BaseCLIAgent surface tools.clink relies on in read-only mode.
            fs_violation_ignore_patterns: tuple[str, ...] = ()

            def get_read_only_args(self) -> list[str]:
                return []

            async def run(self, **kwargs):
                capture.prompt = kwargs.get("prompt")
                return AgentOutput(
                    parsed=ParsedCLIResponse(content="ok", metadata={}),
                    sanitized_command=["dummy"],
                    returncode=0,
                    stdout="ok",
                    stderr="",
                    duration_seconds=0.01,
                    parser_name="dummy",
                    output_file_content=None,
                )

        monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())
        return self


async def _capture_prompt(monkeypatch, **overrides) -> str:
    """Run clink with a dummy agent and return the assembled prompt."""
    tool = CLinkTool()
    capture = _PromptCapture().install(monkeypatch)

    if overrides.get("read_only"):
        # The real snapshot walks the whole working tree; irrelevant here and slow.
        monkeypatch.setattr("tools.clink.capture_snapshot", lambda *a, **k: {})

    arguments = {
        "prompt": "Say hello",
        "role": "default",
        "absolute_file_paths": [],
        "images": [],
    }
    arguments.update(overrides)

    await tool.execute(arguments)
    assert capture.prompt is not None, "agent was never invoked"
    return capture.prompt


# ---------------------------------------------------------------------------
# 1.1 Guidance section is present for every configured CLI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cli_name", _configured_clis())
async def test_guidance_section_present_for_every_cli(monkeypatch, cli_name):
    prompt = await _capture_prompt(monkeypatch, cli_name=cli_name)
    assert "deliver the final answer" in prompt.lower() or "final answer" in prompt.lower()


# ---------------------------------------------------------------------------
# 1.2 Guidance must not name a CLI other than the one being invoked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cli_name", [name for name in _configured_clis() if name != "gemini"])
async def test_guidance_does_not_name_gemini_for_other_clis(monkeypatch, cli_name):
    prompt = await _capture_prompt(monkeypatch, cli_name=cli_name)
    assert "Gemini CLI agent" not in prompt, f"{cli_name} was told it is the Gemini CLI agent"


@pytest.mark.asyncio
@pytest.mark.parametrize("cli_name", _configured_clis())
async def test_guidance_names_the_invoked_cli(monkeypatch, cli_name):
    prompt = await _capture_prompt(monkeypatch, cli_name=cli_name)
    assert cli_name in prompt, f"guidance never names the invoked CLI {cli_name!r}"


# ---------------------------------------------------------------------------
# 1.3 Guidance must not assert capabilities the registry cannot verify
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cli_name", _configured_clis())
async def test_guidance_makes_no_unverifiable_capability_claim(monkeypatch, cli_name):
    prompt = await _capture_prompt(monkeypatch, cli_name=cli_name)
    lowered = prompt.lower()
    assert "web search" not in lowered, f"{cli_name} was promised web search"
    assert "full suite" not in lowered, f"{cli_name} was promised a full suite of capabilities"


# ---------------------------------------------------------------------------
# 1.4 Read-only instruction must not enumerate CLI-specific tool names
# ---------------------------------------------------------------------------


#: Tool names previously hardcoded into the read-only block. None of these match
#: any current target — Gemini CLI 0.46.0 exposes ``write_file`` and ``replace``.
_STALE_TOOL_NAMES = ("EditFile", "WriteFile", "CreateFile", "DeleteFile", "ReplaceInFile")


@pytest.mark.asyncio
@pytest.mark.parametrize("cli_name", _configured_clis())
async def test_read_only_instruction_names_no_cli_specific_tools(monkeypatch, cli_name):
    prompt = await _capture_prompt(monkeypatch, cli_name=cli_name, read_only=True)
    assert "READ-ONLY MODE" in prompt
    for name in _STALE_TOOL_NAMES:
        assert name not in prompt, f"read-only instruction still names {name!r}"


@pytest.mark.asyncio
async def test_read_only_instruction_identical_across_clis(monkeypatch):
    clis = _configured_clis()
    assert len(clis) >= 2, "need at least two configured CLIs"

    def _extract(prompt: str) -> str:
        marker = "=== READ-ONLY MODE ==="
        start = prompt.index(marker)
        return prompt[start:].split("\n\n")[0]

    first = _extract(await _capture_prompt(monkeypatch, cli_name=clis[0], read_only=True))
    second = _extract(await _capture_prompt(monkeypatch, cli_name=clis[1], read_only=True))
    assert first == second


@pytest.mark.asyncio
async def test_read_only_instruction_absent_when_false(monkeypatch):
    prompt = await _capture_prompt(monkeypatch, cli_name=_configured_clis()[0], read_only=False)
    assert "READ-ONLY MODE" not in prompt


# ---------------------------------------------------------------------------
# Omitted cli_name in a multi-client registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_omitted_cli_name_rejected_in_multi_client_registry(monkeypatch):
    """With several CLIs configured, an omitted cli_name must be an explicit error.

    The generated schema has always marked ``cli_name`` required once more than
    one client is configured; before this change nothing enforced it and the
    request silently dispatched to a vendor-preferred default.
    """
    tool = CLinkTool()
    assert len(tool._cli_names) > 1, "this test assumes a multi-client registry"
    _PromptCapture().install(monkeypatch)

    with pytest.raises(Exception) as excinfo:
        await tool.execute({"prompt": "Hello", "absolute_file_paths": [], "images": []})

    message = str(excinfo.value)
    assert "cli_name" in message
    for name in tool._cli_names:
        assert name in message, f"error should enumerate configured client {name!r}"


@pytest.mark.asyncio
async def test_omitted_cli_name_never_reaches_registry_unresolved():
    """The prompt-preparation hook must not pass an unresolved name to the registry.

    ``get_client()`` annotates its parameter ``str`` and calls ``.lower()``, so a
    ``None`` arriving there raises AttributeError instead of an actionable error.
    """
    from tools.clink import CLinkRequest
    from tools.shared.exceptions import ToolExecutionError

    tool = CLinkTool()
    request = CLinkRequest(prompt="Hello")
    assert request.cli_name is None

    with pytest.raises(ToolExecutionError):
        await tool.prepare_prompt(request)


def test_resolve_client_rejects_unknown_name():
    from tools.shared.exceptions import ToolExecutionError

    tool = CLinkTool()
    with pytest.raises(ToolExecutionError) as excinfo:
        tool._resolve_client("not-a-real-cli")
    assert "not-a-real-cli" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Tool description reflects configured targets
# ---------------------------------------------------------------------------


def test_description_names_no_unconfigured_cli():
    tool = CLinkTool()
    description = tool.get_description()
    enum_values = {name.lower() for name in tool.get_input_schema()["properties"]["cli_name"]["enum"]}

    # Any CLI-ish name the description mentions must be a configured target.
    candidates = ("qwen", "gemini", "claude", "codex", "aider", "crush", "amp", "opencode", "copilot")
    for candidate in candidates:
        if candidate not in enum_values:
            assert candidate not in description.lower(), f"description names unconfigured CLI {candidate!r}"


def test_description_does_not_name_qwen():
    """Regression: the description advertised Qwen, which was never a target."""
    assert "qwen" not in CLinkTool().get_description().lower()
