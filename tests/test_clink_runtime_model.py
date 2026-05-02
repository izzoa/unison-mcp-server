"""Tests for the runtime model selection contract on clink agents."""

from __future__ import annotations

import json

import pytest

from clink import get_registry
from clink.agents.base import BaseCLIAgent
from clink.agents.claude import ClaudeAgent
from clink.agents.codex import CodexAgent
from clink.agents.gemini import GeminiAgent
from clink.agents.opencode import OpencodeAgent

# ---------------------------------------------------------------------------
# 8.3a — render_model_args + model_flag_aliases per agent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_cls,expected_flag,expected_aliases",
    [
        (GeminiAgent, "--model", ("--model",)),
        (CodexAgent, "-m", ("-m",)),
        (ClaudeAgent, "--model", ("--model",)),
        (OpencodeAgent, "-m", ("-m",)),
    ],
)
def test_each_agent_renders_its_model_flag(agent_cls, expected_flag, expected_aliases):
    registry = get_registry()
    cli_name = {
        GeminiAgent: "gemini",
        CodexAgent: "codex",
        ClaudeAgent: "claude",
        OpencodeAgent: "opencode",
    }[agent_cls]
    agent = agent_cls(registry.get_client(cli_name))
    assert agent.render_model_args("X/y") == [expected_flag, "X/y"]
    assert agent.model_flag_aliases == expected_aliases


def test_base_agent_render_model_args_is_noop():
    # Build a throwaway resolved client just to instantiate the base class.
    client = get_registry().get_client("gemini")  # any will do
    base_agent = BaseCLIAgent(client)
    assert base_agent.render_model_args("anything") == []
    assert BaseCLIAgent.model_flag_aliases == ()


# ---------------------------------------------------------------------------
# 8.3b — _strip_model_flags edge cases
# ---------------------------------------------------------------------------


def test_strip_removes_long_form_pair_anywhere_in_command():
    agent = ClaudeAgent(get_registry().get_client("claude"))
    cmd = ["claude", "--print", "--model", "sonnet", "--permission-mode", "plan"]
    assert agent._strip_model_flags(cmd) == [
        "claude",
        "--print",
        "--permission-mode",
        "plan",
    ]


def test_strip_removes_short_form_pair():
    agent = CodexAgent(get_registry().get_client("codex"))
    cmd = ["codex", "exec", "-m", "o3", "--json"]
    assert agent._strip_model_flags(cmd) == ["codex", "exec", "--json"]


def test_strip_removes_multiple_occurrences():
    agent = ClaudeAgent(get_registry().get_client("claude"))
    cmd = ["claude", "--model", "first", "--print", "--model", "second"]
    assert agent._strip_model_flags(cmd) == ["claude", "--print"]


def test_strip_drops_bare_alias_at_end_and_warns(caplog):
    agent = OpencodeAgent(get_registry().get_client("opencode"))
    cmd = ["opencode", "run", "-m"]
    with caplog.at_level("WARNING"):
        result = agent._strip_model_flags(cmd)
    assert result == ["opencode", "run"]
    assert any("model flag alias" in rec.message for rec in caplog.records)


def test_strip_leaves_unrelated_args_unchanged():
    agent = GeminiAgent(get_registry().get_client("gemini"))
    cmd = ["gemini", "-o", "json", "--yolo"]
    assert agent._strip_model_flags(cmd) == cmd


def test_base_agent_with_no_aliases_passes_through():
    client = get_registry().get_client("gemini")
    base = BaseCLIAgent(client)
    cmd = ["gemini", "--model", "X"]
    # Base has no aliases — nothing should be stripped
    assert base._strip_model_flags(cmd) == cmd


# ---------------------------------------------------------------------------
# 8.3c — _build_command with and without model
# ---------------------------------------------------------------------------


def test_build_command_strips_pre_existing_model_then_appends_runtime():
    client = get_registry().get_client("claude")  # has --model sonnet baked in
    agent = ClaudeAgent(client)
    role = client.get_role("default")
    cmd = agent._build_command(role=role, system_prompt="hi", model="opus")
    # original 'sonnet' must be gone
    assert "sonnet" not in cmd
    # exactly one --model flag with the runtime value
    assert cmd.count("--model") == 1
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "opus"


def test_build_command_with_no_model_is_byte_for_byte_unchanged():
    client = get_registry().get_client("claude")
    agent = ClaudeAgent(client)
    role = client.get_role("default")
    cmd_baseline = agent._build_command(role=role, system_prompt="hi")
    cmd_none = agent._build_command(role=role, system_prompt="hi", model=None)
    cmd_empty = agent._build_command(role=role, system_prompt="hi", model="")
    assert cmd_baseline == cmd_none == cmd_empty


# ---------------------------------------------------------------------------
# 8.3d — agent statelessness
# ---------------------------------------------------------------------------


def test_agent_does_not_carry_model_across_calls():
    client = get_registry().get_client("opencode")
    agent = OpencodeAgent(client)
    role = client.get_role("default")
    with_model = agent._build_command(role=role, system_prompt=None, model="X/y")
    fresh_agent = OpencodeAgent(client)
    fresh_no_model = fresh_agent._build_command(role=role, system_prompt=None)
    # After the with-model call, the next call without a model on the SAME instance
    # must match a freshly-instantiated agent.
    after_no_model = agent._build_command(role=role, system_prompt=None)
    assert with_model[-2:] == ["-m", "X/y"]
    assert after_no_model == fresh_no_model
    assert "-m" not in after_no_model
    assert "X/y" not in after_no_model


# ---------------------------------------------------------------------------
# 8.3e — supported_models allowlist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowlist_rejects_disallowed_model(monkeypatch, tmp_path):
    """Manifest with non-empty supported_models rejects out-of-list values."""
    manifest = {
        "name": "opencode",
        "command": "opencode run",
        "additional_args": [],
        "env": {},
        "roles": {"default": {"prompt_path": "systemprompts/clink/default.txt"}},
        "supported_models": ["openai/gpt-5"],
    }
    cfg_path = tmp_path / "opencode.json"
    cfg_path.write_text(json.dumps(manifest))
    monkeypatch.setenv("CLI_CLIENTS_CONFIG_PATH", str(tmp_path))

    import clink.registry as r

    monkeypatch.setattr(r, "_REGISTRY", None)

    from tools.clink import CLinkTool
    from tools.shared.exceptions import ToolExecutionError

    tool = CLinkTool()
    arguments = {
        "prompt": "hi",
        "cli_name": "opencode",
        "model": "anthropic/claude-sonnet-4-5",
        "absolute_file_paths": [],
        "images": [],
    }
    with pytest.raises(ToolExecutionError) as exc_info:
        await tool.execute(arguments)
    payload = json.loads(exc_info.value.payload)
    assert payload["status"] == "error"
    assert "anthropic/claude-sonnet-4-5" in payload["content"]
    assert "openai/gpt-5" in payload["content"]


def test_empty_allowlist_skips_validation():
    """A client with no supported_models forwards any model verbatim."""
    client = get_registry().get_client("opencode")
    assert client.supported_models == []  # opencode.json has no allowlist


# ---------------------------------------------------------------------------
# 8.3f — omitted / empty-string model behavior in CLinkTool.execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_omitted_model_produces_no_model_requested_metadata(monkeypatch):
    from clink.agents import AgentOutput
    from clink.parsers.base import ParsedCLIResponse
    from tools.clink import CLinkTool

    captured: dict = {}

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

    monkeypatch.setattr("tools.clink.create_agent", lambda c: DummyAgent())

    tool = CLinkTool()
    arguments = {"prompt": "hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []}
    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    assert "model_requested" not in payload.get("metadata", {})
    assert captured["model"] is None


@pytest.mark.asyncio
async def test_empty_string_model_treated_as_omitted(monkeypatch):
    from clink.agents import AgentOutput
    from clink.parsers.base import ParsedCLIResponse
    from tools.clink import CLinkTool

    captured: dict = {}

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

    monkeypatch.setattr("tools.clink.create_agent", lambda c: DummyAgent())

    tool = CLinkTool()
    arguments = {
        "prompt": "hi",
        "cli_name": "gemini",
        "model": "",
        "absolute_file_paths": [],
        "images": [],
    }
    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    assert "model_requested" not in payload.get("metadata", {})
    assert captured["model"] is None


@pytest.mark.asyncio
async def test_provided_model_records_model_requested_and_passes_to_agent(monkeypatch):
    from clink.agents import AgentOutput
    from clink.parsers.base import ParsedCLIResponse
    from tools.clink import CLinkTool

    captured: dict = {}

    class DummyAgent:
        async def run(self, **kwargs):
            captured.update(kwargs)
            return AgentOutput(
                parsed=ParsedCLIResponse(
                    content="ok",
                    metadata={"model_used": "anthropic/claude-sonnet-4-5"},
                ),
                sanitized_command=["opencode"],
                returncode=0,
                stdout="{}",
                stderr="",
                duration_seconds=0.1,
                parser_name="opencode_jsonl",
                output_file_content=None,
            )

    monkeypatch.setattr("tools.clink.create_agent", lambda c: DummyAgent())

    tool = CLinkTool()
    arguments = {
        "prompt": "hi",
        "cli_name": "opencode",
        "model": "anthropic/claude-sonnet-4-5",
        "absolute_file_paths": [],
        "images": [],
    }
    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    metadata = payload.get("metadata", {})
    assert metadata.get("model_requested") == "anthropic/claude-sonnet-4-5"
    assert metadata.get("model_used") == "anthropic/claude-sonnet-4-5"
    assert captured["model"] == "anthropic/claude-sonnet-4-5"
