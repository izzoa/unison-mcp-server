"""Tests for clink/agents/opencode.py."""

from __future__ import annotations

import json

from clink import get_registry
from clink.agents.opencode import OpencodeAgent


def _client():
    return get_registry().get_client("opencode")


def test_get_read_only_args_returns_empty_list_for_opencode():
    """Opencode has no CLI flag for read-only-while-still-executing mode.

    The previously-shipped ``["--agent", "plan"]`` switched the agent persona to
    opencode's plan agent (which produces planning-language instead of executing
    the requested task), so it was not a true read-only sandbox. Layer-1
    enforcement is intentionally absent for opencode; layers 2 (prompt) and 3
    (fs snapshot diff) provide enforcement.
    """
    agent = OpencodeAgent(_client())
    assert agent.get_read_only_args() == []


def test_fs_violation_ignore_patterns_declared():
    """Opencode declares the explicit, enumerated set of bootstrap paths.

    Tight enumeration (not a directory-wide glob) so model writes to
    .opencode/skills/ or .opencode/commands/ correctly classify as by_model.
    """
    assert OpencodeAgent.fs_violation_ignore_patterns == (
        ".opencode/.gitignore",
        ".opencode/package.json",
        ".opencode/package-lock.json",
        ".opencode/node_modules/**",
        ".git/opencode",
    )


def test_render_model_args_returns_short_flag():
    agent = OpencodeAgent(_client())
    assert agent.render_model_args("anthropic/claude-sonnet-4-5") == [
        "-m",
        "anthropic/claude-sonnet-4-5",
    ]


def test_model_flag_aliases_declares_short_form():
    assert OpencodeAgent.model_flag_aliases == ("-m",)


def test_recover_from_error_returns_parsed_output_when_stdout_parses():
    agent = OpencodeAgent(_client())
    stdout = json.dumps({"type": "text", "text": "recovered"})
    output = agent._recover_from_error(
        returncode=1,
        stdout=stdout,
        stderr="oops",
        sanitized_command=["opencode", "run"],
        duration_seconds=0.1,
        output_file_content=None,
    )
    assert output is not None
    assert output.parsed.content == "recovered"
    assert output.returncode == 1
    assert output.parser_name == "opencode_jsonl"


def test_recover_from_error_returns_none_on_unparseable_stdout():
    agent = OpencodeAgent(_client())
    output = agent._recover_from_error(
        returncode=1,
        stdout="garbage and more garbage",
        stderr="failed",
        sanitized_command=["opencode", "run"],
        duration_seconds=0.1,
        output_file_content=None,
    )
    assert output is None


def test_build_command_appends_runtime_model_with_short_flag():
    agent = OpencodeAgent(_client())
    role = _client().get_role("default")
    cmd = agent._build_command(role=role, system_prompt=None, model="openai/gpt-5")
    # opencode manifest has no pre-existing -m, so the override is just appended
    assert cmd[-2:] == ["-m", "openai/gpt-5"]
