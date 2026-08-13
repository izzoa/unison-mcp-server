"""Tests for clink read-only sandbox: agent flags, prompt injection, metadata."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from clink.agents.base import BaseCLIAgent
from clink.agents.claude import ClaudeAgent
from clink.agents.codex import CodexAgent
from clink.agents.gemini import GeminiAgent
from clink.agents.opencode import OpencodeAgent


def _make_mock_client(name: str = "test") -> MagicMock:
    client = MagicMock()
    client.name = name
    client.parser = "gemini_json"
    client.executable = ["test-cli"]
    client.internal_args = []
    client.config_args = []
    client.env = {}
    client.working_dir = None
    client.timeout_seconds = 30
    client.output_to_file = None
    return client


# -----------------------------------------------------------------------
# 2.6 Agent read-only flag tests
# -----------------------------------------------------------------------


class TestBaseCLIAgentReadOnly:
    def test_default_returns_empty_list(self):
        agent = BaseCLIAgent(_make_mock_client())
        assert agent.get_read_only_args() == []


class TestGeminiAgentReadOnly:
    def test_returns_plan_approval_mode(self):
        agent = GeminiAgent(_make_mock_client("gemini"))
        args = agent.get_read_only_args()
        assert args == ["--approval-mode", "plan"]

    def test_apply_read_only_strips_yolo(self):
        """--yolo must be removed before --approval-mode plan is appended."""
        agent = GeminiAgent(_make_mock_client("gemini"))
        cmd = ["gemini", "-o", "json", "--yolo"]
        result = agent._apply_read_only(cmd)
        assert "--yolo" not in result
        assert "-y" not in result
        assert result[-2:] == ["--approval-mode", "plan"]

    def test_apply_read_only_strips_short_flag(self):
        agent = GeminiAgent(_make_mock_client("gemini"))
        cmd = ["gemini", "-o", "json", "-y"]
        result = agent._apply_read_only(cmd)
        assert "-y" not in result
        assert result[-2:] == ["--approval-mode", "plan"]


class TestClaudeAgentReadOnly:
    def test_returns_plan_mode(self):
        agent = ClaudeAgent(_make_mock_client("claude"))
        args = agent.get_read_only_args()
        assert args == ["--permission-mode", "plan"]


class TestCodexAgentReadOnly:
    def test_returns_sandbox_read_only(self):
        """Codex exec supports --sandbox read-only; use it as layer-1 enforcement."""
        agent = CodexAgent(_make_mock_client("codex"))
        args = agent.get_read_only_args()
        assert args == ["--sandbox", "read-only"]

    def test_apply_read_only_strips_dangerous_bypass_flag(self):
        """The manifest's --dangerously-bypass-approvals-and-sandbox must be removed
        so read_only=True does not run Codex fully unsandboxed."""
        agent = CodexAgent(_make_mock_client("codex"))
        cmd = ["codex", "exec", "--json", "--dangerously-bypass-approvals-and-sandbox"]
        result = agent._apply_read_only(cmd)
        assert "--dangerously-bypass-approvals-and-sandbox" not in result
        assert result[-2:] == ["--sandbox", "read-only"]

    def test_apply_read_only_strips_conflicting_sandbox_pair(self):
        """A pre-existing --sandbox <mode> pair is replaced, not duplicated."""
        agent = CodexAgent(_make_mock_client("codex"))
        cmd = ["codex", "exec", "--sandbox", "workspace-write", "--full-auto"]
        result = agent._apply_read_only(cmd)
        assert "workspace-write" not in result
        assert "--full-auto" not in result
        assert result.count("--sandbox") == 1
        assert result[-2:] == ["--sandbox", "read-only"]


class TestOpencodeAgentReadOnly:
    def test_returns_empty_list(self):
        """Opencode has no CLI flag for read-only-while-still-executing mode.

        v11.8.0 used ``--agent plan`` but that switches the agent persona
        (producing planning-language instead of executing the task), so it was
        not a true read-only sandbox. Layers 2 (prompt) and 3 (fs snapshot)
        provide enforcement.
        """
        agent = OpencodeAgent(_make_mock_client("opencode"))
        assert agent.get_read_only_args() == []

    def test_command_is_unchanged_in_read_only_mode(self):
        """Without a layer-1 flag, the executed command is identical to a
        non-read-only call for the same inputs."""
        client = _make_mock_client("opencode")
        client.parser = "opencode_jsonl"
        agent = OpencodeAgent(client)
        # The base agent's _apply_read_only just appends get_read_only_args()
        # to the command; with [] returned, the command is unchanged.
        cmd = ["opencode", "run", "--format", "json"]
        result = agent._apply_read_only(cmd.copy())
        assert result == cmd


# -----------------------------------------------------------------------
# 3.7 Clink tool read-only integration tests
# -----------------------------------------------------------------------


class TestCLinkRequestReadOnly:
    def test_read_only_field_defaults_false(self):
        from tools.clink import CLinkRequest

        req = CLinkRequest(prompt="test")
        assert req.read_only is False

    def test_read_only_field_accepts_true(self):
        from tools.clink import CLinkRequest

        req = CLinkRequest(prompt="test", read_only=True)
        assert req.read_only is True


class TestCLinkPromptInjection:
    """Assertions on the assembled prompt itself.

    These previously only checked ``request.read_only``, duplicating
    TestCLinkRequestReadOnly and providing no coverage of the injected text.
    """

    @staticmethod
    async def _assemble(read_only: bool) -> str:
        from tools.clink import CLinkRequest, CLinkTool

        tool = CLinkTool()
        client = tool._resolve_client("gemini")
        role = client.get_role("default")
        request = CLinkRequest(prompt="Analyze this code", cli_name="gemini", read_only=read_only)
        return await tool._prepare_prompt_for_role(
            request,
            role,
            client=client,
            system_prompt=role.prompt_path.read_text(encoding="utf-8"),
            include_system_prompt=True,
        )

    @pytest.mark.asyncio
    async def test_read_only_instruction_injected(self):
        prompt = await self._assemble(read_only=True)
        assert "=== READ-ONLY MODE ===" in prompt
        assert "MUST NOT create, modify, delete, or rename any file" in prompt

    @pytest.mark.asyncio
    async def test_read_only_instruction_not_injected_when_false(self):
        prompt = await self._assemble(read_only=False)
        assert "READ-ONLY MODE" not in prompt

    @pytest.mark.asyncio
    async def test_read_only_instruction_names_no_cli_specific_tools(self):
        prompt = await self._assemble(read_only=True)
        for name in ("EditFile", "WriteFile", "CreateFile", "DeleteFile", "ReplaceInFile"):
            assert name not in prompt


class TestCLinkInputSchema:
    def test_schema_includes_read_only(self):
        from tools.clink import CLinkTool

        tool = CLinkTool()
        schema = tool.get_input_schema()
        assert "read_only" in schema["properties"]
        assert schema["properties"]["read_only"]["type"] == "boolean"
        assert schema["properties"]["read_only"]["default"] is False
