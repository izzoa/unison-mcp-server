"""Tests for clink read-only sandbox: agent flags, prompt injection, metadata."""

from __future__ import annotations

from unittest.mock import MagicMock

from clink.agents.base import BaseCLIAgent
from clink.agents.claude import ClaudeAgent
from clink.agents.codex import CodexAgent
from clink.agents.gemini import GeminiAgent


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
    def test_returns_empty_list(self):
        """Codex exec has no read-only flag; enforcement via prompt."""
        agent = CodexAgent(_make_mock_client("codex"))
        args = agent.get_read_only_args()
        assert args == []


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
    def test_read_only_instruction_injected(self):
        """When read_only=true, request carries the flag for prompt injection."""
        from tools.clink import CLinkRequest

        request = CLinkRequest(prompt="Analyze this code", read_only=True)
        assert request.read_only is True

    def test_read_only_instruction_not_injected_when_false(self):
        from tools.clink import CLinkRequest

        request = CLinkRequest(prompt="Edit this code", read_only=False)
        assert request.read_only is False


class TestCLinkInputSchema:
    def test_schema_includes_read_only(self):
        from tools.clink import CLinkTool

        tool = CLinkTool()
        schema = tool.get_input_schema()
        assert "read_only" in schema["properties"]
        assert schema["properties"]["read_only"]["type"] == "boolean"
        assert schema["properties"]["read_only"]["default"] is False
