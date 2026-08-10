"""Smoke tests for the GitHub Copilot CLI clink integration.

Fixtures are real JSONL stdout captured from GitHub Copilot CLI **1.0.78** via
its BYOK path (`COPILOT_PROVIDER_BASE_URL` against a local stub provider), which
bypasses GitHub model routing. The verbose `session.skills_loaded` event is
omitted from the fixtures: it enumerates the user's personal skill names and
filesystem paths, and the parser must never surface it.

Mocked binary, no real Copilot invocation in CI.
"""

from __future__ import annotations

import base64
import os

import pytest

from clink import get_registry
from clink.agents import create_agent
from clink.agents.base import CLIAgentError
from clink.agents.copilot import READ_ONLY_AVAILABLE_TOOLS, CopilotAgent
from clink.parsers.base import ParserError
from clink.parsers.copilot import CopilotJSONLParser

# ---------------------------------------------------------------------------
# Fixtures captured from real Copilot invocations (CLI 1.0.78)
# ---------------------------------------------------------------------------

COPILOT_HAPPY_PATH_FIXTURE = """{"type":"session.tools_updated","data":{"model":"stub-model"},"ephemeral":true,"id":"589c640d","timestamp":"2026-08-10T14:16:14.1Z","parentId":"101809a4"}
{"type":"user.message","data":{"content":"probe","attachments":[]},"id":"8585caaa","timestamp":"2026-08-10T14:16:14.181Z","parentId":"c3d56b2a"}
{"type":"assistant.turn_start","data":{"turnId":"0","interactionId":"91b6447b"},"id":"b335c8a6","timestamp":"2026-08-10T14:16:14.2Z","parentId":"8585caaa"}
{"type":"assistant.message_delta","data":{"messageId":"7f8c415d","deltaContent":"PONG"},"ephemeral":true,"id":"ba1f24f3","timestamp":"2026-08-10T14:16:14.265Z","parentId":"431a672d"}
{"type":"assistant.message","data":{"messageId":"7f8c415d","model":"stub-model","content":"PONG","toolRequests":[],"interactionId":"91b6447b","turnId":"0","rte":false,"apiCallId":"stub"},"id":"872581eb","timestamp":"2026-08-10T14:16:14.269Z","parentId":"431a672d"}
{"type":"assistant.turn_end","data":{"turnId":"0"},"id":"6a7e5605","timestamp":"2026-08-10T14:16:14.3Z","parentId":"872581eb"}
{"type":"result","timestamp":"2026-08-10T14:16:14.306Z","sessionId":"91b2665b","exitCode":0,"usage":{"premiumRequests":0,"totalApiDurationMs":43,"sessionDurationMs":317,"codeChanges":{"linesAdded":0,"linesRemoved":0,"filesModified":[]}}}
"""

#: A run where the model delegated through the `task` tool. The subagent's
#: durable assistant.message carries `agentId` on the envelope and
#: `parentToolCallId` in data; the root agent's messages carry neither.
COPILOT_DELEGATION_FIXTURE = """{"type":"user.message","data":{"content":"probe","attachments":[]},"id":"4101f395","timestamp":"2026-08-10T14:16:19.696Z","parentId":"744b13d4"}
{"type":"assistant.message","data":{"messageId":"58f286b6","model":"stub-model","content":"","toolRequests":[{"toolCallId":"call_stub_1","name":"task","type":"function"}],"interactionId":"e99d2f4f","turnId":"0","rte":false,"apiCallId":"stub"},"id":"aaa1","timestamp":"2026-08-10T14:16:19.75Z","parentId":"744b13d4"}
{"type":"subagent.started","data":{"toolCallId":"call_stub_1","agentName":"explore"},"id":"sub1","timestamp":"2026-08-10T14:16:19.76Z","parentId":"aaa1"}
{"type":"assistant.message","data":{"messageId":"7b342fae","model":"stub-model","content":"SUBAGENT-OR-ROOT-2","toolRequests":[],"interactionId":"5cef1ccb","turnId":"0","rte":false,"apiCallId":"stub","parentToolCallId":"call_stub_1"},"agentId":"call_stub_1","id":"93dd07b3","timestamp":"2026-08-10T14:16:19.803Z","parentId":"6abdac53"}
{"type":"subagent.completed","data":{"toolCallId":"call_stub_1","agentName":"explore","durationMs":64},"id":"sub2","timestamp":"2026-08-10T14:16:19.81Z","parentId":"aaa1"}
{"type":"assistant.message","data":{"messageId":"f03ada2c","model":"stub-model","content":"SUBAGENT-OR-ROOT-3","toolRequests":[],"interactionId":"e99d2f4f","turnId":"1","rte":false,"apiCallId":"stub"},"id":"170e5ad5","timestamp":"2026-08-10T14:16:19.844Z","parentId":"8b5c4ded"}
{"type":"result","timestamp":"2026-08-10T14:16:19.858Z","sessionId":"9d022a77","exitCode":0,"usage":{"premiumRequests":0,"totalApiDurationMs":27,"sessionDurationMs":263,"codeChanges":{"linesAdded":0,"linesRemoved":0,"filesModified":[]}}}
"""

#: Upstream failure: five retries, no assistant message, exitCode 1.
COPILOT_ERROR_FIXTURE = """{"type":"model.call_failure","data":{"model":"stub-model","statusCode":500,"failureKind":"api","errorMessage":"500 stub upstream failure"},"ephemeral":true,"id":"f1","timestamp":"2026-08-10T14:18:00Z","parentId":"p1"}
{"type":"assistant.turn_retry","data":{"attempt":1},"ephemeral":true,"id":"r1","timestamp":"2026-08-10T14:18:01Z","parentId":"p1"}
{"type":"session.error","data":{"errorType":"query","message":"Failed to get response from the AI model; retried 5 times (total retry wait time: 33.16 seconds) Last error: 500 stub upstream failure","statusCode":500},"id":"881a482d","timestamp":"2026-08-10T14:18:20.879Z","parentId":"039bdb7f"}
{"type":"result","timestamp":"2026-08-10T14:18:20.936Z","sessionId":"eff19770","exitCode":1,"usage":{"premiumRequests":0,"totalApiDurationMs":0,"sessionDurationMs":33659,"codeChanges":{"linesAdded":0,"linesRemoved":0,"filesModified":[]}}}
"""

#: Real event that must never reach metadata — it enumerates the user's skills.
COPILOT_SKILLS_LEAK_FIXTURE = (
    '{"type":"session.skills_loaded","data":{"skills":[{"name":"private-skill",'
    '"path":"/Users/someone/.agents/skills/private-skill/SKILL.md"}]},"ephemeral":true,'
    '"id":"s1","timestamp":"2026-08-10T14:16:14.0Z","parentId":"p0"}\n' + COPILOT_HAPPY_PATH_FIXTURE
)


class TestCopilotParser:
    def test_response_is_the_assistant_message_content(self):
        parsed = CopilotJSONLParser().parse(COPILOT_HAPPY_PATH_FIXTURE, "")
        assert parsed.content == "PONG"

    def test_model_is_reported(self):
        parsed = CopilotJSONLParser().parse(COPILOT_HAPPY_PATH_FIXTURE, "")
        assert parsed.metadata["model_used"] == "stub-model"

    def test_flat_result_event_supplies_session_and_usage(self):
        parsed = CopilotJSONLParser().parse(COPILOT_HAPPY_PATH_FIXTURE, "")
        assert parsed.metadata["session_id"] == "91b2665b"
        assert parsed.metadata["exit_code"] == 0
        assert parsed.metadata["usage"]["codeChanges"]["filesModified"] == []

    def test_streaming_deltas_are_not_duplicated_into_content(self):
        parsed = CopilotJSONLParser().parse(COPILOT_HAPPY_PATH_FIXTURE, "")
        assert parsed.content.count("PONG") == 1

    def test_subagent_message_is_not_returned_as_the_response(self):
        parsed = CopilotJSONLParser().parse(COPILOT_DELEGATION_FIXTURE, "")
        assert "SUBAGENT-OR-ROOT-2" not in parsed.content
        assert parsed.content == "SUBAGENT-OR-ROOT-3"

    def test_skills_event_never_reaches_metadata(self):
        parsed = CopilotJSONLParser().parse(COPILOT_SKILLS_LEAK_FIXTURE, "")
        blob = repr(parsed.metadata)
        assert "private-skill" not in blob
        assert "SKILL.md" not in blob
        assert "skills" not in blob

    def test_failure_surfaces_diagnostic_not_a_bare_parse_error(self):
        with pytest.raises(ParserError) as excinfo:
            CopilotJSONLParser().parse(COPILOT_ERROR_FIXTURE, "")
        assert "retried 5 times" in str(excinfo.value)

    def test_empty_stdout_falls_back_to_stderr(self):
        with pytest.raises(ParserError) as excinfo:
            CopilotJSONLParser().parse("", "Access denied by policy settings")
        assert "Access denied by policy settings" in str(excinfo.value)

    def test_malformed_lines_are_skipped(self):
        noisy = "not json\n" + COPILOT_HAPPY_PATH_FIXTURE + "{bad\n"
        assert CopilotJSONLParser().parse(noisy, "").content == "PONG"


class TestCopilotRegistration:
    def test_copilot_is_a_configured_client(self):
        assert "copilot" in get_registry().list_clients()

    def test_factory_returns_the_copilot_agent(self):
        """Guards the silent BaseCLIAgent fallback in the agent factory."""
        agent = create_agent(get_registry().get_client("copilot"))
        assert isinstance(agent, CopilotAgent)

    def test_roles_are_available(self):
        assert sorted(get_registry().list_roles("copilot")) == ["codereviewer", "default", "planner"]


class TestCopilotCommand:
    @staticmethod
    def _command(**kwargs):
        client = get_registry().get_client("copilot")
        agent = create_agent(client)
        return agent, agent._build_command(role=client.get_role("default"), system_prompt=None, **kwargs)

    def test_pinned_invariant_flags_present(self):
        _agent, command = self._command(model=None)
        assert "--output-format" in command and "json" in command
        assert "--allow-all-tools" in command
        assert "--no-auto-update" in command
        assert "--no-color" in command

    def test_prompt_is_not_passed_on_argv(self):
        agent, command = self._command(model=None)
        assert "-p" not in command
        assert "--prompt" not in command
        plan = agent.prepare_invocation("secret prompt text", [], [])
        assert plan.kind == "stdin"
        assert "secret prompt text" not in plan.extra_args

    def test_model_is_rendered_once(self):
        _agent, command = self._command(model="claude-sonnet-4.6")
        assert command.count("--model") == 1
        assert command[command.index("--model") + 1] == "claude-sonnet-4.6"


class TestCopilotReadOnly:
    def test_allowlist_and_denials_present(self):
        agent = create_agent(get_registry().get_client("copilot"))
        args = agent.get_read_only_args()
        assert args[0] == "--available-tools"
        assert args[1] == ",".join(READ_ONLY_AVAILABLE_TOOLS)
        assert args.count("--deny-tool") == 2
        assert "write" in args and "shell" in args

    def test_mutation_tools_are_not_in_the_allowlist(self):
        for tool in ("bash", "create", "edit", "write_agent", "task", "skill"):
            assert tool not in READ_ONLY_AVAILABLE_TOOLS

    def test_permission_broadening_flags_are_stripped(self):
        agent = create_agent(get_registry().get_client("copilot"))
        command = ["copilot", "--yolo", "--allow-all-paths", "--allow-all-tools", "--allow-all-urls"]
        result = agent._apply_read_only(list(command))
        for flag in ("--yolo", "--allow-all-paths", "--allow-all-urls"):
            assert flag not in result
        # Required for non-interactive operation; denials outrank it.
        assert "--allow-all-tools" in result
        assert "--available-tools" in result

    def test_no_restrictions_applied_when_not_read_only(self):
        _agent, command = TestCopilotCommand._command(model=None)
        assert "--available-tools" not in command
        assert "--deny-tool" not in command


class TestCopilotAttachments:
    def test_image_paths_become_attachment_flags(self, tmp_path):
        one = tmp_path / "a.png"
        two = tmp_path / "b.jpg"
        one.write_bytes(b"x")
        two.write_bytes(b"y")

        agent = create_agent(get_registry().get_client("copilot"))
        plan = agent.prepare_invocation("hi", [], [str(one), str(two)])
        assert plan.extra_args == ["--attachment", str(one), "--attachment", str(two)]
        assert plan.kind == "stdin"

    def test_no_images_produces_no_flags(self):
        agent = create_agent(get_registry().get_client("copilot"))
        assert agent.prepare_invocation("hi", [], []).extra_args == []

    def test_base64_blob_is_materialized_then_cleaned_up(self):
        blob = base64.b64encode(b"fake-png-bytes").decode()
        agent = create_agent(get_registry().get_client("copilot"))
        plan = agent.prepare_invocation("hi", [], [blob])

        assert plan.extra_args[0] == "--attachment"
        path = plan.extra_args[1]
        assert os.path.exists(path)

        agent.cleanup_attachments()
        assert not os.path.exists(path)

    def test_data_uri_prefix_is_honored(self):
        blob = "data:image/jpeg;base64," + base64.b64encode(b"jpeg").decode()
        agent = create_agent(get_registry().get_client("copilot"))
        plan = agent.prepare_invocation("hi", [], [blob])
        try:
            assert plan.extra_args[1].endswith(".jpg")
        finally:
            agent.cleanup_attachments()

    def test_unsupported_format_is_filtered_and_recorded(self, tmp_path):
        bad = tmp_path / "notes.txt"
        bad.write_text("nope")
        agent = create_agent(get_registry().get_client("copilot"))
        plan = agent.prepare_invocation("hi", [], [str(bad)])
        assert plan.extra_args == []
        assert str(bad) in agent._skipped_attachments


class TestCopilotErrorRecovery:
    @staticmethod
    def _agent():
        return create_agent(get_registry().get_client("copilot"))

    def test_nonzero_exit_with_a_response_is_salvaged(self):
        out = self._agent()._recover_from_error(
            returncode=1,
            stdout=COPILOT_HAPPY_PATH_FIXTURE,
            stderr="",
            sanitized_command=["copilot"],
            duration_seconds=0.1,
            output_file_content=None,
        )
        assert out is not None
        assert out.parsed.content == "PONG"

    def test_policy_denial_is_reported_with_its_message(self):
        agent = self._agent()
        with pytest.raises(CLIAgentError) as excinfo:
            agent._recover_from_error(
                returncode=1,
                stdout="",
                stderr="Error: Access denied by policy settings (Request ID: X)",
                sanitized_command=["copilot"],
                duration_seconds=0.1,
                output_file_content=None,
            )
        message = str(excinfo.value)
        assert "Access denied by policy settings" in message
        assert "parse" not in message.lower()

    def test_upstream_failure_reports_the_session_error(self):
        agent = self._agent()
        with pytest.raises(CLIAgentError) as excinfo:
            agent._recover_from_error(
                returncode=1,
                stdout=COPILOT_ERROR_FIXTURE,
                stderr="",
                sanitized_command=["copilot"],
                duration_seconds=0.1,
                output_file_content=None,
            )
        assert "retried 5 times" in str(excinfo.value)


class TestCopilotRecursionGuard:
    def test_depth_is_propagated_to_the_subprocess(self, monkeypatch):
        from clink.constants import CLINK_DEPTH_ENV_VAR

        monkeypatch.delenv(CLINK_DEPTH_ENV_VAR, raising=False)
        agent = create_agent(get_registry().get_client("copilot"))
        env = agent._build_environment()
        assert env[CLINK_DEPTH_ENV_VAR] == "1"

    def test_guard_fires_past_max_depth(self, monkeypatch):
        from clink.constants import CLINK_DEPTH_ENV_VAR
        from tools.clink import _check_recursion_guard
        from tools.shared.exceptions import ToolExecutionError

        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "2")
        with pytest.raises(ToolExecutionError):
            _check_recursion_guard()


def test_tests_do_not_require_the_copilot_binary():
    """Nothing here shells out; the suite runs without copilot installed."""
    parsed = CopilotJSONLParser().parse(COPILOT_HAPPY_PATH_FIXTURE, "")
    assert parsed.content == "PONG"
