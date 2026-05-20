"""Unit tests for the cross-cutting clink recursion guard.

The guard prevents loops where a clink-spawned CLI re-invokes Unison via MCP.
Depth is tracked via ``UNISON_CLINK_DEPTH``, incremented in
``BaseCLIAgent._build_environment`` and checked in
``tools.clink._check_recursion_guard`` at ``CLinkTool.execute()`` entry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clink.agents.base import BaseCLIAgent
from clink.constants import CLINK_DEPTH_ENV_VAR, CLINK_MAX_DEPTH_ENV_VAR, DEFAULT_CLINK_MAX_DEPTH
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from tools.clink import _check_recursion_guard
from tools.shared.exceptions import ToolExecutionError


def _make_agent_with_env(env: dict[str, str]) -> BaseCLIAgent:
    role = ResolvedCLIRole(
        name="default",
        prompt_path=Path("systemprompts/clink/default.txt").resolve(),
        role_args=[],
    )
    client = ResolvedCLIClient(
        name="test-agent",
        executable=["x"],
        internal_args=[],
        config_args=[],
        env=env,
        timeout_seconds=30,
        parser="codex_jsonl",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    return BaseCLIAgent(client)


# ---------------------------------------------------------------------------
# _check_recursion_guard: depth boundary cases
# ---------------------------------------------------------------------------


class TestRecursionGuardBoundaries:
    def test_depth_0_succeeds(self, monkeypatch) -> None:
        monkeypatch.delenv(CLINK_DEPTH_ENV_VAR, raising=False)
        monkeypatch.delenv(CLINK_MAX_DEPTH_ENV_VAR, raising=False)
        _check_recursion_guard()  # no exception

    def test_depth_1_succeeds_at_default_max(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "1")
        monkeypatch.delenv(CLINK_MAX_DEPTH_ENV_VAR, raising=False)
        # default max is 1; depth 1 is NOT > 1, so passes
        _check_recursion_guard()

    def test_depth_2_fails_at_default_max(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "2")
        monkeypatch.delenv(CLINK_MAX_DEPTH_ENV_VAR, raising=False)
        with pytest.raises(ToolExecutionError, match="recursion limit exceeded"):
            _check_recursion_guard()

    def test_error_message_includes_remediation(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "5")
        monkeypatch.delenv(CLINK_MAX_DEPTH_ENV_VAR, raising=False)
        with pytest.raises(ToolExecutionError) as exc_info:
            _check_recursion_guard()
        msg = str(exc_info.value)
        assert "UNISON_CLINK_DEPTH=5" in msg
        assert "MCP" in msg or "mcp" in msg  # mentions the MCP loop scenario
        assert "CLINK_MAX_RECURSION_DEPTH" in msg  # tells user how to raise the limit


class TestRecursionGuardConfigurable:
    def test_max_depth_3_allows_depth_3(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "3")
        monkeypatch.setenv(CLINK_MAX_DEPTH_ENV_VAR, "3")
        _check_recursion_guard()  # depth 3 is NOT > max 3

    def test_max_depth_3_rejects_depth_4(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "4")
        monkeypatch.setenv(CLINK_MAX_DEPTH_ENV_VAR, "3")
        with pytest.raises(ToolExecutionError, match="recursion limit"):
            _check_recursion_guard()


class TestRecursionGuardInvalidInputs:
    def test_invalid_depth_treated_as_zero(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "not-a-number")
        monkeypatch.delenv(CLINK_MAX_DEPTH_ENV_VAR, raising=False)
        _check_recursion_guard()  # invalid → 0 → no exception

    def test_invalid_max_treated_as_default(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "1")
        monkeypatch.setenv(CLINK_MAX_DEPTH_ENV_VAR, "garbage")
        # Default max is 1, depth 1 is NOT > 1, so passes
        _check_recursion_guard()

    def test_empty_string_depth_treated_as_zero(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "")
        monkeypatch.delenv(CLINK_MAX_DEPTH_ENV_VAR, raising=False)
        _check_recursion_guard()

    def test_default_max_matches_constant(self) -> None:
        # Sanity: ensure DEFAULT_CLINK_MAX_DEPTH and the guard's parsing agree
        # — fail loudly if someone changes the default in only one place.
        assert DEFAULT_CLINK_MAX_DEPTH == 1


# ---------------------------------------------------------------------------
# Env-var propagation in BaseCLIAgent._build_environment
# ---------------------------------------------------------------------------


class TestEnvVarPropagation:
    def test_unset_depth_becomes_one(self, monkeypatch) -> None:
        monkeypatch.delenv(CLINK_DEPTH_ENV_VAR, raising=False)
        agent = _make_agent_with_env({})
        env = agent._build_environment()
        assert env[CLINK_DEPTH_ENV_VAR] == "1"

    def test_depth_zero_becomes_one(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "0")
        agent = _make_agent_with_env({})
        env = agent._build_environment()
        assert env[CLINK_DEPTH_ENV_VAR] == "1"

    def test_depth_increment_propagates(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "2")
        agent = _make_agent_with_env({})
        env = agent._build_environment()
        assert env[CLINK_DEPTH_ENV_VAR] == "3"

    def test_client_env_can_override_depth(self, monkeypatch) -> None:
        """Client-specific env vars are applied BEFORE the depth increment.

        If a manifest declares ``UNISON_CLINK_DEPTH`` (uncommon but possible),
        the agent reads from the merged env and increments from there.
        """
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "0")
        agent = _make_agent_with_env({CLINK_DEPTH_ENV_VAR: "5"})
        env = agent._build_environment()
        assert env[CLINK_DEPTH_ENV_VAR] == "6"

    def test_invalid_env_value_safe(self, monkeypatch) -> None:
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "garbage")
        agent = _make_agent_with_env({})
        env = agent._build_environment()
        # Invalid input → treated as 0 → increment → 1
        assert env[CLINK_DEPTH_ENV_VAR] == "1"

    def test_other_env_vars_preserved(self, monkeypatch) -> None:
        monkeypatch.setenv("MY_OTHER_VAR", "preserved")
        monkeypatch.delenv(CLINK_DEPTH_ENV_VAR, raising=False)
        agent = _make_agent_with_env({"CLIENT_VAR": "from_manifest"})
        env = agent._build_environment()
        assert env["MY_OTHER_VAR"] == "preserved"
        assert env["CLIENT_VAR"] == "from_manifest"
        assert env[CLINK_DEPTH_ENV_VAR] == "1"


# ---------------------------------------------------------------------------
# End-to-end-ish: simulated nested invocation chain
# ---------------------------------------------------------------------------


class TestNestedInvocationChain:
    """Simulate Unison → spawned CLI → that CLI re-invokes Unison via MCP."""

    def test_chain_depth0_to_depth1_succeeds(self, monkeypatch) -> None:
        """Depth-0 Unison entry → spawn CLI with depth=1 in env → depth-1 entry succeeds."""
        # Top-level Unison invocation: depth 0
        monkeypatch.delenv(CLINK_DEPTH_ENV_VAR, raising=False)
        monkeypatch.delenv(CLINK_MAX_DEPTH_ENV_VAR, raising=False)
        _check_recursion_guard()

        # Spawned CLI sees env with depth=1
        agent = _make_agent_with_env({})
        env = agent._build_environment()
        assert env[CLINK_DEPTH_ENV_VAR] == "1"

        # If that CLI invoked Unison-via-MCP, the child Unison process would
        # see depth=1 at its entry — simulate by re-running the guard.
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, env[CLINK_DEPTH_ENV_VAR])
        _check_recursion_guard()  # depth 1, max 1 → passes

    def test_chain_depth1_to_depth2_fails(self, monkeypatch) -> None:
        """Depth-1 Unison entry → spawn CLI with depth=2 → depth-2 entry fails."""
        # Mid-chain Unison invocation: depth 1
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, "1")
        monkeypatch.delenv(CLINK_MAX_DEPTH_ENV_VAR, raising=False)
        _check_recursion_guard()  # depth 1, max 1 → passes

        # Spawned CLI sees env with depth=2
        agent = _make_agent_with_env({})
        env = agent._build_environment()
        assert env[CLINK_DEPTH_ENV_VAR] == "2"

        # Child Unison-via-MCP would see depth=2 → fails
        monkeypatch.setenv(CLINK_DEPTH_ENV_VAR, env[CLINK_DEPTH_ENV_VAR])
        with pytest.raises(ToolExecutionError, match="recursion limit"):
            _check_recursion_guard()
