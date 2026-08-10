"""Tests for InvocationPlan.extra_args and its materialization.

The regression test here is the important one: `extra_args` defaults to empty,
so every pre-existing agent must produce byte-identical argv to before the
field existed.
"""

from __future__ import annotations

import pytest

from clink import get_registry
from clink.agents import create_agent
from clink.agents.base import InvocationPlan


def _materialize(agent, plan, prompt="hello", files=(), images=()):
    return agent._materialize_plan(plan, prompt, list(files), list(images))


@pytest.mark.parametrize("cli_name", get_registry().list_clients())
def test_default_plan_contributes_no_argv(cli_name):
    """Every existing agent's default plan adds nothing to the command."""
    client = get_registry().get_client(cli_name)
    agent = create_agent(client)

    plan = agent.prepare_invocation("hello", [], [])
    extra_args, _stdin, cleanup = _materialize(agent, plan)
    try:
        assert extra_args == [] or plan.kind == "message_file"
    finally:
        cleanup()


@pytest.mark.parametrize("cli_name", get_registry().list_clients())
def test_existing_agent_command_is_byte_identical(cli_name):
    """Command construction is unchanged by the extra_args addition."""
    client = get_registry().get_client(cli_name)
    agent = create_agent(client)
    role = client.get_role("default")

    command = agent._build_command(role=role, system_prompt=None, model=None)

    expected = list(client.executable) + list(client.internal_args) + list(client.config_args) + list(role.role_args)
    assert command == expected


def test_stdin_plan_with_extra_args_appends_argv_and_keeps_stdin():
    client = get_registry().get_client("gemini")
    agent = create_agent(client)

    plan = InvocationPlan(kind="stdin", extra_args=["--attachment", "/tmp/a.png"])
    extra_args, stdin_data, cleanup = _materialize(agent, plan, prompt="describe this")
    try:
        assert extra_args == ["--attachment", "/tmp/a.png"]
        assert stdin_data == b"describe this"
    finally:
        cleanup()


def test_stdin_plan_without_extra_args_is_unchanged():
    client = get_registry().get_client("gemini")
    agent = create_agent(client)

    extra_args, stdin_data, cleanup = _materialize(agent, InvocationPlan(kind="stdin"))
    try:
        assert extra_args == []
        assert stdin_data == b"hello"
    finally:
        cleanup()


def test_argv_plan_puts_extra_args_before_the_positional_prompt():
    client = get_registry().get_client("gemini")
    agent = create_agent(client)

    plan = InvocationPlan(kind="argv", flag="-p", extra_args=["--attachment", "/tmp/a.png"])
    extra_args, _stdin, cleanup = _materialize(agent, plan, prompt="hi")
    try:
        assert extra_args == ["--attachment", "/tmp/a.png", "-p", "hi"]
    finally:
        cleanup()


def test_message_file_plan_honors_extra_args():
    client = get_registry().get_client("aider")
    agent = create_agent(client)

    plan = InvocationPlan(kind="message_file", flag="--message-file", extra_args=["--foo", "bar"])
    extra_args, stdin_data, cleanup = _materialize(agent, plan)
    try:
        assert extra_args[:2] == ["--foo", "bar"]
        assert extra_args[2] == "--message-file"
        assert stdin_data == b""
    finally:
        cleanup()


def test_stream_json_plan_honors_extra_args():
    client = get_registry().get_client("amp")
    agent = create_agent(client)

    plan = InvocationPlan(kind="stream_json", extra_args=["--foo"])
    extra_args, stdin_data, cleanup = _materialize(agent, plan)
    try:
        assert extra_args == ["--foo"]
        assert b"messages" in stdin_data
    finally:
        cleanup()
