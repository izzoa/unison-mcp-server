import json

import pytest

from clink import get_registry
from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from tools.clink import MAX_RESPONSE_CHARS, CLinkTool


@pytest.mark.asyncio
async def test_clink_tool_execute(monkeypatch):
    tool = CLinkTool()

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content="Hello from Gemini", metadata={"model_used": "gemini-2.5-pro"}),
            sanitized_command=["gemini", "-o", "json"],
            returncode=0,
            stdout='{"response": "Hello from Gemini"}',
            stderr="",
            duration_seconds=0.42,
            parser_name="gemini_json",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    def fake_create_agent(client):
        return DummyAgent()

    monkeypatch.setattr("tools.clink.create_agent", fake_create_agent)

    arguments = {
        "prompt": "Summarize the project",
        "cli_name": "gemini",
        "role": "default",
        "absolute_file_paths": [],
        "images": [],
    }

    results = await tool.execute(arguments)
    assert len(results) == 1

    payload = json.loads(results[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    assert "Hello from Gemini" in payload["content"]
    metadata = payload.get("metadata", {})
    assert metadata.get("cli_name") == "gemini"
    assert metadata.get("command") == ["gemini", "-o", "json"]


def test_registry_lists_roles():
    registry = get_registry()
    clients = registry.list_clients()
    assert {"codex", "gemini"}.issubset(set(clients))
    roles = registry.list_roles("gemini")
    assert "default" in roles
    assert "default" in registry.list_roles("codex")
    codex_client = registry.get_client("codex")
    # Verify codex uses --enable web_search_request (not --search which is unsupported by exec)
    assert codex_client.config_args == [
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--enable",
        "web_search_request",
    ]


@pytest.mark.asyncio
async def test_clink_tool_defaults_to_first_cli(monkeypatch):
    tool = CLinkTool()

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content="Default CLI response", metadata={"events": ["foo"]}),
            sanitized_command=["gemini"],
            returncode=0,
            stdout='{"response": "Default CLI response"}',
            stderr="",
            duration_seconds=0.1,
            parser_name="gemini_json",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Hello",
        "absolute_file_paths": [],
        "images": [],
    }

    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    metadata = payload.get("metadata", {})
    assert metadata.get("cli_name") == tool._default_cli_name
    assert metadata.get("events_removed_for_normal") is True


@pytest.mark.asyncio
async def test_clink_tool_truncates_large_output(monkeypatch):
    tool = CLinkTool()

    summary_section = "<SUMMARY>This is the condensed summary.</SUMMARY>"
    long_text = "A" * (MAX_RESPONSE_CHARS + 500) + summary_section

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content=long_text, metadata={"events": ["event1", "event2"]}),
            sanitized_command=["codex"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.2,
            parser_name="codex_jsonl",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Summarize",
        "cli_name": tool._default_cli_name,
        "absolute_file_paths": [],
        "images": [],
    }

    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    assert payload["content"].strip() == "This is the condensed summary."
    metadata = payload.get("metadata", {})
    assert metadata.get("output_summarized") is True
    assert metadata.get("events_removed_for_normal") is True
    assert metadata.get("output_original_length") == len(long_text)


def test_input_schema_includes_optional_model_property():
    tool = CLinkTool()
    schema = tool.get_input_schema()
    properties = schema.get("properties", {})
    assert "model" in properties
    model_prop = properties["model"]
    assert model_prop.get("type") == "string"
    assert "enum" not in model_prop
    assert "model" not in schema.get("required", [])


@pytest.mark.asyncio
async def test_read_only_violations_have_classified_shape(monkeypatch):
    """Read-only response metadata uses the nested {by_model, by_cli_bookkeeping} shape."""
    from utils.fs_snapshot import SnapshotDiff

    tool = CLinkTool()

    # Simulate the snapshot diff that opencode produces on first-run bootstrap
    # plus one model-driven write.
    fake_diff = SnapshotDiff(
        created=[
            ".opencode/package.json",
            ".opencode/node_modules/foo.json",
            ".git/opencode",
            "src/main.py",  # genuine model write
        ],
    )

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content="ok", metadata={}),
            sanitized_command=["opencode", "run", "--format", "json"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.1,
            parser_name="opencode_jsonl",
            output_file_content=None,
        )

    class DummyAgent:
        fs_violation_ignore_patterns = (
            ".opencode/.gitignore",
            ".opencode/package.json",
            ".opencode/package-lock.json",
            ".opencode/node_modules/**",
            ".git/opencode",
        )

        def get_read_only_args(self):
            return []

        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda c: DummyAgent())
    monkeypatch.setattr("tools.clink.capture_snapshot", lambda d: {})
    monkeypatch.setattr("tools.clink.diff_snapshots", lambda a, b: fake_diff)

    arguments = {
        "prompt": "review",
        "cli_name": "opencode",
        "read_only": True,
        "absolute_file_paths": [],
        "images": [],
    }
    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    metadata = payload["metadata"]

    # Shape: nested object with both buckets always present
    assert metadata["read_only_enforced"] is True
    assert metadata["read_only_sandbox_flags"] == []
    violations = metadata["read_only_violations"]
    assert isinstance(violations, dict)
    assert "by_model" in violations
    assert "by_cli_bookkeeping" in violations

    # Each bucket has the three categories
    for bucket_name in ("by_model", "by_cli_bookkeeping"):
        bucket = violations[bucket_name]
        for category in ("created", "modified", "deleted"):
            assert category in bucket
            assert isinstance(bucket[category], list)

    # Routing: bootstrap files -> bookkeeping; src/main.py -> model
    assert "src/main.py" in violations["by_model"]["created"]
    assert ".opencode/package.json" in violations["by_cli_bookkeeping"]["created"]
    assert ".opencode/node_modules/foo.json" in violations["by_cli_bookkeeping"]["created"]
    assert ".git/opencode" in violations["by_cli_bookkeeping"]["created"]
    # No leakage in either direction
    assert "src/main.py" not in violations["by_cli_bookkeeping"]["created"]
    assert ".opencode/package.json" not in violations["by_model"]["created"]


@pytest.mark.asyncio
async def test_warning_logged_only_for_model_driven_violations(monkeypatch, caplog):
    """The WARNING fires for by_model content; bookkeeping-only changes are silent."""
    from utils.fs_snapshot import SnapshotDiff

    tool = CLinkTool()

    bookkeeping_only = SnapshotDiff(
        created=[".opencode/package.json", ".git/opencode"],
    )

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content="ok", metadata={}),
            sanitized_command=["opencode"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.1,
            parser_name="opencode_jsonl",
            output_file_content=None,
        )

    class DummyAgent:
        fs_violation_ignore_patterns = (".opencode/package.json", ".git/opencode")

        def get_read_only_args(self):
            return []

        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda c: DummyAgent())
    monkeypatch.setattr("tools.clink.capture_snapshot", lambda d: {})
    monkeypatch.setattr("tools.clink.diff_snapshots", lambda a, b: bookkeeping_only)

    arguments = {
        "prompt": "review",
        "cli_name": "opencode",
        "read_only": True,
        "absolute_file_paths": [],
        "images": [],
    }

    with caplog.at_level("WARNING", logger="tools.clink"):
        await tool.execute(arguments)
    warnings = [r for r in caplog.records if "Read-only violation" in r.message]
    assert warnings == [], (
        "WARNING should not fire when only CLI bookkeeping changed; " f"got {[r.message for r in warnings]}"
    )


@pytest.mark.asyncio
async def test_clink_tool_truncates_without_summary(monkeypatch):
    tool = CLinkTool()

    long_text = "B" * (MAX_RESPONSE_CHARS + 1000)

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content=long_text, metadata={"events": ["event"]}),
            sanitized_command=["codex"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.2,
            parser_name="codex_jsonl",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Summarize",
        "cli_name": tool._default_cli_name,
        "absolute_file_paths": [],
        "images": [],
    }

    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    assert "exceeding the configured clink limit" in payload["content"]
    metadata = payload.get("metadata", {})
    assert metadata.get("output_truncated") is True
    assert metadata.get("events_removed_for_normal") is True
    assert metadata.get("output_original_length") == len(long_text)
