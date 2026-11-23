"""Published text must not name things the tool does not have.

The schema and doc text that MCP clients and agents read has drifted from the code before (consensus described a
sequential flow it no longer had; the shared temperature text named ``execution_mode`` on tools that publish no such
field). These checks pin the class of mistake, not one instance.
"""

import json
import re
from pathlib import Path

import pytest

from tools.registry import ToolRegistry

REPO = Path(__file__).resolve().parent.parent

# Fields only a batch-capable tool publishes; text mentioning one must be on a tool that has it.
_BATCH_FIELDS = ("execution_mode", "batch_handle")


def _tool_names():
    return sorted(ToolRegistry().get_tool_names())


def _tool(name):
    return ToolRegistry().get_tool_instance(name)


def _descriptions(node, path=""):
    """Yield (path, description) for every description string in a JSON schema."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                yield path, value
            else:
                yield from _descriptions(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _descriptions(value, f"{path}[{index}]")


@pytest.mark.parametrize("name", _tool_names())
def test_a_property_description_names_a_batch_field_only_on_a_tool_that_publishes_it(name):
    tool = _tool(name)
    schema = tool.get_input_schema()
    published = set(schema.get("properties", {}))
    for path, text in _descriptions(schema.get("properties", {})):
        for field in _BATCH_FIELDS:
            if field in text and f"no {field}" not in text:  # clink's text says it has none, which is true
                assert field in published, f"{name}{path} mentions {field!r} but {name} publishes no such field"


def test_the_pre_batch_schema_fixture_names_no_batch_field_either():
    fixture = json.loads((REPO / "tests/fixtures/tool_schemas_pre_batch.json").read_text())
    for name, schema in fixture.items():
        for path, text in _descriptions(schema.get("properties", {})):
            for field in _BATCH_FIELDS:
                assert field not in text, f"{name}{path} mentions {field!r}"


# Only SimpleTool's chat-style prompt appends web-search guidance (tools/simple/base.py); the workflow tools and
# the rest never do, so their docs must not promise it.
_NO_WEB_SEARCH_DOCS = (
    "analyze",
    "codereview",
    "consensus",
    "debug",
    "docgen",
    "precommit",
    "refactor",
    "secaudit",
    "testgen",
    "thinkdeep",
    "tracer",
)


@pytest.mark.parametrize("doc", _NO_WEB_SEARCH_DOCS)
def test_workflow_tool_docs_make_no_web_search_claim(doc):
    text = (REPO / "docs/tools" / f"{doc}.md").read_text().lower()
    assert not re.search(r"web.?search", text), f"docs/tools/{doc}.md claims web search"
    assert "recommended searches for claude" not in text


def test_no_workflow_tool_code_or_prompt_adds_web_search_guidance():
    """The premise of the doc test above: nothing under tools/workflow or a workflow tool asks for searches."""
    for relative in (
        "tools/workflow",
        "tools/analyze.py",
        "tools/debug.py",
        "tools/thinkdeep.py",
        "tools/codereview.py",
    ):
        path = REPO / relative
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            assert "websearch" not in file.read_text().lower().replace("_", ""), str(file)
