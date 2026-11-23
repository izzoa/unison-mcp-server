"""run-server.sh lists enabled/disabled tools by scanning tools/*.py; that scan must match the registry."""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _excluded_by_script() -> set[str]:
    text = (ROOT / "run-server.sh").read_text()
    start = text.index("for tool_file in tools/*.py; do")
    block = text[start : text.index("done", start)]
    return set(re.findall(r'"\$tool_name" != "(\w+)"', block))


def test_script_tool_discovery_matches_registry():
    from tools.registry import TOOL_DEFINITIONS

    discovered = {p.stem for p in (ROOT / "tools").glob("*.py")} - _excluded_by_script()
    # listmodels and version are registered but can never be disabled, so the summary omits them.
    expected = set(TOOL_DEFINITIONS) - {"listmodels", "version"}
    assert discovered == expected
