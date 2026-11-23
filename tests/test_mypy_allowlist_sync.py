"""The strict-mypy file list is copied into three scripts; they must not drift apart."""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY_PATH = re.compile(r"[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*\.py")


def _paths(text: str) -> set[str]:
    return set(PY_PATH.findall(text))


def _sh_list() -> set[str]:
    text = (ROOT / "code_quality_checks.sh").read_text()
    start = text.index("$MYPY \\")
    return _paths(text[start : text.index("echo", start)])


def _ps1_list() -> set[str]:
    text = (ROOT / "code_quality_checks.ps1").read_text()
    start = text.index("$mypyFiles = @(")
    return _paths(text[start : text.index("\n        )", start)])


def _workflow_list() -> set[str]:
    text = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    start = text.index("mypy \\")
    lines = []
    for line in text[start:].splitlines()[1:]:
        if not line.strip().endswith(("\\", ".py")):
            break
        lines.append(line)
    return _paths("\n".join(lines))


def test_allowlist_is_non_trivial():
    assert len(_sh_list()) >= 45


def test_powershell_allowlist_matches_shell_script():
    assert _ps1_list() == _sh_list()


def test_workflow_allowlist_matches_shell_script():
    assert _workflow_list() == _sh_list()
