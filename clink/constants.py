"""Internal defaults and constants for clink."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_STREAM_LIMIT = 10 * 1024 * 1024  # 10MB per stream

# Recursion-guard environment variables (see clink-multi-cli-infrastructure).
# A clink-spawned CLI that itself wires Unison as an MCP server creates a
# potential infinite loop. We propagate a depth counter via env var so the
# child Unison process can detect the recursion at CLinkTool.execute() entry.
CLINK_DEPTH_ENV_VAR = "UNISON_CLINK_DEPTH"
CLINK_MAX_DEPTH_ENV_VAR = "CLINK_MAX_RECURSION_DEPTH"
DEFAULT_CLINK_MAX_DEPTH = 1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_PROMPTS_DIR = PROJECT_ROOT / "systemprompts" / "clink"
CONFIG_DIR = PROJECT_ROOT / "conf" / "cli_clients"
USER_CONFIG_DIR = Path.home() / ".unison" / "cli_clients"


@dataclass(frozen=True)
class CLIInternalDefaults:
    """Internal defaults applied to a CLI client during registry load."""

    parser: str
    additional_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    default_role_prompt: str | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    runner: str | None = None


INTERNAL_DEFAULTS: dict[str, CLIInternalDefaults] = {
    "gemini": CLIInternalDefaults(
        parser="gemini_json",
        additional_args=["-o", "json"],
        default_role_prompt="systemprompts/clink/default.txt",
        runner="gemini",
    ),
    "codex": CLIInternalDefaults(
        parser="codex_jsonl",
        additional_args=["exec"],
        default_role_prompt="systemprompts/clink/default.txt",
        runner="codex",
    ),
    "claude": CLIInternalDefaults(
        parser="claude_json",
        additional_args=["--print", "--output-format", "json"],
        default_role_prompt="systemprompts/clink/default.txt",
        runner="claude",
    ),
    "opencode": CLIInternalDefaults(
        parser="opencode_jsonl",
        additional_args=["--format", "json"],
        default_role_prompt="systemprompts/clink/default.txt",
        runner="opencode",
    ),
    "aider": CLIInternalDefaults(
        parser="aider_text",
        additional_args=[],
        default_role_prompt="systemprompts/clink/default.txt",
        runner="aider",
    ),
}
