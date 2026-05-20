"""Crush CLI agent for clink.

Crush (https://github.com/charmbracelet/crush) is Charm's multi-provider
terminal AI assistant. We invoke it via ``crush run [prompt] --quiet`` which
is its documented non-interactive mode (the bare ``crush`` invocation launches
the TUI). Prompt delivery uses the default stdin transport — ``crush run``
accepts piped stdin per Charm's docs.

Read-only enforcement: Crush has no native dry-run flag (as of v0.70.0).
Returns ``[]`` from ``get_read_only_args()`` so enforcement falls back to
layer-2 (prompt instruction) and layer-3 (filesystem snapshot diff) —
identical to opencode's approach.

Model selection: Crush supports ``-m`` / ``--model`` and accepts both bare
model names ("gpt-4o") and ``provider/model`` form ("openai/gpt-4o") for
disambiguation across providers — same syntax pattern as opencode.
"""

from __future__ import annotations

from .base import BaseCLIAgent


class CrushAgent(BaseCLIAgent):
    """Crush clink agent — stdin transport, prompt-only read-only."""

    model_flag_aliases: tuple[str, ...] = ("-m", "--model")

    # Crush stores per-project state under .crush/ on first run. The directory
    # structure documented as of v0.70.0 includes session state and provider
    # caches — classify as bookkeeping so they don't show up as model writes
    # in read-only violation reports.
    fs_violation_ignore_patterns: tuple[str, ...] = (".crush/**",)

    def get_read_only_args(self) -> list[str]:
        """Crush has no documented dry-run / plan flag as of v0.70.0.

        Enforcement falls back to prompt instruction + filesystem snapshot
        diff (the same defence-in-depth opencode uses).
        """
        return []

    def render_model_args(self, model: str) -> list[str]:
        return ["--model", model]
