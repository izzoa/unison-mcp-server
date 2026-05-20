"""Amp CLI agent for clink.

Amp (https://ampcode.com) is Sourcegraph's AI coding agent CLI. We invoke it
via ``amp --execute --stream-json`` which is its documented non-interactive
mode with parseable JSONL output. Prompt delivery uses stdin by default;
``prepare_invocation`` switches to ``stream_json`` plan when images are
present so Amp's ``--stream-json-input`` schema is used.

**Authentication:** Amp requires Sourcegraph account credentials via the
``AMP_API_KEY`` environment variable (for non-interactive use) or a prior
``amp login`` flow (for interactive setup). Unison does NOT manage Amp's
auth state — the user sets ``AMP_API_KEY`` in their environment before
launching Unison.

**Model selection:** Amp uses *modes* (named agent personas) rather than
explicit model names. The four documented modes — ``deep``, ``large``,
``rush``, ``smart`` — are exposed via clink's ``model`` parameter and
emitted as ``--mode <name>``. The manifest's ``supported_models``
allowlist constrains valid values per the ``clink-runtime-model-selection``
mechanics.

**Read-only mode:** Amp has no single read-only flag. Permissions are
managed via ``amp permissions`` (configured ahead of time, not at
invocation). For ``read_only=True``, this agent returns ``[]`` so
enforcement falls back to prompt instruction + filesystem-snapshot
diff (same as opencode and Crush).

**Recursion guard:** Amp IS MCP-aware (``amp mcp`` subcommand exists,
``amp.mcpServers`` config supports user-defined servers including
Unison). The cross-cutting guard in ``CLinkTool.execute()`` covers this
— this agent does NOT implement an Amp-specific guard.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import BaseCLIAgent, InvocationPlan


class AmpAgent(BaseCLIAgent):
    """Amp clink agent — stdin transport for text, stream_json for images."""

    model_flag_aliases: tuple[str, ...] = ("-m", "--mode")

    # Amp stores per-user session state under ~/.config/amp/, not in the
    # working directory. No project-local bookkeeping files known as of
    # the implementation spike (amp 0.0.1775837683-g6ddb8e).
    fs_violation_ignore_patterns: tuple[str, ...] = ()

    def prepare_invocation(
        self,
        prompt: str,
        files: Sequence[str],
        images: Sequence[str],
    ) -> InvocationPlan:
        """Use stream_json plan when images are present, stdin otherwise.

        Amp's ``--execute`` mode accepts a prompt via stdin for text-only
        use, but image input requires the structured ``--stream-json-input``
        schema. We switch transports based on whether the caller has
        attached images.
        """
        _ = (prompt, files)
        if images:
            return InvocationPlan(kind="stream_json")
        return InvocationPlan(kind="stdin")

    def get_read_only_args(self) -> list[str]:
        """Amp has no single read-only flag — permissions are managed via
        ``amp permissions`` (configured separately, not at invocation).
        Enforcement falls back to prompt instruction + filesystem snapshot.
        """
        return []

    def render_model_args(self, model: str) -> list[str]:
        return ["--mode", model]
