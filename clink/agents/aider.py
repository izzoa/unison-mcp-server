"""Aider CLI agent for clink.

Aider (https://aider.chat) is a git-aware AI pair programmer. Unlike the other
clink targets, Aider has no stdin scripting mode — its non-interactive interface
is ``--message-file <path>`` which reads the prompt from a file on disk. We use
the ``message_file`` invocation plan introduced in Phase 0 to handle this.

Read-only mode uses Aider's native ``--dry-run`` flag, which tells Aider to
compute edits without writing them to disk. The filesystem-snapshot post-check
still runs as defence-in-depth (in case Aider's dry-run leaks auxiliary writes
to chat-history or input-history files; those classify as bookkeeping rather
than model writes).

Auto-commit suppression: clink always passes ``--no-auto-commits`` because a
clink-spawned Aider invocation acts on behalf of a calling CLI's user — the
user controls commit timing themselves, surprising them with a side-effect
commit feels wrong. (Manifest decision, not negotiable per-call.)
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import BaseCLIAgent, InvocationPlan


class AiderAgent(BaseCLIAgent):
    """Aider clink agent — uses --message-file transport and --dry-run for read-only."""

    model_flag_aliases: tuple[str, ...] = ("--model",)

    # Bookkeeping files Aider creates in the working directory during normal
    # operation. Listed so the post-execution filesystem-snapshot check
    # classifies them as ``by_cli_bookkeeping`` rather than ``by_model``
    # violations (Aider users expect these files; clink callers should not
    # see them surface in read-only reports as actual model writes).
    fs_violation_ignore_patterns: tuple[str, ...] = (
        ".aider.chat.history.md",
        ".aider.input.history",
        ".aider.llm.history",
        ".aider.tags.cache.v4/**",
        ".aider.tags.cache.v3/**",
    )

    def prepare_invocation(
        self,
        prompt: str,
        files: Sequence[str],
        images: Sequence[str],
    ) -> InvocationPlan:
        """Use Aider's --message-file scripting flag.

        Aider does NOT read prompts from stdin — its documented non-interactive
        mode is ``--message-file <path>``. Phase 0's ``message_file`` plan kind
        writes the prompt to a temp file, appends the flag + path to the
        command line, and cleans up the file post-execution.
        """
        _ = (prompt, files, images)
        return InvocationPlan(kind="message_file", flag="--message-file")

    def get_read_only_args(self) -> list[str]:
        """Aider has a documented ``--dry-run`` flag that prevents file writes."""
        return ["--dry-run"]

    def render_model_args(self, model: str) -> list[str]:
        return ["--model", model]
