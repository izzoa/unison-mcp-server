"""Opencode-specific CLI agent hooks."""

from __future__ import annotations

from clink.parsers.base import ParserError

from .base import AgentOutput, BaseCLIAgent


class OpencodeAgent(BaseCLIAgent):
    """Opencode CLI agent with JSONL recovery support.

    Read-only enforcement for opencode is via layer-2 (prompt instruction) and
    layer-3 (filesystem snapshot diff) only — opencode has no CLI flag for
    read-only-while-still-executing semantics. ``--agent plan`` was tried in
    v11.8.0 but switches the agent persona (producing planning-language instead
    of executing the requested task), so it is not a true read-only sandbox.
    """

    model_flag_aliases: tuple[str, ...] = ("-m",)

    # Paths opencode creates as bookkeeping on first-run bootstrap in any
    # working directory. Captured against opencode v0.18.x — see the regression
    # fixture in tests/test_clink_opencode_parser.py for the source. NOT a
    # directory-wide glob over .opencode/ because .opencode/skills/ and
    # .opencode/commands/ are user-extensions where model writes MUST classify
    # as by_model, not bookkeeping.
    fs_violation_ignore_patterns: tuple[str, ...] = (
        ".opencode/.gitignore",
        ".opencode/package.json",
        ".opencode/package-lock.json",
        ".opencode/node_modules/**",
        ".git/opencode",
    )

    def get_read_only_args(self) -> list[str]:
        """Opencode has no CLI flag for read-only-while-still-executing mode.

        ``--agent plan`` was tried in v11.8.0 but switches the agent persona to
        opencode's plan agent (which produces planning-language instead of
        executing the requested task), so it is not a true read-only sandbox.
        Layer-1 enforcement is intentionally absent for opencode; layer-2
        (prompt instruction) and layer-3 (filesystem snapshot diff) provide
        enforcement, both of which are CLI-agnostic.
        """
        return []

    def render_model_args(self, model: str) -> list[str]:
        return ["-m", model]

    def _recover_from_error(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        sanitized_command: list[str],
        duration_seconds: float,
        output_file_content: str | None,
    ) -> AgentOutput | None:
        try:
            parsed = self._parser.parse(stdout, stderr)
        except ParserError:
            return None

        return AgentOutput(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration_seconds,
            parser_name=self._parser.name,
            output_file_content=output_file_content,
        )
