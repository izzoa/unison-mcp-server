"""Opencode-specific CLI agent hooks."""

from __future__ import annotations

from clink.parsers.base import ParserError

from .base import AgentOutput, BaseCLIAgent


class OpencodeAgent(BaseCLIAgent):
    """Opencode CLI agent with plan-mode read-only and JSONL recovery support."""

    model_flag_aliases: tuple[str, ...] = ("-m",)

    def get_read_only_args(self) -> list[str]:
        """Restrict opencode to its plan agent.

        Note: ``--agent plan`` references a user-configurable agent definition in
        ``~/.opencode/config.json``. Layer-1 enforcement is best-effort; layers 2
        (prompt instruction) and 3 (filesystem snapshot diff) provide
        defense-in-depth.
        """
        return ["--agent", "plan"]

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
