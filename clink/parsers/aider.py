"""Parser for Aider CLI plain-text output.

Aider's non-interactive output has a stable shape:

    [preamble: Warning / Aider vX.X.X / Model: ... / Git repo: ... / Repo-map: ...]
    Added <file> to the chat.

    [response prose]
    [optionally: filename then ```diff ... ``` blocks for edits]

    Tokens: X sent, Y received. Cost: $...
    [optionally: Applied edit to <file> lines for each edit]

We split on the ``Tokens:`` summary line — everything before is preamble +
response body, everything after lists the actual files Aider modified.
"""

from __future__ import annotations

from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError

# Substrings that identify preamble lines we strip from the response body.
# Matched as substring (``in``) because Aider's preamble has minor variations
# across versions but the anchor phrases are stable.
_PREAMBLE_MARKERS: tuple[str, ...] = (
    "Input is not a terminal",
    "You can skip this check",
    "Added .aider",  # ".gitignore" addition message
    "Aider v",
    "Main model:",
    "Model:",
    "Weak model:",
    "Editor model:",
    "Git repo:",
    "Repo-map:",
    "to the chat.",
    "https://aider.chat/",
    "Shell cwd was reset",  # appears at the very end of some runs
)

_TOKENS_LINE_PREFIX = "Tokens:"
_APPLIED_EDIT_PREFIX = "Applied edit to "


class AiderTextParser(BaseParser):
    """Parse plain-text stdout from ``aider --message-file ... --no-pretty --no-stream``."""

    name = "aider_text"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        lines = (stdout or "").splitlines()

        # Locate the Tokens: summary line — splits "preamble + body" from "applied edits".
        tokens_idx: int | None = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith(_TOKENS_LINE_PREFIX):
                tokens_idx = i
                break

        if tokens_idx is None:
            raise ParserError(
                "Aider output did not include a 'Tokens:' summary line — "
                "this usually means Aider failed before producing a response. "
                "Check stderr for errors."
            )

        # Body: lines before Tokens:, with preamble lines stripped.
        body_lines: list[str] = []
        for line in lines[:tokens_idx]:
            if _is_preamble(line):
                continue
            body_lines.append(line)

        response_text = "\n".join(body_lines).strip()

        # Edits: every "Applied edit to <path>" line that appears at or after the Tokens: line.
        edits: list[str] = []
        for line in lines[tokens_idx:]:
            if line.startswith(_APPLIED_EDIT_PREFIX):
                edits.append(line[len(_APPLIED_EDIT_PREFIX) :].strip())

        # Token-usage summary is the line at tokens_idx itself.
        usage_line = lines[tokens_idx].strip()

        if not response_text and not edits:
            raise ParserError(
                "Aider output contained no response text and no applied edits "
                "(only the Tokens: summary). This usually means the prompt "
                "produced an empty response."
            )

        # If Aider produced only edits with no prose, synthesize a brief summary.
        if not response_text:
            response_text = f"Edited {len(edits)} file(s): {', '.join(edits)}"

        metadata: dict[str, Any] = {
            "edits": edits,
            "usage_line": usage_line,
        }
        if stderr and stderr.strip():
            metadata["stderr"] = stderr.strip()

        return ParsedCLIResponse(content=response_text, metadata=metadata)


def _is_preamble(line: str) -> bool:
    """Return True if ``line`` matches any of the documented preamble markers."""
    return any(marker in line for marker in _PREAMBLE_MARKERS)
