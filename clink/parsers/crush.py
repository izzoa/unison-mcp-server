"""Parser for Crush CLI plain-text output.

Crush's ``crush run --quiet`` produces remarkably clean output: the model's
response text directly on stdout, no preamble, no token-usage footer, no
chrome of any kind. Errors go to stderr in a Charm-styled box with the word
"ERROR" and a colon-prefixed message.

The parser is correspondingly minimal — return stripped stdout as content,
capture any stderr as diagnostics metadata.
"""

from __future__ import annotations

from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


class CrushTextParser(BaseParser):
    """Parse plain-text stdout from ``crush run --quiet``."""

    name = "crush_text"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        content = (stdout or "").strip()
        stderr_stripped = (stderr or "").strip()

        if not content:
            # Empty stdout with a non-zero exit code is handled by
            # BaseCLIAgent.run; if we reach the parser with empty stdout,
            # surface the stderr message (Crush's error format) as the
            # parse error so the user sees what happened.
            if stderr_stripped:
                raise ParserError(f"Crush produced no stdout content. stderr: {stderr_stripped}")
            raise ParserError("Crush produced no stdout content and no stderr message.")

        metadata: dict[str, Any] = {}
        if stderr_stripped:
            metadata["stderr"] = stderr_stripped

        return ParsedCLIResponse(content=content, metadata=metadata)
